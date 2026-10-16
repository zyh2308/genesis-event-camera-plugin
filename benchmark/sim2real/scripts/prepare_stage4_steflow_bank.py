#!/usr/bin/env python3
"""Create fixed 30 ms STE windows and paired training/validation manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from prepare_steflow_stage0 import gaussian_count_representation


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def load_raw_events(generator: str, group: h5py.Group, v2e_root: Path, rollout: str) -> np.ndarray:
    if generator == "eventgenesis":
        raw = np.asarray(group["eventgenesis_events_txyp"], dtype=np.float64)
        return raw.reshape(-1, 4)
    path = v2e_root / f"{rollout}.h5"
    with h5py.File(path, "r") as source:
        raw = np.asarray(source["events"], dtype=np.uint32)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError(f"V2E {path} expected [N,4], got {raw.shape}")
    return np.stack((raw[:, 0].astype(np.float64) * 1e-6,
                     raw[:, 1].astype(np.float64), raw[:, 2].astype(np.float64),
                     np.where(raw[:, 3] > 0, 1.0, -1.0)), axis=1)


def convert_rollout(generator: str, rollout: str, shared: h5py.File, v2e_root: Path, out: Path) -> dict:
    group = shared[f"episodes/{rollout}"]
    times = np.asarray(group["frame_timestamps_s"], dtype=np.float64)
    if times.shape != (201,) or not np.allclose(np.diff(times), .01, atol=1e-7, rtol=0):
        raise ValueError(f"{rollout}: expected 201 uniform 100 Hz endpoints")
    events = load_raw_events(generator, group, v2e_root, rollout)
    if len(events) and (not np.isfinite(events).all() or not np.all(np.diff(events[:, 0]) >= 0)):
        raise ValueError(f"{generator}/{rollout}: raw events are nonfinite or unsorted")
    height, width = np.asarray(group["v2e_display_luma_u8"]).shape[1:]
    if len(events) and (events[:, 1].min() < 0 or events[:, 1].max() >= width or events[:, 2].min() < 0 or events[:, 2].max() >= height):
        raise ValueError(f"{generator}/{rollout}: event coordinates out of bounds")
    # The fixed 66-window protocol spans 0.00--1.98 s (66 nonoverlapping
    # 30 ms windows); the final 20 ms of a 2 s rollout is a registered tail.
    tail_endpoint = float(times[198])
    discarded_tail_events = int(np.count_nonzero((events[:, 0] > tail_endpoint) & (events[:, 0] <= times[-1])))
    keep = (events[:, 0] > times[0]) & (events[:, 0] <= tail_endpoint)
    events = events[keep]
    cursor = 0
    reps, counts, lefts, rights = [], [], [], []
    window_indices = list(range(0, 198, 3))
    for i in window_indices:
        left, right = float(times[i]), float(times[i + 3])
        stop = int(np.searchsorted(events[:, 0], right, side="right"))
        window = events[cursor:stop]
        if len(window) and (np.any(window[:, 0] <= left) or np.any(window[:, 0] > right)):
            raise RuntimeError(f"{generator}/{rollout}: event window violates (t_i,t_i+3] at {i}")
        # Independently count directly from the complete raw timestamp array,
        # rather than trusting generator interval counters or cursor offsets.
        direct_begin = int(np.searchsorted(events[:, 0], left, side="right"))
        direct_end = int(np.searchsorted(events[:, 0], right, side="right"))
        if len(window) != direct_end - direct_begin:
            raise RuntimeError(f"{generator}/{rollout}: direct timestamp recount disagrees at sample {i // 3}")
        local = window.copy()
        if len(local):
            local[:, 0] -= left
        reps.append(gaussian_count_representation(local, height, width))
        counts.append(int(len(window))); lefts.append(left); rights.append(right)
        cursor = stop
    if cursor != len(events):
        raise RuntimeError(f"{generator}/{rollout}: {len(events)-cursor} events were not assigned")
    if len(reps) != 66:
        raise RuntimeError(f"{rollout}: expected 66 stride-3 windows, got {len(reps)}")

    rgb_gray = np.asarray(group["v2e_display_luma_u8"], dtype=np.float32) / 255.0
    gray_prev = rgb_gray[np.asarray(window_indices)]
    gray_next = rgb_gray[np.asarray(window_indices) + 3]
    flow = np.asarray(group["flow_forward_30ms_px"], dtype=np.float32)
    valid = np.asarray(group["flow_valid_30ms"], dtype=np.uint8)
    if flow.shape != (66, 2, height, width) or valid.shape != (66, height, width):
        raise ValueError(f"{rollout}: shared flow contract shape mismatch")
    if not np.isfinite(flow).all() or not np.isin(valid, (0, 1)).all():
        raise ValueError(f"{rollout}: invalid shared flow labels/mask")
    flow = flow.copy(); flow *= valid[:, None].astype(np.float32)
    label_sha = hashlib.sha256((array_hash(flow) + array_hash(valid)).encode()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, events=np.stack(reps), gray_prev=gray_prev, gray_next=gray_next,
        flow_gt=flow, flow_valid=valid, event_counts=np.asarray(counts, dtype=np.int32),
        event_start_s=np.asarray(lefts, dtype=np.float64), event_end_s=np.asarray(rights, dtype=np.float64),
        sample_ids=np.asarray([f"{rollout}/dt1_{i:03d}" for i in window_indices]),
        resolution_hw=np.asarray([height, width], dtype=np.int32),
        camera_intrinsics_K_px=np.asarray(group["camera_intrinsics_K_px"], dtype=np.float64),
        label_sha256=np.asarray(label_sha),
    )
    return {
        "generator": generator, "rollout_id": rollout, "split": str(group.attrs["split"]),
        "stratum": str(group.attrs["stratum"]), "path": str(out), "sha256": sha256(out),
        "samples": len(reps), "events_total_in_training_horizon": int(len(events)),
        "events_in_registered_20ms_tail_not_used": discarded_tail_events,
        "event_counts_sha256": array_hash(np.asarray(counts, dtype=np.int32)),
        "window_seconds": .03, "window_rule": "(t_i,t_i+3]", "label_sha256": label_sha,
        "flow_shape": list(flow.shape), "representation_shape": [len(reps), 2, height, width, 5],
        "representation_dtype": "uint8", "valid_pixels_mean": float(valid.reshape(len(valid), -1).sum(axis=1).mean()),
        "sample_id_first_last": [f"{rollout}/dt1_000", f"{rollout}/dt1_195"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank-root", type=Path, required=True)
    p.add_argument("--v2e-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    bank, v2e, out = args.bank_root.resolve(), args.v2e_root.resolve(), args.output_root.resolve()
    if not (bank / "bank_manifest.json").is_file() or not (bank / "shared_trajectory.h5").is_file():
        raise RuntimeError("refusing adapter preparation before complete immutable shared bank manifest")
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    with h5py.File(bank / "shared_trajectory.h5", "r") as shared:
        episode_ids = sorted(shared["episodes"].keys())
        if len(episode_ids) != 32:
            raise RuntimeError(f"expected 32 rollout groups, got {len(episode_ids)}")
        for generator in ("eventgenesis", "v2e"):
            for rollout in episode_ids:
                split = str(shared[f"episodes/{rollout}"].attrs["split"])
                path = out / generator / split / f"{rollout}.npz"
                entries.append(convert_rollout(generator, rollout, shared, v2e, path))
    lookup = {(e["generator"], e["rollout_id"]): e for e in entries}
    for rollout in sorted({e["rollout_id"] for e in entries}):
        left, right = lookup[("eventgenesis", rollout)], lookup[("v2e", rollout)]
        if left["label_sha256"] != right["label_sha256"]:
            raise RuntimeError(f"shared labels differ across generators for {rollout}")
    train_n = sum(e["samples"] for e in entries if e["generator"] == "eventgenesis" and e["split"] == "train")
    val_n = sum(e["samples"] for e in entries if e["generator"] == "eventgenesis" and e["split"] == "validation")
    if (train_n, val_n) != (1584, 528):
        raise RuntimeError(f"fixed sample counts mismatch: train={train_n}, validation={val_n}")
    manifest = {
        "schema": "stage4_steflow_30ms_windows_v1", "shared_bank_sha256": sha256(bank / "shared_trajectory.h5"),
        "split_sha256": sha256(bank / "registered_split.json"), "raw_v2e_manifest": str(v2e / "v2e_complete_manifest.json"),
        "train_windows_per_generator": train_n, "validation_windows_per_generator": val_n,
        "paired_labels_identical": True, "event_count_filtering": "none",
        "event_window": "(t_i,t_i+3] on shared 100Hz endpoints; starts at i=0,3,...,195; the remaining 20ms of each 2s rollout is excluded by the registered 66-window count",
        "flow_unit": "pixels per 30ms endpoint interval", "entries": entries,
    }
    path = out / "training_bank_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["manifest_sha256"] = sha256(path)
    (out / "training_bank_manifest.sha256").write_text(manifest["manifest_sha256"] + "  training_bank_manifest.json\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, indent=2))


if __name__ == "__main__":
    main()
