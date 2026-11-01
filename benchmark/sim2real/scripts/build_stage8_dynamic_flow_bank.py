#!/usr/bin/env python3
"""Bind the same Genesis camera/object rollout and GT flow to EventGenesis and V2E."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_digest(flow: np.ndarray, valid: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(flow, dtype=np.float32).view(np.uint8))
    digest.update(np.ascontiguousarray(valid, dtype=np.uint8).view(np.uint8))
    return digest.hexdigest()


def load_events(episode: dict, arm: str) -> np.ndarray:
    motion = Path(episode["motion_dir"])
    if arm == "eventgenesis":
        raw = np.asarray(np.load(motion / "eventgenesis_events_seconds_xy_p.npy", allow_pickle=False), dtype=np.float64)
        events = raw.reshape(-1, 4)
    else:
        event_path = Path(episode["v2e_h5"])
        with h5py.File(event_path, "r") as h5:
            raw = np.asarray(h5["events"], dtype=np.uint32)
        events = np.stack((raw[:, 0].astype(np.float64) * 1e-6,
                           raw[:, 1].astype(np.float64), raw[:, 2].astype(np.float64),
                           np.where(raw[:, 3] > 0, 1.0, -1.0)), axis=1) if len(raw) else np.empty((0, 4), dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != 4 or not np.isfinite(events).all():
        raise ValueError(f"{arm} events must be finite [N,4], got {events.shape}")
    if len(events) and np.any(np.diff(events[:, 0]) < 0):
        raise ValueError(f"{arm} events are not timestamp-sorted")
    h, w = np.load(motion / "depth_frames.npy", mmap_mode="r").shape[1:]
    keep = (events[:, 0] > 0.0) & (events[:, 0] <= 0.03)
    events = events[keep]
    if len(events):
        if np.any((events[:, 1] < 0) | (events[:, 1] >= w) | (events[:, 2] < 0) | (events[:, 2] >= h)):
            raise ValueError(f"{arm} event coordinate outside registered {w}x{h} raster")
        if not np.isin(events[:, 3], (-1, 1)).all():
            raise ValueError(f"{arm} polarity did not map to -1/+1")
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo / "benchmark/sim2real/scripts"))
    from prepare_steflow_stage0 import gaussian_count_representation

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    episodes = config["data"]["episodes"]
    if len(episodes) < 2 or {row["split"] for row in episodes} != {"train", "validation"}:
        raise ValueError("bank requires at least one train and one independent validation rollout")
    if config.get("real_data_used") is not False:
        raise ValueError("this diagnostic bank must be synthetic only")
    out.mkdir(parents=True)
    entries, audits, shared_hashes = [], [], {}
    for episode in episodes:
        rollout_id = episode["rollout_id"]
        if rollout_id in shared_hashes:
            raise ValueError(f"duplicate rollout ID {rollout_id}")
        motion = Path(episode["motion_dir"]).resolve()
        required = [motion / name for name in (
            "radiance_frames.npy", "depth_frames.npy", "segmentation_physical_link_ids.npy",
            "gt_flow_px_t0_to_t30ms.npy", "gt_flow_valid.npy",
            "eventgenesis_events_seconds_xy_p.npy", "shared_intrinsics_pixel_index.npy",
            "frame_timestamps_s.npy", "moving_object_poses_cv.npy", "moving_link_id.txt",
        )]
        missing = [str(p) for p in required if not p.is_file()]
        v2e_path = Path(episode["v2e_h5"]).resolve()
        if missing or not v2e_path.is_file():
            raise FileNotFoundError(f"{rollout_id}: missing {missing + ([] if v2e_path.is_file() else [str(v2e_path)])}")
        radiance = np.load(motion / "radiance_frames.npy", allow_pickle=False)
        depth = np.load(motion / "depth_frames.npy", allow_pickle=False)
        seg = np.load(motion / "segmentation_physical_link_ids.npy", allow_pickle=False)
        flow = np.asarray(np.load(motion / "gt_flow_px_t0_to_t30ms.npy", allow_pickle=False), dtype=np.float32)
        valid = np.asarray(np.load(motion / "gt_flow_valid.npy", allow_pickle=False), dtype=np.uint8)
        times = np.asarray(np.load(motion / "frame_timestamps_s.npy", allow_pickle=False), dtype=np.float64)
        if radiance.shape != (4, 240, 320, 3) or radiance.dtype != np.float32 or not np.isfinite(radiance).all():
            raise ValueError(f"{rollout_id}: malformed shared radiance {radiance.shape}/{radiance.dtype}")
        if depth.shape != (4, 240, 320) or seg.shape != depth.shape or flow.shape != (2, 240, 320) or valid.shape != (240, 320):
            raise ValueError(f"{rollout_id}: mismatched depth/seg/flow/mask shapes")
        if not np.allclose(times, [0, .01, .02, .03], atol=1e-7, rtol=0):
            raise ValueError(f"{rollout_id}: expected exact [0,10,20,30] ms endpoints, got {times.tolist()}")
        label_sha = label_digest(flow, valid)
        shared_hashes[rollout_id] = label_sha
        supports, raw_counts, artifacts = {}, {}, {}
        for arm in ("eventgenesis", "v2e"):
            raw_events = load_events(episode, arm)
            representation = gaussian_count_representation(raw_events, 240, 320)
            if representation.shape != (2, 240, 320, 5):
                raise RuntimeError(f"unexpected STE event representation {representation.shape}")
            supports[arm] = np.any(representation != 0, axis=(0, 3))
            raw_counts[arm] = int(len(raw_events))
            arm_dir = out / arm
            arm_dir.mkdir(exist_ok=True)
            artifact = arm_dir / f"{rollout_id}.npz"
            np.savez_compressed(
                artifact, events=representation, flow_gt=flow, flow_valid=valid,
                sample_ids=np.asarray(f"{rollout_id}/dt30"), label_sha256=np.asarray(label_sha),
                event_count=np.asarray(len(raw_events), dtype=np.int64),
            )
            artifacts[arm] = artifact
            entries.append({
                "generator": arm, "rollout_id": rollout_id, "scene_id": episode["scene_id"],
                "split": episode["split"], "sample_id": f"{rollout_id}/dt30",
                "path": str(artifact), "sha256": sha256(artifact),
                "size_bytes": artifact.stat().st_size, "samples": 1,
                "event_count": len(raw_events), "shared_label_sha256": label_sha,
                "source_radiance_sha256": sha256(motion / "radiance_frames.npy"),
                "source_flow_sha256": sha256(motion / "gt_flow_px_t0_to_t30ms.npy"),
                "source_validity_sha256": sha256(motion / "gt_flow_valid.npy"),
                "source_event_artifact_sha256": sha256(motion / "eventgenesis_events_seconds_xy_p.npy") if arm == "eventgenesis" else sha256(v2e_path),
                "event_only_inference": True,
            })
        if not np.array_equal(np.load(artifacts["eventgenesis"]) ["flow_gt"], np.load(artifacts["v2e"]) ["flow_gt"]):
            raise RuntimeError(f"paired generator artifacts have different flow labels: {rollout_id}")
        active_union = supports["eventgenesis"] | supports["v2e"]
        valid_bool = valid.astype(bool)
        magnitude = np.hypot(flow[0], flow[1])
        motion_mask = valid_bool & (magnitude >= .5)
        luma = .2126 * radiance[..., 0] + .7152 * radiance[..., 1] + .0722 * radiance[..., 2]
        gradients = [np.hypot(*np.gradient(frame)) for frame in luma]
        textured = (np.mean(np.stack(gradients), axis=0) > .005)
        textured_motion = motion_mask & textured
        audits.append({
            "rollout_id": rollout_id, "scene_id": episode["scene_id"], "split": episode["split"],
            "flow_unit": "pixels per 30ms endpoint interval", "flow_valid_pixels": int(valid_bool.sum()),
            "gt_motion_ge_0_5px_pixels": int(motion_mask.sum()),
            "texture_and_motion_pixels": int(textured_motion.sum()),
            "event_counts": raw_counts,
            "active_pixels": {arm: int(supports[arm].sum()) for arm in supports},
            "paired_union_active_valid_pixels": int((active_union & valid_bool).sum()),
            "paired_union_active_motion_pixels": int((active_union & motion_mask).sum()),
            "paired_union_active_texture_motion_pixels": int((active_union & textured_motion).sum()),
            "union_coverage_valid": float((active_union & valid_bool).sum() / max(1, valid_bool.sum())),
            "union_coverage_gt_motion": float((active_union & motion_mask).sum() / max(1, motion_mask.sum())),
            "union_coverage_texture_motion": float((active_union & textured_motion).sum() / max(1, textured_motion.sum())),
            "gt_magnitude_px_quantiles_valid": np.percentile(magnitude[valid_bool], [0, 25, 50, 75, 90, 99, 100]).tolist() if valid_bool.any() else [],
            "label_sha256": label_sha,
            "source_run_id": episode["run_id"],
            "source_motion_dir": str(motion),
            "source_v2e_h5": str(v2e_path),
        })
    train_ids = sorted({e["rollout_id"] for e in episodes if e["split"] == "train"})
    val_ids = sorted({e["rollout_id"] for e in episodes if e["split"] == "validation"})
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train and validation rollout IDs overlap")
    manifest = {
        "schema": "stage8_replicad_dynamic_flow_training_bank_v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "frozen_simulator_core": "e79e656ac00f7847b98db9e12bf43b093f3afff1",
        "flow_unit": "pixels per 30ms endpoint interval",
        "event_representation": "pinned STE equal-event-count five-bin Gaussian-count representation, separately per polarity",
        "event_window": "left-open, right-closed (0,30ms]",
        "label_contract": "shared per-link SE3 renderer-depth labels with target-depth visibility check; identical for EventGenesis and V2E",
        "split_unit": "independent Genesis rollout and layout family; no adjacent-window split",
        "train_rollouts": train_ids, "validation_rollouts": val_ids,
        "real_data_used": False, "entries": entries, "coverage_audit": audits,
    }
    (out / "training_bank_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "manifest": str(out / "training_bank_manifest.json"),
                      "train_rollouts": train_ids, "validation_rollouts": val_ids,
                      "coverage": audits}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
