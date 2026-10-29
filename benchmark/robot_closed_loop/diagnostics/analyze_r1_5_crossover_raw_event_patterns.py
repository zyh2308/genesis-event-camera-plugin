#!/usr/bin/env python3
"""Post-hoc raw-event and saved-input characterization for R1.5 crossover.

This script does not regenerate events, alter model inputs, or run inference.
Segmentation is consulted only after event generation to describe event regions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PERIOD_S = 0.010
PHASE_BIN_S = 0.0005


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_queries(path: Path) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["source"] != "direct_hdr" or row["checkpoint"] != "v2e":
                continue
            out.setdefault(row["episode"], []).append(row)
    for rows in out.values():
        rows.sort(key=lambda row: float(row["query_s"]))
    return out


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else (1.0 if not np.any(a) and not np.any(b) else 0.0)


def describe_stream(events: np.ndarray, timestamps: np.ndarray,
                    segmentations: np.ndarray, target_id: int,
                    robot_ids: set[int], start_s: float) -> dict[str, Any]:
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"expected event shape (N,4), got {events.shape}")
    if len(events) == 0:
        return {"event_count": 0}
    t, x0, y0, p = events.T
    x, y = np.rint(x0).astype(np.int64), np.rint(y0).astype(np.int64)
    h, w = segmentations.shape[1:]
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    safe_x, safe_y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    frame_idx = np.abs(t[:, None] - timestamps[None, :]).argmin(axis=1)
    labels = segmentations[frame_idx, safe_y, safe_x]
    spatial = np.full(len(events), "static_or_scene", dtype=object)
    spatial[~in_bounds] = "out_of_bounds"
    spatial[labels == target_id] = "target"
    spatial[np.isin(labels, list(robot_ids))] = "robot"
    spatial[(labels == 0) & in_bounds] = "unlabeled"

    phase = np.mod(t - start_s, PERIOD_S)
    phase_bin = np.floor(phase / PHASE_BIN_S).astype(np.int64)
    phase_bin = np.clip(phase_bin, 0, int(PERIOD_S / PHASE_BIN_S) - 1)
    period_bin = np.floor((t - start_s) / PERIOD_S).astype(np.int64)
    five_ms_bin = np.floor((t - start_s) / 0.005).astype(np.int64)
    polarity = np.where(p > 0, "on", "off")

    class_counts = {name: int(np.count_nonzero(spatial == name)) for name in
                    ("target", "robot", "static_or_scene", "unlabeled", "out_of_bounds")}
    class_polarity = {
        name: {pol: int(np.count_nonzero((spatial == name) & (polarity == pol)))
               for pol in ("on", "off")}
        for name in ("target", "robot", "static_or_scene", "unlabeled", "out_of_bounds")
    }

    # Count consecutive polarity reversals per pixel after stable time sorting.
    order = np.lexsort((t, y, x))
    xs, ys, ps = x[order], y[order], p[order]
    same_pixel_next = (xs[1:] == xs[:-1]) & (ys[1:] == ys[:-1])
    reversals = int(np.count_nonzero(same_pixel_next & (ps[1:] != ps[:-1])))
    pixel_code = safe_y.astype(np.int64) * w + safe_x.astype(np.int64)
    unique_pixels = int(np.unique(pixel_code[in_bounds]).size)
    return {
        "event_count": int(len(events)),
        "time_range_s": [float(t.min()), float(t.max())],
        "unique_event_timestamps": int(np.unique(t).size),
        "unique_event_pixels": unique_pixels,
        "polarity_counts": {pol: int(np.count_nonzero(polarity == pol)) for pol in ("on", "off")},
        "consecutive_same_pixel_polarity_reversals": reversals,
        "events_by_posthoc_region": class_counts,
        "events_by_posthoc_region_and_polarity": class_polarity,
        "events_per_0p5ms_phase_bin_of_10ms_native_endpoint_period":
            np.bincount(phase_bin, minlength=20).astype(int).tolist(),
        "events_per_10ms_interval": {
            str(int(i)): int(np.count_nonzero(period_bin == i))
            for i in range(max(0, int(period_bin.max()) + 1))
        },
        "events_per_5ms_bin": {
            str(int(i)): int(np.count_nonzero(five_ms_bin == i))
            for i in range(max(0, int(five_ms_bin.max()) + 1))
        },
        "nearest_direct_state_frame_dt_ms": {
            "median": float(np.median(np.abs(t - timestamps[frame_idx])) * 1000.0),
            "p95": float(np.percentile(np.abs(t - timestamps[frame_idx]), 95) * 1000.0),
            "max": float(np.max(np.abs(t - timestamps[frame_idx])) * 1000.0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.result_root.resolve()
    out = (args.output_dir.resolve() if args.output_dir else root / "posthoc_raw_event_patterns")
    out.mkdir(parents=True, exist_ok=False)
    all_queries = read_queries(root / "all_four_group_queries.csv")
    summary: dict[str, Any] = {
        "status": "completed",
        "scope": "post-hoc characterization of saved raw events and saved five-bin inputs only",
        "segmentation_boundary": "post-inference region labels; segmentation is not an event/model input",
        "region_label_caveat": "static_or_scene groups every visible non-target, non-robot segmentation ID; it is not a shadow mask",
        "phase_histogram_definition": {
            "period_s": PERIOD_S,
            "bin_width_s": PHASE_BIN_S,
            "anchor_s": "episode direct-HDR trace start",
            "purpose": "describe event timing relative to the archived 100 Hz endpoint cadence; not an interpolation sample schedule",
        },
        "episodes": {},
        "input_source_cosine": {},
        "input_source_cosine_csv": "input_source_cosine_by_query.csv",
        "input_file_hashes": {},
    }
    input_rows: list[dict[str, Any]] = []

    for episode_dir in sorted((root / "episodes").glob("episode_*")):
        episode = episode_dir.name.split("_", 2)[2]
        queries = all_queries[episode]
        direct_h5 = episode_dir / "direct_hdr_exact_state_frames.h5"
        input_npz = episode_dir / "four_group_inputs_heatmaps.npz"
        native_npz = episode_dir / "raw_events_ours_native_hdr.npz"
        direct_npz = episode_dir / "raw_events_direct_hdr.npz"
        with h5py.File(direct_h5, "r") as h5:
            ts = np.asarray(h5["timestamps_s"][:], dtype=np.float64)
            seg = np.asarray(h5["segmentation_post_inference_only"][:], dtype=np.int32)
            pose_ids = set(map(int, np.asarray(h5.attrs["segmentation_pose_ids"]).tolist()))

        # Identify the target segment from small post-hoc crops around the known
        # projected target center; the robust majority vote is repeated per episode.
        target_votes: list[int] = []
        for row in queries:
            qi = int(np.argmin(np.abs(ts - float(row["query_s"]))))
            u, v = int(round(float(row["gt_u_px"]))), int(round(float(row["gt_v_px"])))
            crop = seg[qi, max(0, v - 7):min(seg.shape[1], v + 8),
                       max(0, u - 7):min(seg.shape[2], u + 8)]
            vals, counts = np.unique(crop[crop != 0], return_counts=True)
            if len(vals):
                target_votes.append(int(vals[np.argmax(counts)]))
        if not target_votes:
            raise RuntimeError(f"could not identify target segment for {episode}")
        target_id = int(np.bincount(np.asarray(target_votes, dtype=np.int64)).argmax())
        robot_ids = pose_ids - {target_id}
        episode_result: dict[str, Any] = {
            "episode": episode,
            "exact_state_frames": int(len(ts)),
            "state_dt_ms": {
                "min": float(np.diff(ts).min() * 1000),
                "median": float(np.median(np.diff(ts)) * 1000),
                "max": float(np.diff(ts).max() * 1000),
            },
            "posthoc_target_segmentation_id": target_id,
            "posthoc_robot_segmentation_ids": sorted(robot_ids),
            "target_id_vote_fraction": float(np.mean(np.asarray(target_votes) == target_id)),
            "streams": {},
        }
        for source, path in (("ours_native_hdr", native_npz), ("direct_hdr", direct_npz)):
            events = np.asarray(np.load(path)["events"], dtype=np.float64)
            episode_result["streams"][source] = describe_stream(
                events, ts, seg, target_id, robot_ids, float(ts[0]))
            summary["input_file_hashes"][str(path.relative_to(root))] = sha256(path)
        for path in (direct_h5, input_npz):
            summary["input_file_hashes"][str(path.relative_to(root))] = sha256(path)

        with np.load(input_npz) as data:
            count_key = "__voxel_counts"
            keys = data.files
            source_keys = {
                source: [k for k in keys if k.startswith(source + "__") and k.endswith(count_key)]
                for source in ("ours_native_hdr", "direct_hdr")
            }
            if any(len(v) != 2 for v in source_keys.values()):
                raise RuntimeError(f"unexpected saved input keys for {episode}: {source_keys}")
            for checkpoint in ("ours_native_hdr", "v2e"):
                native_key = f"ours_native_hdr__{checkpoint}__voxel_counts"
                direct_key = f"direct_hdr__{checkpoint}__voxel_counts"
                native_counts = np.asarray(data[native_key], dtype=np.float32)
                direct_counts = np.asarray(data[direct_key], dtype=np.float32)
                for qi in range(native_counts.shape[0]):
                    row = {
                        "episode": episode,
                        "query_index": qi,
                        "query_s": float(data["query_times_s"][qi]),
                        "checkpoint": checkpoint,
                        "five_bin_count_tensor_cosine": safe_cosine(native_counts[qi], direct_counts[qi]),
                    }
                    per_bin = [safe_cosine(native_counts[qi, b], direct_counts[qi, b]) for b in range(5)]
                    row["bin_0_cosine"], row["bin_1_cosine"], row["bin_2_cosine"], row["bin_3_cosine"], row["bin_4_cosine"] = per_bin
                    row["native_nonzero_bins"] = int(np.count_nonzero(np.sum(native_counts[qi], axis=(1, 2, 3))))
                    row["direct_nonzero_bins"] = int(np.count_nonzero(np.sum(direct_counts[qi], axis=(1, 2, 3))))
                    input_rows.append(row)
        summary["episodes"][episode] = episode_result

    cosine_values = [r["five_bin_count_tensor_cosine"] for r in input_rows]
    summary["input_source_cosine"] = {
        "query_rows": len(input_rows),
        "mean": float(np.mean(cosine_values)),
        "p50": float(np.median(cosine_values)),
        "p05": float(np.percentile(cosine_values, 5)),
        "min": float(np.min(cosine_values)),
        "interpretation": "paired cosine between saved raw-count voxel tensors at identical query times, stratified by frozen checkpoint; not a performance metric",
    }
    with (out / "input_source_cosine_by_query.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(input_rows[0]))
        writer.writeheader()
        writer.writerows(input_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in manifest.items()), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "episodes": list(summary["episodes"]),
                      "input_cosine": summary["input_source_cosine"],
                      "summary_sha256": sha256(out / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
