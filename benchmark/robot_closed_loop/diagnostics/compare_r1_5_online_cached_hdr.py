#!/usr/bin/env python3
"""Compare online event-source HDR frames with cached-state renderer frames.

No Genesis renderer, physics, checkpoint, segmentation, or depth is used here.
The comparison joins only identical timestamps and measures linear HDR and
log-luma differences; raw event matching is an auxiliary diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"mean": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)), "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def log_luma(rgb: np.ndarray) -> np.ndarray:
    luminance = (np.asarray(rgb, dtype=np.float64)[..., 0] * 0.2126 +
                 np.asarray(rgb, dtype=np.float64)[..., 1] * 0.7152 +
                 np.asarray(rgb, dtype=np.float64)[..., 2] * 0.0722)
    return np.log(np.maximum(luminance, 1e-6))


def event_match(reference: np.ndarray, candidate: np.ndarray,
                tolerance_s: float = 0.0005) -> dict[str, Any]:
    groups: list[dict[tuple[int, int, int], list[float]]] = [{}, {}]
    for target, events in zip(groups, (reference, candidate)):
        for time_s, x, y, polarity in events:
            target.setdefault((int(x), int(y), int(polarity)), []).append(float(time_s))
    matched = 0
    time_deltas: list[float] = []
    for key in groups[0].keys() & groups[1].keys():
        left, right = sorted(groups[0][key]), sorted(groups[1][key])
        i = j = 0
        while i < len(left) and j < len(right):
            delta = left[i] - right[j]
            if abs(delta) <= tolerance_s:
                matched += 1
                time_deltas.append(abs(delta) * 1000.0)
                i += 1
                j += 1
            elif delta < 0:
                i += 1
            else:
                j += 1
    return {
        "online_events": int(len(reference)), "cached_events": int(len(candidate)),
        "matched": int(matched),
        "online_match_fraction": None if not len(reference) else matched / len(reference),
        "cached_match_fraction": None if not len(candidate) else matched / len(candidate),
        "same_pixel_polarity_tolerance_ms": tolerance_s * 1000.0,
        "matched_abs_time_ms": stats(np.asarray(time_deltas, dtype=np.float64)),
    }


def load_frames(path: Path, *, time_key: str, frame_key: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        times = np.asarray(archive[time_key], dtype=np.float64)
        frames = np.asarray(archive[frame_key], dtype=np.float32)
    if frames.ndim != 4 or frames.shape[0] != len(times) or frames.shape[-1] != 3:
        raise ValueError(f"unexpected frames in {path}: times={times.shape}, frames={frames.shape}")
    if not np.isfinite(frames).all() or np.any(frames < 0):
        raise ValueError(f"non-finite or negative HDR in {path}")
    return times, frames


def load_events(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        events = np.asarray(archive[key], dtype=np.float64).reshape(-1, 4)
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-frames", type=Path, required=True)
    parser.add_argument("--online-events", type=Path, required=True)
    parser.add_argument("--cached-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-s", type=float, nargs=2, default=(0.5505, 0.5905))
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start_s, end_s = map(float, args.window_s)
    if start_s >= end_s:
        raise ValueError("window start must be less than end")

    online_times, online_frames = load_frames(
        args.online_frames, time_key="times_s", frame_key="hdr_rgb")
    with np.load(args.online_events, allow_pickle=False) as archive:
        online_events = np.asarray(archive["events_2000hz"], dtype=np.float64).reshape(-1, 4)
    window_events = online_events[(online_events[:, 0] >= start_s) & (online_events[:, 0] <= end_s)]
    frame_window = (online_times >= start_s - 1e-9) & (online_times <= end_s + 1e-9)
    online_times = online_times[frame_window]
    online_frames = online_frames[frame_window]
    if len(online_times) != 81:
        raise ValueError(f"expected 81 online HDR frames in window, found {len(online_times)}")

    conditions = {
        "msaa_on_sparse47": "frames_msaa_on_sparse47.npz",
        "msaa_on_dense2khz": "frames_msaa_on_dense2khz.npz",
        "msaa_off_dense2khz": "frames_msaa_off_dense2khz.npz",
    }
    rows: list[dict[str, Any]] = []
    condition_summary: dict[str, Any] = {}
    for name, filename in conditions.items():
        cached_path = args.cached_root / filename
        cached_times, cached_frames = load_frames(
            cached_path, time_key="times_s", frame_key="hdr_rgb")
        indices: list[int] = []
        online_indices: list[int] = []
        for index, time_s in enumerate(cached_times):
            matches = np.flatnonzero(np.abs(online_times - time_s) <= 1e-9)
            if len(matches) != 1:
                raise ValueError(f"{name}: expected one online frame at {time_s}, found {len(matches)}")
            indices.append(index)
            online_indices.append(int(matches[0]))
        online_subset = online_frames[online_indices]
        cached_subset = cached_frames[indices]
        hdr_delta = np.abs(online_subset.astype(np.float64) - cached_subset.astype(np.float64))
        online_log = np.stack([log_luma(frame) for frame in online_subset])
        cached_log = np.stack([log_luma(frame) for frame in cached_subset])
        log_delta = np.abs(online_log - cached_log)
        per_frame: list[dict[str, Any]] = []
        for i, (time_s, hdr_frame, log_frame) in enumerate(zip(cached_times, hdr_delta, log_delta)):
            record = {
                "condition": name, "time_s": float(time_s),
                "hdr_abs_mean": float(hdr_frame.mean()),
                "hdr_abs_p99": float(np.percentile(hdr_frame, 99)),
                "hdr_abs_max": float(hdr_frame.max()),
                "log_luma_abs_mean": float(log_frame.mean()),
                "log_luma_abs_p95": float(np.percentile(log_frame, 95)),
                "log_luma_abs_p99": float(np.percentile(log_frame, 99)),
                "log_luma_abs_max": float(log_frame.max()),
                "log_luma_fraction_over_0_2": float(np.mean(log_frame > 0.2)),
            }
            rows.append(record)
            per_frame.append(record)
        event_path = args.cached_root / ("events_" + name + ".npz")
        cached_events = load_events(event_path, "events")
        cached_events = cached_events[(cached_events[:, 0] >= start_s) & (cached_events[:, 0] <= end_s)]
        condition_summary[name] = {
            "frame_count": len(cached_times),
            "timestamps_exact_subset_of_online": True,
            "linear_hdr_abs_delta": stats(hdr_delta.ravel()),
            "log_luma_abs_delta": stats(log_delta.ravel()),
            "log_luma_fraction_over_0_2": float(np.mean(log_delta > 0.2)),
            "event_match_to_online": event_match(window_events, cached_events),
            "cached_frame_sha256": sha256(cached_path),
            "cached_event_sha256": sha256(event_path),
            "per_frame": per_frame,
        }

    csv_path = out / "per_frame_hdr_comparison.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "PASS",
        "experiment": "online Direct-HDR buffers versus exact-state cached re-renders; image comparison and event agreement",
        "window_s": [start_s, end_s],
        "online_frames": int(len(online_times)),
        "online_interval_s": stats(np.diff(online_times)),
        "online_frame_shape": list(online_frames.shape),
        "frozen_core_modified": False,
        "no_GT_inputs_used_for_comparison": True,
        "inputs": {
            "online_frames_sha256": sha256(args.online_frames),
            "online_events_sha256": sha256(args.online_events),
            "cached_artifact_root": str(args.cached_root.resolve()),
        },
        "conditions": condition_summary,
        "per_frame_csv": csv_path.name,
        "interpretation_boundary": "same timestamps are required; event match is supporting evidence only and cannot identify DVS history state by itself",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "conditions": {
        name: {k: v for k, v in data.items() if k not in {"per_frame", "cached_frame_sha256", "cached_event_sha256"}}
        for name, data in condition_summary.items()
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
