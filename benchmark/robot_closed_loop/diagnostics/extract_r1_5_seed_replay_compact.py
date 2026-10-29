#!/usr/bin/env python3
"""Extract compact, hash-linked evidence from a seed-matched R0.5 rerun.

This runs beside the shared raw event files so only a small causal event window
needs to move to a local analysis machine. GT is exported for post-inference
scoring only; the event payload contains no GT-derived pixels or masks.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


QUERY_TIMES_S = [0.5855, 0.5860, 0.5865, 0.5875, 0.5885, 0.5895, 0.5904999999999999]
EVENT_WINDOW_START_S = 0.5605
EVENT_WINDOW_END_S = 0.5905
MATCH_WINDOW_START_S = 0.5505
MATCH_WINDOW_END_S = 0.5905
ARCHIVE_EVENT_SHA256 = "b698817a8347ca48ad75015f0cb81c3bfc711958a60d3387d468c70456a7b2b6"
ARCHIVE_TRACE_SHA256 = "310837ea07ab2f819f590ccce91b3952dd9e85ea5005537bc1f4d1a339bff28c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def event_arrays(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        events = np.asarray(data["events_2000hz"], dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"invalid events_2000hz shape {events.shape} at {path}")
    return events


def event_match(a: np.ndarray, b: np.ndarray, tolerance_s: float = 0.0005) -> dict[str, Any]:
    grouped_a: dict[tuple[int, int, int], list[float]] = {}
    grouped_b: dict[tuple[int, int, int], list[float]] = {}
    for groups, events in ((grouped_a, a), (grouped_b, b)):
        for t, x, y, polarity in events:
            groups.setdefault((int(x), int(y), int(polarity)), []).append(float(t))
    matched = 0
    deltas: list[float] = []
    for key in grouped_a.keys() & grouped_b.keys():
        left = sorted(grouped_a[key])
        right = sorted(grouped_b[key])
        i = j = 0
        while i < len(left) and j < len(right):
            delta = left[i] - right[j]
            if abs(delta) <= tolerance_s:
                matched += 1
                deltas.append(abs(delta))
                i += 1
                j += 1
            elif delta < 0:
                i += 1
            else:
                j += 1
    return {
        "archive_event_count": int(len(a)), "rerun_event_count": int(len(b)),
        "matched_count": matched,
        "archive_matched_fraction": None if not len(a) else matched / len(a),
        "rerun_matched_fraction": None if not len(b) else matched / len(b),
        "same_pixel_polarity_time_tolerance_s": tolerance_s,
        "matched_abs_time_error_ms": stats([d * 1000 for d in deltas]),
    }


def exact_or_interpolated_label(trace: list[dict[str, Any]], query_s: float) -> dict[str, Any]:
    times = np.asarray([float(row["time_s"]) for row in trace], dtype=np.float64)
    hi = int(np.searchsorted(times, query_s, side="left"))
    if hi < len(times) and abs(times[hi] - query_s) <= 1e-9:
        row = trace[hi]
        method = "exact logged physical state row"
        position = np.asarray(row["object"]["position_m"], dtype=np.float64)
        velocity = np.asarray(row["object"]["velocity_m_s"], dtype=np.float64)
    else:
        if hi == 0 or hi == len(times):
            raise ValueError(f"query {query_s} outside saved trace")
        lo = hi - 1
        alpha = (query_s - times[lo]) / (times[hi] - times[lo])
        position = np.asarray(trace[lo]["object"]["position_m"], dtype=np.float64) * (1 - alpha)
        position += np.asarray(trace[hi]["object"]["position_m"], dtype=np.float64) * alpha
        velocity = np.asarray(trace[lo]["object"]["velocity_m_s"], dtype=np.float64) * (1 - alpha)
        velocity += np.asarray(trace[hi]["object"]["velocity_m_s"], dtype=np.float64) * alpha
        method = "linear interpolation between adjacent saved GT rows for offline scoring"
    return {"time_s": query_s, "position_m": position.tolist(),
            "velocity_m_s": velocity.tolist(), "label_method": method}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    new_episode = args.rerun_root / "episodes" / "episode_000"
    old_episode = args.archive_root / "episodes" / "episode_005"
    new_trace_path = new_episode / "state_control_trace.jsonl"
    old_trace_path = old_episode / "state_control_trace.jsonl"
    new_event_path = new_episode / "direct_hdr_events_1khz_2khz.npz"
    old_event_path = old_episode / "direct_hdr_events_1khz_2khz.npz"
    for path in (new_trace_path, old_trace_path, new_event_path, old_event_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    old_event_hash = sha256(old_event_path)
    old_trace_hash = sha256(old_trace_path)
    if old_event_hash != ARCHIVE_EVENT_SHA256 or old_trace_hash != ARCHIVE_TRACE_SHA256:
        raise RuntimeError("archived episode 005 does not match the audited input hashes")

    old_trace, new_trace = read_trace(old_trace_path), read_trace(new_trace_path)
    old_by_time = {round(float(row["time_s"]), 9): row for row in old_trace}
    new_by_time = {round(float(row["time_s"]), 9): row for row in new_trace}
    common_times = sorted(old_by_time.keys() & new_by_time.keys())
    if not common_times:
        raise RuntimeError("no exact shared state timestamps")

    position_error, velocity_error = [], []
    robot_qpos_error, robot_qvel_error, tcp_error = [], [], []
    contact_mismatch = 0
    for time_s in common_times:
        old, new = old_by_time[time_s], new_by_time[time_s]
        position_error.append(float(np.linalg.norm(
            np.asarray(new["object"]["position_m"]) - np.asarray(old["object"]["position_m"]))))
        velocity_error.append(float(np.linalg.norm(
            np.asarray(new["object"]["velocity_m_s"]) - np.asarray(old["object"]["velocity_m_s"]))))
        robot_qpos_error.append(float(np.linalg.norm(np.asarray(new["robot_qpos"]) - np.asarray(old["robot_qpos"]))))
        robot_qvel_error.append(float(np.linalg.norm(np.asarray(new["robot_qvel"]) - np.asarray(old["robot_qvel"]))))
        tcp_error.append(float(np.linalg.norm(np.asarray(new["tcp_pad_center_m"]) - np.asarray(old["tcp_pad_center_m"]))))
        contact_mismatch += int(new["contacts"] != old["contacts"])

    old_events_all, new_events_all = event_arrays(old_event_path), event_arrays(new_event_path)
    match_mask_old = ((old_events_all[:, 0] >= MATCH_WINDOW_START_S) &
                      (old_events_all[:, 0] <= MATCH_WINDOW_END_S))
    match_mask_new = ((new_events_all[:, 0] >= MATCH_WINDOW_START_S) &
                      (new_events_all[:, 0] <= MATCH_WINDOW_END_S))
    matching = event_match(old_events_all[match_mask_old], new_events_all[match_mask_new])
    input_old = old_events_all[(old_events_all[:, 0] >= EVENT_WINDOW_START_S) &
                               (old_events_all[:, 0] <= EVENT_WINDOW_END_S)]
    input_new = new_events_all[(new_events_all[:, 0] >= EVENT_WINDOW_START_S) &
                               (new_events_all[:, 0] <= EVENT_WINDOW_END_S)]
    buffer = io.BytesIO()
    np.savez_compressed(buffer, archive_events=input_old, rerun_events=input_new,
                        query_times_s=np.asarray(QUERY_TIMES_S, dtype=np.float64))
    query_labels = {
        "archive": [exact_or_interpolated_label(old_trace, t) for t in QUERY_TIMES_S],
        "seed_matched_rerun": [exact_or_interpolated_label(new_trace, t) for t in QUERY_TIMES_S],
    }
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump({
            "schema": "r1_5_seed_replay_compact_v1",
            "episode": {"seed": 2026103425, "speed_m_s": 0.4, "launch_y_m": 0.055},
            "source_paths": {"archive_root": str(args.archive_root.resolve()),
                             "rerun_root": str(args.rerun_root.resolve())},
            "source_sha256": {
                "archive_event_file": old_event_hash, "archive_trace": old_trace_hash,
                "rerun_event_file": sha256(new_event_path), "rerun_trace": sha256(new_trace_path),
            },
            "state_trace_comparison": {
                "archive_rows": len(old_trace), "rerun_rows": len(new_trace),
                "exact_common_timestamps": len(common_times),
                "archive_time_range_s": [min(old_by_time), max(old_by_time)],
                "rerun_time_range_s": [min(new_by_time), max(new_by_time)],
                "target_position_error_m": stats(position_error),
                "target_velocity_error_m_s": stats(velocity_error),
                "robot_qpos_l2_error": stats(robot_qpos_error),
                "robot_qvel_l2_error": stats(robot_qvel_error),
                "tcp_pad_center_error_m": stats(tcp_error),
                "contact_record_mismatch_rows": contact_mismatch,
            },
            "event_source_match_5505_5905ms": matching,
            "event_input_window_s": [EVENT_WINDOW_START_S, EVENT_WINDOW_END_S],
            "event_payload_npz_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "event_payload_dtype": "float64; original absolute timestamps preserved exactly",
            "event_payload_counts": {"archive": int(len(input_old)), "seed_matched_rerun": int(len(input_new))},
            "post_inference_gt_labels": query_labels,
            "boundary": "GT labels are for scoring only; archived/rerun event payload contains no GT mask/depth input",
        }, stream, separators=(",", ":"))
        stream.write("\n")
    with args.output.open("rb") as stream:
        print(json.dumps({"output": str(output), "output_bytes": output.stat().st_size,
                          "output_sha256": hashlib.sha256(stream.read()).hexdigest(),
                          "state": json.loads(output.read_text(encoding="utf-8"))["state_trace_comparison"],
                          "event_match": matching}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
