#!/usr/bin/env python3
"""Separate HDR-to-log transform from comparator-history initialization.

This CPU diagnostic consumes saved online HDR frames/events and cached-state
frames for one exact interval. It compares the cached physical-log input route
with the online precomputed-log route, then reconstructs the online comparator
reference at the interval start from its launch frame and causal event history.
It does not initialize Genesis or use target pose, segmentation, or depth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.radiometry import linear_luminance


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
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "p99": float(np.percentile(array, 99)),
            "max": float(array.max())}


def online_log_gray(hdr: np.ndarray) -> np.ndarray:
    rgb = np.asarray(hdr, dtype=np.float32)
    gray = np.clip(rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722,
                   0.0, None)
    return np.log(gray.astype(np.float64) + 1e-6)


def online_config() -> DvsNoiseConfig:
    return DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
        cutoff_hz=0.0, leak_rate_hz=0.0, leak_jitter_fraction=0.0,
        shot_noise_rate_hz=0.0, refractory_period_s=0.0,
        log_input=True, readout_model="ideal_readout",
    )


def cached_config() -> DvsNoiseConfig:
    return DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
        cutoff_hz=0.0, leak_rate_hz=0.0, leak_jitter_fraction=0.0,
        shot_noise_rate_hz=0.0, noise_rate_cov_decades=0.0,
        refractory_period_s=0.0, log_input=False, radiance_input=True,
        response_mode="physical_log", radiometric_gain=1.0,
        radiometric_offset=0.0, radiometric_floor=1e-6,
        radiometric_calibration_status="uncalibrated",
        readout_model="ideal_readout", seed=42,
    )


def run_stream(frames: np.ndarray, times: np.ndarray, config: DvsNoiseConfig,
               *, already_log: bool, initial_base: np.ndarray | None = None,
               initial_lp: np.ndarray | None = None) -> np.ndarray:
    height, width = frames.shape[1:3]
    emulator = DvsEmulator(res=(height, width), config=config, device="cpu")

    def convert(frame: np.ndarray) -> np.ndarray:
        return online_log_gray(frame) if already_log else linear_luminance(frame)

    emulator.initialize(convert(frames[0]), t=float(times[0]))
    if initial_base is not None:
        if initial_base.shape != (height, width):
            raise ValueError("reconstructed comparator state shape mismatch")
        emulator.base_log_frame = torch.as_tensor(initial_base, dtype=torch.float64,
                                                   device=emulator.device).clone()
    if initial_lp is not None:
        if initial_lp.shape != (height, width):
            raise ValueError("reconstructed photoreceptor state shape mismatch")
        emulator.lp_log_frame = torch.as_tensor(initial_lp, dtype=torch.float64,
                                                 device=emulator.device).clone()
    chunks: list[np.ndarray] = []
    for index in range(1, len(times)):
        events = emulator.generate_events(convert(frames[index]), float(times[index]))
        if len(events):
            chunks.append(np.asarray(events, dtype=np.float64))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)


def event_match(reference: np.ndarray, candidate: np.ndarray,
                tolerance_s: float = 0.0005) -> dict[str, Any]:
    grouped: list[dict[tuple[int, int, int], list[float]]] = [{}, {}]
    for groups, events in zip(grouped, (reference, candidate)):
        for time_s, x, y, polarity in events:
            groups.setdefault((int(x), int(y), int(polarity)), []).append(float(time_s))
    matched = 0
    deltas: list[float] = []
    for key in grouped[0].keys() & grouped[1].keys():
        left, right = sorted(grouped[0][key]), sorted(grouped[1][key])
        i = j = 0
        while i < len(left) and j < len(right):
            delta = left[i] - right[j]
            if abs(delta) <= tolerance_s:
                matched += 1
                deltas.append(abs(delta) * 1000.0)
                i += 1
                j += 1
            elif delta < 0:
                i += 1
            else:
                j += 1
    return {
        "online_events": int(len(reference)), "replay_events": int(len(candidate)),
        "matched": int(matched),
        "online_match_fraction": None if not len(reference) else matched / len(reference),
        "replay_match_fraction": None if not len(candidate) else matched / len(candidate),
        "match_tolerance_ms": tolerance_s * 1000.0,
        "matched_abs_time_ms": stats(np.asarray(deltas, dtype=np.float64)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-frames", type=Path, required=True)
    parser.add_argument("--online-events", type=Path, required=True)
    parser.add_argument("--online-snapshots", type=Path, required=True)
    parser.add_argument("--cached-frames", type=Path, required=True)
    parser.add_argument("--cached-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    with np.load(args.online_frames, allow_pickle=False) as archive:
        online_times = np.asarray(archive["times_s"], dtype=np.float64)
        online_frames = np.asarray(archive["hdr_rgb"], dtype=np.float32)
    with np.load(args.cached_frames, allow_pickle=False) as archive:
        cached_times = np.asarray(archive["times_s"], dtype=np.float64)
        cached_frames = np.asarray(archive["hdr_rgb"], dtype=np.float32)
    if len(online_times) != len(cached_times) or online_frames.shape != cached_frames.shape:
        raise RuntimeError("online and cached dense HDR arrays have different shapes")
    cached_indices: list[int] = []
    timestamp_deltas: list[float] = []
    for time_s in online_times:
        matches = np.flatnonzero(np.abs(cached_times - time_s) <= 1e-9)
        if len(matches) != 1:
            raise RuntimeError(f"expected one cached frame near online t={time_s}, got {len(matches)}")
        cached_indices.append(int(matches[0]))
        timestamp_deltas.append(abs(float(cached_times[matches[0]] - time_s)))
    cached_frames_aligned = cached_frames[cached_indices]
    if not np.array_equal(online_frames, cached_frames_aligned):
        raise RuntimeError("dense cached HDR pixels differ from online frames at shared timestamps")
    with np.load(args.online_events, allow_pickle=False) as archive:
        online_events = np.asarray(archive["events_2000hz"], dtype=np.float64).reshape(-1, 4)
    with np.load(args.cached_events, allow_pickle=False) as archive:
        cached_events = np.asarray(archive["events"], dtype=np.float64).reshape(-1, 4)
    with np.load(args.online_snapshots, allow_pickle=False) as archive:
        labels = np.asarray(archive["labels"]).astype(str).tolist()
        snapshot_times = np.asarray(archive["times_s"], dtype=np.float64)
        snapshots = np.asarray(archive["hdr"], dtype=np.float32)
    if "launch" not in labels:
        raise ValueError(f"launch snapshot missing; labels={labels}")
    launch_index = labels.index("launch")
    if abs(float(snapshot_times[launch_index])) > 1e-9:
        raise ValueError(f"expected launch HDR at t=0, got {snapshot_times[launch_index]}")

    start_s, end_s = float(online_times[0]), float(online_times[-1])
    source_window = online_events[(online_events[:, 0] > start_s) &
                                  (online_events[:, 0] <= end_s)]
    history = online_events[online_events[:, 0] <= start_s]
    initial_log = online_log_gray(snapshots[launch_index])
    frame_log_start = online_log_gray(online_frames[0])
    positive = history[history[:, 3] > 0]
    negative = history[history[:, 3] < 0]
    # Rebuild the live comparator in the original 0.5ms call order. The DVS
    # updates all ON counts and then all OFF counts at each sample, so a single
    # whole-history sum can differ by ulps near an event threshold.
    step_count = int(round(start_s / 0.0005))
    history_ticks = np.arange(1, step_count + 1, dtype=np.float64) * 0.0005
    tick_index = np.searchsorted(history_ticks, history[:, 0], side="left")
    for event_index, event_time in enumerate(history[:, 0]):
        candidate = int(tick_index[event_index])
        if candidate > 0 and abs(history_ticks[candidate - 1] - event_time) <= 1e-12:
            tick_index[event_index] = candidate - 1
        elif candidate < len(history_ticks) and abs(history_ticks[candidate] - event_time) <= 1e-12:
            tick_index[event_index] = candidate
    if np.any(tick_index >= len(history_ticks)):
        raise RuntimeError("a history event lies beyond the requested DVS state cutoff")
    order = np.argsort(tick_index, kind="stable")
    sorted_history = history[order]
    sorted_ticks = tick_index[order]
    tick_counts = np.bincount(sorted_ticks, minlength=len(history_ticks))
    tick_starts = np.concatenate(([0], np.cumsum(tick_counts)))
    seeded_base_flat = initial_log.reshape(-1).copy()
    width = frame_log_start.shape[1]
    for tick in range(len(history_ticks)):
        sample_events = sorted_history[tick_starts[tick]:tick_starts[tick + 1]]
        for polarity, sign in ((1, 1.0), (-1, -1.0)):
            selected = sample_events[sample_events[:, 3] == polarity]
            if not len(selected):
                continue
            pixels = selected[:, 2].astype(np.int64) * width + selected[:, 1].astype(np.int64)
            unique_pixels, counts = np.unique(pixels, return_counts=True)
            seeded_base_flat[unique_pixels] += sign * counts.astype(np.float64) * 0.2
    seeded_base = seeded_base_flat.reshape(frame_log_start.shape)
    residual = frame_log_start - seeded_base

    cold_physical = run_stream(cached_frames, cached_times, cached_config(), already_log=False)
    cold_online = run_stream(online_frames, online_times, online_config(), already_log=True)
    warm_online = run_stream(online_frames, online_times, online_config(), already_log=True,
                              initial_base=seeded_base)
    online_input_log = np.stack([online_log_gray(frame) for frame in online_frames])
    cached_input_log = np.stack([
        np.log(np.maximum(linear_luminance(frame).astype(np.float64), 1e-6))
        for frame in cached_frames_aligned
    ])
    warm_physical = run_stream(cached_frames, cached_times, cached_config(), already_log=False,
                               initial_base=seeded_base, initial_lp=cached_input_log[0])
    transform_delta = np.abs(online_input_log - cached_input_log)

    comparison = {
        "cold_physical_transform": event_match(source_window, cold_physical),
        "cold_online_log_transform": event_match(source_window, cold_online),
        "history_seeded_online_log_transform": event_match(source_window, warm_online),
        "history_seeded_physical_log_transform": event_match(source_window, warm_physical),
        "cold_physical_vs_saved_cached_stream": event_match(cached_events, cold_physical),
    }
    out_events = output / "history_seeded_online_events.npz"
    np.savez_compressed(out_events, events=warm_online)
    out_physical_events = output / "history_seeded_physical_events.npz"
    np.savez_compressed(out_physical_events, events=warm_physical)
    input_frames_sha256 = sha256(args.online_frames)
    per_sample: list[dict[str, Any]] = []
    previous_t = float(online_times[0])
    for t in online_times[1:]:
        row: dict[str, Any] = {"sample_end_s": float(t), "sample_start_s": previous_t}
        for name, events in (("online", source_window), ("cold_physical", cold_physical),
                             ("cold_online", cold_online), ("history_seeded_online", warm_online),
                             ("history_seeded_physical", warm_physical)):
            row[f"{name}_events"] = int(np.count_nonzero((events[:, 0] > previous_t) &
                                                          (events[:, 0] <= float(t))))
        per_sample.append(row)
        previous_t = float(t)
    csv_path = output / "per_sample_event_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_sample[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_sample)

    module_path = Path(inspect.getfile(DvsEmulator)).resolve()
    summary = {
        "status": "PASS",
        "experiment": "same HDR frames with cold versus reconstructed live DVS comparator state; online-log versus physical-log routes",
        "window_s": [start_s, end_s],
        "frame_count": int(len(online_times)),
        "frames_bitwise_equal_at_shared_timestamps": True,
        "max_shared_timestamp_delta_s": float(max(timestamp_deltas)),
        "source_events_in_window_excluding_start": int(len(source_window)),
        "history_events_used_through_window_start": int(len(history)),
        "history_events_on": int(len(positive)),
        "history_events_off": int(len(negative)),
        "history_state_reconstruction": "0.5ms per-source-step ON update then OFF update",
        "state_seed_residual_from_live_frame": {
            "abs_stats": stats(np.abs(residual)),
            "max_above_positive_threshold": float(np.max(residual - 0.2)),
            "max_below_negative_threshold": float(np.max(-0.2 - residual)),
        },
        "input_log_transform_abs_delta": {
            "stats": stats(transform_delta.ravel()),
            "fraction_nonzero": float(np.mean(transform_delta != 0)),
            "fraction_above_1e_7": float(np.mean(transform_delta > 1e-7)),
        },
        "event_comparisons": comparison,
        "dvs_implementation": {"path": str(module_path), "sha256": sha256(module_path)},
        "input_hashes": {str(path.resolve()): sha256(path) for path in (
            args.online_frames, args.online_events, args.online_snapshots,
            args.cached_frames, args.cached_events,
        )},
        "event_output": out_events.name,
        "event_output_sha256": sha256(out_events),
        "physical_event_output": out_physical_events.name,
        "physical_event_output_sha256": sha256(out_physical_events),
        "per_sample_csv": csv_path.name,
        "no_physics_or_renderer": True,
        "GT_segmentation_depth_or_future_state_used": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"input_hashes"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
