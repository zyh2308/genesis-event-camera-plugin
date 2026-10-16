#!/usr/bin/env python3
"""Replay saved Direct-HDR event tracks through the causal five-bin state path.

Ground truth is opened only after each event estimate is made, for offline
evaluation. It is never passed to EventKinematicsTracker or the predictor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from benchmark.robot_closed_loop.event_state import EventKinematicsTracker
from benchmark.robot_closed_loop.predictor import FixedDecelerationPredictor, MotionState


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_to_world_cv(config: Dict[str, Any]) -> np.ndarray:
    """Reconstruct Genesis' look-at camera basis from its fixed scene config."""
    camera = config["camera"]
    position = np.asarray(camera["position_m"], dtype=np.float64)
    forward = np.asarray(camera["lookat_m"], dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    up = np.asarray(camera["up_m"], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    # Genesis stores OpenGL camera axes [right, up, -forward]; the plugin's
    # camera_to_world_cv adapter flips columns y/z for OpenCV coordinates.
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((right, -camera_up, forward))
    transform[:3, 3] = position
    return transform


def _camera_args(config: Dict[str, Any]) -> Tuple[np.ndarray, Sequence[int], float]:
    camera = config["camera"]
    return (
        _camera_to_world_cv(config), camera["resolution_wh"],
        float(camera["vertical_fov_deg"]),
    )


def _make_tracker(config: Dict[str, Any]) -> EventKinematicsTracker:
    target = config["target"]
    controller = config["controller"]
    x_lo = float(controller["intercept_x_m"]) - 0.10
    x_hi = float(target["launch_x_range_m"][1]) + 0.04
    y_limit = max(abs(float(v)) for v in target["launch_y_range_m"]) + 0.06
    return EventKinematicsTracker(
        *_camera_args(config),
        (x_lo, x_hi), (-y_limit, y_limit),
        float(target["size_m"][2]) / 2.0,
        window_s=0.030,
        bin_s=0.005,
        max_state_age_s=float(controller.get("maximum_event_state_age_s", 0.030)),
    )


def _read_trace(path: Path) -> List[Dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _truth_at(trace: Sequence[Dict[str, Any]], time_s: float) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray([float(row["time_s"]) for row in trace], dtype=np.float64)
    right = int(np.searchsorted(times, float(time_s), side="left"))
    if right <= 0:
        rows = (trace[0], trace[0])
    elif right >= len(trace):
        rows = (trace[-1], trace[-1])
    else:
        rows = (trace[right - 1], trace[right])
    t0, t1 = (float(row["time_s"]) for row in rows)
    weight = 0.0 if t1 <= t0 else float(np.clip((time_s - t0) / (t1 - t0), 0.0, 1.0))
    p0 = np.asarray(rows[0]["object"]["position_m"], dtype=np.float64)
    p1 = np.asarray(rows[1]["object"]["position_m"], dtype=np.float64)
    v0 = np.asarray(rows[0]["object"]["velocity_m_s"], dtype=np.float64)
    v1 = np.asarray(rows[1]["object"]["velocity_m_s"], dtype=np.float64)
    return (1.0 - weight) * p0 + weight * p1, (1.0 - weight) * v0 + weight * v1


def _phase(time_s: float, contact_s: Optional[float], lift_s: Optional[float]) -> str:
    if contact_s is None or time_s < contact_s:
        return "pre_contact"
    if lift_s is None or time_s < lift_s:
        return "contact_to_lift"
    return "post_lift"


def _percentiles(values: Iterable[float]) -> Dict[str, Optional[float]]:
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"n": len(rows)}
    for key in ("position_error_xy_m", "velocity_error_xy_m_s", "age_ms", "arrival_error_s"):
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            result[key] = _percentiles(values)
    return result


def _five_bin_tensor(events: np.ndarray, cutoff_s: float, width: int, height: int) -> np.ndarray:
    """Build causal (time, polarity, y, x) 5x2x64x64 event counts."""
    bins = 5
    bin_s = 0.005
    start = float(cutoff_s) - bins * bin_s
    arr = np.asarray(events, dtype=np.float64).reshape(-1, 4)
    keep = (arr[:, 0] >= start) & (arr[:, 0] < float(cutoff_s))
    arr = arr[keep]
    output = np.zeros((bins, 2, 64, 64), dtype=np.float32)
    if not len(arr):
        return output
    time_idx = np.floor((arr[:, 0] - start) / bin_s).astype(np.int64)
    x_idx = np.clip((arr[:, 1] * 64 / width).astype(np.int64), 0, 63)
    y_idx = np.clip((arr[:, 2] * 64 / height).astype(np.int64), 0, 63)
    pol_idx = np.where(arr[:, 3] > 0, 0, 1)
    flat = (((time_idx * 2 + pol_idx) * 64 + y_idx) * 64 + x_idx)
    output = np.bincount(flat, minlength=bins * 2 * 64 * 64).reshape(output.shape).astype(np.float32)
    return output


def _five_bin_compare(events_1k: np.ndarray, events_2k: np.ndarray, duration_s: float) -> Dict[str, Any]:
    width, height = 320, 240
    windows = []
    cutoff = 0.040
    last_cutoff = duration_s - 0.010
    while cutoff <= last_cutoff + 1e-9:
        a = _five_bin_tensor(events_1k, cutoff, width, height)
        b = _five_bin_tensor(events_2k, cutoff, width, height)
        count_a, count_b = float(a.sum()), float(b.sum())
        if count_a == 0 and count_b == 0:
            category, cosine, count_delta = "both_empty", 1.0, 0.0
        elif count_a == 0 or count_b == 0:
            category, cosine, count_delta = "one_empty", None, 1.0
        else:
            category = "both_nonempty"
            av, bv = a.ravel(), b.ravel()
            cosine = float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv)))
            count_delta = abs(count_a - count_b) / max(count_a, count_b)
        windows.append({"category": category, "cosine": cosine, "count_delta": count_delta})
        cutoff += 0.010
    active = [row for row in windows if row["category"] == "both_nonempty"]
    all_rows = windows
    return {
        "representation": "causal five temporal bins x two polarities x 64 x 64 counts",
        "bin_width_ms": 5.0,
        "window_width_ms": 25.0,
        "controller_cadence_ms": 10.0,
        "cutoff_latency_ms": 10.0,
        "all_controller_windows": len(all_rows),
        "both_nonempty_windows": len(active),
        "one_empty_windows": sum(r["category"] == "one_empty" for r in all_rows),
        "both_empty_windows": sum(r["category"] == "both_empty" for r in all_rows),
        "event_count_relative_difference_denominator": "max(count_1khz,count_2khz)",
        "event_count_relative_difference_mean_all_windows": float(np.mean([r["count_delta"] for r in all_rows])) if all_rows else None,
        "event_count_relative_difference_mean_both_nonempty": float(np.mean([r["count_delta"] for r in active])) if active else None,
        "event_count_relative_difference_p95_both_nonempty": float(np.percentile([r["count_delta"] for r in active], 95)) if active else None,
        "voxel_cosine_mean_both_nonempty": float(np.mean([r["cosine"] for r in active])) if active else None,
        "voxel_cosine_p10_both_nonempty": float(np.percentile([r["cosine"] for r in active], 10)) if active else None,
    }


def _event_target_proximity(
    event_streams: Dict[int, np.ndarray], trace: Sequence[Dict[str, Any]], config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluator-only check of how much raw event activity projects near the GT cube."""
    camera_to_world, _resolution, _fov = _camera_args(config)
    width, height = (int(v) for v in config["camera"]["resolution_wh"])
    focal = height / (2.0 * math.tan(math.radians(float(config["camera"]["vertical_fov_deg"])) / 2.0))
    fx = fy = focal
    cx, cy = width / 2.0, height / 2.0
    trace_t = np.asarray([float(row["time_s"]) for row in trace], dtype=np.float64)
    trace_p = np.asarray([row["object"]["position_m"] for row in trace], dtype=np.float64)
    target_z = float(config["target"]["size_m"][2]) / 2.0
    result = {}
    for rate_hz, events in event_streams.items():
        arr = np.asarray(events, dtype=np.float64).reshape(-1, 4)
        camera_points = np.column_stack(((arr[:, 1] - cx) / fx, (arr[:, 2] - cy) / fy, np.ones(len(arr))))
        direction_world = camera_points @ camera_to_world[:3, :3].T
        denom = direction_world[:, 2]
        scale = np.divide(
            target_z - camera_to_world[2, 3], denom,
            out=np.full_like(denom, np.nan), where=np.abs(denom) > 1e-9,
        )
        projectable = np.isfinite(scale) & (scale > 0)
        arr = arr[projectable]
        direction_world = direction_world[projectable]
        scale = scale[projectable]
        projected = camera_to_world[:3, 3] + direction_world * scale[:, None]
        truth_xy = np.column_stack((
            np.interp(arr[:, 0], trace_t, trace_p[:, 0]),
            np.interp(arr[:, 0], trace_t, trace_p[:, 1]),
        ))
        distance = np.linalg.norm(projected[:, :2] - truth_xy, axis=1)
        result[str(rate_hz)] = {
            "raw_event_count": int(len(events)),
            "projectable_event_count": int(len(arr)),
            "projected_to_target_center_plane_distance_median_mm": float(np.median(distance) * 1000.0),
            "fraction_within_target_center_radius_30mm": float(np.mean(distance <= 0.030)),
            "fraction_within_target_center_radius_50mm": float(np.mean(distance <= 0.050)),
            "fraction_within_target_center_radius_80mm": float(np.mean(distance <= 0.080)),
            "fraction_within_target_center_radius_120mm": float(np.mean(distance <= 0.120)),
        }
    return {
        "purpose": "evaluator-only spatial attribution diagnostic; projected event points are compared with GT cube center after the event estimate is formed",
        "projection_assumption": "all event rays intersect a fixed plane at target center z; target-face parallax remains",
        "by_sampling_hz": result,
    }


def _reprojection_summary(tracker: EventKinematicsTracker, config: Dict[str, Any]) -> Dict[str, Any]:
    xs = np.linspace(0.55, 0.82, 5)
    ys = np.linspace(-0.06, 0.06, 5)
    errors = []
    for x in xs:
        for y in ys:
            point = np.asarray([x, y, tracker.target_center_z_m], dtype=np.float64)
            px, py = tracker._project_world(point)
            recovered = tracker._event_to_world(px, py)
            errors.append(float(np.linalg.norm(recovered[:2] - point[:2])))
    old_tracker = EventKinematicsTracker(
        *_camera_args(config), (0.45, 0.86), (-0.12, 0.12),
        tracker.target_center_z_m,
        window_s=tracker.window_s, bin_s=tracker.bin_s,
    )
    # Translate an identical pixel grid with the previous (W-1)/2 principal
    # point and report the induced workspace bias against Genesis W/2,H/2.
    old_tracker.cx = (old_tracker.width - 1) / 2.0
    old_tracker.cy = (old_tracker.height - 1) / 2.0
    principal_bias = []
    for x in xs:
        for y in ys:
            point = np.asarray([x, y, tracker.target_center_z_m], dtype=np.float64)
            px, py = tracker._project_world(point)
            recovered = old_tracker._event_to_world(px, py)
            principal_bias.append(float(np.linalg.norm(recovered[:2] - point[:2])))
    camera = config["camera"]
    return {
        "method": "fixed-config Genesis look-at basis + Genesis Camera.intrinsics convention; fixed target-center plane",
        "workspace_grid_points": len(errors),
        "roundtrip_position_error_xy_mm_max": float(max(errors) * 1000.0),
        "old_half_pixel_center_bias_xy_mm_p50": float(np.percentile(principal_bias, 50) * 1000.0),
        "old_half_pixel_center_bias_xy_mm_max": float(max(principal_bias) * 1000.0),
        "intrinsics_fx_fy_cx_cy": [
            tracker.fx, tracker.fy, tracker.cx, tracker.cy,
        ],
        "camera_pose_source": "reconstructed offline from fixed scene position/lookat/up; old run did not serialize camera.transform",
        "principal_point_source": "Genesis Camera.cx=W/2 and Camera.cy=H/2",
    }


def _freshness_contract_probe(config: Dict[str, Any]) -> Dict[str, Any]:
    """Exercise all fail-closed state labels with small synthetic event batches."""
    no_event_tracker = _make_tracker(config)
    no_event = no_event_tracker.observe(0.0, 0.0)

    lost_tracker = _make_tracker(config)
    point = np.asarray([0.70, 0.0, lost_tracker.target_center_z_m], dtype=np.float64)
    px, py = lost_tracker._project_world(point)
    lost_events = np.asarray([
        [0.0020, px, py, 1], [0.0025, px, py, -1], [0.0030, px, py, 1],
        [0.0070, px, py, 1], [0.0075, px, py, -1], [0.0080, px, py, 1],
    ], dtype=np.float64)
    lost_tracker.push(lost_events, 0.010)
    lost = lost_tracker.observe(0.010, 0.010)

    fresh_tracker = _make_tracker(config)
    fresh_events = []
    for bin_idx in range(4):
        center_t = (bin_idx + 0.5) * fresh_tracker.bin_s
        for delta_t, polarity in ((-0.0004, 1), (0.0, -1), (0.0004, 1)):
            fresh_events.append([center_t + delta_t, px, py, polarity])
    fresh_tracker.push(np.asarray(fresh_events, dtype=np.float64), 0.020)
    fresh = fresh_tracker.observe(0.020, 0.020)
    fresh_tracker.push(np.empty((0, 4), dtype=np.float64), 0.200)
    stale = fresh_tracker.observe(0.200, 0.200)
    return {
        "synthetic_interface_probe_only": True,
        "initial_no_event": {
            "state_status": no_event.state_status, "tracking_status": no_event.tracking_status,
        },
        "events_without_minimum_track_support": {
            "state_status": lost.state_status, "tracking_status": lost.tracking_status,
            "event_count_window": lost.event_count_window,
            "time_bin_count_window": lost.time_bin_count_window,
        },
        "fresh_estimate": {
            "state_status": fresh.state_status, "tracking_status": fresh.tracking_status,
            "age_ms": None if fresh.age_s is None else fresh.age_s * 1000.0,
            "sample_generated_at_s": fresh.sample_generated_at_s,
        },
        "expired_estimate_after_silence": {
            "state_status": stale.state_status, "tracking_status": stale.tracking_status,
            "age_ms": None if stale.age_s is None else stale.age_s * 1000.0,
        },
        "controller_policy": "only fresh has a usable state; stale/unavailable never yields a MotionState",
    }


def _replay_episode(root: Path, episode_id: int, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    episode_dir = root / "episodes" / f"episode_{episode_id:03d}"
    trace = _read_trace(episode_dir / "state_control_trace.jsonl")
    with (episode_dir / "episode.csv").open(newline="") as stream:
        episode = next(csv.DictReader(stream))
    data_path = episode_dir / "direct_hdr_events_1khz_2khz.npz"
    with np.load(data_path) as archive:
        event_streams = {
            1000: np.asarray(archive["events_1000hz"], dtype=np.float64),
            2000: np.asarray(archive["events_2000hz"], dtype=np.float64),
        }

    contact_s = float(episode["first_contact_s"])
    lift_s = None if not episode["lift_start_s"] else float(episode["lift_start_s"])
    planned_arrival = float(episode["planned_free_slide_arrival_t_s"])
    controller_cfg = config["controller"]
    latency_s = float(controller_cfg["observation_latency_s"])
    controller_period_s = 1.0 / float(controller_cfg["high_level_hz"])
    physics_dt_s = float(config["physics"]["dt_s"])
    predictor = FixedDecelerationPredictor(
        float(controller_cfg["intercept_x_m"]),
        float(controller_cfg["initial_deceleration_assumption_m_s2"]),
    )
    duration_s = float(episode["sim_duration_s"])
    rate_rows: Dict[int, List[Dict[str, Any]]] = {1000: [], 2000: []}
    estimates_by_rate: Dict[int, List[Dict[str, Any]]] = {1000: [], 2000: []}

    for rate_hz, events in event_streams.items():
        tracker = _make_tracker(config)
        events = events[np.argsort(events[:, 0], kind="stable")]
        cursor = 0
        control_steps = int(round(controller_period_s / physics_dt_s))
        burn_step = int(math.ceil(float(controller_cfg["event_burn_in_s"]) / controller_period_s)) * control_steps
        max_steps = int(math.floor(duration_s / physics_dt_s))
        for step in range(burn_step, max_steps + 1, control_steps):
            now_s = step * physics_dt_s
            cutoff_s = now_s - latency_s
            end = int(np.searchsorted(events[:, 0], cutoff_s + 1e-12, side="right"))
            batch = events[cursor:end]
            cursor = end
            # The controller only consumes one observation every 10 ms. At
            # each poll, replay all newly available events through the causal
            # latency cutoff. This is equivalent to the saved 0.5/1 ms source
            # updates at the controller decision instants, without replaying
            # every intermediate tracker computation offline.
            tracker.push(batch, cutoff_s)
            obs = tracker.observe(cutoff_s, now_s)
            row: Dict[str, Any] = {
                "episode_id": episode_id,
                "sampling_hz": rate_hz,
                "evaluation_time_s": now_s,
                "observation_cutoff_s": cutoff_s,
                "phase": _phase(now_s, contact_s, lift_s),
                "state_status": obs.state_status,
                "tracking_status": obs.tracking_status,
                "age_ms": None if obs.age_s is None else obs.age_s * 1000.0,
                "sample_generated_at_s": obs.sample_generated_at_s,
                "event_count_window": obs.event_count_window,
                "time_bin_count_window": obs.time_bin_count_window,
                "position_error_xy_m": None,
                "velocity_error_xy_m_s": None,
                "arrival_error_s": None,
                "arrival_reachable": None,
            }
            if obs.state is not None:
                prediction = predictor.predict(obs.state, float(obs.age_s))
                position, velocity = _truth_at(trace, obs.state.time_s)
                row["position_error_xy_m"] = float(np.linalg.norm(obs.state.position_m[:2] - position[:2]))
                row["velocity_error_xy_m_s"] = float(np.linalg.norm(obs.state.velocity_m_s[:2] - velocity[:2]))
                if row["phase"] == "pre_contact" and prediction.reachable:
                    row["arrival_error_s"] = now_s + prediction.ttc_s - planned_arrival
                    row["arrival_reachable"] = True
                elif row["phase"] == "pre_contact":
                    row["arrival_reachable"] = False
                row["estimate_time_s"] = obs.state.time_s
                row["estimate_position_m"] = obs.state.position_m.tolist()
                row["estimate_velocity_m_s"] = obs.state.velocity_m_s.tolist()
                estimate_row = {
                    "episode_id": episode_id,
                    "sampling_hz": rate_hz,
                    "estimate_time_s": obs.state.time_s,
                    "phase": _phase(obs.state.time_s, contact_s, lift_s),
                    "event_count_window": obs.event_count_window,
                    "time_bin_count_window": obs.time_bin_count_window,
                    "age_ms": float(obs.age_s) * 1000.0,
                    "position_error_xy_m": row["position_error_xy_m"],
                    "velocity_error_xy_m_s": row["velocity_error_xy_m_s"],
                }
                estimates_by_rate[rate_hz].append(estimate_row)
            rate_rows[rate_hz].append(row)

    rate_summary: Dict[str, Any] = {}
    for rate_hz in (1000, 2000):
        state_rows = estimates_by_rate[rate_hz]
        poll_rows = rate_rows[rate_hz]
        phases: Dict[str, Any] = {}
        for phase in ("pre_contact", "contact_to_lift", "post_lift"):
            phases[phase] = {
                "fresh_controller_observation_samples": _summary([r for r in state_rows if r["phase"] == phase]),
                "controller_poll_samples": len([r for r in poll_rows if r["phase"] == phase]),
                "fresh_controller_poll_samples": len([r for r in poll_rows if r["phase"] == phase and r["state_status"] == "fresh"]),
                "stale_controller_poll_samples": len([r for r in poll_rows if r["phase"] == phase and r["state_status"] == "stale"]),
                "unavailable_controller_poll_samples": len([r for r in poll_rows if r["phase"] == phase and r["state_status"] == "unavailable"]),
                "no_event_poll_samples": len([r for r in poll_rows if r["phase"] == phase and r["tracking_status"] == "no_event"]),
                "tracking_lost_poll_samples": len([r for r in poll_rows if r["phase"] == phase and r["tracking_status"] == "tracking_lost"]),
                "tracking_poll_samples": len([r for r in poll_rows if r["phase"] == phase and r["tracking_status"] == "tracking"]),
            }
            eligible = [r for r in poll_rows if r["phase"] == phase]
            phases[phase]["fresh_state_fraction"] = (
                None if not eligible else phases[phase]["fresh_controller_poll_samples"] / len(eligible)
            )
            phases[phase]["controller_observation_age_ms"] = _percentiles(
                row["age_ms"] for row in eligible if row["age_ms"] is not None
            )
        precontact_poll_rows = [r for r in poll_rows if r["phase"] == "pre_contact"]
        arrivals = [r["arrival_error_s"] for r in precontact_poll_rows if r["arrival_error_s"] is not None]
        phases["pre_contact"]["arrival_error_s"] = _percentiles(arrivals)
        phases["pre_contact"]["reachable_arrival_prediction_fraction"] = (
            None if not precontact_poll_rows else
            sum(r["arrival_reachable"] is True for r in precontact_poll_rows) / len(precontact_poll_rows)
        )
        rate_summary[str(rate_hz)] = phases

    trace_rows = [row for row in trace if row.get("object")]
    proximity = _event_target_proximity(event_streams, trace_rows, config)
    return ({
        "episode_id": episode_id,
        "seed": int(episode["seed"]),
        "speed_m_s": float(episode["launch_speed_m_s"]),
        "launch_y_m": float(episode["start_y_m"]),
        "first_contact_s": contact_s,
        "lift_start_s": lift_s,
        "planned_free_slide_arrival_t_s_evaluator_only": planned_arrival,
        "strict_shadow_success": episode["success_strict_r0_5"].lower() == "true",
        "input_sha256": _sha256(data_path),
        "state_accuracy_and_status_by_sampling_hz": rate_summary,
    }, rate_rows[1000] + rate_rows[2000], {
        "duration_s": duration_s,
        "events_1000hz": len(event_streams[1000]),
        "events_2000hz": len(event_streams[2000]),
        "five_bin_input_comparison": _five_bin_compare(event_streams[1000], event_streams[2000], duration_s),
        "event_target_proximity_evaluator_only": proximity,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", default="0,4,8")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    source_root = args.source_results
    output = args.out
    output.mkdir(parents=True, exist_ok=False)
    episode_ids = [int(value) for value in args.episodes.split(",")]
    rows = []
    all_poll_rows = []
    sampling_rows = []
    started = time.perf_counter()
    for episode_id in episode_ids:
        episode_summary, poll_rows, sample_summary = _replay_episode(source_root, episode_id, config)
        rows.append(episode_summary)
        all_poll_rows.extend(poll_rows)
        sample_summary["episode_id"] = episode_id
        sampling_rows.append(sample_summary)
    replay_wall = time.perf_counter() - started

    tracker = _make_tracker(config)
    reprojection = _reprojection_summary(tracker, config)
    # Preserve the regression defect as a transparent numeric check.
    times = np.arange(6, dtype=np.float64) * 0.005 + 0.0025
    centers = np.column_stack((0.4 * times, np.zeros(6)))
    old_time = times - times[-1]
    old_velocity = old_time @ (centers - centers.mean(axis=0)) / np.dot(old_time, old_time)
    centered_time = times - times.mean()
    fixed_velocity = centered_time @ centers / np.dot(centered_time, centered_time)

    checksum_manifest = source_root / "SHA256SUMS"
    expected = {}
    if checksum_manifest.exists():
        for line in checksum_manifest.read_text().splitlines():
            pieces = line.split(None, 1)
            if len(pieces) == 2:
                expected[pieces[1].lstrip("* ")] = pieces[0]
    for row in rows:
        episode_id = row["episode_id"]
        rel = f"episodes/episode_{episode_id:03d}/direct_hdr_events_1khz_2khz.npz"
        if rel in expected and row["input_sha256"] != expected[rel]:
            raise RuntimeError(f"raw event checksum mismatch for {rel}")

    per_episode_path = output / "per_episode.json"
    per_episode_path.write_text(json.dumps(rows, indent=2) + "\n")
    poll_path = output / "causal_observation_polls.csv"
    if all_poll_rows:
        with poll_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_poll_rows[0]))
            writer.writeheader()
            writer.writerows(all_poll_rows)
    sampling_path = output / "five_bin_sampling_comparison.json"
    sampling_path.write_text(json.dumps(sampling_rows, indent=2) + "\n")
    manifest = {
        "phase": "R0.6 offline causal event-state diagnosis",
        "source_g1_results": "~/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/",
        "episodes": episode_ids,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "command": [sys.executable, *sys.argv],
        "config_sha256": _sha256(args.config),
        "raw_event_sha256_manifest_sha256": None if not checksum_manifest.exists() else _sha256(checksum_manifest),
        "raw_events_verified_against_server_manifest": bool(expected),
        "estimator_inputs": "timestamped raw events and fixed config-derived pinhole calibration only",
        "ground_truth_boundary": "GT traces/arrival time are read only after estimates exist, for offline evaluator alignment",
        "controller_sampling_hz": 100,
        "observation_latency_ms": 10.0,
        "five_bin_comparison": "same causal 25 ms, five x 5 ms bins at 10 ms controller cadence; cutoff is 10 ms behind simulated now",
        "replay_wall_s": replay_wall,
        "old_six_bin_slope_m_s": old_velocity.tolist(),
        "fixed_six_bin_slope_m_s": fixed_velocity.tolist(),
        "calibration_reprojection": reprojection,
        "freshness_contract_probe": _freshness_contract_probe(config),
        "output_hashes": {
            "per_episode.json": _sha256(per_episode_path),
            "causal_observation_polls.csv": _sha256(poll_path) if poll_path.exists() else None,
            "five_bin_sampling_comparison.json": _sha256(sampling_path),
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    generated_files = [per_episode_path, poll_path, sampling_path, manifest_path]
    checksum_path = output / "SHA256SUMS"
    checksum_path.write_text("".join(
        f"{_sha256(path)}  {path.name}\n" for path in generated_files if path.exists()
    ))
    manifest["artifact_checksum_file_sha256"] = _sha256(checksum_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
