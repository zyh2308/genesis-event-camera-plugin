#!/usr/bin/env python3
"""Replay paired 1/2 kHz raw events through the causal five-bin controller."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.event_state import LearnedEventStateEstimator
from benchmark.robot_closed_loop.predictor import BoundedLateralCommand, FixedDecelerationPredictor


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _interp_trace(trace: list[dict], time_s: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([float(row["time_s"]) for row in trace], dtype=np.float64)
    before = max(0, min(len(times) - 1, int(np.searchsorted(times, time_s, side="right") - 1)))
    after = min(len(times) - 1, before + 1)
    left, right = trace[before], trace[after]
    t0, t1 = float(left["time_s"]), float(right["time_s"])
    alpha = 0.0 if t1 <= t0 else float(np.clip((time_s - t0) / (t1 - t0), 0.0, 1.0))
    p0 = np.asarray(left["object"]["position_m"], dtype=np.float64)
    p1 = np.asarray(right["object"]["position_m"], dtype=np.float64)
    v0 = np.asarray(left["object"]["velocity_m_s"], dtype=np.float64)
    v1 = np.asarray(right["object"]["velocity_m_s"], dtype=np.float64)
    return p0 * (1.0 - alpha) + p1 * alpha, v0 * (1.0 - alpha) + v1 * alpha


def _controller_replay(estimator, events, trace, row, cfg, hz):
    controller = cfg["controller"]
    predictor = FixedDecelerationPredictor(
        float(controller["intercept_x_m"]),
        float(controller["initial_deceleration_assumption_m_s2"]),
    )
    robot = cfg["robot"]
    y_cmd = BoundedLateralCommand(
        0.0, float(robot["cartesian_speed_limit_m_s"]),
        float(robot["cartesian_acceleration_limit_m_s2"]),
    )
    lead_cfg = controller["closure_lead_speed_schedule"]
    latency = float(controller["observation_latency_s"])
    control_dt = 1.0 / float(controller["high_level_hz"])
    last_cutoff = -float("inf")
    speed_reference = None
    terminal_commit = False
    close_started = None
    rows = []
    polls = [r for r in trace if r.get("controller_update") and r.get("event_observation")]
    for item in polls:
        observed = item["event_observation"]
        poll_time = float(item["time_s"])
        cutoff = observed.get("state_time_s")
        if cutoff is None:
            cutoff = poll_time - latency
        cutoff = float(cutoff)
        causal_chunk = events[(events[:, 0] >= last_cutoff) & (events[:, 0] < cutoff)]
        estimator.push(causal_chunk, cutoff)
        last_cutoff = cutoff
        obs = estimator.observe(cutoff, now_s=cutoff + float(observed.get("age_s") or latency))
        estimate = obs.state
        gt_position, gt_velocity = _interp_trace(trace, cutoff)
        predicted_ttc = None
        predicted_y = None
        close_lead = float(lead_cfg["max_lead_s"])
        close_decision = False
        if estimate is not None:
            prediction = predictor.predict(estimate, float(obs.age_s or 0.0))
            if speed_reference is None and prediction.propagated_state is not None:
                speed_reference = float(np.linalg.norm(prediction.propagated_state.velocity_m_s[:2]))
            if speed_reference is not None:
                close_lead = float(np.clip(
                    float(lead_cfg["reference_lead_s"])
                    + float(lead_cfg["slope_s_per_mps"])
                    * (speed_reference - float(lead_cfg["reference_speed_m_s"])),
                    float(lead_cfg["min_lead_s"]), float(lead_cfg["max_lead_s"]),
                ))
            if prediction.reachable:
                predicted_ttc = float(prediction.ttc_s)
                predicted_y = float(prediction.position_m[1])
                if not terminal_commit:
                    y_cmd.step(predicted_y, control_dt)
                    if prediction.ttc_s <= float(controller["terminal_commit_guard_s"]):
                        terminal_commit = True
                close_decision = bool(0.0 <= prediction.ttc_s <= close_lead)
                if close_started is None and close_decision:
                    close_started = poll_time
        gt_prediction = predictor.predict(
            __import__("benchmark.robot_closed_loop.predictor", fromlist=["MotionState"]).MotionState(
                cutoff, gt_position, gt_velocity,
            ), float(obs.age_s or 0.0),
        )
        ttc_error = None if predicted_ttc is None or not gt_prediction.reachable else predicted_ttc - gt_prediction.ttc_s
        rows.append({
            "sampling_hz": hz,
            "poll_time_s": poll_time,
            "cutoff_time_s": cutoff,
            "status": obs.state_status,
            "event_count": obs.event_count_window,
            "time_bin_count": obs.time_bin_count_window,
            "state_age_ms": None if obs.age_s is None else obs.age_s * 1000.0,
            "estimated_position_m": None if estimate is None else estimate.position_m.tolist(),
            "estimated_velocity_m_s": None if estimate is None else estimate.velocity_m_s.tolist(),
            "position_error_xy_mm": None if estimate is None else float(np.linalg.norm(estimate.position_m[:2] - gt_position[:2]) * 1000.0),
            "velocity_error_xy_m_s": None if estimate is None else float(np.linalg.norm(estimate.velocity_m_s[:2] - gt_velocity[:2])),
            "predicted_ttc_s": predicted_ttc,
            "gt_current_ttc_s": None if not gt_prediction.reachable else float(gt_prediction.ttc_s),
            "ttc_error_ms": None if ttc_error is None else ttc_error * 1000.0,
            "predicted_intercept_y_m": predicted_y,
            "commanded_y_m": y_cmd.position_m,
            "close_lead_s": close_lead,
            "close_decision": close_decision,
            "close_started_s": close_started,
            "velocity_source": estimator.latest_diagnostics.get("velocity_source"),
        })
    return rows


def _summary(values):
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    return {"n": int(len(arr)), "median": None if not len(arr) else float(np.median(arr)),
            "p95_abs": None if not len(arr) else float(np.percentile(np.abs(arr), 95))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads(args.scene_config.read_text(encoding="utf-8"))
    scene["shared_control"] = json.loads(args.control_config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    calibration = checkpoint["spatial_calibration"]
    camera = np.asarray(calibration["camera_to_world_cv"], dtype=np.float64)
    intrinsics = calibration["intrinsics_fx_fy_cx_cy"]
    all_rows = []
    episode_summaries = []
    for episode_dir in sorted((args.rollout_root / "episodes").glob("episode_*")):
        raw_path = episode_dir / "direct_hdr_events_1khz_2khz.npz"
        trace_path = episode_dir / "state_control_trace.jsonl"
        csv_path = episode_dir / "episode.csv"
        if not (raw_path.is_file() and trace_path.is_file() and csv_path.is_file()):
            continue
        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            episode_summary = next(csv.DictReader(stream))
        contact_time_s = (
            None if not episode_summary.get("first_contact_s")
            else float(episode_summary["first_contact_s"])
        )
        with np.load(raw_path) as archive:
            arrays = {hz: np.asarray(archive[f"events_{hz}hz"], dtype=np.float64) for hz in (1000, 2000)}
        per_hz = {}
        for hz in (1000, 2000):
            estimator = LearnedEventStateEstimator(
                str(args.checkpoint), scene["camera"]["resolution_wh"],
                max_state_age_s=float(scene["controller"]["maximum_event_state_age_s"]),
                fixed_height_m=float(scene["shared_control"]["close_time_estimator"]["heatmap_world_height_m"]),
                velocity_window_s=float(scene["shared_control"]["close_time_estimator"]["velocity_window_s"]),
                minimum_regression_samples=int(scene["shared_control"]["close_time_estimator"]["minimum_regression_samples"]),
                camera_to_world_cv=camera,
                intrinsics_fx_fy_cx_cy=intrinsics,
            )
            rows = _controller_replay(estimator, arrays[hz], trace, None, scene, hz)
            all_rows.extend({"episode": episode_dir.name, **row} for row in rows)
            precontact = [r for r in rows if contact_time_s is None or r["poll_time_s"] < contact_time_s]
            per_hz[str(hz)] = {
                "polls": len(rows),
                "fresh_fraction": None if not rows else sum(r["status"] == "fresh" for r in rows) / len(rows),
                "precontact_position_error_xy_mm": _summary([r["position_error_xy_mm"] for r in precontact]),
                "precontact_velocity_error_xy_m_s": _summary([r["velocity_error_xy_m_s"] for r in precontact]),
                "precontact_ttc_error_ms": _summary([r["ttc_error_ms"] for r in precontact]),
                "close_poll_time_s": next((r["close_started_s"] for r in rows if r["close_started_s"] is not None), None),
                "close_lead_s_at_trigger": next((r["close_lead_s"] for r in rows if r["close_started_s"] is not None), None),
                "close_ttc_error_vs_actual_crossing_ms": (
                    None if not episode_summary.get("actual_crossing_t_s")
                    else (next((r["predicted_ttc_s"] for r in rows if r["close_started_s"] is not None), 0.0)
                          - float(episode_summary["actual_crossing_t_s"])
                          + float(next((r["close_started_s"] for r in rows if r["close_started_s"] is not None), 0.0))) * 1000.0
                ),
                "inference_ms_p95": None if not estimator.inference_ms else float(np.percentile(estimator.inference_ms, 95)),
            }
        rows_1 = [r for r in all_rows if r["episode"] == episode_dir.name and r["sampling_hz"] == 1000]
        rows_2 = [r for r in all_rows if r["episode"] == episode_dir.name and r["sampling_hz"] == 2000]
        paired = [(a, b) for a, b in zip(rows_1, rows_2) if a["status"] == b["status"] == "fresh"]
        episode_summaries.append({
            "episode": episode_dir.name,
            "sampling_by_hz": per_hz,
            "paired_fresh_poll_count": len(paired),
            "paired_state_xy_difference_mm": _summary([
                np.linalg.norm(np.asarray(a["estimated_position_m"])[:2] - np.asarray(b["estimated_position_m"])[:2]) * 1000
                for a, b in paired
            ]),
            "paired_state_velocity_xy_difference_m_s": _summary([
                np.linalg.norm(np.asarray(a["estimated_velocity_m_s"])[:2] - np.asarray(b["estimated_velocity_m_s"])[:2])
                for a, b in paired
            ]),
            "paired_ttc_difference_ms": _summary([
                None if a["predicted_ttc_s"] is None or b["predicted_ttc_s"] is None else
                (a["predicted_ttc_s"] - b["predicted_ttc_s"]) * 1000 for a, b in paired
            ]),
            "close_action_agreement": None if not paired else float(np.mean([a["close_decision"] == b["close_decision"] for a, b in paired])),
            "commanded_y_difference_mm_p95": None if not paired else float(np.percentile([
                abs(a["commanded_y_m"] - b["commanded_y_m"]) * 1000 for a, b in paired
            ], 95)),
        })
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "decision_poll_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]) if all_rows else ["episode", "sampling_hz"])
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "status": "PASS" if episode_summaries else "STOP",
        "episodes": episode_summaries,
        "input_contract": "causal five-bin 64x64 tensors; event timestamps satisfy t < cutoff; labels only used after inference",
        "decision_metric": "fixed-deceleration TTC, 100 Hz lateral command, closure schedule, and trigger agreement are replayed from raw 1/2 kHz event streams",
        "paired_pixel_model": "both source streams already use the shared Direct-HDR ideal DvsEmulator response; this tests cadence/model-input sufficiency only",
        "input_hashes": {
            str(p.relative_to(args.rollout_root)): sha256(p)
            for p in sorted(args.rollout_root.glob("episodes/episode_*/direct_hdr_events_1khz_2khz.npz"))
        },
        "label_and_control_trace_hashes": {
            str(p.relative_to(args.rollout_root)): sha256(p)
            for pattern in ("episodes/episode_*/state_control_trace.jsonl", "episodes/episode_*/episode.csv")
            for p in sorted(args.rollout_root.glob(pattern))
        },
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "scene_config_sha256": sha256(args.scene_config),
        "shared_control_config_sha256": sha256(args.control_config),
    }
    (args.output / "sampling_decision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "episodes": len(episode_summaries), "output": str(args.output)}, indent=2))
    return 0 if episode_summaries else 2


if __name__ == "__main__":
    raise SystemExit(main())
