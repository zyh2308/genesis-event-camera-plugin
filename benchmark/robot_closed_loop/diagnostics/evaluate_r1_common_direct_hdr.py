#!/usr/bin/env python3
"""Evaluate two event-state checkpoints on the same Direct-HDR reference.

This is a development evaluator. Both estimators receive the same cached event
slice at each poll and are scored against the same post-inference GT state.
The arrival/intercept reference applies the fixed predictor to GT at the causal
cutoff; it is not measured plane-crossing ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.replay_event_state import _camera_to_world_cv
from benchmark.robot_closed_loop.event_state import LearnedEventStateEstimator
from benchmark.robot_closed_loop.predictor import FixedDecelerationPredictor, MotionState


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_hashes(reference: Path) -> Dict[str, str]:
    expected = {}
    for line in (reference / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        expected[relative.lstrip("* ")] = digest
    return expected


def _checked(path: Path, reference: Path, expected: Dict[str, str]) -> str:
    relative = str(path.relative_to(reference))
    actual = _sha256(path)
    if actual != expected.get(relative):
        raise RuntimeError(f"reference hash mismatch: {relative}")
    return actual


def _stats(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": int(len(array)), "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
    }


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for source in ("ours", "v2e"):
        selected = [row for row in rows if row["source"] == source]
        reachable = [row for row in selected if row["reference_reachable"]]
        state_rows = [row for row in selected if row["state_available"]]
        predicted_reachable = [row for row in selected if row.get("prediction_reachable") is True]
        false_unreachable = [row for row in reachable if row.get("prediction_reachable") is False]
        selected_metrics = {
            key: _stats(row[key] for row in selected if row.get(key) is not None)
            for key in (
                "position_xy_error_m", "velocity_xy_error_m_s",
                "intercept_xy_error_m", "intercept_y_signed_error_m",
                "arrival_signed_error_s", "arrival_abs_error_s", "state_age_ms",
            )
        }
        selected_metrics = {key: value for key, value in selected_metrics.items() if value["n"] or key in (
            "position_xy_error_m", "velocity_xy_error_m_s", "intercept_xy_error_m",
            "intercept_y_signed_error_m", "arrival_signed_error_s", "arrival_abs_error_s", "state_age_ms")}
        unreachable = [row for row in selected if not row["reference_reachable"]]
        summary[source] = {
            "poll_count": len(selected),
            "state_available_count": len(state_rows),
            "state_available_fraction_all_polls": len(state_rows) / len(selected) if selected else None,
            "state_status_counts": {
                status: sum(row["state_status"] == status for row in selected)
                for status in sorted({row["state_status"] for row in selected})
            },
            "gt_predictor_reachable_poll_count": len(reachable),
            "gt_predictor_unreachable_poll_count": len(unreachable),
            "predicted_reachable_count_all_polls": len(predicted_reachable),
            "reachable_decision_coverage_on_gt_reachable": {
                "correct_reachable": sum(row.get("prediction_reachable") is True for row in reachable),
                "false_unreachable": len(false_unreachable),
                "missing_state": sum(not row["state_available"] for row in reachable),
                "fraction_correct_reachable": (
                    sum(row.get("prediction_reachable") is True for row in reachable) / len(reachable)
                    if reachable else None
                ),
            },
            "unreachable_decision_coverage_on_gt_unreachable": {
                "correct_unreachable": sum(row.get("prediction_reachable") is False for row in unreachable),
                "false_reachable": sum(row.get("prediction_reachable") is True for row in unreachable),
                "missing_state": sum(not row["state_available"] for row in unreachable),
                "fraction_correct_unreachable": (
                    sum(row.get("prediction_reachable") is False for row in unreachable) / len(unreachable)
                    if unreachable else None
                ),
            },
            "metrics": selected_metrics,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--ours-checkpoint", type=Path, required=True)
    parser.add_argument("--v2e-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deceleration", type=float, required=True)
    parser.add_argument("--latency", type=float, default=0.01)
    parser.add_argument("--poll-period", type=float, default=0.01)
    parser.add_argument("--first-poll", type=float, default=0.04)
    parser.add_argument("--max-state-age", type=float, default=0.03)
    args = parser.parse_args()

    scene_path = args.scene_config.resolve()
    control_path = args.control_config.resolve()
    reference = args.reference_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(scene_path.read_text(encoding="utf-8"))
    control = json.loads(control_path.read_text(encoding="utf-8"))
    config["shared_control"] = control
    summary_path = reference / "r0_5_reference_summary.json"
    episodes = json.loads(summary_path.read_text(encoding="utf-8"))["episodes"]
    expected_hashes = _reference_hashes(reference)
    camera = config["camera"]
    width, height = (int(v) for v in camera["resolution_wh"])
    focal = height / (2.0 * np.tan(np.deg2rad(float(camera["vertical_fov_deg"])) / 2.0))
    camera_pose = _camera_to_world_cv(config)
    predictor = FixedDecelerationPredictor(config["controller"]["intercept_x_m"], args.deceleration)

    def make_estimators():
        # Reset temporal state at every episode boundary while reusing the same
        # checkpoint and calibration. This prevents event/history carryover.
        return {
            source: LearnedEventStateEstimator(
                str(checkpoint.resolve()), [width, height], max_state_age_s=args.max_state_age,
                fixed_height_m=float(config["target"]["size_m"][2]) / 2.0,
                velocity_window_s=0.05, minimum_regression_samples=4,
                camera_to_world_cv=camera_pose,
                intrinsics_fx_fy_cx_cy=(focal, focal, width / 2.0, height / 2.0),
            )
            for source, checkpoint in (("ours", args.ours_checkpoint), ("v2e", args.v2e_checkpoint))
        }

    rows: List[Dict[str, Any]] = []
    provenance = []
    started = time.perf_counter()
    for episode in episodes:
        estimators = make_estimators()
        episode_id = int(episode["episode_id"])
        episode_dir = reference / "episodes" / f"episode_{episode_id:03d}"
        event_path = episode_dir / "direct_hdr_events_1khz_2khz.npz"
        trace_path = episode_dir / "state_control_trace.jsonl"
        event_hash = _checked(event_path, reference, expected_hashes)
        trace_hash = _checked(trace_path, reference, expected_hashes)
        events = np.load(event_path)["events_2000hz"]
        events = events[np.argsort(events[:, 0], kind="stable")]
        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        times = np.asarray([item["time_s"] for item in trace], dtype=np.float64)
        positions = np.asarray([item["object"]["position_m"] for item in trace], dtype=np.float64)
        velocities = np.asarray([item["object"]["velocity_m_s"] for item in trace], dtype=np.float64)
        contact_s = float(episode["first_contact_s"])
        polls = np.arange(args.first_poll, contact_s - 1e-9, args.poll_period)
        provenance.append({
            "episode_id": episode_id, "seed": int(episode["seed"]),
            "speed_m_s": float(episode["speed_m_s"]), "launch_y_m": float(episode["launch_y_m"]),
            "events_sha256": event_hash, "trace_sha256": trace_hash,
            "poll_count": int(len(polls)), "first_contact_s": contact_s,
        })
        cursor = 0
        for now_s in polls:
            cutoff_s = round(float(now_s) - args.latency, 10)
            stop = int(np.searchsorted(events[:, 0], now_s, side="left"))
            observations = {}
            for source, estimator in estimators.items():
                estimator.push(events[cursor:stop], float(now_s))
                observations[source] = estimator.observe(cutoff_s, float(now_s))
            cursor = stop

            # Ground truth is queried only after both event-only estimators ran.
            true_position = np.asarray([np.interp(cutoff_s, times, positions[:, k]) for k in range(3)])
            true_velocity = np.asarray([np.interp(cutoff_s, times, velocities[:, k]) for k in range(3)])
            gt_prediction = predictor.predict(MotionState(cutoff_s, true_position, true_velocity), args.latency)
            for source, observation in observations.items():
                row: Dict[str, Any] = {
                    "source": source, "episode_id": episode_id, "seed": int(episode["seed"]),
                    "speed_m_s": float(episode["speed_m_s"]), "launch_y_m": float(episode["launch_y_m"]),
                    "poll_s": float(now_s), "cutoff_s": cutoff_s,
                    "state_status": observation.state_status,
                    "tracking_status": observation.tracking_status,
                    "state_age_s": observation.age_s,
                    "state_age_ms": (None if observation.age_s is None else observation.age_s * 1000.0),
                    "state_available": observation.state is not None,
                    "reference_reachable": bool(gt_prediction.reachable),
                    "prediction_reachable": None,
                    "position_xy_error_m": None, "velocity_xy_error_m_s": None,
                    "intercept_xy_error_m": None, "intercept_y_signed_error_m": None,
                    "arrival_signed_error_s": None, "arrival_abs_error_s": None,
                    "last_100ms_before_contact": bool(now_s >= contact_s - 0.1),
                }
                if observation.state is not None:
                    position_error = observation.state.position_m - true_position
                    velocity_error = observation.state.velocity_m_s - true_velocity
                    row["position_xy_error_m"] = float(np.linalg.norm(position_error[:2]))
                    row["velocity_xy_error_m_s"] = float(np.linalg.norm(velocity_error[:2]))
                    prediction = predictor.predict(observation.state, args.latency)
                    row["prediction_reachable"] = bool(prediction.reachable)
                    if prediction.reachable and gt_prediction.reachable:
                        intercept_delta = prediction.position_m[:2] - gt_prediction.position_m[:2]
                        row["intercept_xy_error_m"] = float(np.linalg.norm(intercept_delta))
                        row["intercept_y_signed_error_m"] = float(intercept_delta[1])
                        row["arrival_signed_error_s"] = float(prediction.ttc_s - gt_prediction.ttc_s)
                        row["arrival_abs_error_s"] = abs(row["arrival_signed_error_s"])
                rows.append(row)

    overall = _summarize(rows)
    summary = {
        "purpose": "R1 common Direct-HDR reference development evaluation; paired training is not a ranking",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scene_config_sha256": _sha256(scene_path), "control_config_sha256": _sha256(control_path),
        "checkpoint_sha256": {"ours": _sha256(args.ours_checkpoint.resolve()), "v2e": _sha256(args.v2e_checkpoint.resolve())},
        "reference_root": str(reference), "reference_manifest_sha256": _sha256(reference / "SHA256SUMS"),
        "reference_summary_sha256": _sha256(summary_path), "sampling_hz": 2000,
        "poll_hz": 1.0 / args.poll_period, "latency_s": args.latency,
        "deceleration_m_s2": args.deceleration,
        "arrival_and_intercept_reference": "the same frozen predictor applied to evaluation GT state at the causal cutoff; not measured plane-crossing ground truth",
        "reference_events_are_common_to_both_models": True,
        "provenance": provenance,
        "overall": overall,
        "last_100ms_before_contact": _summarize([row for row in rows if row["last_100ms_before_contact"]]),
        "by_speed": {
            str(speed): _summarize([row for row in rows if row["speed_m_s"] == speed])
            for speed in sorted({float(ep["speed_m_s"]) for ep in episodes})
        },
        "by_episode": {
            str(episode["episode_id"]): _summarize([row for row in rows if row["episode_id"] == episode["episode_id"]])
            for episode in episodes
        },
        "wall_s": time.perf_counter() - started,
    }
    (out / "r1_common_direct_hdr_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "r1_common_direct_hdr_polls.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output_files = [path for path in sorted(out.iterdir()) if path.is_file()]
    (out / "SHA256SUMS").write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in output_files), encoding="utf-8")
    print(json.dumps({"output": str(out), "overall": overall, "wall_s": summary["wall_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
