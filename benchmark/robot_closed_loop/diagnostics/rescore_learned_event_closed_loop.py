#!/usr/bin/env python3
"""Evaluator-only phase metrics for a saved learned-event closed-loop run.

The logged event estimate and controller prediction are read first. Simulator
GT is then used only to align and score those outputs; it is never sent back
through the estimator, predictor, controller, or lift gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from benchmark.robot_closed_loop.predictor import FixedDecelerationPredictor, MotionState


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _read_trace(path: Path) -> List[Dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _truth_arrays(trace: Sequence[Dict[str, Any]]):
    rows = [row for row in trace if row.get("object") and row.get("time_s") is not None]
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    positions = np.asarray([row["object"]["position_m"] for row in rows], dtype=np.float64)
    velocities = np.asarray([row["object"]["velocity_m_s"] for row in rows], dtype=np.float64)
    return times, positions, velocities


def _truth_at(times, positions, velocities, timestamp: float) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([np.interp(timestamp, times, positions[:, axis]) for axis in range(3)]),
        np.asarray([np.interp(timestamp, times, velocities[:, axis]) for axis in range(3)]),
    )


def _phase(now_s: float, contact_s: Optional[float], lift_s: Optional[float]) -> str:
    if contact_s is None or now_s < contact_s:
        return "pre_contact"
    if lift_s is None or now_s < lift_s:
        return "contact_to_lift_or_episode_end"
    return "post_lift"


def _percentiles(values: Iterable[float]) -> Dict[str, Optional[float]]:
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": int(len(values)), "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def _optional_float(value: str) -> Optional[float]:
    return None if value in (None, "") else float(value)


def _load_episode(path: Path) -> Dict[str, str]:
    with path.open(newline="") as stream:
        return next(csv.DictReader(stream))


def _score_episode(
    run_root: Path,
    episode_id: int,
    config: Dict[str, Any],
    predictor: FixedDecelerationPredictor,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, str]]:
    episode_dir = run_root / "episodes" / f"episode_{episode_id:03d}"
    trace_path = episode_dir / "state_control_trace.jsonl"
    episode_path = episode_dir / "episode.csv"
    trace = _read_trace(trace_path)
    episode = _load_episode(episode_path)
    times, positions, velocities = _truth_arrays(trace)
    contact_s = _optional_float(episode.get("first_contact_s", ""))
    lift_s = _optional_float(episode.get("lift_start_s", ""))
    latency_limit_s = float(config["controller"].get("maximum_event_state_age_s", 0.030))
    rows: List[Dict[str, Any]] = []

    for frame in trace:
        if not frame.get("controller_update"):
            continue
        now_s = float(frame["time_s"])
        status = frame.get("event_observation") or {}
        estimate = (frame.get("event_state_by_rate") or {}).get("2000")
        current_state = frame.get("controller_observation_state")
        state_status = str(status.get("state_status", "unavailable"))
        age_s = None if status.get("age_s") is None else float(status["age_s"])
        row: Dict[str, Any] = {
            "episode_id": episode_id,
            "time_s": now_s,
            "phase": _phase(now_s, contact_s, lift_s),
            "event_state_status": state_status,
            "tracking_status": status.get("tracking_status"),
            "event_count_window": status.get("event_count_window"),
            "time_bin_count_window": status.get("time_bin_count_window"),
            "event_input_age_ms": None if age_s is None else age_s * 1000.0,
            "usable_fresh_state": bool(
                state_status == "fresh" and age_s is not None and age_s <= latency_limit_s + 1e-9
                and estimate is not None and current_state is not None
            ),
            "estimate_timestamp_s": None if estimate is None else float(estimate["time_s"]),
            "position_error_xy_m": None,
            "velocity_error_xy_m_s": None,
            "event_prediction_reachable": None,
            "gt_same_cutoff_prediction_reachable": None,
            "absolute_arrival_error_s": None,
            "absolute_intercept_y_error_m": None,
            "event_prediction_ttc_s": None,
            "gt_same_cutoff_prediction_ttc_s": None,
            "event_intercept_y_m": None,
            "gt_same_cutoff_intercept_y_m": None,
            "closing": (frame.get("command") or {}).get("closing"),
            "terminal_commit": (frame.get("command") or {}).get("terminal_commit"),
            "lift_gate_eligible": ((frame.get("command") or {}).get("lift_gate") or {}).get("eligible"),
        }
        if estimate is not None and age_s is not None:
            estimate_t = float(estimate["time_s"])
            gt_position, gt_velocity = _truth_at(times, positions, velocities, estimate_t)
            estimate_position = np.asarray(estimate["position_m"], dtype=np.float64)
            estimate_velocity = np.asarray(estimate["velocity_m_s"], dtype=np.float64)
            row["position_error_xy_m"] = float(np.linalg.norm(estimate_position[:2] - gt_position[:2]))
            row["velocity_error_xy_m_s"] = float(np.linalg.norm(estimate_velocity[:2] - gt_velocity[:2]))
            event_prediction = predictor.predict(
                MotionState(estimate_t, estimate_position, estimate_velocity), age_s,
            )
            gt_prediction = predictor.predict(
                MotionState(estimate_t, gt_position, gt_velocity), age_s,
            )
            row["event_prediction_reachable"] = bool(event_prediction.reachable)
            row["gt_same_cutoff_prediction_reachable"] = bool(gt_prediction.reachable)
            if event_prediction.reachable and gt_prediction.reachable:
                row["event_prediction_ttc_s"] = float(event_prediction.ttc_s)
                row["gt_same_cutoff_prediction_ttc_s"] = float(gt_prediction.ttc_s)
                row["event_intercept_y_m"] = float(event_prediction.position_m[1])
                row["gt_same_cutoff_intercept_y_m"] = float(gt_prediction.position_m[1])
                row["absolute_arrival_error_s"] = abs(float(event_prediction.ttc_s - gt_prediction.ttc_s))
                row["absolute_intercept_y_error_m"] = abs(
                    float(event_prediction.position_m[1] - gt_prediction.position_m[1])
                )
        rows.append(row)

    score = json.loads(episode.get("strict_score", "{}") or "{}")
    first_contact_position = json.loads(episode.get("first_contact_position_m", "null") or "null")
    contact_location_error = json.loads(episode.get("first_contact_location_error", "{}") or "{}")
    result = {
        "episode_id": episode_id,
        "seed": int(episode["seed"]),
        "speed_m_s": float(episode["launch_speed_m_s"]),
        "launch_y_m": float(episode["start_y_m"]),
        "strict_success": episode.get("success_strict_r0_5", episode.get("success", "False")) == "True",
        "strict_score_version": episode.get("score_version"),
        "failure_primary": episode.get("failure_primary", score.get("failure_primary")),
        "first_contact_s": contact_s,
        "first_bilateral_contact_s": _optional_float(episode.get("first_bilateral_contact_s", "")),
        "close_start_s": _optional_float(episode.get("close_start_s", "")),
        "terminal_commit_s": _optional_float(episode.get("terminal_commit_s", "")),
        "lift_start_s": lift_s,
        "planned_free_slide_arrival_t_s": _optional_float(episode.get("planned_free_slide_arrival_t_s", "")),
        "close_minus_planned_arrival_s": _optional_float(episode.get("close_start_minus_planned_arrival_s", "")),
        "contact_minus_planned_arrival_s": _optional_float(episode.get("first_contact_minus_planned_arrival_s", "")),
        "contact_position_m": first_contact_position,
        "contact_location_error": contact_location_error,
        "contact_relative_speed_m_s": _optional_float(episode.get("first_contact_relative_speed_m_s", "")),
        "hold_s": _optional_float(episode.get("hold_s", "")),
        "minimum_true_bottom_clearance_m": score.get("minimum_true_bottom_clearance_m"),
        "trace_sha256": _sha256(trace_path),
        "episode_csv_sha256": _sha256(episode_path),
    }
    close_s = _optional_float(episode.get("close_start_s", ""))
    before_close = [row for row in rows if close_s is not None and row["time_s"] <= close_s and row["usable_fresh_state"]]
    result["last_fresh_poll_at_or_before_close"] = None if not before_close else before_close[-1]
    return result, rows, episode


def _phase_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"controller_polls": len(rows)}
    usable = [row for row in rows if row["usable_fresh_state"]]
    result["fresh_state_count"] = len(usable)
    result["fresh_state_fraction"] = None if not rows else len(usable) / len(rows)
    ages = [row["event_input_age_ms"] for row in rows if row["event_input_age_ms"] is not None]
    result["event_input_age_ms"] = _percentiles(ages)
    for key, target in (
        ("position_error_xy_m", "position_error_xy_m"),
        ("velocity_error_xy_m_s", "velocity_error_xy_m_s"),
        ("absolute_arrival_error_s", "absolute_arrival_error_s"),
        ("absolute_intercept_y_error_m", "absolute_intercept_y_error_m"),
    ):
        result[target] = _percentiles(row[key] for row in rows if row[key] is not None)
    valid_pair = [row for row in rows if row["event_prediction_reachable"] and row["gt_same_cutoff_prediction_reachable"]]
    result["reachable_prediction_pair_count"] = len(valid_pair)
    result["reachable_prediction_pair_fraction_of_polls"] = None if not rows else len(valid_pair) / len(rows)
    gates = [row["lift_gate_eligible"] for row in rows if row["lift_gate_eligible"] is not None]
    result["lift_gate_evaluations"] = len(gates)
    result["lift_gate_eligible_count"] = sum(bool(value) for value in gates)
    result["lift_gate_eligible_fraction"] = None if not gates else sum(bool(value) for value in gates) / len(gates)
    result["closing_poll_count"] = sum(bool(row["closing"]) for row in rows)
    result["terminal_commit_poll_count"] = sum(bool(row["terminal_commit"]) for row in rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-results", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_results.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    episode_ids = sorted(
        int(path.name.removeprefix("episode_"))
        for path in (run_root / "episodes").glob("episode_*") if path.is_dir()
    )
    controller = config["controller"]
    predictor = FixedDecelerationPredictor(
        float(controller["intercept_x_m"]),
        float(controller["initial_deceleration_assumption_m_s2"]),
    )
    episode_results: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    rows_by_episode: Dict[int, List[Dict[str, Any]]] = {}
    for episode_id in episode_ids:
        result, rows, _episode = _score_episode(run_root, episode_id, config, predictor)
        episode_results.append(result)
        all_rows.extend(rows)
        rows_by_episode[episode_id] = rows

    phase_names = ("pre_contact", "contact_to_lift_or_episode_end", "post_lift")
    phase_summary = {}
    for phase in phase_names:
        phase_rows = [row for row in all_rows if row["phase"] == phase]
        phase_summary[phase] = _phase_summary(phase_rows)
        phase_summary[phase]["per_episode"] = {
            str(episode_id): _phase_summary([row for row in rows if row["phase"] == phase])
            for episode_id, rows in rows_by_episode.items()
        }

    successful_ids = {row["episode_id"] for row in episode_results if row["strict_success"]}
    phase_by_outcome = {}
    for outcome_name, episode_set in (("strict_success", successful_ids),
                                      ("strict_failure", set(episode_ids) - successful_ids)):
        phase_by_outcome[outcome_name] = {
            phase: _phase_summary([
                row for row in all_rows
                if row["phase"] == phase and row["episode_id"] in episode_set
            ])
            for phase in phase_names
        }

    args.out.mkdir(parents=True, exist_ok=False)
    fields = list(all_rows[0]) if all_rows else []
    with (args.out / "controller_poll_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "phase": "R0.7 learned-event closed-loop evaluator-only rescore",
        "code_commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "run_code_commit": manifest.get("code_commit_at_run_start"),
        "run_manifest_sha256": _sha256(manifest_path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": manifest.get("runtime", {}).get("event_model_checkpoint_sha256"),
        "episode_count": len(episode_results),
        "strict_success_count": sum(bool(row["strict_success"]) for row in episode_results),
        "episodes": episode_results,
        "phase_metrics": phase_summary,
        "phase_metrics_by_strict_outcome": phase_by_outcome,
        "metric_boundary": (
            "The saved 2 kHz learned state is scored against interpolated GT at its causal state timestamp. "
            "Arrival and intercept errors compare the learned state and evaluator-only GT state at the same "
            "cutoff through the same fixed predictor with the logged event input age; no future GT arrival "
            "is used. Input age is read from EventObservation.age_s, not the propagated state age at each "
            "0.5 ms log frame."
        ),
        "status_boundary": "Fresh status means the event-window count/bin and age rules passed; it does not imply localization accuracy.",
        "outputs": {
            "controller_poll_metrics.csv": {
                "bytes": (args.out / "controller_poll_metrics.csv").stat().st_size,
                "sha256": _sha256(args.out / "controller_poll_metrics.csv"),
            },
        },
    }
    (args.out / "closed_loop_rescore.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "SHA256SUMS").write_text(
        "".join(f"{_sha256(args.out / name)}  {name}\n" for name in ("controller_poll_metrics.csv", "closed_loop_rescore.json")),
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(args.out.resolve()),
        "strict_success": f"{summary['strict_success_count']}/{summary['episode_count']}",
        "phases": {key: {
            "polls": value["controller_polls"],
            "fresh_fraction": value["fresh_state_fraction"],
            "age_p95_ms": value["event_input_age_ms"]["p95"],
            "position_p95_mm": None if value["position_error_xy_m"]["p95"] is None else value["position_error_xy_m"]["p95"] * 1000,
            "arrival_p95_ms": None if value["absolute_arrival_error_s"]["p95"] is None else value["absolute_arrival_error_s"]["p95"] * 1000,
            "intercept_y_p95_mm": None if value["absolute_intercept_y_error_m"]["p95"] is None else value["absolute_intercept_y_error_m"]["p95"] * 1000,
        } for key, value in phase_summary.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
