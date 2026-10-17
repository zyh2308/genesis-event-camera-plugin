#!/usr/bin/env python3
"""Exercise the shared grip gate on true holds and evaluator-only negatives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.grip_gate import ProprioceptiveGripGate
from benchmark.robot_closed_loop.robot_env import as_numpy, build_robot_scene
from benchmark.robot_closed_loop.run import jsonable, run_static_trial, save_gif


DEFAULT_SCENE = ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"
DEFAULT_CONTROL = ROOT / "benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_negative(env, name: str, position_xyz: list[float], duration_s: float, output: Path) -> dict:
    cfg = env.config
    env.reset_episode(position_xyz, 0.0)
    steps = max(1, int(round(0.05 / env.dt_s)))
    for _ in range(steps):
        env.hold_robot_home()
        env.scene.step()
    started = float(env.scene.cur_t.item() if hasattr(env.scene.cur_t, "item") else env.scene.cur_t)
    gate = ProprioceptiveGripGate(cfg["shared_control"]["proprioceptive_grip_gate"], env.dt_s)
    first_eligible_s = None
    first_contact_s = None
    bilateral_steps = 0
    left_only_steps = 0
    right_only_steps = 0
    any_contact_steps = 0
    rows = []
    total_steps = int(round(duration_s / env.dt_s))
    sample_steps = max(1, int(round(0.001 / env.dt_s)))
    for step in range(total_steps):
        env.robot.control_dofs_position(env.home_qpos[:7], env.arm_dofs)
        env.command_fingers_close()
        env.scene.step()
        now = float(env.scene.cur_t.item() if hasattr(env.scene.cur_t, "item") else env.scene.cur_t)
        gate_result = gate.update(
            now - started, 0.0, env.finger_joint_positions(), env.finger_joint_velocities(),
        )
        # The contact query is evaluator-only: it labels the diagnostic and is
        # never passed into ProprioceptiveGripGate.
        contacts = env.contacts_with_target()
        bilateral_steps += int(contacts["bilateral"])
        left_only_steps += int(contacts["left"] and not contacts["right"])
        right_only_steps += int(contacts["right"] and not contacts["left"])
        any_contact_steps += int(contacts["left"] or contacts["right"])
        if (contacts["left"] or contacts["right"]) and first_contact_s is None:
            first_contact_s = now - started
        if gate_result["eligible"] and first_eligible_s is None:
            first_eligible_s = now - started
        if step % sample_steps == 0:
            rows.append({
                "scenario": name,
                "time_since_close_s": now - started,
                "finger_joint_position_m": gate_result["finger_joint_position_m"],
                "finger_joint_velocity_m_s": gate_result["finger_joint_velocity_m_s"],
                "jaw_opening_m": env.jaw_opening_m(),
                "gate_eligible": gate_result["eligible"],
                "gate_reason": gate_result["reason"],
                "gate_stable_for_s": gate_result["stable_for_s"],
                "evaluator_left_contact": contacts["left"],
                "evaluator_right_contact": contacts["right"],
            })
    return {
        "scenario": name,
        "kind": "negative_close_only",
        "target_initial_position_m": position_xyz,
        "first_evaluator_contact_s": first_contact_s,
        "evaluator_any_contact_sample_count": any_contact_steps,
        "evaluator_left_only_sample_count": left_only_steps,
        "evaluator_right_only_sample_count": right_only_steps,
        "evaluator_bilateral_contact_sample_count": bilateral_steps,
        "first_proprioceptive_gate_eligible_s": first_eligible_s,
        "false_positive": first_eligible_s is not None,
        "duration_s": duration_s,
        "trace_rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--control-config", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-duration", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=2026104100)
    args = parser.parse_args()

    os_env = __import__("os").environ
    os_env.setdefault("PYOPENGL_PLATFORM", "egl")
    scene_cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    control_cfg = json.loads(args.control_config.read_text(encoding="utf-8"))
    scene_cfg["shared_control"] = control_cfg
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = build_robot_scene(scene_cfg)
    try:
        positive, positive_trace, positive_frames = run_static_trial(
            env, 0, args.seed, capture=True,
        )
        positive["scenario"] = "centered_true_hold"
        save_gif(out / "centered_true_hold.gif", positive_frames)
        trace_rows = [{
            "scenario": "centered_true_hold",
            "time_s": row.get("time_s"),
            "jaw_opening_m": row.get("jaw_opening_m"),
            "finger_q_m": row.get("finger_q_m"),
            "finger_qd_m_s": row.get("finger_qd_m_s"),
            "grip_completion_gate": (row.get("command") or {}).get("grip_completion_gate"),
            "contacts": row.get("contacts"),
            "score": row.get("score"),
        } for row in positive_trace]

        # Exercise front-edge contact through the actual lift and strict
        # scorer. This geometry is only a candidate negative; physical
        # success must be decided by the unchanged task scorer.
        front_edge_trial, front_edge_trace, front_edge_frames = run_static_trial(
            env, 1, args.seed + 1, capture=True,
            target_offset_xy_m=(0.025, 0.0),
        )
        front_edge_trial["scenario"] = "front_edge_x_plus_25mm_full_lift"
        save_gif(out / "front_edge_x_plus_25mm_full_lift.gif", front_edge_frames)
        with (out / "front_edge_x_plus_25mm_full_lift_trace.jsonl").open("w", encoding="utf-8") as stream:
            for row in front_edge_trace:
                stream.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")

        # The y-offset is a candidate side-contact fixture. Record its full
        # task outcome rather than assuming that the offset is single-finger.
        offset_y_trial, offset_y_trace, _ = run_static_trial(
            env, 2, args.seed + 2, capture=False,
            target_offset_xy_m=(0.0, 0.015),
        )
        offset_y_trial["scenario"] = "side_offset_y_plus_15mm_full_trial"
        with (out / "side_offset_y_plus_15mm_trace.jsonl").open("w", encoding="utf-8") as stream:
            for row in offset_y_trace:
                stream.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")

        side_offset_full_trials = []
        for index, offset_y_m in enumerate((0.020, 0.025, 0.030), start=4):
            side_trial, side_trace, side_frames = run_static_trial(
                env, index, args.seed + index, capture=True,
                target_offset_xy_m=(0.0, offset_y_m),
            )
            scenario = f"side_offset_y_plus_{int(round(offset_y_m * 1000))}mm_full_trial"
            side_trial["scenario"] = scenario
            save_gif(out / f"{scenario}.gif", side_frames)
            with (out / f"{scenario}_trace.jsonl").open("w", encoding="utf-8") as stream:
                for row in side_trace:
                    stream.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
            side_offset_full_trials.append(side_trial)

        front_edge_30_trial, front_edge_30_trace, front_edge_30_frames = run_static_trial(
            env, 3, args.seed + 3, capture=True,
            target_offset_xy_m=(0.030, 0.0),
        )
        front_edge_30_trial["scenario"] = "front_edge_x_plus_30mm_full_lift"
        save_gif(out / "front_edge_x_plus_30mm_full_lift.gif", front_edge_30_frames)
        with (out / "front_edge_x_plus_30mm_full_lift_trace.jsonl").open("w", encoding="utf-8") as stream:
            for row in front_edge_30_trace:
                stream.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")

        target_size = float(scene_cfg["target"]["size_m"][0])
        intercept_x = float(scene_cfg["controller"]["intercept_x_m"])
        negative_specs = [
            ("empty_close", [intercept_x + 0.20, 0.0, target_size / 2]),
            ("side_offset_y_plus_20mm", [intercept_x, 0.020, target_size / 2]),
            ("side_offset_y_plus_25mm", [intercept_x, 0.025, target_size / 2]),
            ("side_offset_y_plus_30mm", [intercept_x, 0.030, target_size / 2]),
            ("side_offset_y_plus_35mm", [intercept_x, 0.035, target_size / 2]),
            ("front_edge_x_plus_25mm", [intercept_x + 0.025, 0.0, target_size / 2]),
        ]
        negatives = [run_negative(env, name, position, args.negative_duration, out)
                     for name, position in negative_specs]
        for negative in negatives:
            trace_rows.extend(negative.pop("trace_rows"))
        with (out / "grip_gate_trace.jsonl").open("w", encoding="utf-8") as stream:
            for row in trace_rows:
                stream.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
        edge_false_positive = bool(
            front_edge_trial["lift_trigger_s"] is not None
            and not front_edge_trial["success"]
            and not front_edge_trial.get("probe_aborted", False)
        )
        front_edge_probe_rejected = bool(front_edge_trial.get("probe_aborted", False))
        if front_edge_trial.get("grasp_stability_abort_s") is not None:
            front_edge_monitor_abort_phase = "full_lift"
        elif front_edge_trial.get("probe_aborted", False):
            front_edge_monitor_abort_phase = "40mm_probe"
        else:
            front_edge_monitor_abort_phase = None
        front_edge_outcome = (
            "rejected_during_full_lift" if front_edge_monitor_abort_phase == "full_lift" else
            "probe_rejected_as_unstable" if front_edge_monitor_abort_phase == "40mm_probe" else
            "probe_confirmed_strict_success" if front_edge_trial.get("success", False) else
            "probe_confirmed_but_strict_failure" if front_edge_trial.get("lift_trigger_s") is not None else
            "no_probe_or_lift"
        )
        front_edge_30_false_positive = bool(
            front_edge_30_trial["lift_trigger_s"] is not None and not front_edge_30_trial["success"]
        )
        empty_false_positive = any(
            row["scenario"] == "empty_close" and row["false_positive"] for row in negatives
        )
        side_offset_false_positives = [
            row["scenario"] for row in side_offset_full_trials
            if row["lift_trigger_s"] is not None and not row["success"] and not row.get("probe_aborted", False)
        ]
        result = {
            "status": "PASS" if positive["success"] and not empty_false_positive
                      and not edge_false_positive and not front_edge_30_false_positive
                      and not side_offset_false_positives else "FAIL",
            "score_version": positive["score_version"],
            "true_hold": positive,
            "front_edge_full_lift_trial": front_edge_trial,
            "front_edge_gate_false_positive": edge_false_positive,
            "front_edge_probe_rejected": front_edge_probe_rejected,
            "front_edge_monitor_abort_phase": front_edge_monitor_abort_phase,
            "front_edge_outcome": front_edge_outcome,
            "side_offset_y_plus_15mm_full_trial": offset_y_trial,
            "front_edge_x_plus_30mm_full_lift_trial": front_edge_30_trial,
            "front_edge_x_plus_30mm_gate_false_positive": front_edge_30_false_positive,
            "side_offset_y_plus_20_25_30mm_full_trials": side_offset_full_trials,
            "side_offset_full_task_gate_false_positives": side_offset_false_positives,
            "negative_scenarios": negatives,
            "negative_scenario_gate_trigger_scope": "close-only eligibility is a request to start the shared lift probe, not a confirmed grasp or strict-success label",
            "gate_input_boundary": "gate receives only finger joint position/velocity and close-command timestamp; contacts are evaluator labels only",
            "strict_scorer_modified": False,
        }
        (out / "grip_gate_regression.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        manifest = {
            "code_commit": commit,
            "scene_config": str(args.scene_config.resolve()),
            "scene_config_sha256": sha256(args.scene_config.resolve()),
            "shared_control_config": str(args.control_config.resolve()),
            "shared_control_config_sha256": sha256(args.control_config.resolve()),
            "seed": args.seed,
            "negative_duration_s": args.negative_duration,
            "runtime": {"python": sys.version.split()[0], "genesis": getattr(env.gs, "__version__", "unknown"),
                        "physics_dt_s": env.dt_s, "sim_backend": "cpu"},
        }
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (out / "SHA256SUMS").write_text("".join(
            f"{sha256(path)}  {path.name}\n" for path in sorted(out.iterdir())
            if path.is_file() and path.name != "SHA256SUMS"
        ), encoding="utf-8")
        print(json.dumps({"status": result["status"], "strict_hold": positive["success"],
                          "close_only_probe_candidates": [r["scenario"] for r in negatives if r["false_positive"]],
                          "side_offset_full_task_false_positives": side_offset_false_positives,
                          "output": str(out)}, indent=2))
        return 0 if result["status"] == "PASS" else 2
    finally:
        env.gs.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
