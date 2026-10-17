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
from benchmark.robot_closed_loop.run import run_static_trial, save_gif


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

        target_size = float(scene_cfg["target"]["size_m"][0])
        intercept_x = float(scene_cfg["controller"]["intercept_x_m"])
        negative_specs = [
            ("empty_close", [intercept_x + 0.20, 0.0, target_size / 2]),
            ("one_finger_side_edge_y_plus_35mm", [intercept_x, 0.035, target_size / 2]),
            ("front_edge_x_plus_25mm", [intercept_x + 0.025, 0.0, target_size / 2]),
        ]
        negatives = [run_negative(env, name, position, args.negative_duration, out)
                     for name, position in negative_specs]
        for negative in negatives:
            trace_rows.extend(negative.pop("trace_rows"))
        with (out / "grip_gate_trace.jsonl").open("w", encoding="utf-8") as stream:
            for row in trace_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = {
            "status": "PASS" if positive["success"] and not any(row["false_positive"] for row in negatives) else "FAIL",
            "score_version": positive["score_version"],
            "true_hold": positive,
            "negative_scenarios": negatives,
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
                          "gate_false_positives": [r["scenario"] for r in negatives if r["false_positive"]],
                          "output": str(out)}, indent=2))
        return 0 if result["status"] == "PASS" else 2
    finally:
        env.gs.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
