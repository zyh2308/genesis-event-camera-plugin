#!/usr/bin/env python3
"""Reproducible R0 runners and a short direct-HDR event-source pilot."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmark.robot_closed_loop.predictor import (
    BoundedLateralCommand,
    FixedDecelerationPredictor,
    MotionState,
    fit_common_deceleration,
)
from benchmark.robot_closed_loop.robot_env import (
    FreeSlideScene,
    RobotScene,
    as_numpy,
    build_free_slide_scene,
    build_robot_scene,
    object_state,
)
from benchmark.robot_closed_loop.scoring import (
    LEGACY_SCORE_VERSION,
    STRICT_SCORE_VERSION,
    StrictHoldScorer,
    cube_bottom_clearance_m,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "scene_pilot_r0_v1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def get_sim_time(env: RobotScene) -> float:
    return float(as_numpy(env.scene.cur_t).reshape(()))


def contact_snapshot(env: RobotScene) -> Dict[str, Any]:
    c = env.contacts_with_target()
    return {"left": c["left"], "right": c["right"], "bilateral": c["bilateral"],
            "robot_link_ids": c["robot_link_ids"], "support": env.support_contact()}


def _record_state(
    env: RobotScene, t_s: float, stage: str, command: Dict[str, Any],
    contacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    obj = object_state(env.target, t_s)
    q = as_numpy(env.robot.get_dofs_position()).reshape(-1)
    qv = as_numpy(env.robot.get_dofs_velocity()).reshape(-1)
    return {
        "time_s": float(t_s), "stage": stage, "object": obj,
        "true_bottom_clearance_m": cube_bottom_clearance_m(
            obj["position_m"], obj["orientation_wxyz"], env.config["target"]["size_m"],
            float(env.config["physics"].get("table_top_z_m", 0.0)),
        ),
        "hand_link_position_m": env.hand_position(), "tcp_pad_center_m": env.tcp_position(),
        "jaw_opening_m": env.jaw_opening_m(), "finger_q_m": q[7:9],
        "finger_qd_m_s": qv[7:9], "contacts": contact_snapshot(env) if contacts is None else contacts,
        "robot_qpos": q, "robot_qvel": qv, "command": command,
    }


def _home_hand_goal(env: RobotScene) -> np.ndarray:
    tcp = np.asarray(env.config["robot"]["controller_tcp_reference_m"], dtype=np.float64)
    offset = float(env.config["robot"]["hand_link_z_offset_to_pad_center_m"])
    return tcp + np.asarray([0.0, 0.0, offset])


def _set_episode_origin(env: RobotScene, position: Sequence[float], yaw: float = 0.0) -> None:
    env.reset_episode(position, yaw)
    env.hold_robot_home()


def _settle(env: RobotScene, duration_s: float = 0.05) -> None:
    steps = max(1, int(round(duration_s / env.dt_s)))
    for _ in range(steps):
        env.hold_robot_home()
        env.scene.step()


def _settle_free_slide(env: FreeSlideScene, duration_s: float = 0.05) -> None:
    steps = max(1, int(round(duration_s / env.dt_s)))
    for _ in range(steps):
        env.scene.step()


def _gripper_stall(env: RobotScene) -> bool:
    q = env.finger_joint_positions()
    qd = env.finger_joint_velocities()
    return bool(np.all((q > 0.012) & (q < 0.029)) and np.max(np.abs(qd)) < 0.03)


def _lift_gate(
    estimated_state: Optional[MotionState],
    hand_y_m: float,
    intercept_x_m: float,
    gate_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Gate lifting with the same estimated target state available to a policy."""
    if estimated_state is None:
        return {"eligible": False, "reason": "no_current_state_estimate"}
    p = np.asarray(estimated_state.position_m, dtype=np.float64)
    v = np.asarray(estimated_state.velocity_m_s, dtype=np.float64)
    dx = float(p[0] - intercept_x_m)
    dy = float(p[1] - hand_y_m)
    planar_speed = float(np.linalg.norm(v[:2]))
    x_ok = abs(dx) <= float(gate_cfg["x_tolerance_m"])
    y_ok = abs(dy) <= float(gate_cfg["y_tolerance_m"])
    speed_ok = planar_speed <= float(gate_cfg["planar_speed_max_m_s"])
    return {
        "eligible": bool(x_ok and y_ok and speed_ok),
        "estimated_position_m": p.tolist(),
        "estimated_planar_speed_m_s": planar_speed,
        "dx_m": dx,
        "dy_m": dy,
        "x_ok": x_ok,
        "y_ok": y_ok,
        "speed_ok": speed_ok,
    }


def run_static_trial(
    env: RobotScene, episode_id: int, seed: int, *, capture: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[np.ndarray]]:
    cfg = env.config
    rng = np.random.default_rng(seed)
    target_y = float(rng.uniform(-0.002, 0.002))
    yaw = float(np.deg2rad(rng.uniform(-1.0, 1.0)))
    size = float(cfg["target"]["size_m"][2])
    x = float(cfg["controller"]["intercept_x_m"])
    _set_episode_origin(env, (x, target_y, size / 2), yaw)
    _settle(env)
    trial_start_t = get_sim_time(env)

    trace: List[Dict[str, Any]] = []
    frames: List[np.ndarray] = []
    stage = "close"
    first_contact_s = None
    bilateral = False
    stall_s = 0.0
    lift_q = None
    lift_trigger_s = None
    max_steps = int(round(float(cfg["success"].get("episode_timeout_s", 3.0)) / env.dt_s))
    sample_steps = max(1, int(round(0.001 / env.dt_s)))
    video_steps = max(1, int(round(0.05 / env.dt_s)))
    scorer = StrictHoldScorer(
        float(cfg["success"]["target_bottom_clearance_m"]),
        float(cfg["success"]["maximum_object_to_tcp_slip_m"]),
        float(cfg["success"]["continuous_hold_s"]),
    )
    strict_score = dict(scorer.last)
    legacy_hold_steps = 0
    legacy_stop_s = None
    legacy_success = False
    legacy_bilateral_at_stop = False
    legacy_clearance_at_stop = None
    legacy_true_bottom_min_at_stop = None
    legacy_window_true_bottom_min_m = float("inf")
    bilateral_seen = False
    max_true_clearance = -float("inf")

    for step in range(max_steps):
        now = get_sim_time(env)
        if lift_q is None:
            env.robot.control_dofs_position(env.home_qpos[:7], env.arm_dofs)
            env.command_fingers_close()
        else:
            env.robot.control_dofs_position(lift_q[:7], env.arm_dofs)
            env.command_fingers_close()
            stage = "lift_hold"
        env.scene.step()
        now = get_sim_time(env)
        c = contact_snapshot(env)
        if c["bilateral"]:
            bilateral = True
            bilateral_seen = True
        if (c["left"] or c["right"]) and first_contact_s is None:
            first_contact_s = now
        if lift_q is None and _gripper_stall(env):
            stall_s += env.dt_s
        elif lift_q is None:
            stall_s = 0.0
        if lift_q is None and stall_s >= 0.03:
            goal = _home_hand_goal(env)
            goal[2] += float(cfg["robot"]["lift_hand_delta_z_m"])
            lift_q = as_numpy(env.command_hand(goal)).reshape(-1)
            lift_trigger_s = now
            stage = "lift_hold"
        obj = object_state(env.target, now)
        pos = np.asarray(obj["position_m"], dtype=np.float64)
        clearance = cube_bottom_clearance_m(
            pos, obj["orientation_wxyz"], cfg["target"]["size_m"],
            float(cfg["physics"].get("table_top_z_m", 0.0)),
        )
        max_true_clearance = max(max_true_clearance, clearance)
        tcp = env.tcp_position()
        strict_score = scorer.update(
            now, pos - tcp, clearance, bool(c["bilateral"]), bool(c["support"]), env.dt_s,
        )
        center_bottom = float(pos[2] - size / 2)
        if center_bottom >= float(cfg["success"]["target_bottom_clearance_m"]):
            legacy_hold_steps += 1
        else:
            legacy_hold_steps = 0
        if legacy_stop_s is None and legacy_hold_steps * env.dt_s + 1e-12 >= float(cfg["success"]["continuous_hold_s"]):
            legacy_stop_s = now
            legacy_success = bool(now - trial_start_t <= 1.8 + 1e-9 and bilateral_seen and center_bottom >= float(cfg["success"]["target_bottom_clearance_m"]))
            legacy_bilateral_at_stop = bool(bilateral_seen)
            legacy_clearance_at_stop = center_bottom
        if step % sample_steps == 0 or c["bilateral"]:
            frame = _record_state(env, now, stage, {"finger_force_each": -5.0, "lift": lift_q is not None}, contacts=c)
            frame["score"] = strict_score
            trace.append(frame)
        if capture and step % video_steps == 0:
            rgb, _, _, _ = env.render(hdr=False)
            frames.append(np.asarray(rgb, dtype=np.uint8))
        if scorer.success_at(env.dt_s):
            break

    target_final = object_state(env.target, get_sim_time(env))
    target_p = target_final["position_m"]
    bottom = cube_bottom_clearance_m(
        target_p, target_final["orientation_wxyz"], cfg["target"]["size_m"],
        float(cfg["physics"].get("table_top_z_m", 0.0)),
    )
    strict_summary = scorer.summary(env.dt_s, step + 1)
    success = bool(scorer.success_at(env.dt_s) and bottom >= cfg["success"]["target_bottom_clearance_m"])
    result = {
        "episode_id": episode_id, "seed": seed, "phase": "static", "speed_m_s": 0.0,
        "target_initial_y_m": target_y, "target_yaw_deg": float(np.rad2deg(yaw)),
        "target_mass_kg": float(as_numpy(env.target.get_mass()).reshape(-1)[0]),
        "first_contact_s": first_contact_s, "bilateral_contact": bilateral,
        "measured_net_opening_initial_m": float(cfg["robot"]["initial_net_opening_m"]),
        "measured_net_opening_after_close_m": env.jaw_opening_m(),
        "lift_trigger_s": lift_trigger_s, "target_bottom_clearance_m": bottom,
        "hold_s": strict_summary["continuous_qualified_hold_s"],
        "maximum_true_bottom_clearance_m": max_true_clearance,
        "score_version": STRICT_SCORE_VERSION, "strict_score": strict_summary,
        "legacy_score_version": LEGACY_SCORE_VERSION, "success_legacy_r0": legacy_success,
        "legacy_first_stop_s": legacy_stop_s,
        "legacy_score": {
            "score_version": LEGACY_SCORE_VERSION, "success": legacy_success,
            "first_stop_s": legacy_stop_s, "center_based_bottom_clearance_at_stop_m": legacy_clearance_at_stop,
            "historical_bilateral_contact_by_stop": legacy_bilateral_at_stop,
        },
        "success": success, "success_strict_r0_5": success,
        "failure_primary": "success" if success else ("unstable_grasp" if bilateral else "spatial_miss"),
        "object_final_position_m": target_p.tolist(), "sim_duration_s": get_sim_time(env),
    }
    return result, trace, frames


def _sample_initial(rng: np.random.Generator, cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    target = cfg["target"]
    x = float(rng.uniform(*target["launch_x_range_m"]))
    y = float(rng.uniform(*target["launch_y_range_m"]))
    theta = float(np.deg2rad(rng.uniform(*target["heading_from_negative_x_range_deg"])))
    return x, y, theta


def _crossing(states: Sequence[Dict[str, Any]], x_int: float) -> Optional[Dict[str, Any]]:
    for left, right in zip(states[:-1], states[1:]):
        x0 = float(left["position_m"][0] - x_int)
        x1 = float(right["position_m"][0] - x_int)
        if x0 >= 0 and x1 <= 0 and x1 != x0:
            alpha = x0 / (x0 - x1)
            t = left["time_s"] + alpha * (right["time_s"] - left["time_s"])
            p = np.asarray(left["position_m"]) + alpha * (np.asarray(right["position_m"]) - np.asarray(left["position_m"]))
            v = np.asarray(left["velocity_m_s"]) + alpha * (np.asarray(right["velocity_m_s"]) - np.asarray(left["velocity_m_s"]))
            return {"time_s": float(t), "position_m": p, "velocity_m_s": v}
    return None


def run_slide_trial(
    env: FreeSlideScene, episode_id: int, seed: int, speed: float, *, duration_s: float = 2.5,
    deceleration_m_s2: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[MotionState]]:
    rng = np.random.default_rng(seed)
    x, y, theta = _sample_initial(rng, env.config)
    size = float(env.config["target"]["size_m"][2])
    env.reset_episode((x, y, size / 2), 0.0)
    _settle_free_slide(env)
    launch_t = get_sim_time(env)
    env.set_initial_target_velocity((-speed * np.cos(theta), speed * np.sin(theta), 0.0))
    history: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    sample_steps = max(1, int(round(0.01 / env.dt_s)))
    max_steps = int(round(duration_s / env.dt_s))
    crossed = False
    stopped = False
    for step in range(max_steps):
        env.scene.step()
        t = get_sim_time(env) - launch_t
        state = object_state(env.target, t)
        history.append(state)
        if step % sample_steps == 0:
            trace.append({**state, "support_contact": env.support_contact()})
        if state["position_m"][0] <= float(env.config["controller"]["intercept_x_m"]):
            crossed = True
            break
        if np.linalg.norm(state["velocity_m_s"][:2]) < 0.03 and t > 0.1:
            stopped = True
            break
    cross = _crossing(history, float(env.config["controller"]["intercept_x_m"]))
    velocities = np.asarray([s["velocity_m_s"][:2] for s in history])
    speeds = np.linalg.norm(velocities, axis=1)
    # Exclude the first 20 ms of solver/contact settling in the per-episode diagnostic.
    fit_start = max(0, int(round(0.02 / env.dt_s)))
    fit_t = np.asarray([s["time_s"] for s in history[fit_start:]])
    fit_v = speeds[fit_start:]
    moving = fit_v > 0.04
    local_a = float(-np.polyfit(fit_t[moving] - fit_t[moving][0], fit_v[moving], 1)[0]) if moving.sum() >= 4 else math.nan
    state0 = MotionState(0.0, np.asarray(history[0]["position_m"]), np.asarray(history[0]["velocity_m_s"]))
    a_candidate = float(deceleration_m_s2 if deceleration_m_s2 is not None else env.config["controller"]["initial_deceleration_assumption_m_s2"])
    initial_prediction = FixedDecelerationPredictor(
        float(env.config["controller"]["intercept_x_m"]), a_candidate,
    ).predict(state0, 0.0)
    pred_t = initial_prediction.ttc_s if initial_prediction.reachable else math.nan
    pred_y = float(initial_prediction.position_m[1]) if initial_prediction.reachable else math.nan
    result = {
        "episode_id": episode_id, "seed": seed, "phase": "slide_calibration_candidate",
        "launch_speed_m_s": speed, "start_x_m": x, "start_y_m": y,
        "heading_deg": float(np.rad2deg(theta)), "crossed_intercept": cross is not None,
        "crossing_t_s": None if cross is None else cross["time_s"],
        "crossing_y_m": None if cross is None else float(cross["position_m"][1]),
        "crossing_speed_m_s": None if cross is None else float(np.linalg.norm(cross["velocity_m_s"][:2])),
        "predicted_ttc_initial_s": pred_t, "predicted_y_initial_m": pred_y,
        "prediction_ttc_error_s": None if cross is None or not np.isfinite(pred_t) else float(pred_t - cross["time_s"]),
        "prediction_y_error_m": None if cross is None or not np.isfinite(pred_y) else float(pred_y - cross["position_m"][1]),
        "a_eff_episode_m_s2": local_a, "stopped_before_intercept": stopped and cross is None,
        "final_x_m": float(history[-1]["position_m"][0]),
        "final_speed_m_s": float(np.linalg.norm(history[-1]["velocity_m_s"][:2])),
        "sim_duration_s": float(history[-1]["time_s"]),
    }
    return result, trace, [MotionState(s["time_s"], np.asarray(s["position_m"]), np.asarray(s["velocity_m_s"])) for s in history]


def _delayed_observation(history: deque, cutoff_t: float) -> Optional[MotionState]:
    selected = None
    for item in history:
        if item.time_s <= cutoff_t + 1e-12:
            selected = item
        else:
            break
    return selected


def run_oracle_trial(
    env: RobotScene,
    episode_id: int,
    seed: int,
    speed: float,
    deceleration: float,
    *,
    capture: bool = False,
    latency_s: Optional[float] = None,
    position_bias_y_m: float = 0.0,
    ttc_bias_s: float = 0.0,
    extra_latency_s: float = 0.0,
    event_source: Optional[Any] = None,
    initial_y_m: Optional[float] = None,
    visibility_evaluator: Optional[Any] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[np.ndarray]]:
    cfg = env.config
    rng = np.random.default_rng(seed)
    x, y, theta = _sample_initial(rng, cfg)
    if initial_y_m is not None:
        y = float(initial_y_m)
    target_size = np.asarray(cfg["target"]["size_m"], dtype=np.float64)
    _set_episode_origin(env, (x, y, target_size[2] / 2), 0.0)
    _settle(env, float(cfg["controller"]["event_burn_in_s"]))
    launch_t = get_sim_time(env)
    initial_velocity = np.asarray([-speed * np.cos(theta), speed * np.sin(theta), 0.0], dtype=np.float64)
    env.set_initial_target_velocity(initial_velocity)
    if event_source is not None:
        event_source.initialize(env.camera, 0.0)
        event_source.capture_snapshot("launch", 0.0)

    controller_cfg = cfg["controller"]
    predictor = FixedDecelerationPredictor(float(controller_cfg["intercept_x_m"]), deceleration)
    initial_prediction = predictor.predict(
        MotionState(0.0, np.asarray([x, y, target_size[2] / 2]), initial_velocity), 0.0,
    )
    planned_arrival_s = float(initial_prediction.ttc_s) if initial_prediction.reachable else None
    latency = float(controller_cfg["observation_latency_s"] if latency_s is None else latency_s) + float(extra_latency_s)
    command = BoundedLateralCommand(
        0.0,
        float(cfg["robot"]["cartesian_speed_limit_m_s"]),
        float(cfg["robot"]["cartesian_acceleration_limit_m_s2"]),
    )
    control_steps = max(1, int(round((1.0 / float(controller_cfg["high_level_hz"])) / env.dt_s)))
    # Score every physics step and preserve 1 kHz state/control evidence.
    sample_steps = max(1, int(round(0.001 / env.dt_s)))
    video_steps = max(1, int(round(0.05 / env.dt_s)))
    history: deque = deque(maxlen=max(32, int(math.ceil((latency + 0.03) / env.dt_s)) + 4))
    history.append(MotionState(0.0, np.asarray([x, y, target_size[2] / 2]), initial_velocity))
    trace: List[Dict[str, Any]] = []
    frames: List[np.ndarray] = []
    close_started_s = None
    first_contact_s = None
    first_bilateral_s = None
    first_contact_position_m = None
    first_contact_tcp_m = None
    first_contact_relative_velocity_m_s = None
    first_bilateral_position_m = None
    lift_started_s = None
    lift_q = None
    lift_stall_time = 0.0
    stall_time = 0.0
    terminal_commit = False
    last_prediction = None
    estimated_current_state = None
    closure_speed_reference_m_s = None
    closure_lead_used_s = None
    last_obs = history[-1]
    last_command = {"y_command_m": 0.0, "closing": False, "lifting": False}
    crossing_actual = None
    object_history: List[Dict[str, Any]] = []
    strict_scorer = StrictHoldScorer(
        float(cfg["success"]["target_bottom_clearance_m"]),
        float(cfg["success"]["maximum_object_to_tcp_slip_m"]),
        float(cfg["success"]["continuous_hold_s"]),
    )
    strict_score = dict(strict_scorer.last)
    maximum_true_clearance = -float("inf")
    legacy_hold_steps = 0
    legacy_stop_s = None
    legacy_hold_at_stop_s = 0.0
    legacy_success = False
    legacy_bilateral_at_stop = False
    legacy_clearance_at_stop = None
    legacy_true_bottom_min_at_stop = None
    bilateral_seen = False
    previous_contacts = (False, False, False, False)
    tcp_velocity = np.zeros(3, dtype=np.float64)
    prev_obj = None
    timeout_s = float(cfg["success"].get("episode_timeout_s", 3.0))
    max_steps = int(round(timeout_s / env.dt_s))
    close_schedule = controller_cfg["closure_lead_speed_schedule"]
    guard = float(controller_cfg["terminal_commit_guard_s"])
    y_bounds = tuple(float(v) for v in controller_cfg["legal_intercept_y_range_m"])
    home_hand = _home_hand_goal(env)
    lift_hand_goal = None

    for step in range(max_steps):
        t_abs = get_sim_time(env)
        rel_t = t_abs - launch_t
        state = object_state(env.target, rel_t)
        obj_state = MotionState(rel_t, np.asarray(state["position_m"]), np.asarray(state["velocity_m_s"]))
        history.append(obj_state)
        object_history.append(state)
        if prev_obj is not None and prev_obj["position_m"][0] >= controller_cfg["intercept_x_m"] >= state["position_m"][0]:
            crossing_actual = _crossing([prev_obj, state], float(controller_cfg["intercept_x_m"]))
        prev_obj = state

        if step % control_steps == 0 and rel_t + 1e-9 >= float(controller_cfg["event_burn_in_s"]):
            obs = event_source.observe(rel_t - latency) if event_source is not None else _delayed_observation(history, rel_t - latency)
            if obs is not None:
                last_obs = MotionState(obs.time_s, obs.position_m.copy(), obs.velocity_m_s.copy())
                last_obs.position_m[1] += float(position_bias_y_m)
                observation_age = max(0.0, rel_t - last_obs.time_s) if event_source is not None else latency
                prediction = predictor.predict(last_obs, observation_age)
                last_prediction = prediction
                estimated_current_state = prediction.propagated_state
                if closure_speed_reference_m_s is None and estimated_current_state is not None:
                    closure_speed_reference_m_s = float(np.linalg.norm(estimated_current_state.velocity_m_s[:2]))
                if closure_speed_reference_m_s is not None:
                    close_lead = float(np.clip(
                        float(close_schedule["reference_lead_s"])
                        + float(close_schedule["slope_s_per_mps"])
                        * (closure_speed_reference_m_s - float(close_schedule["reference_speed_m_s"])),
                        float(close_schedule["min_lead_s"]),
                        float(close_schedule["max_lead_s"]),
                    ))
                else:
                    close_lead = float(close_schedule["max_lead_s"])
                if prediction.reachable:
                    ttc_cmd = prediction.ttc_s + float(ttc_bias_s)
                    target_y = float(np.clip(prediction.position_m[1], *y_bounds))
                    if not terminal_commit:
                        y_cmd, y_speed = command.step(target_y, control_steps * env.dt_s)
                        if prediction.ttc_s <= guard:
                            terminal_commit = True
                    else:
                        y_cmd, y_speed = command.position_m, command.velocity_m_s
                    hand_goal = np.asarray([float(controller_cfg["intercept_x_m"]), y_cmd, home_hand[2]])
                    if lift_q is None:
                        env.command_hand(hand_goal)
                    if close_started_s is None and 0.0 <= ttc_cmd <= close_lead:
                        close_started_s = rel_t
                        closure_lead_used_s = close_lead
                    if close_started_s is not None:
                        env.command_fingers_close()
                    else:
                        env.command_fingers_open()
                    last_command = {
                        "y_target_predicted_m": target_y, "y_command_m": y_cmd,
                        "y_command_velocity_m_s": y_speed, "ttc_predicted_s": ttc_cmd,
                        "closure_lead_s": close_lead,
                        "closure_speed_reference_m_s": closure_speed_reference_m_s,
                        "observation_age_s": observation_age,
                        "hand_goal_m": hand_goal, "closing": close_started_s is not None,
                        "lifting": lift_q is not None, "terminal_commit": terminal_commit,
                    }
                elif lift_q is None:
                    env.command_hand(np.asarray([float(controller_cfg["intercept_x_m"]), command.position_m, home_hand[2]]))
                    if close_started_s is None:
                        env.command_fingers_open()
                    else:
                        # Once closure begins, loss of a crossing prediction must
                        # not reopen the gripper while it is enclosing the target.
                        env.command_fingers_close()
                    last_command = {
                        "reachable": False, "y_command_m": command.position_m,
                        "closure_lead_s": close_lead,
                        "closure_speed_reference_m_s": closure_speed_reference_m_s,
                        "observation_age_s": observation_age,
                        "closing": close_started_s is not None, "lifting": False,
                    }

        if close_started_s is not None and lift_q is None and rel_t - close_started_s >= 0.05:
            lift_gate = _lift_gate(
                estimated_current_state, command.position_m,
                float(controller_cfg["intercept_x_m"]), controller_cfg["lift_gate"],
            )
            if _gripper_stall(env) and lift_gate["eligible"]:
                stall_time += env.dt_s
            else:
                stall_time = 0.0
            last_command = {**last_command, "lift_gate": lift_gate,
                            "gripper_stalled": _gripper_stall(env),
                            "lift_gate_stall_s": stall_time}
            if stall_time >= 0.03:
                lift_hand_goal = np.asarray([float(controller_cfg["intercept_x_m"]), command.position_m, home_hand[2] + float(cfg["robot"]["lift_hand_delta_z_m"])])
                lift_q = as_numpy(env.command_hand(lift_hand_goal)).reshape(-1)
                lift_started_s = rel_t
                last_command = {**last_command, "lifting": True, "lift_hand_goal_m": lift_hand_goal}
        elif lift_q is not None:
            env.robot.control_dofs_position(lift_q[:7], env.arm_dofs)
            env.command_fingers_close()

        tcp_before_step = env.tcp_position()
        env.scene.step()
        next_t = get_sim_time(env)
        next_rel_t = next_t - launch_t
        c = contact_snapshot(env)
        contact_tuple = (bool(c["left"]), bool(c["right"]), bool(c["bilateral"]), bool(c["support"]))
        obj_after = object_state(env.target, next_rel_t)
        obj_position = np.asarray(obj_after["position_m"], dtype=np.float64)
        obj_velocity = np.asarray(obj_after["velocity_m_s"], dtype=np.float64)
        tcp_after = env.tcp_position()
        tcp_velocity = (tcp_after - tcp_before_step) / env.dt_s
        exact_clearance = cube_bottom_clearance_m(
            obj_position, obj_after["orientation_wxyz"], target_size,
            float(cfg["physics"].get("table_top_z_m", 0.0)),
        )
        maximum_true_clearance = max(maximum_true_clearance, exact_clearance)
        if (c["left"] or c["right"]) and first_contact_s is None:
            first_contact_s = next_rel_t
            first_contact_position_m = obj_position.copy()
            first_contact_tcp_m = tcp_after.copy()
            first_contact_relative_velocity_m_s = obj_velocity - tcp_velocity
        if c["bilateral"] and first_bilateral_s is None:
            first_bilateral_s = next_rel_t
            first_bilateral_position_m = obj_position.copy()
        bilateral_seen = bilateral_seen or bool(c["bilateral"])
        strict_score = strict_scorer.update(
            next_rel_t,
            obj_position - tcp_after,
            exact_clearance,
            bool(c["bilateral"]),
            bool(c["support"]),
            env.dt_s,
        )
        if event_source is not None:
            event_source.advance(env.camera, next_rel_t)
            if first_contact_s == next_rel_t:
                event_source.capture_snapshot("first_contact", next_rel_t)
            if first_bilateral_s == next_rel_t:
                event_source.capture_snapshot("bilateral_contact", next_rel_t)
            if lift_started_s is not None and abs(lift_started_s - rel_t) <= env.dt_s + 1e-9:
                event_source.capture_snapshot("lift_start", next_rel_t)
        if visibility_evaluator is not None and step % visibility_evaluator.sample_every_steps == 0:
            visibility_evaluator.sample(env, next_rel_t)

        # Reproduce the old R0 scorer at its original first stop point so the
        # historical result remains interpretable alongside the strict score.
        legacy_center_bottom = float(obj_position[2] - target_size[2] / 2)
        if legacy_center_bottom >= float(cfg["success"]["target_bottom_clearance_m"]):
            if legacy_hold_steps == 0:
                legacy_window_true_bottom_min_m = exact_clearance
            else:
                legacy_window_true_bottom_min_m = min(legacy_window_true_bottom_min_m, exact_clearance)
            legacy_hold_steps += 1
        else:
            legacy_hold_steps = 0
            legacy_window_true_bottom_min_m = float("inf")
        if legacy_stop_s is None and legacy_hold_steps * env.dt_s + 1e-12 >= float(cfg["success"]["continuous_hold_s"]):
            legacy_stop_s = next_rel_t
            legacy_hold_at_stop_s = legacy_hold_steps * env.dt_s
            legacy_success = bool(next_rel_t <= 2.5 + 1e-9 and bilateral_seen and legacy_center_bottom >= float(cfg["success"]["target_bottom_clearance_m"]))
            legacy_bilateral_at_stop = bool(bilateral_seen)
            legacy_clearance_at_stop = legacy_center_bottom
            legacy_true_bottom_min_at_stop = legacy_window_true_bottom_min_m

        contact_changed = contact_tuple != previous_contacts
        should_log = step % sample_steps == 0 or contact_changed or (lift_q is not None and step % control_steps == 0)
        if should_log:
            frame = _record_state(
                env, next_rel_t,
                "lift_hold" if lift_q is not None else ("closing" if close_started_s is not None else "intercept"),
                last_command, contacts=c,
            )
            frame["score"] = strict_score
            frame["legacy_score"] = {
                "score_version": LEGACY_SCORE_VERSION,
                "center_based_bottom_clearance_m": legacy_center_bottom,
                "current_center_clearance_hold_s": legacy_hold_steps * env.dt_s,
                "minimum_true_bottom_clearance_in_current_center_hold_m": None
                if not np.isfinite(legacy_window_true_bottom_min_m) else legacy_window_true_bottom_min_m,
            }
            frame["tcp_velocity_m_s"] = tcp_velocity.tolist()
            frame["state_source"] = "direct_hdr_events" if event_source is not None else "current_gt_delayed"
            if event_source is not None:
                frame["event_state_by_rate"] = {}
                for rate_hz in (1000, 2000):
                    estimate = event_source.latest(rate_hz)
                    frame["event_state_by_rate"][str(rate_hz)] = None if estimate is None else {
                        "time_s": estimate.time_s,
                        "position_m": estimate.position_m.tolist(),
                        "velocity_m_s": estimate.velocity_m_s.tolist(),
                        "age_s": float(next_rel_t - estimate.time_s),
                        "event_count_window": int(event_source.trackers[rate_hz].last_confidence["event_count"]),
                        "time_bins_window": int(event_source.trackers[rate_hz].last_confidence["time_bin_count"]),
                    }
            if last_prediction is not None:
                frame["observation_cutoff_s"] = last_obs.time_s
                frame["controller_observation_state"] = {
                    "time_s": last_obs.time_s,
                    "position_m": last_obs.position_m.tolist(),
                    "velocity_m_s": last_obs.velocity_m_s.tolist(),
                    "age_at_log_s": float(next_rel_t - last_obs.time_s),
                }
                if estimated_current_state is not None:
                    frame["controller_estimated_current_state"] = {
                        "time_s": estimated_current_state.time_s,
                        "position_m": estimated_current_state.position_m.tolist(),
                        "velocity_m_s": estimated_current_state.velocity_m_s.tolist(),
                        "age_at_log_s": float(next_rel_t - estimated_current_state.time_s),
                    }
                frame["prediction"] = {
                    "reachable": last_prediction.reachable,
                    "ttc_s": last_prediction.ttc_s if last_prediction.reachable else None,
                    "position_m": None if last_prediction.position_m is None else last_prediction.position_m.tolist(),
                }
            trace.append(frame)
        previous_contacts = contact_tuple
        if capture and step % video_steps == 0:
            rgb, _, _, _ = env.render(hdr=False)
            frames.append(np.asarray(rgb, dtype=np.uint8))
        if strict_scorer.success_at(env.dt_s):
            if event_source is not None:
                event_source.capture_snapshot("strict_hold_complete", next_rel_t)
            break
        if np.linalg.norm(object_state(env.target, next_rel_t)["velocity_m_s"][:2]) < 0.015 and next_rel_t > 0.4 and lift_q is None:
            # Continue until timeout so an ungrasped stop is explicitly classified.
            pass

    final_state = object_state(env.target, get_sim_time(env) - launch_t)
    final_bottom = cube_bottom_clearance_m(
        final_state["position_m"], final_state["orientation_wxyz"], target_size,
        float(cfg["physics"].get("table_top_z_m", 0.0)),
    )
    contacts = contact_snapshot(env)
    strict_summary = strict_scorer.summary(env.dt_s, max_steps if len(trace) == 0 else step + 1)
    success = bool(strict_scorer.success_at(env.dt_s) and final_bottom >= cfg["success"]["target_bottom_clearance_m"])
    crossing_y = None if crossing_actual is None else float(crossing_actual["position_m"][1])
    t_contact_error = None if first_contact_s is None or crossing_actual is None else float(first_contact_s - crossing_actual["time_s"])
    y_tcp = float(env.tcp_position()[1])
    close_to_crossing = None if close_started_s is None or crossing_actual is None else float(close_started_s - crossing_actual["time_s"])
    contact_to_planned_arrival = None if first_contact_s is None or planned_arrival_s is None else float(first_contact_s - planned_arrival_s)
    close_to_planned_arrival = None if close_started_s is None or planned_arrival_s is None else float(close_started_s - planned_arrival_s)
    contact_to_close_s = None if first_contact_s is None or close_started_s is None else float(first_contact_s - close_started_s)
    first_contact_relative_speed = None if first_contact_relative_velocity_m_s is None else float(np.linalg.norm(first_contact_relative_velocity_m_s[:2]))
    first_contact_location_error = None if first_contact_position_m is None or first_contact_tcp_m is None else {
        "object_x_minus_intercept_m": float(first_contact_position_m[0] - float(controller_cfg["intercept_x_m"])),
        "object_y_minus_tcp_y_m": float(first_contact_position_m[1] - first_contact_tcp_m[1]),
        "object_minus_tcp_m": (first_contact_position_m - first_contact_tcp_m).tolist(),
    }
    if success:
        primary = "success"
    elif first_contact_s is not None:
        if first_bilateral_s is None or lift_started_s is None:
            primary = "unstable_grasp"
        elif strict_scorer.support_contact_when_clearance_met_samples > 0 and strict_scorer.qualified_samples == 0:
            primary = "support_contact_violation"
        elif strict_scorer.max_slip_m > float(cfg["success"]["maximum_object_to_tcp_slip_m"]):
            primary = "relative_slip"
        elif maximum_true_clearance >= float(cfg["success"]["target_bottom_clearance_m"]):
            primary = "strict_hold_constraint_failed"
        else:
            primary = "insufficient_clearance"
    elif not last_prediction or not last_prediction.reachable:
        primary = "oracle_unreachable"
    elif first_contact_s is None:
        primary = "spatial_miss" if crossing_y is not None and abs(crossing_y - y_tcp) > 0.02 else "late_closure"
    else:
        primary = "unstable_grasp"
    result = {
        "episode_id": episode_id, "seed": seed, "phase": "gt_oracle",
        "launch_speed_m_s": speed, "start_x_m": x, "start_y_m": y,
        "heading_deg": float(np.rad2deg(theta)), "observation_latency_s": latency,
        "position_bias_y_m": position_bias_y_m, "ttc_bias_s": ttc_bias_s,
        "first_contact_s": first_contact_s, "first_bilateral_contact_s": first_bilateral_s,
        "first_contact_position_m": None if first_contact_position_m is None else first_contact_position_m.tolist(),
        "first_bilateral_contact_position_m": None if first_bilateral_position_m is None else first_bilateral_position_m.tolist(),
        "first_contact_tcp_position_m": None if first_contact_tcp_m is None else first_contact_tcp_m.tolist(),
        "first_contact_location_error": first_contact_location_error,
        "first_contact_relative_velocity_m_s": None if first_contact_relative_velocity_m_s is None else first_contact_relative_velocity_m_s.tolist(),
        "first_contact_relative_speed_m_s": first_contact_relative_speed,
        "planned_free_slide_arrival_t_s": planned_arrival_s,
        "close_start_s": close_started_s,
        "close_start_minus_planned_arrival_s": close_to_planned_arrival,
        "first_contact_minus_planned_arrival_s": contact_to_planned_arrival,
        "first_contact_minus_close_start_s": contact_to_close_s,
        "actual_crossing_t_s": None if crossing_actual is None else crossing_actual["time_s"],
        "actual_crossing_y_m": crossing_y,
        "contact_minus_crossing_s": t_contact_error,
        "close_start_minus_crossing_s": close_to_crossing,
        "closure_speed_reference_m_s": closure_speed_reference_m_s,
        "closure_lead_used_s": closure_lead_used_s,
        "last_predicted_ttc_s": None if last_prediction is None or not last_prediction.reachable else last_prediction.ttc_s,
        "last_predicted_y_m": None if last_prediction is None or not last_prediction.reachable else float(last_prediction.position_m[1]),
        "final_tcp_y_m": y_tcp, "measured_net_opening_m": env.jaw_opening_m(),
        "lift_start_s": lift_started_s, "target_bottom_clearance_m": final_bottom,
        "maximum_target_bottom_clearance_m": maximum_true_clearance,
        "hold_s": strict_summary["continuous_qualified_hold_s"],
        "bilateral_contact_at_end": contacts["bilateral"],
        "score_version": STRICT_SCORE_VERSION, "success": success,
        "success_strict_r0_5": success,
        "strict_score": strict_summary,
        "legacy_score_version": LEGACY_SCORE_VERSION,
        "success_legacy_r0": legacy_success,
        "legacy_first_stop_s": legacy_stop_s,
        "legacy_hold_at_stop_s": legacy_hold_at_stop_s,
        "legacy_minimum_true_bottom_clearance_during_hold_m": legacy_true_bottom_min_at_stop,
        "legacy_score": {
            "score_version": LEGACY_SCORE_VERSION, "success": legacy_success,
            "first_stop_s": legacy_stop_s, "continuous_center_clearance_hold_s": legacy_hold_at_stop_s,
            "center_based_bottom_clearance_at_stop_m": legacy_clearance_at_stop,
            "minimum_true_bottom_clearance_during_center_hold_m": legacy_true_bottom_min_at_stop,
            "historical_bilateral_contact_by_stop": legacy_bilateral_at_stop,
        },
        "failure_primary": primary, "final_object_position_m": final_state["position_m"].tolist(),
        "final_object_velocity_m_s": final_state["velocity_m_s"].tolist(),
        "sim_duration_s": get_sim_time(env) - launch_t,
    }
    return result, trace, frames


def _event_config():
    from genesis_event_plugin.noise_model import DvsNoiseConfig
    return DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
        cutoff_hz=0.0, leak_rate_hz=0.0, leak_jitter_fraction=0.0,
        shot_noise_rate_hz=0.0, refractory_period_s=0.0,
        log_input=True, readout_model="ideal_readout",
    )


def _event_match_summary(events_1k: np.ndarray, events_2k: np.ndarray, tolerance_s: float = 0.0005) -> Dict[str, Any]:
    groups: Dict[Tuple[int, int, int], List[float]] = {}
    for event in events_2k:
        key = (int(event[1]), int(event[2]), int(event[3]))
        groups.setdefault(key, []).append(float(event[0]))
    matched = 0
    deltas = []
    for values in groups.values():
        values.sort()
    for event in events_1k:
        key = (int(event[1]), int(event[2]), int(event[3]))
        times = groups.get(key)
        if not times:
            continue
        idx = min(range(len(times)), key=lambda i: abs(times[i] - float(event[0])))
        delta = abs(times[idx] - float(event[0]))
        if delta <= tolerance_s:
            matched += 1
            deltas.append(delta)
            times.pop(idx)
    return {
        "events_1khz": int(len(events_1k)), "events_2khz": int(len(events_2k)),
        "count_relative_difference": None if not len(events_2k) else float(abs(len(events_1k) - len(events_2k)) / len(events_2k)),
        "matched_same_pixel_polarity_within_tolerance": int(matched),
        "fraction_1khz_matched": None if not len(events_1k) else float(matched / len(events_1k)),
        "fraction_2khz_matched": None if not len(events_2k) else float(matched / len(events_2k)),
        "matched_event_time_mae_ms": None if not deltas else float(np.mean(deltas) * 1000),
        "matched_event_time_p95_ms": None if not deltas else float(np.percentile(deltas, 95) * 1000),
        "time_tolerance_ms": float(tolerance_s * 1000.0),
    }


def run_reference(
    env: RobotScene,
    duration_s: float,
    speed: float,
    seed: int,
    out: Path,
    *,
    speeds: Optional[Sequence[float]] = None,
    episodes_per_speed: int = 1,
    y_values: Optional[Sequence[float]] = None,
    controller_hz: int = 2000,
    visibility_hz: float = 50.0,
) -> Dict[str, Any]:
    from benchmark.robot_closed_loop.event_state import DirectHdrEventSource, voxel_convergence_summary

    cfg = env.config
    speed_grid = [float(v) for v in (speeds or [speed])]
    y_grid = list(float(v) for v in y_values) if y_values is not None else []
    if y_grid and len(y_grid) != len(speed_grid) * int(episodes_per_speed):
        raise ValueError("--reference-y-values must provide one value per reference episode")
    visibility = DynamicVisibilityEvaluator(cfg, env.dt_s, sample_hz=visibility_hz)
    episode_rows = []
    reference_started = time.perf_counter()
    sample_index = 0
    for speed_index, launch_speed in enumerate(speed_grid):
        for repeat in range(int(episodes_per_speed)):
            episode_id = sample_index
            episode_seed = int(seed) + sample_index
            # The default 3-speed sweep includes both configured lateral edges.
            default_y = [0.0, -0.055, 0.055][speed_index % 3] if len(speed_grid) >= 3 else 0.0
            launch_y = y_grid[sample_index] if y_grid else default_y
            episode_out = out / "episodes" / f"episode_{episode_id:03d}"
            episode_out.mkdir(parents=True, exist_ok=False)
            source = DirectHdrEventSource(cfg, env.dt_s, controller_hz=controller_hz)
            visibility_start = len(visibility.rows)
            episode_started = time.perf_counter()
            row, trace, frames = run_oracle_trial(
                env, episode_id, episode_seed, launch_speed,
                float(cfg["controller"]["initial_deceleration_assumption_m_s2"]),
                capture=True, event_source=source, initial_y_m=launch_y,
                visibility_evaluator=visibility,
            )
            episode_wall_s = time.perf_counter() - episode_started
            row.update({
                "phase": "direct_hdr_event_state_smoke",
                "task_rollout_wall_s": episode_wall_s,
                "task_rollout_wall_boundary": "complete physics and control rollout, including per-step HDR rendering, both DVS streams, event tracking, dynamic visibility sampling, and video frame capture; excludes output serialization and simulator initialization",
                "reference_requested_duration_s_ignored_until_task_end": float(duration_s),
                "reference_controller_sampling_hz": int(controller_hz),
            })
            write_csv(episode_out / "episode.csv", [row])
            write_trace(episode_out / "state_control_trace.jsonl", trace)
            save_gif(episode_out / "rollout.gif", frames)
            source_summary = source.save(episode_out)
            events_1k = np.concatenate(source.events[1000], axis=0) if source.events[1000] else np.empty((0, 4), dtype=np.float64)
            events_2k = np.concatenate(source.events[2000], axis=0) if source.events[2000] else np.empty((0, 4), dtype=np.float64)
            conv = _event_match_summary(events_1k, events_2k)
            conv["voxel_convergence"] = voxel_convergence_summary(events_1k, events_2k, cfg["camera"]["resolution_wh"])
            state_errors = []
            post_contact_event_errors: Dict[int, List[Dict[str, float]]] = {1000: [], 2000: []}
            post_contact_controller_errors: List[Dict[str, float]] = []
            post_contact_frame_count = 0
            post_contact_gate_sample_count = 0
            post_contact_gate_eligible_count = 0
            gt_trace = [f for f in trace if f.get("object") and f.get("time_s") is not None]
            for frame in trace:
                estimates = frame.get("event_state_by_rate", {})
                frame_t = float(frame["time_s"])
                contact_t = row.get("first_contact_s")
                lift_t = row.get("lift_start_s")
                in_post_contact_window = bool(
                    contact_t is not None and frame_t >= float(contact_t)
                    and (lift_t is None or frame_t <= float(lift_t))
                )
                if in_post_contact_window:
                    post_contact_frame_count += 1
                    lift_gate = frame.get("command", {}).get("lift_gate")
                    if lift_gate is not None:
                        post_contact_gate_sample_count += 1
                        post_contact_gate_eligible_count += int(bool(lift_gate.get("eligible", False)))
                if estimates and gt_trace:
                    for hz in (1000, 2000):
                        estimate = estimates.get(str(hz))
                        if estimate is None:
                            continue
                        aligned = min(gt_trace, key=lambda item: abs(float(item["time_s"]) - float(estimate["time_s"])))
                        aligned_object = aligned["object"]
                        gt_position = np.asarray(aligned_object["position_m"], dtype=np.float64)
                        gt_velocity = np.asarray(aligned_object["velocity_m_s"], dtype=np.float64)
                        error_row = {
                            "position_error_xy_m": float(np.linalg.norm(np.asarray(estimate["position_m"][:2]) - gt_position[:2])),
                            "velocity_error_xy_m_s": float(np.linalg.norm(np.asarray(estimate["velocity_m_s"][:2]) - gt_velocity[:2])),
                            "estimate_age_ms": float(estimate["age_s"] * 1000.0),
                        }
                        state_errors.append({
                            "episode_id": episode_id, "evaluation_time_s": float(frame["time_s"]),
                            "estimate_time_s": float(estimate["time_s"]), "sampling_hz": hz,
                            **error_row,
                            "nearest_gt_sample_time_error_ms": float(abs(float(aligned["time_s"]) - float(estimate["time_s"])) * 1000.0),
                            "event_count_window": int(estimate["event_count_window"]),
                            "time_bins_window": int(estimate["time_bins_window"]),
                        })
                        if in_post_contact_window:
                            post_contact_event_errors[hz].append(error_row)
                controlled = frame.get("controller_estimated_current_state")
                if in_post_contact_window and controlled and gt_trace:
                    aligned_current = min(gt_trace, key=lambda item: abs(float(item["time_s"]) - frame_t))
                    gt_current = aligned_current["object"]
                    controlled_pos = np.asarray(controlled["position_m"], dtype=np.float64)
                    controlled_vel = np.asarray(controlled["velocity_m_s"], dtype=np.float64)
                    post_contact_controller_errors.append({
                        "position_error_xy_m": float(np.linalg.norm(controlled_pos[:2] - np.asarray(gt_current["position_m"], dtype=np.float64)[:2])),
                        "velocity_error_xy_m_s": float(np.linalg.norm(controlled_vel[:2] - np.asarray(gt_current["velocity_m_s"], dtype=np.float64)[:2])),
                        "age_ms": float(controlled["age_at_log_s"] * 1000.0),
                    })
            write_csv(episode_out / "event_state_error.csv", state_errors)
            errors_by_rate = {}
            post_contact_event_errors_by_rate = {}
            for hz in (1000, 2000):
                rows = [r for r in state_errors if r["sampling_hz"] == hz]
                errors_by_rate[str(hz)] = {
                    "n": len(rows),
                    "position_error_xy_mean_m": None if not rows else float(np.mean([r["position_error_xy_m"] for r in rows])),
                    "position_error_xy_p95_m": None if not rows else float(np.percentile([r["position_error_xy_m"] for r in rows], 95)),
                    "velocity_error_xy_mean_m_s": None if not rows else float(np.mean([r["velocity_error_xy_m_s"] for r in rows])),
                    "velocity_error_xy_p95_m_s": None if not rows else float(np.percentile([r["velocity_error_xy_m_s"] for r in rows], 95)),
                    "estimate_age_p95_ms": None if not rows else float(np.percentile([r["estimate_age_ms"] for r in rows], 95)),
                }
                contact_rows = post_contact_event_errors[hz]
                post_contact_event_errors_by_rate[str(hz)] = {
                    "n_event_states_evaluated": len(contact_rows),
                    "position_error_xy_mean_m": None if not contact_rows else float(np.mean([r["position_error_xy_m"] for r in contact_rows])),
                    "position_error_xy_p95_m": None if not contact_rows else float(np.percentile([r["position_error_xy_m"] for r in contact_rows], 95)),
                    "velocity_error_xy_mean_m_s": None if not contact_rows else float(np.mean([r["velocity_error_xy_m_s"] for r in contact_rows])),
                    "velocity_error_xy_p95_m_s": None if not contact_rows else float(np.percentile([r["velocity_error_xy_m_s"] for r in contact_rows], 95)),
                    "event_state_age_p95_ms": None if not contact_rows else float(np.percentile([r["estimate_age_ms"] for r in contact_rows], 95)),
                }
            post_control = post_contact_controller_errors
            post_contact_controller_summary = {
                "first_contact_to_lift_start_or_episode_end_frame_count": post_contact_frame_count,
                "controller_state_available_count": len(post_control),
                "controller_state_available_fraction": None if not post_contact_frame_count else float(len(post_control) / post_contact_frame_count),
                "lift_gate_evaluation_count": post_contact_gate_sample_count,
                "lift_gate_eligible_count": post_contact_gate_eligible_count,
                "lift_gate_eligible_fraction": None if not post_contact_gate_sample_count else float(post_contact_gate_eligible_count / post_contact_gate_sample_count),
                "position_error_xy_mean_m": None if not post_control else float(np.mean([r["position_error_xy_m"] for r in post_control])),
                "position_error_xy_p95_m": None if not post_control else float(np.percentile([r["position_error_xy_m"] for r in post_control], 95)),
                "velocity_error_xy_mean_m_s": None if not post_control else float(np.mean([r["velocity_error_xy_m_s"] for r in post_control])),
                "velocity_error_xy_p95_m_s": None if not post_control else float(np.percentile([r["velocity_error_xy_m_s"] for r in post_control], 95)),
                "controller_state_age_p95_ms": None if not post_control else float(np.percentile([r["age_ms"] for r in post_control], 95)),
            }
            visibility_summary = visibility.summary(visibility_start)
            (episode_out / "reference_metrics.json").write_text(json.dumps(jsonable({
                "episode_id": episode_id, "seed": episode_seed, "speed_m_s": launch_speed,
                "launch_y_m": launch_y, "physics_dt_s": env.dt_s,
                "strict_score_version": row["score_version"],
                "strict_success": row["success"], "legacy_success_at_legacy_stop": row["success_legacy_r0"],
                "task_rollout_wall_s": episode_wall_s,
                "simulated_duration_s": row["sim_duration_s"],
                "real_time_factor": row["sim_duration_s"] / max(episode_wall_s, 1e-9),
                "event_source": source_summary,
                "event_convergence": conv,
                "event_state_accuracy_evaluator_only": errors_by_rate,
                "post_contact_event_state_accuracy_evaluator_only": post_contact_event_errors_by_rate,
                "post_contact_controller_state_accuracy_evaluator_only": post_contact_controller_summary,
                "post_contact_controller_state_boundary": "scores the actual causal event-derived observation propagated by the fixed predictor and used by the lift gate; evaluator GT aligns position/velocity after the episode and never enters the controller",
                "dynamic_visibility_evaluator_only": visibility_summary,
                "reference_duration_setting_s": duration_s,
            }), indent=2), encoding="utf-8")
            episode_rows.append({
                "episode_id": episode_id, "seed": episode_seed, "speed_m_s": launch_speed,
                "launch_y_m": launch_y, "strict_success": row["success"],
                "success_legacy_r0": row["success_legacy_r0"],
                "first_contact_s": row["first_contact_s"], "lift_start_s": row["lift_start_s"],
                "sim_duration_s": row["sim_duration_s"], "task_rollout_wall_s": episode_wall_s,
                "event_count_1khz": len(events_1k), "event_count_2khz": len(events_2k),
                "state_control_trace": str((episode_out / "state_control_trace.jsonl").relative_to(out)),
                "raw_events": str((episode_out / "direct_hdr_events_1khz_2khz.npz").relative_to(out)),
                "dynamic_visibility_min": visibility_summary["minimum_visible_fraction"],
                "state_accuracy_1khz_p95_m": errors_by_rate["1000"]["position_error_xy_p95_m"],
                "state_accuracy_2khz_p95_m": errors_by_rate["2000"]["position_error_xy_p95_m"],
            })
            sample_index += 1
    total_wall_s = time.perf_counter() - reference_started
    write_csv(out / "r0_5_reference_episodes.csv", episode_rows)
    summary = {
        "mode": "r0_5_direct_hdr_reference_and_event_state_smoke",
        "code_scope": "robot benchmark; frozen simulator core unchanged",
        "physics_dt_s": env.dt_s, "paired_event_rates_hz": [1000, 2000],
        "controller_state_source": "2 kHz causal event tracker" if controller_hz == 2000 else "1 kHz causal event tracker",
        "episodes": episode_rows,
        "strict_success_by_speed": {
            str(s): {"n": sum(1 for r in episode_rows if r["speed_m_s"] == s),
                     "successes": sum(bool(r["strict_success"]) for r in episode_rows if r["speed_m_s"] == s)}
            for s in speed_grid
        },
        "episode_loop_artifact_wall_s": total_wall_s,
        "mean_task_rollout_wall_s": None if not episode_rows else float(np.mean([r["task_rollout_wall_s"] for r in episode_rows])),
        "cost_boundary": "episode_loop_artifact_wall_s includes complete rollouts and per-episode trace/video/event/CSV/JSON artifact generation, but excludes environment initialization and final run_manifest/SHA256SUMS finalization. The launch wrapper must record the full Python process wall time as the end-to-end cost.",
        "no_rollout_scale_extrapolation": True,
        "pixel_response_model_boundary": "Direct linear-HDR input and DvsEmulator are shared with the plugin; Ours temporal reconstruction and V2E frame interpolation are bypassed. This ideal synthetic threshold model is uncalibrated to EVK4.",
        "visibility_evaluation_boundary": "uses evaluator-only segmentation and matched-pose target-only render; those data do not enter the event state source or controller",
        "small_smoke_only": True,
    }
    (out / "r0_5_reference_summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    (out / "dynamic_visibility.json").write_text(json.dumps(jsonable(visibility.summary()), indent=2), encoding="utf-8")
    return summary


def _resolve_segmentation_ids(scene: Any, entity: Any) -> List[int]:
    def visit(value: Any) -> set[int]:
        found: set[int] = set()
        for name in ("idx", "entity_idx", "link_idx"):
            if hasattr(value, name):
                try:
                    found.add(int(getattr(value, name)))
                except (ValueError, TypeError):
                    pass
        if isinstance(value, dict):
            for v in value.values():
                found.update(visit(v))
        elif isinstance(value, (tuple, list, set)):
            for v in value:
                found.update(visit(v))
        elif isinstance(value, (int, np.integer)):
            found.add(int(value))
        return found
    seg_map = getattr(scene, "segmentation_idx_dict", {}) or {}
    expected = {int(getattr(entity, "idx", -1))}
    links = getattr(entity, "links", [])
    try:
        expected.update(int(link.idx) for link in links)
    except TypeError:
        pass
    return sorted(int(raw) for raw, descriptor in seg_map.items() if expected.intersection(visit(descriptor)))


def _mask_and_bbox(seg: np.ndarray, ids: Sequence[int]) -> Tuple[np.ndarray, Optional[List[int]]]:
    arr = np.asarray(seg)
    if arr.ndim == 3:
        candidates = [arr[..., i] for i in range(arr.shape[-1])]
        masks = [np.isin(c, ids) for c in candidates]
        mask = np.logical_or.reduce(masks)
    else:
        mask = np.isin(arr, ids)
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return mask, None
    return mask, [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]


class DynamicVisibilityEvaluator:
    """Paired segmentation audit; its GT input never reaches the event controller."""

    def __init__(self, config: Dict[str, Any], dt_s: float, sample_hz: float = 50.0):
        self.reference = build_free_slide_scene(config, dt_s=dt_s, with_camera=True)
        if self.reference.camera is None:
            raise RuntimeError("could not construct target-only visibility scene")
        self.sample_every_steps = max(1, int(round(1.0 / (float(sample_hz) * float(dt_s)))))
        self.reference.camera.render(rgb=True, depth=False, segmentation=True, force_render=True)
        self.target_only_ids = _resolve_segmentation_ids(self.reference.scene, self.reference.target)
        if not self.target_only_ids:
            raise RuntimeError("target-only segmentation IDs missing in dynamic visibility audit")
        self.rows: List[Dict[str, Any]] = []
        self.render_ms: List[float] = []

    def sample(self, env: RobotScene, time_s: float) -> None:
        started = time.perf_counter()
        _rgb, _depth, actual_seg, _normal = env.camera.render(
            rgb=True, depth=False, segmentation=True, force_render=True,
        )
        actual_ids = _resolve_segmentation_ids(env.scene, env.target)
        actual_mask, actual_bbox = _mask_and_bbox(actual_seg, actual_ids)
        qpos = as_numpy(env.target.get_qpos()).astype(np.float32)
        self.reference.target.set_qpos(qpos, zero_velocity=True)
        self.reference.target.set_dofs_velocity(np.zeros(6, dtype=np.float32))
        _rgb, _depth, clear_seg, _normal = self.reference.camera.render(
            rgb=True, depth=False, segmentation=True, force_render=True,
        )
        clear_mask, clear_bbox = _mask_and_bbox(clear_seg, self.target_only_ids)
        actual_pixels = int(actual_mask.sum())
        clear_pixels = int(clear_mask.sum())
        self.render_ms.append((time.perf_counter() - started) * 1000.0)
        self.rows.append({
            "time_s": float(time_s),
            "target_position_m": as_numpy(env.target.get_qpos()).reshape(-1)[:3].tolist(),
            "visible_target_pixels": actual_pixels,
            "target_only_pixels": clear_pixels,
            "visible_fraction_of_unoccluded_mask": None if clear_pixels == 0 else float(actual_pixels / clear_pixels),
            "robot_occlusion_fraction": None if clear_pixels == 0 else float(1.0 - actual_pixels / clear_pixels),
            "visible_bbox_xyxy": actual_bbox,
            "unoccluded_bbox_xyxy": clear_bbox,
        })

    def summary(self, start_index: int = 0) -> Dict[str, Any]:
        rows = self.rows[int(start_index):]
        render_ms = self.render_ms[int(start_index):]
        fractions = [r["visible_fraction_of_unoccluded_mask"] for r in rows if r["visible_fraction_of_unoccluded_mask"] is not None]
        return {
            "method": "live robot segmentation divided by target-only segmentation at matching GT pose; evaluator-only",
            "sample_hz": 1.0 / (self.sample_every_steps * self.reference.dt_s),
            "sample_count": len(rows),
            "minimum_visible_fraction": None if not fractions else float(np.min(fractions)),
            "median_visible_fraction": None if not fractions else float(np.median(fractions)),
            "mean_visibility_render_pair_ms": None if not render_ms else float(np.mean(render_ms)),
            "rows": rows,
        }


def run_preflight(env: RobotScene, out: Path) -> Dict[str, Any]:
    from PIL import Image
    cfg = env.config
    size = float(cfg["target"]["size_m"][0])
    _set_episode_origin(env, (0.55, 0.0, size / 2), 0.0)
    _settle(env)
    rgb, depth, seg, _ = env.camera.render(
        rgb=True, depth=True, segmentation=True, radiance=False,
    )
    hdr, _, _, _ = env.camera.render(
        rgb=True, depth=True, segmentation=False, radiance=True,
    )
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth)
    seg = np.asarray(seg)
    hdr = np.asarray(hdr, dtype=np.float32)
    ids = _resolve_segmentation_ids(env.scene, env.target)
    if not ids:
        raise RuntimeError("camera segmentation did not resolve target ID; refusing to guess")
    mask, bbox = _mask_and_bbox(seg, ids)
    Image.fromarray(rgb).save(out / "camera_preflight_rgb.png")
    np.savez_compressed(out / "camera_preflight_hdr_depth_seg.npz", hdr=hdr, depth=depth, segmentation=seg)

    # Render a target-only reference with the exact same camera and target
    # geometry. Do not use Panda's all-zero model qpos: it is outside joint
    # limits and is not a meaningful robot posture.
    q_home = as_numpy(env.robot.get_dofs_position()).reshape(-1).copy()
    reference_env = build_free_slide_scene(cfg, dt_s=env.dt_s, with_camera=True)
    if reference_env.camera is None:
        raise RuntimeError("target-only camera reference was not constructed")
    current_position = as_numpy(env.target.get_pos(relative=False)).reshape(-1)[:3]
    reference_env.reset_episode(current_position, 0.0)
    _, _, seg_clear, _ = reference_env.camera.render(
        rgb=True, depth=False, segmentation=True, radiance=False,
    )
    reference_ids = _resolve_segmentation_ids(reference_env.scene, reference_env.target)
    if not reference_ids:
        raise RuntimeError("camera segmentation did not resolve target-only reference ID")
    clear_mask, clear_bbox = _mask_and_bbox(np.asarray(seg_clear), reference_ids)
    total_px = int(clear_mask.sum())
    visible_px = int(mask.sum())
    occlusion = None if total_px == 0 else float(max(0, total_px - visible_px) / total_px)

    # Check view coverage over the nominal launch-to-intercept path and the
    # configured lateral launch range. These are static camera samples only;
    # they do not replace the dynamic task rollout or move the target during it.
    target_cfg = cfg["target"]
    _, x_hi = (float(v) for v in target_cfg["launch_x_range_m"])
    y_abs = max(abs(float(v)) for v in target_cfg["launch_y_range_m"])
    intercept_x = float(cfg["controller"]["intercept_x_m"])
    x_samples = sorted({x_hi, 0.5 * (x_hi + intercept_x), intercept_x}, reverse=True)
    y_samples = sorted({-y_abs, 0.0, y_abs})
    view_samples = []
    for sample_x in x_samples:
        for sample_y in y_samples:
            sample_position = np.asarray([sample_x, sample_y, size / 2], dtype=np.float64)
            env.robot.set_qpos(q_home, zero_velocity=True)
            env.target.set_qpos(
                np.concatenate((sample_position, np.asarray([1.0, 0.0, 0.0, 0.0]))),
                zero_velocity=True,
            )
            env.hold_robot_home()
            # Advance one physical step so Genesis synchronizes the new target
            # pose to the renderer before the static view measurement.
            env.scene.step()
            rendered_sample_position = as_numpy(
                env.target.get_pos(relative=False),
            ).reshape(-1)[:3]
            _, _, sample_seg, _ = env.camera.render(
                rgb=True, depth=False, segmentation=True, radiance=False,
            )
            sample_mask, sample_bbox = _mask_and_bbox(np.asarray(sample_seg), ids)
            reference_env.reset_episode(rendered_sample_position, 0.0)
            _, _, sample_reference_seg, _ = reference_env.camera.render(
                rgb=True, depth=False, segmentation=True, radiance=False,
            )
            sample_reference_mask, _ = _mask_and_bbox(
                np.asarray(sample_reference_seg), reference_ids,
            )
            sample_total = int(sample_reference_mask.sum())
            sample_visible = int(sample_mask.sum())
            sample_occlusion = (
                None if sample_total == 0
                else float(max(0, sample_total - sample_visible) / sample_total)
            )
            sample_short_side = (
                None if sample_bbox is None
                else min(sample_bbox[2] - sample_bbox[0], sample_bbox[3] - sample_bbox[1])
            )
            view_samples.append({
                "target_position_m": rendered_sample_position.tolist(),
                "target_pixels_target_only": sample_total,
                "visible_target_pixels_with_robot_home": sample_visible,
                "rendered_robot_occlusion_fraction": sample_occlusion,
                "target_bbox_xyxy_px_with_robot_home": sample_bbox,
                "target_projected_short_side_px_with_robot_home": sample_short_side,
            })
    env.robot.set_qpos(q_home, zero_velocity=True)
    env.target.set_qpos(
        np.concatenate((current_position, np.asarray([1.0, 0.0, 0.0, 0.0]))),
        zero_velocity=True,
    )

    camera_cfg = cfg["camera"]
    w, h = (int(v) for v in camera_cfg["resolution_wh"])
    fy = h / (2.0 * math.tan(math.radians(float(camera_cfg["vertical_fov_deg"])) / 2.0))
    fx = fy
    try:
        from genesis_event_plugin.genesis_adapter import camera_to_world_cv
        camera_to_world = as_numpy(camera_to_world_cv(env.camera)).tolist()
    except Exception as exc:
        camera_to_world = {"error": f"{type(exc).__name__}: {exc}"}
    radiance_ok = bool(hdr.shape == (h, w, 3) and np.isfinite(hdr).all())
    report = {
        "status": "passed" if radiance_ok and total_px > 0 and visible_px > 0 else "failed",
        "camera": {
            "resolution_wh": [w, h], "vertical_fov_deg": camera_cfg["vertical_fov_deg"],
            "intrinsics_fx_fy_cx_cy": [fx, fy, w / 2.0, h / 2.0],
            "camera_to_world_cv": camera_to_world,
        },
        "intercept_render": {
            "resolved_target_segmentation_ids": ids,
            "target_bbox_xyxy_px_with_robot": bbox,
            "target_bbox_xyxy_px_target_only_reference": clear_bbox,
            "visible_target_pixels_with_robot": visible_px,
            "target_pixels_target_only_reference": total_px,
            "rendered_robot_occlusion_fraction": occlusion,
            "target_projected_short_side_px": None if bbox is None else min(bbox[2] - bbox[0], bbox[3] - bbox[1]),
            "nominal_path_view_samples_robot_held_home": view_samples,
        },
        "radiance": {
            "dtype": str(hdr.dtype), "shape": list(hdr.shape),
            "finite": bool(np.isfinite(hdr).all()),
            "min": float(np.nanmin(hdr)) if hdr.size else None,
            "max": float(np.nanmax(hdr)) if hdr.size else None,
            "above_one_pixels": int((hdr > 1.0).any(axis=-1).sum()) if hdr.ndim == 3 else 0,
        },
        "note": "Occlusion compares actual robot-scene target pixels against a target-only scene with identical camera, target geometry, and target pose. Path samples are static view checks with Panda held at the nominal pre-grasp home qpos; they are not task rollouts.",
    }
    (out / "camera_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "passed":
        raise RuntimeError(
            "camera preflight failed: target silhouette/visibility or finite HDR check did not pass; "
            "see camera_preflight.json"
        )
    return report


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(jsonable(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple, np.ndarray)) else v for k, v in row.items()})


def write_trace(path: Path, trace: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in trace:
            f.write(json.dumps(jsonable(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def save_gif(path: Path, frames: Sequence[np.ndarray]) -> None:
    if not frames:
        return
    import imageio.v2 as imageio
    imageio.mimsave(path, list(frames), duration=0.05, loop=0)


def write_run_manifest(args: argparse.Namespace, config: Dict[str, Any], out: Path, finalize: bool = False) -> Dict[str, Any]:
    config_path = Path(args.config).resolve()
    repo = ROOT
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    except Exception:
        commit, branch = "unknown", "unknown"
    import genesis as gs
    import torch
    asset = Path(gs.__file__).resolve().parent / "assets/xml/franka_emika_panda/panda.xml"
    contact = Path(gs.__file__).resolve().parent / "engine/solvers/rigid/collider/contact.py"
    material = Path(gs.__file__).resolve().parent / "engine/materials/rigid.py"
    overlay = Path(__file__).resolve().parents[2] / "integrations/genesis_renderer_hdr_v2_overlay/genesis"
    files = [p for p in overlay.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    manifest = {
        "mode": args.mode, "command": sys.argv,
        "code_commit_at_run_start": commit, "branch": branch,
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "frozen_simulator_core": config["provenance"]["frozen_simulator_core"],
        "runtime": {
            "python": sys.version.split()[0], "genesis": getattr(gs, "__version__", "unknown"),
            "genesis_module": gs.__file__, "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(), "sim_backend": "cpu",
            "panda_asset_in_scene": args.mode != "slide",
            "panda_asset": str(asset), "panda_asset_sha256": sha256(asset),
            "contact_source": str(contact), "contact_source_sha256": sha256(contact),
            "material_source_sha256": sha256(material),
            "overlay_files": {str(p.relative_to(overlay)): sha256(p) for p in files},
        },
    }
    if finalize:
        outputs = {}
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name not in {"run_manifest.json", "SHA256SUMS"}:
                outputs[str(p.relative_to(out))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        manifest["outputs"] = outputs
        (out / "SHA256SUMS").write_text("".join(f"{meta['sha256']}  {name}\n" for name, meta in outputs.items()), encoding="utf-8")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def run_batch(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to mix/overwrite existing run artifacts: {out}")
    out.mkdir(parents=True, exist_ok=True)
    traces_dir = out / "traces"
    videos_dir = out / "videos"
    traces_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)
    write_run_manifest(args, config, out)

    dt = float(args.physics_dt_us) / 1e6 if args.physics_dt_us is not None else None
    env = build_free_slide_scene(config, dt_s=dt) if args.mode == "slide" else build_robot_scene(config, dt_s=dt)
    results: List[Dict[str, Any]] = []
    trace_by_id: Dict[int, List[Dict[str, Any]]] = {}
    motion_histories: List[List[MotionState]] = []
    if args.mode == "preflight":
        run_preflight(env, out)
    elif args.mode == "reference":
        run_reference(
            env, args.reference_duration, args.speed, args.seed, out,
            speeds=args.speeds, episodes_per_speed=args.reference_episodes,
            y_values=args.reference_y_values,
            controller_hz=args.event_controller_hz, visibility_hz=args.visibility_hz,
        )
    elif args.mode == "static":
        for i in range(args.episodes):
            seed = args.seed + i
            row, trace, _ = run_static_trial(env, i, seed)
            row["config_sha256"] = sha256(config_path)
            results.append(row)
            trace_by_id[i] = trace
        write_csv(out / "r0_static.csv", results)
        successes = [r for r in results if r["success"]]
        candidates = ([successes[0]] if successes else []) + ([r for r in results if not r["success"]][:1])
        for row in candidates:
            eid = int(row["episode_id"])
            write_trace(traces_dir / f"static_episode_{eid:04d}_{row['failure_primary']}.jsonl", trace_by_id[eid])
            rerun, _, frames = run_static_trial(env, eid, int(row["seed"]), capture=True)
            save_gif(videos_dir / f"static_episode_{eid:04d}_{rerun['failure_primary']}.gif", frames)
    elif args.mode == "slide":
        if args.split == "holdout" and args.a_eff is None:
            raise ValueError("slide holdout requires the fixed a_eff fitted from a separate calibration split")
        for speed in args.speeds:
            local_ids = range(len(results), len(results) + args.episodes)
            for i in local_ids:
                seed = args.seed + int(i)
                row, trace, history = run_slide_trial(env, int(i), seed, float(speed), deceleration_m_s2=args.a_eff)
                row["split"] = args.split
                row["config_sha256"] = sha256(config_path)
                results.append(row)
                trace_by_id[int(i)] = trace
                motion_histories.append(history)
        write_csv(out / f"r0_slide_{args.split}.csv", results)
        if args.split == "calibration":
            a_eff = fit_common_deceleration(motion_histories)
            summary = {
                "status": "fitted_from_no_hand_physics_trajectories",
                "a_eff_m_s2": a_eff, "episodes": len(results),
                "episodes_by_speed": {str(s): sum(1 for r in results if r["launch_speed_m_s"] == s) for s in args.speeds},
                "median_episode_fit_m_s2": float(np.nanmedian([r["a_eff_episode_m_s2"] for r in results])),
                "iqr_episode_fit_m_s2": [float(np.nanpercentile([r["a_eff_episode_m_s2"] for r in results], 25)), float(np.nanpercentile([r["a_eff_episode_m_s2"] for r in results], 75))],
                "holdout_required": True,
                "warning": "This is a shared pilot motion model, not per-episode future-GT fitting.",
            }
            (out / "r0_slide_calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        elif args.split == "holdout":
            def finite_values(group: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
                values = np.asarray([r[key] for r in group if r.get(key) is not None], dtype=np.float64)
                return values[np.isfinite(values)]

            by_speed = {}
            for speed in args.speeds:
                group = [r for r in results if abs(float(r["launch_speed_m_s"]) - float(speed)) < 1e-9]
                crossed = [r for r in group if r["crossing_t_s"] is not None]
                t_error = finite_values(group, "prediction_ttc_error_s")
                y_error = finite_values(group, "prediction_y_error_m")
                drift = np.asarray([
                    float(r["crossing_y_m"]) - float(r["start_y_m"])
                    for r in crossed
                ], dtype=np.float64)
                crossing_t = finite_values(crossed, "crossing_t_s")
                crossing_speed = finite_values(crossed, "crossing_speed_m_s")
                episode_a = finite_values(group, "a_eff_episode_m_s2")
                by_speed[str(speed)] = {
                    "n": len(group), "crossed_intercept_n": len(crossed),
                    "crossing_rate": None if not group else float(len(crossed) / len(group)),
                    "crossing_time_median_s": None if not len(crossing_t) else float(np.median(crossing_t)),
                    "crossing_speed_median_m_s": None if not len(crossing_speed) else float(np.median(crossing_speed)),
                    "fixed_model_ttc_error_median_ms": None if not len(t_error) else float(np.median(t_error) * 1000.0),
                    "fixed_model_ttc_abs_error_p95_ms": None if not len(t_error) else float(np.percentile(np.abs(t_error), 95) * 1000.0),
                    "fixed_model_lateral_error_median_mm": None if not len(y_error) else float(np.median(y_error) * 1000.0),
                    "fixed_model_lateral_abs_error_p95_mm": None if not len(y_error) else float(np.percentile(np.abs(y_error), 95) * 1000.0),
                    "crossing_lateral_drift_median_mm": None if not len(drift) else float(np.median(drift) * 1000.0),
                    "crossing_lateral_drift_mean_abs_mm": None if not len(drift) else float(np.mean(np.abs(drift)) * 1000.0),
                    "episode_deceleration_fit_median_m_s2": None if not len(episode_a) else float(np.median(episode_a)),
                    "episode_deceleration_fit_iqr_m_s2": None if not len(episode_a) else [float(np.percentile(episode_a, 25)), float(np.percentile(episode_a, 75))],
                }
            holdout_summary = {
                "mode": "fixed_deceleration_no_hand_holdout",
                "fixed_a_eff_m_s2": float(args.a_eff),
                "episodes": len(results), "episodes_by_speed": by_speed,
                "model_refit_on_holdout": False,
                "prediction_boundary": "each arrival prediction uses the fixed no-hand fit and the episode's initial measured state; held-out future crossing is only used by the evaluator",
            }
            (out / "r0_slide_holdout_summary.json").write_text(json.dumps(holdout_summary, indent=2), encoding="utf-8")
        for row in results:
            write_trace(traces_dir / f"slide_episode_{int(row['episode_id']):04d}.jsonl", trace_by_id[int(row["episode_id"])])
    elif args.mode in {"oracle", "sensitivity"}:
        if args.a_eff is None:
            raise ValueError("oracle and sensitivity modes require --a-eff from the no-hand calibration")
        for speed in args.speeds:
            for index in range(args.episodes):
                eid = len(results)
                seed = args.seed + eid
                row, trace, _ = run_oracle_trial(
                    env, eid, seed, float(speed), float(args.a_eff),
                    latency_s=args.latency,
                    position_bias_y_m=args.bias_y,
                    ttc_bias_s=args.ttc_bias_ms / 1000.0,
                    extra_latency_s=args.extra_latency_ms / 1000.0,
                )
                row["config_sha256"] = sha256(config_path)
                results.append(row)
                trace_by_id[eid] = trace
        csv_name = "r0_5_sensitivity.csv" if args.mode == "sensitivity" else "r0_5_oracle_per_episode.csv"
        write_csv(out / csv_name, results)
        reps = []
        seen_failures = set()
        for row in results:
            if row["success"] and not any(r["success"] for r in reps):
                reps.append(row)
            elif not row["success"] and row["failure_primary"] not in seen_failures:
                reps.append(row)
                seen_failures.add(row["failure_primary"])
        for row in reps:
            eid = int(row["episode_id"])
            write_trace(traces_dir / f"oracle_episode_{eid:04d}_{row['failure_primary']}.jsonl", trace_by_id[eid])
            rerun, _, frames = run_oracle_trial(
                env, eid, int(row["seed"]), float(row["launch_speed_m_s"]), float(args.a_eff),
                capture=True, latency_s=args.latency,
                position_bias_y_m=float(row["position_bias_y_m"]),
                ttc_bias_s=float(row["ttc_bias_s"]),
                extra_latency_s=args.extra_latency_ms / 1000.0,
            )
            save_gif(videos_dir / f"oracle_episode_{eid:04d}_{rerun['failure_primary']}.gif", frames)
        grouped = {}
        for speed in args.speeds:
            group = [r for r in results if abs(float(r["launch_speed_m_s"]) - float(speed)) < 1e-9]
            n = len(group)
            k = sum(bool(r["success"]) for r in group)
            # Wilson interval is finite and explicit for small pilot counts.
            z = 1.959963984540054
            phat = k / n if n else math.nan
            denom = 1 + z * z / n if n else math.nan
            center = (phat + z * z / (2 * n)) / denom if n else math.nan
            half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom if n else math.nan
            grouped[str(speed)] = {"n": n, "successes": k, "success_rate": phat, "wilson_95_ci": [center - half, center + half] if n else None}
        summary = {
            "mode": args.mode, "a_eff_m_s2": args.a_eff, "observation_latency_s": args.latency,
            "score_version": STRICT_SCORE_VERSION,
            "legacy_score_version": LEGACY_SCORE_VERSION,
            "success_by_speed": grouped,
            "legacy_success_by_speed": {
                str(speed): {
                    "n": sum(1 for r in results if abs(float(r["launch_speed_m_s"]) - float(speed)) < 1e-9),
                    "successes": sum(bool(r["success_legacy_r0"]) for r in results if abs(float(r["launch_speed_m_s"]) - float(speed)) < 1e-9),
                }
                for speed in args.speeds
            },
            "failure_counts": {key: sum(r["failure_primary"] == key for r in results) for key in sorted(set(r["failure_primary"] for r in results))},
            "not_formal_freeze": bool(args.mode == "sensitivity" or any(v["n"] < 100 for v in grouped.values())),
            "note": "Wilson intervals describe independent rollout outcomes only; current run is a development or single-factor sensitivity result.",
        }
        (out / ("r0_5_sensitivity_summary.json" if args.mode == "sensitivity" else "r0_5_oracle_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_manifest(args, config, out, finalize=True)
    env.gs.destroy()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["static", "slide", "oracle", "sensitivity", "preflight", "reference"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--speeds", type=lambda s: [float(v) for v in s.split(",")], default=[0.3, 0.4, 0.5])
    parser.add_argument("--speed", type=float, default=0.4, help="single speed used by reference pilot")
    parser.add_argument("--split", choices=["calibration", "holdout"], default="calibration")
    parser.add_argument("--a-eff", type=float)
    parser.add_argument("--latency", type=float, default=0.01)
    parser.add_argument("--bias-y", type=float, default=0.0)
    parser.add_argument("--ttc-bias-ms", type=float, default=0.0)
    parser.add_argument("--extra-latency-ms", type=float, default=0.0)
    parser.add_argument("--reference-duration", type=float, default=0.20)
    parser.add_argument("--reference-episodes", type=int, default=1, help="episodes per speed for complete reference rollouts")
    parser.add_argument("--reference-y-values", type=lambda s: [float(v) for v in s.split(",")], help="explicit launch y values, one per reference episode")
    parser.add_argument("--event-controller-hz", type=int, choices=[1000, 2000], default=2000)
    parser.add_argument("--visibility-hz", type=float, default=50.0)
    parser.add_argument("--physics-dt-us", type=float, help="override configured physical timestep in microseconds")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be >=1")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    run_batch(args)


if __name__ == "__main__":
    main()
