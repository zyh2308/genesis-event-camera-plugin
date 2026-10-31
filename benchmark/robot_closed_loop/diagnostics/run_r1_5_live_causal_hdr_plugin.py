#!/usr/bin/env python3
"""Exercise strict HDR event delivery and a minimal controller in actual Genesis time.

Physics is advanced normally at 0.5 ms. Native HDR/depth/segmentation frames are
captured at each observed step, retained only for the current bracket, and the
production GenesisPhysicsEventPlugin is called at 2 ms endpoints. Its strict
provider may read only already captured exact-time samples. A tiny gripper
policy reads returned events and commands the fingers; the command is applied
by the following Genesis step. No pose is overwritten during an episode.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.robot_env import build_robot_scene, quaternion_matrix_wxyz
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.dvs_emulator import DvsNoiseConfig
from genesis_event_plugin.genesis_adapter import camera_to_world_cv
from genesis_event_plugin.physics_interpolator import RenderRequest
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.radiometry import linear_luminance


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def as_numpy(value, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def descriptor_entity_idx(value):
    if hasattr(value, "idx"):
        return int(value.idx)
    if hasattr(value, "entity"):
        return descriptor_entity_idx(value.entity)
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return descriptor_entity_idx(item)
            except (TypeError, ValueError, AttributeError):
                pass
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(f"cannot resolve segmentation descriptor {value!r}")


def current_sim_time(scene) -> float:
    value = scene.cur_t
    return float(value.item() if hasattr(value, "item") else value)


def target_to_world_cv(target) -> np.ndarray:
    qpos = as_numpy(target.get_qpos(), np.float64).reshape(-1)
    pos = as_numpy(target.get_pos(relative=False), np.float64).reshape(-1)[:3]
    if len(qpos) < 7:
        raise RuntimeError(f"free target qpos has no quaternion: shape={qpos.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_matrix_wxyz(qpos[3:7])
    transform[:3, 3] = pos
    return transform


def render_native(camera) -> dict[str, np.ndarray]:
    hdr, depth, seg, _ = camera.render(
        rgb=True, depth=True, segmentation=True, radiance=True, force_render=True,
    )
    hdr, depth, seg = as_numpy(hdr), as_numpy(depth), as_numpy(seg)
    if hdr.dtype != np.float32:
        hdr = hdr.astype(np.float32)
    if not np.isfinite(hdr).all() or np.any(hdr < 0):
        raise RuntimeError("native radiance contains invalid values")
    return {"rgb": hdr, "radiance": hdr, "depth": depth, "seg": seg}


def event_array(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1, 4)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    scene_cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    expected_scene_hash = cfg.get("expected_scene_config_sha256")
    if expected_scene_hash and sha256(args.scene_config) != expected_scene_hash:
        raise RuntimeError(f"scene config hash mismatch: expected={expected_scene_hash}, actual={sha256(args.scene_config)}")
    if args.output.exists():
        raise FileExistsError(args.output)
    dt = float(cfg["physics_dt_s"])
    duration = float(cfg["duration_s"])
    capture_dt = float(cfg["plugin_capture_interval_s"])
    stride_float = capture_dt / dt
    stride = int(round(stride_float))
    if abs(stride_float - stride) > 1e-9 or stride < 2:
        raise ValueError("capture interval must be an integer multiple of physics dt and >=2 steps")
    step_count = int(round(duration / dt))
    if abs(step_count * dt - duration) > 1e-9:
        raise ValueError("duration must be an integer multiple of physics dt")

    import genesis as gs
    from genesis.vis.camera import Camera

    signature = inspect.signature(Camera.render)
    if "radiance" not in signature.parameters:
        raise RuntimeError(f"HDR camera API missing: {sys.executable} {signature}")
    runtime_modules = {"genesis": Path(gs.__file__).resolve(),
                       "camera": Path(inspect.getsourcefile(Camera)).resolve()}
    from genesis.ext.pyrender import offscreen, renderer
    from genesis.vis import rasterizer
    runtime_modules.update({"rasterizer": Path(inspect.getsourcefile(rasterizer)).resolve(),
                            "renderer": Path(inspect.getsourcefile(renderer)).resolve(),
                            "offscreen": Path(inspect.getsourcefile(offscreen)).resolve()})

    args.output.mkdir(parents=True, exist_ok=False)
    shutil_config = args.output / "config.json"
    shutil_config.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    import torch
    torch.set_num_threads(1)
    total_wall_started = time.perf_counter()
    init_started = time.perf_counter()
    gs.init(backend=gs.cpu, logging_level="warning")
    genesis_init_wall = time.perf_counter() - init_started
    scene_started = time.perf_counter()
    env = build_robot_scene(scene_cfg, dt_s=dt, segmentation_level="link")
    scene_build_wall = time.perf_counter() - scene_started
    if getattr(env.camera, "_rasterizer", None) is None:
        raise RuntimeError("causal HDR smoke requires Genesis rasterizer backend")

    index_dict = getattr(env.scene, "segmentation_idx_dict", None)
    if not index_dict:
        raise RuntimeError("Genesis scene does not expose segmentation_idx_dict")
    target_seg_ids, floor_seg_ids = set(), set()
    for seg_id, descriptor in index_dict.items():
        try:
            entity_idx = descriptor_entity_idx(descriptor)
        except (TypeError, ValueError, AttributeError):
            continue
        if entity_idx == int(env.target.idx):
            target_seg_ids.add(int(seg_id))
        if entity_idx == int(env.plane.idx):
            floor_seg_ids.add(int(seg_id))
    if not target_seg_ids:
        raise RuntimeError(f"target segmentation identity unresolved; target entity={env.target.idx}")
    if not floor_seg_ids:
        raise RuntimeError(f"floor segmentation identity unresolved; plane entity={env.plane.idx}")

    response = DvsNoiseConfig(
        pos_thres=float(cfg["sensor"]["positive_threshold"]),
        neg_thres=float(cfg["sensor"]["negative_threshold"]),
        sigma_thres=float(cfg["sensor"]["threshold_mismatch_std"]),
        cutoff_hz=float(cfg["sensor"]["cutoff_hz"]),
        leak_rate_hz=float(cfg["sensor"]["leak_rate_hz"]),
        shot_noise_rate_hz=float(cfg["sensor"]["shot_noise_rate_hz"]),
        refractory_period_s=float(cfg["sensor"]["refractory_period_s"]),
        response_mode=str(cfg["sensor"]["response_mode"]),
        readout_model=str(cfg["sensor"]["readout_model"]),
        seed=int(cfg["sensor"]["seed"]),
    )

    current_payload: dict[str, Any] = {}
    observed: dict[float, dict[str, Any]] = {}
    request_trace: list[dict[str, Any]] = []
    provider_wall_s = 0.0

    def frame_provider(scene, cam):
        if "t" not in current_payload or abs(current_payload["t"] - current_sim_time(scene)) > 1e-8:
            raise RuntimeError("event plugin requested an endpoint other than the current direct observation")
        return current_payload["frame"]

    def motion_provider(scene, cam):
        return {"camera_to_world": camera_to_world_cv(cam),
                "object_to_world": {int(seg_id): target_to_world_cv(env.target) for seg_id in target_seg_ids}}

    def intermediate_state_provider(request: RenderRequest):
        nonlocal provider_wall_s
        started = time.perf_counter()
        provider_started_monotonic = time.monotonic()
        t_req = round(float(request.t), 9)
        matches = [(t, row) for t, row in observed.items() if abs(t - t_req) <= 1e-8]
        if len(matches) != 1:
            raise RuntimeError(f"request t={request.t:.9f} was not captured in actual physics order; candidates={len(matches)}")
        sample_t, sample = matches[0]
        endpoint_t = current_sim_time(env.scene)
        if sample_t > endpoint_t + 1e-9 or sample["captured_wall_s"] > provider_started_monotonic:
            raise RuntimeError("causal provider selected a future state")
        request_trace.append({"requested_t_s": float(request.t), "sample_t_s": sample_t,
                              "sample_available_at_s": sample_t,
                              "sample_captured_before_provider_call": sample["captured_wall_s"] <= provider_started_monotonic,
                              "provider_endpoint_t_s": endpoint_t,
                              "provenance": sample["provenance"]})
        provider_wall_s += time.perf_counter() - started
        return {"t": sample_t, "radiance": sample["frame"]["radiance"],
                "provenance": sample["provenance"], "available_at_s": sample_t}

    plugin = GenesisPhysicsEventPlugin(
        preset="clean", noise_config=response, input_space="linear",
        motion_state_provider=motion_provider, frame_provider=frame_provider,
        intermediate_state_provider=intermediate_state_provider,
        sampler_config=AdaptiveSamplingConfig(min_samples=2, max_samples=128),
        disocclusion_policy="hold", warp_composite="same_surface_log_linear",
        warp_backend="numpy", device="cpu", strict_fidelity=True,
        max_photometric_interval_s=float(cfg["max_photometric_interval_s"]),
        min_coverage_fraction=0.995,
    )
    plugin.attach(env.scene, env.camera)

    interval_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    event_archives: dict[str, np.ndarray] = {}
    episode_summaries: dict[str, Any] = {}
    overall_step_count = 0

    for ep_index, ep in enumerate(cfg["episodes"]):
        name = str(ep["name"])
        env.reset_episode(ep["start_position_m"], heading_rad=0.0)
        env.set_initial_target_velocity(ep["initial_velocity_m_s"])
        env.command_fingers_open()
        env.robot.control_dofs_position(env.home_qpos[:7], env.arm_dofs)
        observed.clear()
        request_trace.clear()
        episode_events: list[np.ndarray] = []
        keyframe_hdr, keyframe_times = [], []
        render_times, capture_times = [], []
        target_pixel_counts, floor_change_p95, floor_change_over = [], [], []
        plugin.start_episode()

        first_obs_started = time.perf_counter()
        first_frame = render_native(env.camera)
        first_t = round(current_sim_time(env.scene), 9)
        current_payload.update({"t": first_t, "frame": first_frame})
        observed[first_t] = {"t": first_t, "frame": first_frame,
                             "captured_wall_s": time.monotonic(),
                             "provenance": f"Genesis native HDR render at actual scene.cur_t={first_t:.9f}"}
        initial_luma = np.maximum(linear_luminance(first_frame["radiance"]), 1e-6)
        initial_log_luma = np.log(initial_luma)
        initial_floor = np.isin(first_frame["seg"], list(floor_seg_ids))
        first_render_wall = time.perf_counter() - first_obs_started
        render_times.append(first_render_wall)
        target_pixel_counts.append(int(np.count_nonzero(np.isin(first_frame["seg"], list(target_seg_ids)))))
        keyframe_hdr.append(first_frame["radiance"].copy())
        keyframe_times.append(first_t)
        first_events = event_array(plugin.capture())
        if len(first_events):
            raise RuntimeError("plugin initialization must not emit events before a second observed frame")

        command_latched = None
        pending_action: dict[str, Any] | None = None
        tail_action_step = 0
        decision_count = 0
        episode_started = time.perf_counter()
        latest_frame_wall = time.monotonic()

        for step in range(1, step_count + 1):
            sim_before_step = current_sim_time(env.scene)
            env.robot.control_dofs_position(env.home_qpos[:7], env.arm_dofs)
            if command_latched == "close":
                env.command_fingers_close()
            else:
                env.command_fingers_open()
            env.scene.step()
            sim_t = round(current_sim_time(env.scene), 9)
            if abs(sim_t - round(sim_before_step + dt, 9)) > 1e-8:
                raise RuntimeError(f"Genesis time did not advance by dt: {sim_before_step} -> {sim_t}")
            overall_step_count += 1
            if pending_action is not None:
                row_index = int(pending_action["interval_row_index"])
                interval_rows[row_index]["command_effective_at_sim_s"] = sim_t
                interval_rows[row_index]["command_effective_after_issued_s"] = (
                    sim_t - float(pending_action["issued_at_sim_s"])
                )
                interval_rows[row_index]["jaw_opening_after_effect_step_m"] = env.jaw_opening_m()
                interval_rows[row_index]["action_effective_order_pass"] = (
                    sim_t > float(pending_action["issued_at_sim_s"])
                )
                pending_action = None

            render_started = time.perf_counter()
            frame = render_native(env.camera)
            render_wall = time.perf_counter() - render_started
            render_times.append(render_wall)
            latest_frame_wall = time.monotonic()
            current_payload.update({"t": sim_t, "frame": frame})
            observed[sim_t] = {"t": sim_t, "frame": frame, "captured_wall_s": latest_frame_wall,
                               "provenance": f"Genesis native HDR render at actual scene.cur_t={sim_t:.9f}"}

            target_pixels = int(np.count_nonzero(np.isin(frame["seg"], list(target_seg_ids))))
            target_pixel_counts.append(target_pixels)
            current_floor = np.isin(frame["seg"], list(floor_seg_ids))
            common_floor = initial_floor & current_floor
            luma = np.maximum(linear_luminance(frame["radiance"]), 1e-6)
            floor_delta = np.abs(np.log(luma[common_floor]) - initial_log_luma[common_floor]) if np.any(common_floor) else np.empty(0)
            floor_p95 = float(np.percentile(floor_delta, 95)) if floor_delta.size else None
            floor_over = int(np.count_nonzero(floor_delta > 0.2)) if floor_delta.size else 0
            floor_change_p95.append(floor_p95)
            floor_change_over.append(floor_over)
            frame_rows.append({"episode": name, "step": step, "timestamp_s": sim_t,
                               "direct_render_wall_s": render_wall, "target_pixels": target_pixels,
                               "common_static_floor_pixels": int(np.count_nonzero(common_floor)),
                               "floor_log_luma_abs_p95_vs_initial": floor_p95,
                               "floor_pixels_log_luma_change_over_0p2": floor_over,
                               "jaw_opening_m": env.jaw_opening_m()})

            if step in {step_count // 2, step_count}:
                keyframe_hdr.append(frame["radiance"].copy())
                keyframe_times.append(sim_t)

            if step % stride == 0:
                capture_started = time.perf_counter()
                events = event_array(plugin.capture())
                capture_wall = time.perf_counter() - capture_started
                uncertainty = plugin.uncertainty
                output_available = float(uncertainty.get("output_available_at_s", sim_t))
                max_event_time = float(events[:, 0].max()) if len(events) else None
                provider_rows = [r for r in request_trace if abs(r["provider_endpoint_t_s"] - sim_t) <= 1e-8]
                requested_times = [float(r["requested_t_s"]) for r in provider_rows]
                if abs(output_available - sim_t) > 1e-8:
                    raise RuntimeError(f"plugin output availability {output_available} != right endpoint {sim_t}")
                if max_event_time is not None and max_event_time > output_available + 1e-9:
                    raise RuntimeError("plugin returned an event timestamp later than its availability time")
                if any(r["sample_available_at_s"] > output_available + 1e-9 for r in provider_rows):
                    raise RuntimeError("strict plugin consumed a future intermediate radiance frame")
                if uncertainty.get("strict_sampling_status") != "actual_state_samples":
                    raise RuntimeError(
                        "strict interval did not resolve from actual sampled states: "
                        f"{uncertainty.get('strict_sampling_status')}"
                    )
                max_anchor_gap = float(uncertainty.get("photometric_max_anchor_gap_s", float("inf")))
                if max_anchor_gap > float(cfg["max_photometric_interval_s"]) + 1e-9:
                    raise RuntimeError(
                        f"strict photometric anchor gap {max_anchor_gap} exceeds configured bound "
                        f"{cfg['max_photometric_interval_s']}"
                    )

                controller_read_t = current_sim_time(env.scene)
                controller_read_wall = time.perf_counter()
                event_count = int(len(events))
                command_issued = command_latched is None and event_count > 0
                if command_issued:
                    command_latched = "close"
                decision_count += 1
                if len(events):
                    episode_events.append(events)
                interval_row = {
                    "episode": name, "interval_index": step // stride,
                    "previous_plugin_endpoint_s": round(sim_t - capture_dt, 9),
                    "plugin_right_endpoint_s": sim_t,
                    "strict_output_available_at_sim_s": output_available,
                    "plugin_capture_wall_s": capture_wall,
                    "controller_read_sim_s": controller_read_t,
                    "controller_read_wall_monotonic_s": controller_read_wall,
                    "events_returned": event_count,
                    "event_time_min_s": float(events[:, 0].min()) if len(events) else None,
                    "event_time_max_s": max_event_time,
                    "request_count": len(requested_times),
                    "request_times_s_json": json.dumps(requested_times),
                    "provider_samples_past_or_current": all(r["sample_t_s"] <= sim_t + 1e-9 for r in provider_rows),
                    "provider_samples_captured_before_call": all(r["sample_captured_before_provider_call"] for r in provider_rows),
                    "strict_sampling_status": uncertainty.get("strict_sampling_status"),
                    "photometric_anchor_count": uncertainty.get("photometric_anchor_count"),
                    "output_ready_wall_after_capture_s": uncertainty.get("output_ready_after_right_endpoint_wall_s"),
                    "controller_command": command_latched or "open",
                    "command_issued_this_packet": command_issued,
                    "command_issued_at_sim_s": controller_read_t if command_issued else None,
                    "command_effective_at_sim_s": None,
                    "command_effective_after_issued_s": None,
                    "jaw_opening_after_effect_step_m": None,
                    "action_effective_order_pass": None,
                    "jaw_opening_at_decision_m": env.jaw_opening_m(),
                    "future_state_used": uncertainty.get("future_state_used", False),
                }
                interval_rows.append(interval_row)
                if command_issued:
                    pending_action = {"command": command_latched,
                                      "issued_at_sim_s": controller_read_t,
                                      "interval_row_index": len(interval_rows) - 1}
            min_keep = round(sim_t - capture_dt - 0.001, 9)
            for old_t in [t for t in observed if t < min_keep]:
                del observed[old_t]

        if pending_action is not None:
            env.robot.control_dofs_position(env.home_qpos[:7], env.arm_dofs)
            env.command_fingers_close()
            env.scene.step()
            overall_step_count += 1
            tail_action_step = 1
            effective_t = round(current_sim_time(env.scene), 9)
            row_index = int(pending_action["interval_row_index"])
            interval_rows[row_index]["command_effective_at_sim_s"] = effective_t
            interval_rows[row_index]["command_effective_after_issued_s"] = (
                effective_t - float(pending_action["issued_at_sim_s"])
            )
            interval_rows[row_index]["jaw_opening_after_effect_step_m"] = env.jaw_opening_m()
            interval_rows[row_index]["action_effective_order_pass"] = (
                effective_t > float(pending_action["issued_at_sim_s"])
            )
            pending_action = None
        episode_wall = time.perf_counter() - episode_started
        joined_events = np.concatenate(episode_events, axis=0) if episode_events else np.empty((0, 4), dtype=np.float64)
        event_archives[name] = joined_events
        target_array = np.asarray(target_pixel_counts, dtype=np.int64)
        floor_p95_values = np.asarray([x for x in floor_change_p95 if x is not None], dtype=np.float64)
        floor_over_array = np.asarray(floor_change_over, dtype=np.int64)
        episode_summaries[name] = {
            "physics_steps": step_count + tail_action_step,
            "tail_physics_steps_for_action_application": tail_action_step,
            "physics_dt_s": dt,
            "plugin_capture_count": decision_count,
            "episode_sim_duration_s": current_sim_time(env.scene),
            "episode_wall_s_excluding_scene_build": episode_wall,
            "direct_hdr_observation_count": step_count + 1,
            "direct_hdr_render_wall_s_sum": float(sum(render_times)),
            "direct_hdr_render_wall_s_mean": float(np.mean(render_times)),
            "direct_hdr_render_wall_s_p95": float(np.percentile(render_times, 95)),
            "plugin_capture_wall_s_total": float(sum(row["plugin_capture_wall_s"] for row in interval_rows if row["episode"] == name)),
            "event_count": int(len(joined_events)),
            "event_on_count": int(np.count_nonzero(joined_events[:, 3] > 0)) if len(joined_events) else 0,
            "event_off_count": int(np.count_nonzero(joined_events[:, 3] < 0)) if len(joined_events) else 0,
            "event_time_max_s": float(joined_events[:, 0].max()) if len(joined_events) else None,
            "target_visible_pixel_min": int(target_array.min()) if target_array.size else None,
            "target_visible_pixel_max": int(target_array.max()) if target_array.size else None,
            "common_floor_log_luma_abs_change_p95_max": float(floor_p95_values.max()) if floor_p95_values.size else None,
            "common_floor_pixels_change_over_0p2_max": int(floor_over_array.max()) if floor_over_array.size else None,
            "latch_command": command_latched,
            "event_triggered_action_count": sum(
                1 for row in interval_rows
                if row["episode"] == name and row["command_issued_this_packet"]
            ),
            "event_only_action_applied_next_step": all(
                row["action_effective_order_pass"] is True
                for row in interval_rows
                if row["episode"] == name and row["command_issued_this_packet"]
            ),
            "controller_input_boundary": "event packets only; GT state and segmentation/depth are never passed to policy",
        }
        if episode_summaries[name]["event_triggered_action_count"] == 0:
            raise RuntimeError(f"episode {name} produced no event-triggered action; action path was not exercised")
        if not episode_summaries[name]["event_only_action_applied_next_step"]:
            raise RuntimeError(f"episode {name} did not apply its event-triggered action on the next physics step")
        if episode_summaries[name]["target_visible_pixel_max"] <= 0:
            raise RuntimeError(f"episode {name} target was never visible to the camera")
        np.savez_compressed(args.output / f"{name}_keyframes.npz",
                            timestamps_s=np.asarray(keyframe_times, dtype=np.float64),
                            hdr_rgb=np.stack(keyframe_hdr).astype(np.float32))
        plugin.end_episode()

    plugin.close()
    np.savez_compressed(args.output / "raw_events.npz", **event_archives)
    with (args.output / "causal_trace.jsonl").open("w", encoding="utf-8") as stream:
        for row in interval_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (args.output / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = list(frame_rows[0]) if frame_rows else []
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(frame_rows)

    all_commands_causal = all(
        row["strict_output_available_at_sim_s"] <= row["controller_read_sim_s"] + 1e-9
        and (not row["command_issued_this_packet"]
             or (row["controller_read_sim_s"] == row["command_issued_at_sim_s"]
                 and row["action_effective_order_pass"] is True
                 and row["command_effective_at_sim_s"] > row["command_issued_at_sim_s"]))
        and row["provider_samples_past_or_current"]
        and row["provider_samples_captured_before_call"]
        and not row["future_state_used"]
        for row in interval_rows
    )
    if not all_commands_causal:
        raise RuntimeError("causal event/control ordering gate failed")

    source_commit = os.environ.get("GENESIS_EVENT_PLUGIN_SOURCE_COMMIT", "unknown")
    summary = {
        "run_id": cfg["run_id"], "source_commit": source_commit,
        "status": "PASS", "causal_order_gate": True,
        "physics_advanced_normally": True, "physics_state_overwrite_during_episode": False,
        "episodes": episode_summaries,
        "aggregate": {
            "physics_steps": overall_step_count,
            "physics_dt_s": dt, "plugin_capture_interval_s": capture_dt,
            "strict_sample_interval_s": float(cfg["max_photometric_interval_s"]),
            "actual_hdr_observations": sum(v["direct_hdr_observation_count"] for v in episode_summaries.values()),
            "controller_reads": len(interval_rows),
            "genesis_initialization_wall_s": genesis_init_wall,
            "scene_build_wall_s": scene_build_wall,
            "provider_wall_s": provider_wall_s,
            "provider_requests": len(request_trace),
            "all_provider_samples_previously_observed": all(r["sample_captured_before_provider_call"] for r in request_trace),
            "all_event_times_no_later_than_right_endpoint": all(
                row["event_time_max_s"] is None or row["event_time_max_s"] <= row["strict_output_available_at_sim_s"] + 1e-9
                for row in interval_rows
            ),
            "total_wall_s_including_scene_build": float(time.perf_counter() - total_wall_started),
            "renderer_and_module_hashes": {name: sha256(path) for name, path in runtime_modules.items()},
        },
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "platform": platform.platform(), "genesis_version": getattr(gs, "__version__", "unknown"),
                    "genesis_path": gs.__file__, "camera_render_signature": str(signature),
                    "egl_device_id": os.environ.get("EGL_DEVICE_ID"),
                    "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
                    "renderer_backend": "rasterizer",
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "config_sha256": sha256(args.config), "scene_config_sha256": sha256(args.scene_config),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "provider_request_trace": request_trace,
        "claim_boundary": cfg["claim_boundary"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"run_id": cfg["run_id"], "source_commit": source_commit,
                "config_sha256": summary["config_sha256"], "scene_config_sha256": summary["scene_config_sha256"],
                "runner_sha256": summary["runner_sha256"], "renderer_and_module_hashes": summary["aggregate"]["renderer_and_module_hashes"],
                "physics_state_overwrite_during_episode": False,
                "future_state_used": False,
                "controller_input": "DVS event packet only; evaluator metrics are not passed to policy"}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = []
    for path in sorted(p for p in args.output.iterdir() if p.is_file() and p.name != "SHA256SUMS"):
        sums.append(f"{sha256(path)}  {path.name}\n")
    (args.output / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")
    print(json.dumps({"run_id": cfg["run_id"], "status": "PASS", "output": str(args.output),
                      "causal_order_gate": all_commands_causal,
                      "provider_requests": len(request_trace),
                      "controller_reads": len(interval_rows)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
