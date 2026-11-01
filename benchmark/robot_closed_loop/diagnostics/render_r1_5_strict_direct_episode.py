#!/usr/bin/env python3
"""Render one exact saved robot trajectory and compare frozen strict vs direct HDR.

This is an offline visual replay of a previously verified Genesis physical
trace. It does not advance physics while rendering, and it never passes GT,
depth, segmentation, or robot state to the event-state network. Exact observed
HDR samples are used by both continuous DVS histories. The frozen core is
loaded from a separate checkout selected by --frozen-core-root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import h5py
import numpy as np


FROZEN_COMMIT = "90f792d166f000ef4f9e072a97633cc4ad39c8b7"
FROZEN_TAG = "r1.5-strict-sampled-20261031"
PROFILE_SHA256 = "3525cad9e84982a2b835b1b7d32428b94be24508fe28faf9f0d79002dfb10818"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _bootstrap_core_path() -> tuple[Path, list[str]]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--frozen-core-root", type=Path, required=True)
    known, remaining = bootstrap.parse_known_args()
    core_root = known.frozen_core_root.resolve()
    if not (core_root / "genesis_event_plugin" / "physics_interpolator.py").is_file():
        raise FileNotFoundError(f"not a plugin source root: {core_root}")
    sys.path.insert(0, str(core_root))
    return core_root, remaining


def _parse_args(core_root: Path, remaining: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-core-root", type=Path, required=True)
    parser.add_argument("--profile-config", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--legacy-cache-root", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--genesis-backend", choices=("cpu", "gpu"), default="cpu")
    return parser.parse_args(remaining)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _state_from_cache(handle: h5py.File, index: int = 0) -> dict[str, Any]:
    return {
        "time_s": float(handle["timestamps_s"][index]),
        "object": {
            "time_s": float(handle["timestamps_s"][index]),
            "position_m": np.asarray(handle["target_position_m"][index], dtype=np.float64),
            "orientation_wxyz": np.asarray(handle["target_orientation_wxyz"][index], dtype=np.float64),
            "velocity_m_s": np.asarray(handle["target_velocity_m_s"][index], dtype=np.float64),
            "angular_velocity_rad_s": np.asarray(handle["target_angular_velocity_rad_s"][index], dtype=np.float64)
            if "target_angular_velocity_rad_s" in handle else np.zeros(3, dtype=np.float64),
        },
        "robot_qpos": np.asarray(handle["robot_qpos"][index], dtype=np.float64),
    }


def _render_exact(env, row: dict[str, Any], pose_bodies: dict[int, Any], pose_fn) -> dict[str, Any]:
    env.robot.set_qpos(np.asarray(row["robot_qpos"], dtype=np.float32), zero_velocity=True)
    obj = row["object"]
    target_qpos = np.asarray([*obj["position_m"], *obj["orientation_wxyz"]], dtype=np.float32)
    env.target.set_qpos(target_qpos, zero_velocity=True)
    hdr, depth, seg, _ = env.camera.render(
        rgb=True, depth=True, segmentation=True, radiance=True, force_render=True,
    )
    hdr = np.asarray(hdr, dtype=np.float32).copy()
    depth = np.asarray(depth, dtype=np.float32).copy()
    seg = np.asarray(seg, dtype=np.int32).copy()
    if hdr.ndim != 3 or hdr.shape[-1] != 3 or not np.isfinite(hdr).all() or np.min(hdr) < 0:
        raise RuntimeError(f"invalid native linear HDR frame at t={row['time_s']}")
    poses = {int(seg_id): np.asarray(pose_fn(body), dtype=np.float64) for seg_id, body in pose_bodies.items()}
    return {"time_s": float(row["time_s"]), "radiance": hdr, "depth": depth, "seg": seg, "poses": poses}


def _event_array(chunks: list[np.ndarray]) -> np.ndarray:
    nonempty = [np.asarray(chunk, dtype=np.float64).reshape(-1, 4) for chunk in chunks if len(chunk)]
    if not nonempty:
        return np.empty((0, 4), dtype=np.float64)
    events = np.concatenate(nonempty, axis=0)
    return events[np.argsort(events[:, 0], kind="stable")]


def _canonical_events(events: np.ndarray) -> np.ndarray:
    value = np.asarray(events, dtype=np.float64).reshape(-1, 4)
    if not len(value):
        return value
    return value[np.lexsort((value[:, 3], value[:, 2], value[:, 1], value[:, 0]))]


def main() -> int:
    core_root, remaining = _bootstrap_core_path()
    args = _parse_args(core_root, remaining)
    source_root = args.source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(1, str(source_root))
    output = args.output.resolve()

    profile_path = (args.profile_config.resolve() if args.profile_config else
                    core_root / "benchmark/robot_closed_loop/configs/r1_5_core_strict_sampled_20261102.json")
    profile = _read_json(profile_path)
    if profile.get("profile") != "scientific_strict_sampled_reference":
        raise RuntimeError(f"unexpected frozen core profile: {profile.get('profile')}")
    if _sha256(profile_path) != PROFILE_SHA256:
        raise RuntimeError("strict profile hash differs from the reviewed frozen profile")

    import genesis as gs
    import genesis_event_plugin.physics_interpolator as physics_interpolator
    import genesis_event_plugin.physics_plugin as physics_plugin
    import genesis_event_plugin.adaptive_sampler as adaptive_sampler
    import genesis_event_plugin.physics_warp as physics_warp
    import genesis_event_plugin.radiometry as radiometry
    import genesis_event_plugin.noise_model as noise_model
    import genesis_event_plugin.genesis_adapter as genesis_adapter
    import genesis_event_plugin.dvs_emulator as dvs_emulator
    from genesis_event_plugin import GenesisPhysicsEventPlugin
    from genesis_event_plugin.dvs_emulator import DvsEmulator
    from genesis_event_plugin.genesis_adapter import camera_to_world_cv
    from genesis_event_plugin.noise_model import DvsNoiseConfig
    from genesis_event_plugin.radiometry import linear_luminance
    from genesis_event_plugin.physics_interpolator import RenderRequiredError
    from benchmark.robot_closed_loop.robot_env import build_robot_scene
    from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import (
        _pose, _resolve_motion_bodies,
    )
    from benchmark.robot_closed_loop.event_state_model import five_bin_tensor

    plugin_module_hashes = {}
    for module in (physics_interpolator, physics_plugin, adaptive_sampler, physics_warp,
                   radiometry, noise_model, genesis_adapter, dvs_emulator):
        if not Path(module.__file__).resolve().is_relative_to(core_root):
            raise RuntimeError(f"frozen core import escaped requested checkout: {module.__file__}")
        module_path = Path(module.__file__).resolve()
        plugin_module_hashes[str(module_path)] = _sha256(module_path)
    try:
        actual_commit = __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=core_root, text=True, stderr=__import__("subprocess").DEVNULL,
        ).strip()
    except Exception:
        commit_file = core_root / "CORE_COMMIT"
        actual_commit = commit_file.read_text(encoding="utf-8").strip() if commit_file.is_file() else ""
    if actual_commit != FROZEN_COMMIT:
        raise RuntimeError(f"frozen core checkout commit {actual_commit} != {FROZEN_COMMIT}")

    scene_cfg = _read_json(args.scene_config.resolve())
    scene_cfg["shared_control"] = _read_json(args.control_config.resolve())
    experiment = _read_json(args.experiment_config.resolve())
    cache_root = args.legacy_cache_root.resolve()
    trace_dir = args.trace_root.resolve() / f"episode_{args.episode_id:03d}"
    cache_dir = cache_root / f"episode_{args.episode_id:03d}"
    cache_path = cache_dir / "trajectory_cache.h5"
    episode_result_path = cache_dir / "episode_result.json"
    trace_path = trace_dir / "state_trace_0p5ms.jsonl"
    for required in (cache_path, episode_result_path, trace_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    trace = _read_trace(trace_path)
    result = _read_json(episode_result_path)
    if not trace:
        raise RuntimeError("empty exact-state trace")
    all_times = np.asarray([float(row["time_s"]) for row in trace], dtype=np.float64)
    dt = float(experiment["training_rollouts"]["strict_trace_recapture_cadence_s"])
    if not np.allclose(np.diff(all_times), dt, atol=1e-9, rtol=0):
        raise RuntimeError("state trace is not uniformly sampled at the declared 0.5 ms")
    with h5py.File(cache_path, "r") as handle:
        cache_times = np.asarray(handle["timestamps_s"], dtype=np.float64)
        t0_indices = np.flatnonzero(np.isclose(cache_times, 0.0, atol=1e-9, rtol=0))
        if len(t0_indices) != 1:
            raise RuntimeError("legacy HDR cache must provide the exact t=0 observed frame")
        initial_row = _state_from_cache(handle, int(t0_indices[0]))
        if not np.isclose(initial_row["time_s"], 0.0, atol=1e-9, rtol=0):
            raise RuntimeError("initial HDR cache frame is not at t=0")
        cache_camera_pose = np.asarray(handle["camera_to_world_cv"][int(t0_indices[0])], dtype=np.float64)
        cache_pose_ids = np.asarray(handle["segmentation_pose_ids"], dtype=np.int32)
        cache_validation = {
            round(float(t), 9): {
                "radiance": np.asarray(handle["radiance"][index], dtype=np.float32),
                "depth": np.asarray(handle["depth"][index], dtype=np.float32),
                "seg": np.asarray(handle["segmentation"][index], dtype=np.int32),
            }
            for index, t in enumerate(cache_times)
        }
    trace_time_keys = {round(float(t), 9) for t in all_times}
    missing_cached_times = [float(t) for t in cache_times
                            if t > 1e-9 and round(float(t), 9) not in trace_time_keys]
    if missing_cached_times:
        raise RuntimeError(f"verified trace misses legacy HDR timestamps: {missing_cached_times[:8]}")

    output.mkdir(parents=True, exist_ok=True)
    ep_out = output / "episodes" / f"episode_{args.episode_id:03d}"
    ep_out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    gs.init(backend=gs.cpu if args.genesis_backend == "cpu" else gs.gpu, logging_level="warning")
    init_wall_s = time.perf_counter() - started
    scene_started = time.perf_counter()
    env = build_robot_scene(scene_cfg, dt_s=float(scene_cfg["physics"]["dt_s"]), segmentation_level="link")
    pose_bodies = _resolve_motion_bodies(env)
    pose_ids = np.asarray(sorted(pose_bodies), dtype=np.int32)
    camera_pose = np.asarray(camera_to_world_cv(env.camera), dtype=np.float64)
    if not np.array_equal(pose_ids, cache_pose_ids):
        raise RuntimeError("render scene segmentation pose IDs do not match the cached rollout")
    if not np.allclose(camera_pose, cache_camera_pose, atol=1e-8, rtol=0):
        raise RuntimeError("render scene camera pose does not match the cached rollout")
    render_wall_s = 0.0
    rendered = 0
    cache_image_checks: list[dict[str, Any]] = []
    try:
        camera_probe = env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True, force_render=True)
        scene_build_wall_s = time.perf_counter() - scene_started

        sensor = DvsNoiseConfig(
            pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0, cutoff_hz=0.0,
            leak_rate_hz=0.0, leak_jitter_fraction=0.0, shot_noise_rate_hz=0.0,
            noise_rate_cov_decades=0.0, refractory_period_s=0.0, log_input=False,
            radiance_input=True, response_mode="physical_log", radiometric_gain=1.0,
            radiometric_offset=0.0, radiometric_floor=1e-6,
            radiometric_calibration_status="uncalibrated",
            readout_model="ideal_readout", seed=42,
        )
        response = {
            "positive_threshold": 0.2, "negative_threshold": 0.2, "threshold_noise": 0.0,
            "cutoff_hz": 0.0, "leak_rate_hz": 0.0, "shot_noise_rate_hz": 0.0,
            "refractory_period_s": 0.0, "radiance_floor": 1e-6, "seed": 42,
            "history": "one continuous history per episode; initialized at actual t=0 frame",
        }

        current = {"frame": None}
        by_time: dict[float, dict[str, Any]] = {}
        direct = DvsEmulator(res=(int(env.camera.res[1]), int(env.camera.res[0])), config=sensor, device="cpu")
        plugin = GenesisPhysicsEventPlugin(
            preset="clean", noise_config=sensor, input_space="linear",
            response_mode="physical_log", radiometric_gain=1.0,
            radiometric_offset=0.0, radiometric_floor=1e-6,
            radiometric_calibration_status="uncalibrated",
            disocclusion_policy="hold", warp_composite="same_surface_log_linear",
            warp_backend="auto", strict_fidelity=True,
            max_photometric_interval_s=0.0005, min_coverage_fraction=0.995,
            motion_state_provider=lambda _scene, _camera: {
                "camera_to_world": camera_pose,
                "object_to_world": current["frame"]["poses"],
            },
            frame_provider=lambda _scene, _camera: {
                "rgb": current["frame"]["radiance"],
                "radiance": current["frame"]["radiance"],
                "depth": current["frame"]["depth"], "seg": current["frame"]["seg"],
            },
            intermediate_state_provider=lambda request: _actual_sample(request, by_time),
        )
        plugin.attach(env.scene, env.camera)

        direct_chunks: list[np.ndarray] = []
        direct_availability_chunks: list[np.ndarray] = []
        strict_chunks: list[np.ndarray] = []
        strict_availability_chunks: list[np.ndarray] = []
        interval_rows: list[dict[str, Any]] = []
        direct_wall_s = 0.0
        strict_wall_s = 0.0
        render_hash = hashlib.sha256()
        # The old t=0 HDR observation is retained as the common real DVS
        # initialization frame, then every later frame is rendered from an
        # exact row in the verified 0.5 ms physical trace.
        initial_started = time.perf_counter()
        env.scene.cur_t = 0.0
        initial = _render_exact(env, initial_row, pose_bodies, _pose)
        render_wall_s += time.perf_counter() - initial_started
        rendered += 1
        with h5py.File(cache_path, "r") as handle:
            cached_hdr = np.asarray(handle["radiance"][int(t0_indices[0])], dtype=np.float32)
        initial_max_error = float(np.max(np.abs(initial["radiance"] - cached_hdr)))
        if initial_max_error > 1e-6:
            raise RuntimeError(f"re-rendered t=0 HDR differs from cached initial frame by {initial_max_error}")
        initial["time_s"] = 0.0
        cache_t0 = cache_validation[0.0]
        cache_image_checks.append({
            "time_s": 0.0, "radiance_max_abs_error": initial_max_error,
            "depth_finite_mask_mismatch": int(np.count_nonzero(np.isfinite(initial["depth"]) != np.isfinite(cache_t0["depth"]))),
            "segmentation_mismatch_pixels": int(np.count_nonzero(initial["seg"] != cache_t0["seg"])),
        })
        current["frame"] = initial
        by_time[0.0] = initial
        render_hash.update(np.ascontiguousarray(initial["radiance"]).tobytes())
        direct.initialize(linear_luminance(initial["radiance"]), t=0.0)
        env.scene.cur_t = 0.0
        plugin.start_episode()
        plugin.capture()

        missing_sample = None
        for row_index, row in enumerate(trace, start=1):
            before = time.perf_counter()
            env.scene.cur_t = float(row["time_s"])
            frame = _render_exact(env, row, pose_bodies, _pose)
            render_wall_s += time.perf_counter() - before
            rendered += 1
            t = float(frame["time_s"])
            key = round(t, 9)
            by_time[key] = frame
            if key in cache_validation:
                reference = cache_validation[key]
                cache_image_checks.append({
                    "time_s": key,
                    "radiance_max_abs_error": float(np.max(np.abs(frame["radiance"] - reference["radiance"]))),
                    "depth_finite_mask_mismatch": int(np.count_nonzero(np.isfinite(frame["depth"]) != np.isfinite(reference["depth"]))),
                    "segmentation_mismatch_pixels": int(np.count_nonzero(frame["seg"] != reference["seg"])),
                })
            current["frame"] = frame
            render_hash.update(np.ascontiguousarray(frame["radiance"]).tobytes())
            direct_started = time.perf_counter()
            direct_event = direct.generate_events(linear_luminance(frame["radiance"]), t)
            direct_wall_s += time.perf_counter() - direct_started
            if len(direct_event):
                direct_chunks.append(np.asarray(direct_event, dtype=np.float64))
                direct_availability_chunks.append(np.full(len(direct_event), t, dtype=np.float64))
            env.scene.cur_t = t
            # Only call the strict wrapper on 1 ms brackets. Its actual
            # midpoint provider returns the exact intervening 0.5 ms row.
            if np.isclose(t / 0.001, round(t / 0.001), atol=1e-7, rtol=0):
                strict_started = time.perf_counter()
                try:
                    strict_event = np.asarray(plugin.capture(), dtype=np.float64).reshape(-1, 4)
                except (RenderRequiredError, KeyError) as exc:
                    strict_wall_s += time.perf_counter() - strict_started
                    sampling = plugin.interpolator.last_sampling
                    requests = [r.as_dict() for r in plugin.interpolator.last_render_requests]
                    missing_sample = {
                        "at_endpoint_s": t,
                        "adaptive_sample_count": None if sampling is None else int(sampling.samples),
                        "error": str(exc), "requests": requests,
                    }
                    break
                strict_wall_s += time.perf_counter() - strict_started
                if len(strict_event):
                    strict_chunks.append(strict_event)
                    strict_availability_chunks.append(np.full(len(strict_event), t, dtype=np.float64))
                interval_rows.append({
                    "endpoint_s": t,
                    "adaptive_sample_count": int(plugin.interpolator.last_sampling.samples),
                    "anchor_count": int(plugin.interpolator.last_uncertainty.get("photometric_anchor_count", 0)),
                    "max_anchor_gap_s": float(plugin.interpolator.last_uncertainty.get("photometric_max_anchor_gap_s", 0.0)),
                    "extra_render_count": int(plugin.interpolator.last_uncertainty.get("extra_render_count", 0)),
                    "output_available_at_s": float(plugin.interpolator.last_uncertainty.get("output_available_at_s", t)),
                    "future_state_used": bool(plugin.interpolator.last_uncertainty.get("future_state_used", False)),
                    "strict_sample_output": plugin.interpolator.last_uncertainty.get("strict_sample_output"),
                    "strict_event_count": int(len(strict_event)),
                })
            for old_time in [value for value in by_time if value < t - 0.0010001]:
                del by_time[old_time]
        plugin.end_episode()
        plugin.close()

        direct_events = _event_array(direct_chunks)
        strict_events = _event_array(strict_chunks)
        direct_availability = np.concatenate(direct_availability_chunks) if direct_availability_chunks else np.empty(0, dtype=np.float64)
        strict_availability = np.concatenate(strict_availability_chunks) if strict_availability_chunks else np.empty(0, dtype=np.float64)
        if len(direct_availability) != len(direct_events) or len(strict_availability) != len(strict_events):
            raise RuntimeError("event timestamp/availability arrays lost alignment")
        np.savez_compressed(ep_out / "direct_hdr_events_1khz_2khz.npz",
                            events_2000hz=direct_events, available_at_s=direct_availability)
        np.savez_compressed(ep_out / "strict_sampled_events.npz",
                            events_2000hz=strict_events, available_at_s=strict_availability)
        _write_json(ep_out / "episode_result.json", result)
        shutil.copy2(trace_path, ep_out / "state_control_trace.jsonl")
        fields = ["episode_id", "seed", "launch_speed_m_s", "start_y_m", "first_contact_s", "lift_start_s", "sim_duration_s"]
        with (ep_out / "episode.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow({key: result.get(key) for key in fields})

        strict_coverage_end_s = float(interval_rows[-1]["endpoint_s"]) if interval_rows else 0.0
        direct_common = direct_events[
            (direct_events[:, 0] <= strict_coverage_end_s + 1e-12)
            & (direct_availability <= strict_coverage_end_s + 1e-12)
        ]
        strict_common = strict_events[
            (strict_events[:, 0] <= strict_coverage_end_s + 1e-12)
            & (strict_availability <= strict_coverage_end_s + 1e-12)
        ]
        direct_canonical = _canonical_events(direct_common)
        strict_canonical = _canonical_events(strict_common)
        identical_shape = direct_canonical.shape == strict_canonical.shape
        identical = bool(identical_shape and np.array_equal(direct_canonical, strict_canonical))
        event_max_abs_delta = None
        event_rows_equal = None
        if identical_shape:
            event_rows_equal = np.all(direct_canonical[:, 1:] == strict_canonical[:, 1:], axis=1) if len(direct_canonical) else np.empty(0, bool)
            event_max_abs_delta = float(np.max(np.abs(direct_canonical - strict_canonical))) if len(direct_canonical) else 0.0
        query_rows = []
        width, height = (int(value) for value in scene_cfg["camera"]["resolution_wh"])
        cadence = 1.0 / float(scene_cfg["controller"]["high_level_hz"])
        latency_s = float(scene_cfg["controller"]["observation_latency_s"])
        burn_in_s = float(scene_cfg["controller"]["event_burn_in_s"])
        duration_s = float(result.get("sim_duration_s", all_times[-1]))
        first_poll = np.ceil(burn_in_s / cadence - 1e-10) * cadence
        for now in np.arange(first_poll, duration_s + 1e-9, cadence):
            cutoff = float(now - latency_s)
            if cutoff <= 0.025:
                continue
            common_support = cutoff <= strict_coverage_end_s + 1e-12
            direct_visible = direct_availability <= cutoff + 1e-12
            strict_visible = strict_availability <= cutoff + 1e-12
            de = direct_events[direct_visible & (direct_events[:, 0] <= cutoff + 1e-12)]
            se = strict_events[strict_visible & (strict_events[:, 0] <= cutoff + 1e-12)]
            dv = five_bin_tensor(de, cutoff, width, height)
            sv = five_bin_tensor(se, cutoff, width, height)
            query_rows.append({
                "now_s": float(now), "cutoff_s": cutoff,
                "common_support": bool(common_support),
                "direct_event_count": int(dv.sum()), "strict_event_count": int(sv.sum()),
                "direct_available_event_count": int(direct_visible.sum()),
                "strict_available_event_count": int(strict_visible.sum()),
                "five_bin_exact": bool(common_support and np.array_equal(dv, sv)),
                "direct_five_bin_sha256": _array_sha(dv),
                "strict_five_bin_sha256": _array_sha(sv),
                "five_bin_abs_difference_sum": int(np.abs(dv.astype(np.int64) - sv.astype(np.int64)).sum()),
            })

        image_consistency_pass = (
            len(cache_image_checks) == len(cache_times)
            and max((r["radiance_max_abs_error"] for r in cache_image_checks), default=float("inf")) <= 1e-6
            and sum(r["depth_finite_mask_mismatch"] for r in cache_image_checks) == 0
            and sum(r["segmentation_mismatch_pixels"] for r in cache_image_checks) == 0
        )
        status = (
            "STRICT_PROVIDER_NEEDS_UNSAVED_ACTUAL_STATE" if missing_sample is not None
            else "DIRECT_RENDER_CACHE_MISMATCH" if not image_consistency_pass
            else "PASS"
        )

        summary = {
            "status": status,
            "episode_id": args.episode_id, "seed": int(result["seed"]),
            "speed_m_s": float(result.get("launch_speed_m_s", result.get("speed_m_s"))),
            "start_y_m": float(result.get("start_y_m", result.get("initial_y_m"))),
            "frozen_core_tag": FROZEN_TAG, "frozen_core_commit": FROZEN_COMMIT,
            "profile": profile.get("profile"), "profile_config_sha256": _sha256(profile_path),
            "source_code_commit": os.environ.get("ROBOT_EXECUTION_CODE_COMMIT", "unknown"),
            "runner_script_sha256": _sha256(Path(__file__).resolve()),
            "scene_config_sha256": _sha256(args.scene_config.resolve()),
            "control_config_sha256": _sha256(args.control_config.resolve()),
            "experiment_config_sha256": _sha256(args.experiment_config.resolve()),
            "source_trace_sha256": _sha256(trace_path), "legacy_cache_sha256": _sha256(cache_path),
            "physical_trace_interval_s": dt, "strict_capture_interval_s": 0.001,
            "direct_actual_frame_interval_s": dt,
            "dvs_response": response, "initial_cached_vs_rerender_hdr_max_abs_error": initial_max_error,
            "actual_rendered_frame_count": rendered, "actual_rendered_radiance_stream_sha256": render_hash.hexdigest(),
            "direct_event_count": int(len(direct_events)), "strict_event_count": int(len(strict_events)),
            "direct_events_sha256": _array_sha(direct_events), "strict_events_sha256": _array_sha(strict_events),
            "canonical_direct_events_sha256": _array_sha(direct_canonical),
            "canonical_strict_events_sha256": _array_sha(strict_canonical),
            "direct_common_support_event_count": int(len(direct_common)),
            "strict_common_support_event_count": int(len(strict_common)),
            "strict_coverage_end_s": strict_coverage_end_s,
            "direct_trace_end_s": float(all_times[-1]),
            "uncovered_direct_tail_s": max(0.0, float(all_times[-1]) - strict_coverage_end_s),
            "raw_events_exactly_equal_on_common_support": identical,
            "raw_event_max_abs_delta_on_common_support": event_max_abs_delta,
            "raw_event_same_pixel_polarity_fraction": None if event_rows_equal is None or not len(event_rows_equal) else float(event_rows_equal.mean()),
            "cache_image_consistency_check_count": len(cache_image_checks),
            "cache_image_consistency_max_radiance_error": max((r["radiance_max_abs_error"] for r in cache_image_checks), default=None),
            "cache_image_consistency_depth_mask_mismatches": sum(r["depth_finite_mask_mismatch"] for r in cache_image_checks),
            "cache_image_consistency_segmentation_mismatches": sum(r["segmentation_mismatch_pixels"] for r in cache_image_checks),
            "cache_image_consistency_pass": image_consistency_pass,
            "cache_image_consistency_checks": cache_image_checks,
            "strict_interval_count": len(interval_rows), "strict_intervals": interval_rows,
            "strict_missing_actual_sample": missing_sample,
            "query_count": len(query_rows),
            "five_bin_exact_query_count": sum(row["five_bin_exact"] for row in query_rows),
            "five_bin_queries": query_rows,
            "runtime": {
                "genesis_init_wall_s": init_wall_s, "scene_build_wall_s": scene_build_wall_s,
                "render_wall_s": render_wall_s,
                "direct_dvs_wall_s": direct_wall_s, "strict_plugin_wall_s": strict_wall_s,
                "whole_episode_wall_s_before_manifest": time.perf_counter() - started,
            },
            "offline_replay_boundary": "exact states and per-time native HDR are reconstructed offline from a verified trajectory; no new physics rollout and no model receives GT/depth/segmentation",
            "module_paths_and_sha256": plugin_module_hashes,
        }
        _write_json(ep_out / "summary.json", summary)
        _write_json(ep_out / "strict_intervals.json", {"intervals": interval_rows, "missing_actual_sample": missing_sample})
        (ep_out / "SHA256SUMS").write_text("".join(
            f"{_sha256(path)}  {path.name}\n" for path in sorted(ep_out.iterdir()) if path.is_file() and path.name != "SHA256SUMS"
        ), encoding="utf-8")
        (output / "SHA256SUMS").write_text("".join(
            f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in sorted(output.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"
        ), encoding="utf-8")
        print(json.dumps({"status": summary["status"], "episode_id": args.episode_id,
                          "direct_events": len(direct_events), "strict_events": len(strict_events),
                          "raw_equal_on_common_support": identical,
                          "five_bin_exact": summary["five_bin_exact_query_count"],
                          "queries": len(query_rows), "missing_sample": missing_sample}, indent=2), flush=True)
        return 0 if status == "PASS" else 2
    finally:
        gs.destroy()


def _actual_sample(request, by_time: dict[float, dict[str, Any]]) -> dict[str, Any]:
    key = round(float(request.t), 9)
    if key not in by_time:
        nearest = sorted(by_time, key=lambda t: abs(t - key))[:4]
        raise KeyError(f"strict requested t={key:.9f}; no actual rendered state at that time; nearest={nearest}")
    frame = by_time[key]
    if abs(float(frame["time_s"]) - float(request.t)) > 1e-8:
        raise KeyError("strict request did not match the timestamp of an actual rendered state")
    return {
        "t": float(request.t), "radiance": frame["radiance"],
        "provenance": "Genesis native scene-linear HDR render at exact saved physical state",
        "available_at_s": float(request.t),
    }


if __name__ == "__main__":
    raise SystemExit(main())
