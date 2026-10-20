#!/usr/bin/env python3
"""Reconstruct cached HDR intervals with selective multi-reference surfaces.

The runner re-renders only exact saved robot/target trace states on g1. It
does not advance physics or synthesize state rows. Selected exact direct frames
are used as source references; every other logged query is reconstructed from
those references. By default, a render request at a logged time is charged as a
full-frame direct-render fallback. The optional ROI diagnostic uses either an
auxiliary camera with exact off-axis intrinsics or a per-frame scissor crop on
the original camera, and charges any request outside the rendered crop as a
full fallback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.diagnose_r1_native_hdr_replay import (
    infer_event_state,
    interpolate_trace_label,
    load_event_models,
    read_trace,
    project_world,
)
from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import (
    _pose,
    _resolve_motion_bodies,
    _sha256,
    _write_hash_manifest,
)
from benchmark.robot_closed_loop.robot_env import build_robot_scene
from benchmark.robot_closed_loop.diagnostics.r1_5_surface_raster_candidate import (
    prepare_surface_raster_source,
    surface_raster_warp,
)
from benchmark.robot_closed_loop.diagnostics.r1_5_multiref_surface_candidate import (
    compose_multireference_scene,
    interpolate_body_poses,
    moving_reference_indices,
    prepare_moving_projection_source,
    prepare_static_reference_composite,
    project_moving_source_to_target,
)
from benchmark.robot_closed_loop.diagnostics.r1_5_direct_roi_render import (
    configure_intrinsics_crop_camera,
    crop_render_to_frame_dict,
    restore_intrinsics_crop_camera,
)
from benchmark.robot_closed_loop.diagnostics.r1_5_scissor_roi_render import (
    MainCameraScissorROI,
    roi_from_mask,
)
from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.genesis_adapter import camera_to_world_cv
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_warp import interpolate_se3
from genesis_event_plugin.radiometry import linear_luminance
from benchmark.robot_closed_loop.event_state_model import five_bin_tensor


def _render_exact_logged_state(env, row: dict, pose_bodies: dict) -> dict:
    """Render only the HDR/depth/segmentation channels used by this method."""
    _set_exact_logged_state(env, row)
    hdr, depth, seg, _ = env.camera.render(
        rgb=True, depth=True, segmentation=True, radiance=True, force_render=True,
    )
    hdr = np.asarray(hdr, dtype=np.float32).copy()
    depth = np.asarray(depth, dtype=np.float32).copy()
    seg = np.asarray(seg, dtype=np.int32).copy()
    if not np.isfinite(hdr).all() or not np.isfinite(depth[np.isfinite(depth)]).all():
        raise RuntimeError("Genesis returned non-finite HDR/depth values")
    return {
        "time_s": round(float(row["time_s"]), 9),
        "hdr_rgb": hdr,
        "radiance_luma": linear_luminance(hdr),
        "depth": depth,
        "segmentation": seg,
        "poses": {int(seg_id): _pose(body) for seg_id, body in pose_bodies.items()},
    }


def _set_exact_logged_state(env, row: dict) -> None:
    """Restore only an already logged robot/target state; never step physics."""
    env.robot.set_qpos(np.asarray(row["robot_qpos"], dtype=np.float32), zero_velocity=True)
    obj = row["object"]
    target_qpos = np.asarray(list(obj["position_m"]) + list(obj["orientation_wxyz"]), dtype=np.float32)
    env.target.set_qpos(target_qpos, zero_velocity=True)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _response_config() -> DvsNoiseConfig:
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


def _events_from_frames(frames: list[dict]) -> tuple[np.ndarray, float]:
    if not frames:
        return np.empty((0, 4), dtype=np.float64), 0.0
    h, w = frames[0]["radiance_luma"].shape
    emulator = DvsEmulator(res=(h, w), config=_response_config(), device="cpu")
    started = time.perf_counter()
    emulator.initialize(frames[0]["radiance_luma"], float(frames[0]["time_s"]))
    chunks = []
    for frame in frames[1:]:
        event = emulator.generate_events(frame["radiance_luma"], float(frame["time_s"]))
        if len(event):
            chunks.append(np.asarray(event, dtype=np.float64))
    elapsed = time.perf_counter() - started
    events = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)
    if len(events):
        events = events[np.argsort(events[:, 0], kind="stable")]
    return events, elapsed


def _region_counts(events: np.ndarray, frame_records: list[dict],
                   target_ids: list[int], robot_ids: list[int]) -> dict[str, int]:
    out = {"target": 0, "robot": 0, "background": 0, "unknown": 0}
    if not len(events):
        return out
    times = np.asarray([x["time_s"] for x in frame_records], dtype=np.float64)
    for event in events:
        idx = int(np.clip(np.searchsorted(times, event[0], side="left"), 0, len(times) - 1))
        x, y = int(event[1]), int(event[2])
        seg = frame_records[idx]["segmentation"]
        if not (0 <= y < seg.shape[0] and 0 <= x < seg.shape[1]):
            out["unknown"] += 1
            continue
        label = int(seg[y, x])
        if label in target_ids:
            out["target"] += 1
        elif label in robot_ids:
            out["robot"] += 1
        elif label < 0:
            out["unknown"] += 1
        else:
            out["background"] += 1
    return out


def _exact_event_match_by_support(
    candidate_events: np.ndarray,
    direct_events: np.ndarray,
    times: np.ndarray,
    request_masks: np.ndarray,
    width: int,
) -> dict:
    """Compare exact (timestamp, pixel, polarity) event multisets by support."""
    def counters(events: np.ndarray) -> dict[str, Counter]:
        result = {"requested_interval": Counter(), "supported_interval": Counter()}
        for event in events:
            t, x, y, p = event
            x_i, y_i = int(x), int(y)
            if not (0 <= x_i < width and 0 <= y_i < request_masks.shape[1]):
                raise ValueError(f"event coordinate {(x_i, y_i)} lies outside request-mask dimensions")
            hi = int(np.clip(np.searchsorted(times, float(t), side="left"), 1, len(times) - 1))
            lo = hi - 1
            requested = bool(request_masks[lo, y_i, x_i] or request_masks[hi, y_i, x_i])
            region = "requested_interval" if requested else "supported_interval"
            key = (int(round(float(t) * 1_000_000_000)), x_i, y_i, int(np.sign(p)))
            result[region][key] += 1
        return result

    candidate = counters(candidate_events)
    direct = counters(direct_events)
    result = {}
    for region in ("requested_interval", "supported_interval"):
        cand_counter = candidate[region]
        direct_counter = direct[region]
        matches = sum(min(count, direct_counter.get(key, 0)) for key, count in cand_counter.items())
        result[region] = {
            "candidate_events": int(sum(cand_counter.values())),
            "direct_events": int(sum(direct_counter.values())),
            "exact_timestamp_pixel_polarity_matches": int(matches),
            "candidate_exact_match_fraction": float(matches / max(1, sum(cand_counter.values()))),
            "direct_exact_recall_fraction": float(matches / max(1, sum(direct_counter.values()))),
        }
    result["timestamp_quantization_s"] = 1e-9
    result["matching_rule"] = "multiset equality on timestamp rounded to 1 ns, x, y, and polarity; partitioned by whether either adjacent full-frame comparison request mask includes the event pixel"
    return result


def _write_event_montage(path: Path, frame: dict, events: np.ndarray,
                         candidate: np.ndarray, unknown: np.ndarray, label: str) -> None:
    h, w = frame["radiance_luma"].shape
    gray = np.clip(np.log1p(np.maximum(frame["radiance_luma"], 0.0)) * 85.0, 0, 255).astype(np.uint8)
    base = Image.fromarray(gray, mode="L").convert("RGB")
    overlay = base.copy()
    pix = np.asarray(overlay).copy()
    pix[np.asarray(unknown, dtype=bool)] = (255, 215, 0)
    if len(events):
        local = events[np.abs(events[:, 0] - frame["time_s"]) <= 0.00055]
        for event in local:
            x, y = int(event[1]), int(event[2])
            if 0 <= x < w and 0 <= y < h:
                pix[y, x] = (255, 40, 40) if event[3] > 0 else (50, 100, 255)
    overlay = Image.fromarray(pix)
    comparison = np.clip(np.log1p(np.maximum(candidate, 0.0)) * 85.0, 0, 255).astype(np.uint8)
    comparison[~np.isfinite(candidate)] = 0
    right = Image.fromarray(comparison, mode="L").convert("RGB")
    canvas = Image.new("RGB", (2 * w, h + 28), (20, 22, 25))
    canvas.paste(overlay, (0, 28))
    canvas.paste(right, (w, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), f"{label}: left direct + events, right surface candidate", fill=(240, 240, 240))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _query_models(events: np.ndarray, models: dict, query_times: list[float],
                  trace: list[dict], camera_pose: np.ndarray, intrinsics: np.ndarray,
                  width: int, height: int, fixed_height: float, source_name: str) -> list[dict]:
    rows = []
    for query_time in query_times:
        label = interpolate_trace_label(trace, query_time)
        position = np.asarray(label["position_m"], dtype=np.float64)
        velocity = np.asarray(label["velocity_m_s"], dtype=np.float64)
        gt_uv = project_world(position, camera_pose, intrinsics)
        for model_name, model in models.items():
            estimate, _ = infer_event_state(
                events, model, query_time, width, height, camera_pose,
                intrinsics, fixed_height,
            )
            row = {
                "query_time_s": float(query_time), "event_source": source_name,
                "model": model_name, "state_available": bool(estimate.get("state_available")),
                "label_method": label["label_method"],
                "event_count_25ms": estimate.get("event_count_25ms"),
                "nonempty_5ms_bins": estimate.get("nonempty_5ms_bins"),
                "position_error_xy_mm": None,
                "velocity_error_xy_m_s": None,
                "heatmap_uv_error_px": None,
                "estimated_position_xyz_m": estimate.get("position_xyz_m"),
                "estimated_velocity_xy_m_s": estimate.get("velocity_xy_m_s"),
                "estimated_uv_px": estimate.get("heatmap_uv_px"),
            }
            if estimate.get("state_available"):
                pred_pos = np.asarray(estimate["position_xyz_m"], dtype=np.float64)
                pred_vel = np.asarray(estimate["velocity_xy_m_s"], dtype=np.float64)
                pred_uv = np.asarray(estimate["heatmap_uv_px"], dtype=np.float64)
                row["position_error_xy_mm"] = float(np.linalg.norm(pred_pos[:2] - position[:2]) * 1000)
                row["velocity_error_xy_m_s"] = float(np.linalg.norm(pred_vel - velocity[:2]))
                row["heatmap_uv_error_px"] = float(np.linalg.norm(pred_uv - gt_uv))
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="previous R1.5 native_hdr_cutoff059 cache with episode_004/005 trace_window.jsonl")
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--roi-repair-config", type=Path, default=None,
                        help="optional exact-ray direct-render crop repair for candidate request pixels")
    parser.add_argument("--ours-checkpoint", type=Path, required=True)
    parser.add_argument("--v2e-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.5505)
    parser.add_argument("--end", type=float, default=0.5905)
    parser.add_argument("--query-start", type=float, default=0.5855)
    parser.add_argument("--source-interval", type=float, default=0.010,
                        help="requested source period; every source time must match an existing saved physical row")
    parser.add_argument("--max-source-views", type=int, default=5)
    parser.add_argument("--moving-source-views", type=int, default=None,
                        help="number of temporally nearest dynamic source views, always including both brackets")
    args = parser.parse_args()
    if args.moving_source_views is None:
        args.moving_source_views = args.max_source_views
    if not (2 <= args.moving_source_views <= args.max_source_views):
        raise ValueError("moving-source-views must be between 2 and max-source-views")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    scene_cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    scene_cfg["shared_control"] = json.loads(args.control_config.read_text(encoding="utf-8"))
    candidate_cfg = json.loads(args.candidate_config.read_text(encoding="utf-8"))
    roi_repair_cfg = (json.loads(args.roi_repair_config.read_text(encoding="utf-8"))
                      if args.roi_repair_config is not None else None)
    roi_cfg = roi_repair_cfg["roi"] if roi_repair_cfg else None
    roi_backend = str(roi_cfg.get("renderer", "auxiliary_intrinsics_camera")) if roi_cfg else None
    main_scissor_enabled = roi_backend == "main_camera_scissor"
    if roi_cfg and roi_backend not in {"auxiliary_intrinsics_camera", "main_camera_scissor"}:
        raise ValueError(f"unsupported ROI renderer {roi_backend!r}")
    roi_camera_name = (str(roi_cfg["camera_name"])
                       if roi_cfg and roi_backend == "auxiliary_intrinsics_camera" else None)
    roi_xyxy = (tuple(int(v) for v in roi_cfg["crop_xyxy"])
                if roi_cfg and roi_backend == "auxiliary_intrinsics_camera" else None)
    if roi_xyxy is not None and len(roi_xyxy) != 4:
        raise ValueError("roi.crop_xyxy must contain four integers")
    moving_edge_guard_px = int(candidate_cfg.get("geometry", {}).get("moving_edge_guard_px", 0))
    if moving_edge_guard_px < 0:
        raise ValueError("candidate geometry moving_edge_guard_px must be non-negative")
    cache_root = args.cache_root.resolve()
    episodes = [4, 5]
    traces = {}
    inputs = {}
    for episode_id in episodes:
        ep_dir = cache_root / f"episode_{episode_id:03d}"
        trace_path = ep_dir / "trace_window.jsonl"
        evidence_path = ep_dir / "rendered_frame_evidence.h5"
        source_path = ep_dir / "same_state_source_cache.h5"
        for path in (trace_path, evidence_path, source_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            inputs[str(path)] = _sha256(path)
        traces[episode_id] = _rows(trace_path)

    width, height = (int(x) for x in scene_cfg["camera"]["resolution_wh"])
    fov = float(scene_cfg["camera"]["vertical_fov_deg"])
    fx = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    intrinsics = np.asarray([fx, fx, width / 2.0, height / 2.0], dtype=np.float64)
    K = np.asarray([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    inputs[str(args.scene_config.resolve())] = _sha256(args.scene_config.resolve())
    inputs[str(args.control_config.resolve())] = _sha256(args.control_config.resolve())
    inputs[str(Path(__file__).resolve())] = _sha256(Path(__file__).resolve())
    candidate_path = Path(__file__).with_name("r1_5_surface_raster_candidate.py")
    inputs[str(candidate_path.resolve())] = _sha256(candidate_path.resolve())
    multiref_path = Path(__file__).with_name("r1_5_multiref_surface_candidate.py")
    inputs[str(multiref_path.resolve())] = _sha256(multiref_path.resolve())
    inputs[str(args.candidate_config.resolve())] = _sha256(args.candidate_config.resolve())
    roi_helper_path = Path(__file__).with_name("r1_5_direct_roi_render.py")
    inputs[str(roi_helper_path.resolve())] = _sha256(roi_helper_path.resolve())
    scissor_helper_path = Path(__file__).with_name("r1_5_scissor_roi_render.py")
    inputs[str(scissor_helper_path.resolve())] = _sha256(scissor_helper_path.resolve())
    if args.roi_repair_config is not None:
        inputs[str(args.roi_repair_config.resolve())] = _sha256(args.roi_repair_config.resolve())
    for path in (args.ours_checkpoint, args.v2e_checkpoint):
        inputs[str(path.resolve())] = _sha256(path.resolve())

    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning")
    env = None
    roi_camera = None
    main_scissor = None
    episode_records = {}
    inference_records = []
    try:
        diagnostic_camera_specs = {}
        if roi_cfg and not main_scissor_enabled:
            x0, y0, x1, y1 = roi_xyxy
            diagnostic_camera_specs[roi_camera_name] = {
                "resolution_wh": [x1 - x0, y1 - y0],
                # Use a broad renderer culling frustum; the auxiliary node is
                # then assigned exact off-axis intrinsics for the crop rays.
                "vertical_fov_deg": float(roi_cfg.get(
                    "culling_fov_deg", scene_cfg["camera"]["vertical_fov_deg"],
                )),
            }
        scene_build_started = time.perf_counter()
        env = build_robot_scene(
            scene_cfg, dt_s=float(scene_cfg["physics"]["dt_s"]), segmentation_level="link",
            diagnostic_camera_specs=diagnostic_camera_specs,
        )
        scene_build_wall_s = time.perf_counter() - scene_build_started
        env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True, force_render=True)
        roi_camera = None
        roi_camera_info = None
        main_scissor_info = None
        roi_camera_setup_wall_s = 0.0
        if roi_cfg and not main_scissor_enabled:
            roi_setup_started = time.perf_counter()
            roi_camera = env.diagnostic_cameras[roi_camera_name]
            roi_camera_info = configure_intrinsics_crop_camera(env.camera, roi_camera, roi_xyxy)
            roi_camera_setup_wall_s = time.perf_counter() - roi_setup_started
        elif main_scissor_enabled:
            roi_setup_started = time.perf_counter()
            main_scissor = MainCameraScissorROI(env.camera)
            main_scissor_info = main_scissor.install()
            roi_camera_setup_wall_s = time.perf_counter() - roi_setup_started
        pose_bodies = _resolve_motion_bodies(env)
        pose_ids = np.asarray(sorted(pose_bodies), dtype=np.int32)
        camera_pose = np.asarray(camera_to_world_cv(env.camera), dtype=np.float64)
        model_records, checkpoint_hashes = load_event_models(
            {"ours_old_checkpoint": args.ours_checkpoint.resolve(),
             "v2e_old_checkpoint": args.v2e_checkpoint.resolve()},
            camera_pose, intrinsics, (width, height),
        )
        fixed_height = float(scene_cfg["target"]["size_m"][2]) / 2.0
        target_ids = [int(seg_id) for seg_id, body in pose_bodies.items() if body is env.target]
        robot_ids = [int(seg_id) for seg_id, body in pose_bodies.items() if body is not env.target]
        for episode_id in episodes:
            all_trace = traces[episode_id]
            trace = [row for row in all_trace
                     if args.start - 1e-9 <= float(row["time_s"]) <= args.end + 1e-9]
            times = np.asarray([round(float(row["time_s"]), 9) for row in trace], dtype=np.float64)
            if not trace or abs(times[0] - args.start) > 1e-8 or abs(times[-1] - args.end) > 1e-8:
                raise RuntimeError(f"episode_{episode_id:03d}: selected window endpoints are not exact trace rows")
            query_times = [float(t) for t in times if args.query_start - 1e-9 <= t <= args.end + 1e-9]
            print(f"[episode_{episode_id:03d}] exact states={len(trace)} queries={len(query_times)}", flush=True)
            direct_frames = []
            render_per_frame = []
            for row in trace:
                started = time.perf_counter()
                frame = _render_exact_logged_state(env, row, pose_bodies)
                elapsed = time.perf_counter() - started
                direct_frames.append(frame)
                render_per_frame.append(elapsed)
            frame_by_time = {round(float(x["time_s"]), 9): x for x in direct_frames}
            render_time_by_time = {round(float(t), 9): float(dt)
                                   for t, dt in zip(times, render_per_frame)}
            ep_cache = cache_root / f"episode_{episode_id:03d}"
            with h5py.File(ep_cache / "same_state_source_cache.h5", "r") as cache:
                cached_source_times = np.asarray(cache["timestamps_s"][:], dtype=np.float64)
                cached_luma = np.asarray(cache["radiance_luma"][:], dtype=np.float32)
                cached_depth = np.asarray(cache["depth"][:], dtype=np.float32)
                cached_seg = np.asarray(cache["segmentation"][:], dtype=np.int32)
            with h5py.File(ep_cache / "rendered_frame_evidence.h5", "r") as cache:
                cached_dense_times = np.asarray(cache["dense_times_s"][:], dtype=np.float64)
                cached_dense_hdr = np.asarray(cache["dense_direct_hdr_rgb"][:], dtype=np.float32)
                cached_dense_seg = np.asarray(cache["dense_direct_segmentation"][:], dtype=np.int32)
            endpoint_rerender_checks = []
            for idx, t in enumerate(cached_source_times):
                if round(float(t), 9) not in frame_by_time:
                    continue
                current = frame_by_time[round(float(t), 9)]
                depth_mask = np.isfinite(cached_depth[idx]) & np.isfinite(current["depth"])
                endpoint_rerender_checks.append({
                    "time_s": float(t),
                    "luma_max_abs_error": float(np.max(np.abs(current["radiance_luma"] - cached_luma[idx]))),
                    "depth_max_abs_error_valid": float(np.max(np.abs(current["depth"][depth_mask] - cached_depth[idx][depth_mask]))) if np.any(depth_mask) else None,
                    "segmentation_mismatch_pixels": int(np.count_nonzero(current["segmentation"] != cached_seg[idx])),
                })
            dense_rerender_checks = []
            for idx, t in enumerate(cached_dense_times):
                if round(float(t), 9) not in frame_by_time:
                    continue
                current = frame_by_time[round(float(t), 9)]
                hdr_diff = np.abs(current["hdr_rgb"] - cached_dense_hdr[idx])
                dense_rerender_checks.append({
                    "time_s": float(t), "hdr_mae": float(hdr_diff.mean()),
                    "hdr_p99_abs": float(np.percentile(hdr_diff, 99)),
                    "hdr_max_abs": float(hdr_diff.max()),
                    "segmentation_mismatch_pixels": int(np.count_nonzero(current["segmentation"] != cached_dense_seg[idx])),
                })
            source_times = np.arange(args.start, args.end + args.source_interval / 2,
                                     args.source_interval, dtype=np.float64)
            source_times = np.asarray([round(float(x), 9) for x in source_times])
            missing_source_times = [float(t) for t in source_times if float(t) not in frame_by_time]
            if missing_source_times:
                raise ValueError(f"source times must match saved physical rows exactly: {missing_source_times}")
            source_frames = [frame_by_time[float(t)] for t in source_times]
            if args.max_source_views < 2 or args.max_source_views > len(source_frames):
                raise ValueError(f"max-source-views must be in [2, {len(source_frames)}]")
            source_render_s = float(sum(render_time_by_time[float(t)] for t in source_times))
            preparation_started = time.perf_counter()
            moving_projection_prepare_started = time.perf_counter()
            prepared_motion_sources = [prepare_moving_projection_source(
                frame["depth"], frame["segmentation"], K, camera_pose, pose_ids,
            ) for frame in source_frames]
            moving_projection_prepare_wall_s = time.perf_counter() - moving_projection_prepare_started
            surface_raster_prepare_started = time.perf_counter()
            prepared_raster_sources = [prepare_surface_raster_source(
                frame["radiance_luma"], frame["depth"], frame["segmentation"], K,
                raster_labels=set(pose_ids.tolist()),
            ) for frame in source_frames]
            surface_raster_prepare_wall_s = time.perf_counter() - surface_raster_prepare_started
            static_refs = [{
                "time_s": float(source["time_s"]), "radiance": source["radiance_luma"],
                "depth": source["depth"], "labels": source["segmentation"],
            } for source in source_frames]
            static_prepare_started = time.perf_counter()
            prepared_static = prepare_static_reference_composite(
                static_refs, pose_ids.tolist(), [float(row["time_s"]) for row in trace],
                composition_backend=str(candidate_cfg.get("geometry", {}).get(
                    "static_composition_backend", "numpy",
                )),
                audit_query_indices_against_numpy=bool(candidate_cfg.get("geometry", {}).get(
                    "audit_static_query_indices_against_numpy", False,
                )),
            )
            static_preparation_wall_s = time.perf_counter() - static_prepare_started
            preparation_wall_s = time.perf_counter() - preparation_started

            candidate_frames = []
            candidate_pre_fallback_frames = []
            candidate_final_frames = []
            candidate_request_masks = []
            frame_metrics = []
            roi_frame_metrics = []
            roi_crop_luma_frames = []
            roi_crop_depth_frames = []
            roi_crop_seg_frames = []
            roi_crop_times = []
            roi_crop_boxes = []
            roi_crop_row_indices = []
            reconstruction_wall_s = 0.0
            projection_wall_s = 0.0
            raster_wall_s = 0.0
            composition_wall_s = 0.0
            moving_composition_wall_s = 0.0
            static_composition_wall_s = 0.0
            visibility_merge_wall_s = 0.0
            moving_pair_numpy_audit_wall_s = 0.0
            roi_render_wall_s = 0.0
            roi_render_pixels = 0
            roi_repaired_pixels = 0
            roi_segmentation_audit_wall_s = 0.0
            roi_segmentation_mismatch_pixels = 0
            fallback_times = []
            query_source_meta = []
            for row, direct in zip(trace, direct_frames):
                t = round(float(row["time_s"]), 9)
                exact_source_index = np.flatnonzero(np.isclose(source_times, t, atol=1e-8, rtol=0))
                request_mask = np.zeros((height, width), dtype=bool)
                candidate_pre_fallback = np.asarray(direct["radiance_luma"], dtype=np.float32).copy()
                candidate_final_luma = candidate_pre_fallback.copy()
                diagnostics = {"mode": "exact_direct_endpoint"}
                roi_metric = None
                full_frame_fallback = False
                if exact_source_index.size:
                    candidate = direct
                    lower_time = upper_time = t
                    alpha = 0.0
                else:
                    hi = int(np.searchsorted(source_times, t, side="right"))
                    lo = hi - 1
                    if lo < 0 or hi >= len(source_times):
                        raise RuntimeError(f"time {t} lies outside the recorded source endpoints")
                    lower_time, upper_time = float(source_times[lo]), float(source_times[hi])
                    alpha = float((t - lower_time) / (upper_time - lower_time))
                    f0, f1 = frame_by_time[lower_time], frame_by_time[upper_time]
                    target_poses = interpolate_body_poses(f0["poses"], f1["poses"], alpha, pose_ids)
                    started = time.perf_counter()
                    Talpha = interpolate_se3(camera_pose, camera_pose, alpha)
                    moving_refs = []
                    raster_diagnostics = []
                    moving_source_indices = moving_reference_indices(
                        lo, hi, len(source_frames), args.moving_source_views, alpha,
                    )
                    for source_index, source in enumerate(source_frames):
                        if source_index not in moving_source_indices:
                            continue
                        projection_started = time.perf_counter()
                        uv, z, projected_valid = project_moving_source_to_target(
                            source["depth"], source["segmentation"], K,
                            camera_pose, Talpha, source["poses"], target_poses, pose_ids,
                            prepared_source=prepared_motion_sources[source_index],
                        )
                        projection_wall_s += time.perf_counter() - projection_started
                        raster_started = time.perf_counter()
                        warped = surface_raster_warp(
                            source["radiance_luma"], source["depth"], source["segmentation"],
                            uv, z, projected_valid, K, raster_labels=set(pose_ids.tolist()),
                            prepared_source=prepared_raster_sources[source_index],
                        )
                        raster_wall_s += time.perf_counter() - raster_started
                        moving_refs.append({
                            "time_s": float(source["time_s"]), "radiance": warped[0],
                            "depth": warped[1], "labels": warped[2], "visible": warped[3],
                        })
                        raster_diagnostics.append(warped[4])
                    composition_started = time.perf_counter()
                    candidate_pre_fallback, candidate_valid, request_mask, comp_diag = compose_multireference_scene(
                        moving_refs, static_refs, pose_ids.tolist(), t,
                        prepared_static=prepared_static,
                        moving_edge_guard_px=moving_edge_guard_px,
                        moving_composition_backend=str(candidate_cfg.get("geometry", {}).get(
                            "moving_composition_backend", "numpy",
                        )),
                        audit_moving_pair_against_numpy=bool(candidate_cfg.get(
                            "runtime", {},
                        ).get("audit_moving_pair_against_numpy", False)),
                    )
                    composition_elapsed = time.perf_counter() - composition_started
                    pair_audit_elapsed = float(comp_diag.get(
                        "moving_pair_numpy_audit_wall_s_excluded_from_candidate_cost", 0.0,
                    ))
                    composition_wall_s += max(0.0, composition_elapsed - pair_audit_elapsed)
                    moving_composition_wall_s += float(comp_diag.get("moving_composition_wall_s", 0.0))
                    static_composition_wall_s += float(comp_diag.get("static_composition_wall_s", 0.0))
                    visibility_merge_wall_s += float(comp_diag.get("visibility_merge_wall_s", 0.0))
                    moving_pair_numpy_audit_wall_s += pair_audit_elapsed
                    comp_summary = {key: value for key, value in comp_diag.items()
                                    if not isinstance(value, np.ndarray)}
                    reconstruction_wall_s += time.perf_counter() - started - pair_audit_elapsed
                    diagnostics = {
                        "mode": "selective_multireference_surface_raster",
                        "alpha": alpha,
                        "primary_source_times_s": [lower_time, upper_time],
                        "reference_source_times_s": [float(x["time_s"]) for x in source_frames],
                        "moving_id_count": int(len(pose_ids)),
                        "moving_raster_diagnostics": raster_diagnostics,
                        "composition": comp_summary,
                        "valid_fraction": float(candidate_valid.mean()),
                        "request_fraction": float(request_mask.mean()),
                    }
                    candidate = dict(direct)
                    candidate_final_luma = np.asarray(candidate_pre_fallback, dtype=np.float32).copy()
                    residual_request = request_mask.copy()
                    if (roi_camera is not None or main_scissor is not None) and np.any(request_mask):
                        _set_exact_logged_state(env, row)
                        roi_started = time.perf_counter()
                        if main_scissor is not None:
                            crop_xyxy = roi_from_mask(request_mask)
                            if crop_xyxy is None:
                                raise RuntimeError("non-empty request mask produced an empty scissor ROI")
                            x0, y0, x1, y1 = crop_xyxy
                            crop_hdr = main_scissor.render_hdr_crop(t, crop_xyxy)
                            crop_luma = linear_luminance(crop_hdr).astype(np.float32, copy=False)
                            crop_depth = None
                            crop_seg = None
                        else:
                            crop_xyxy = roi_xyxy
                            x0, y0, x1, y1 = crop_xyxy
                            crop_frame = crop_render_to_frame_dict(roi_camera, t)
                            crop_luma = np.asarray(crop_frame["radiance_luma"], dtype=np.float32)
                            crop_depth = np.asarray(crop_frame["depth"], dtype=np.float32)
                            crop_seg = np.asarray(crop_frame["segmentation"], dtype=np.int32)
                        roi_elapsed = time.perf_counter() - roi_started
                        roi_render_wall_s += roi_elapsed
                        roi_render_pixels += int((x1 - x0) * (y1 - y0))
                        direct_luma_crop = np.asarray(direct["radiance_luma"], dtype=np.float32)[y0:y1, x0:x1]
                        direct_depth_crop = np.asarray(direct["depth"], dtype=np.float32)[y0:y1, x0:x1]
                        direct_seg_crop = np.asarray(direct["segmentation"], dtype=np.int32)[y0:y1, x0:x1]
                        crop_depth_valid = (
                            np.isfinite(crop_depth) & np.isfinite(direct_depth_crop)
                            if crop_depth is not None else np.zeros_like(direct_depth_crop, dtype=bool)
                        )
                        crop_log_error = np.abs(
                            np.log(np.maximum(crop_luma, 1e-6))
                            - np.log(np.maximum(direct_luma_crop, 1e-6))
                        )
                        local_request = request_mask[y0:y1, x0:x1]
                        candidate_crop = candidate_final_luma[y0:y1, x0:x1]
                        candidate_crop[local_request] = crop_luma[local_request]
                        candidate_final_luma[y0:y1, x0:x1] = candidate_crop
                        residual_request[y0:y1, x0:x1] = False
                        patched = int(local_request.sum())
                        roi_repaired_pixels += patched
                        roi_crop_luma_frames.append(crop_luma.copy())
                        if crop_depth is not None:
                            roi_crop_depth_frames.append(crop_depth.copy())
                            roi_crop_seg_frames.append(crop_seg.copy())
                        roi_crop_times.append(t)
                        roi_crop_boxes.append((x0, y0, x1, y1))
                        roi_crop_row_indices.append(len(candidate_request_masks))
                        roi_metric = {
                            "time_s": t,
                            "render_wall_s": float(roi_elapsed),
                            "crop_xyxy": [x0, y0, x1, y1],
                            "request_pixels_in_crop": patched,
                            "request_pixels_outside_crop": int(residual_request.sum()),
                            "crop_luma_exact_equal_full_frame_crop": bool(np.array_equal(crop_luma, direct_luma_crop)),
                            "crop_luma_abs_max_vs_full_frame_crop": float(np.max(np.abs(crop_luma - direct_luma_crop))),
                            "crop_log_abs_p99_vs_full_frame_crop": float(np.percentile(crop_log_error, 99)),
                            "crop_segmentation_mismatch_pixels_vs_full_frame_crop": (
                                int(np.count_nonzero(crop_seg != direct_seg_crop))
                                if crop_seg is not None else None
                            ),
                            "crop_depth_compared_pixels": int(crop_depth_valid.sum()),
                            "crop_depth_abs_max_m_vs_full_frame_crop": (
                                float(np.max(np.abs(crop_depth[crop_depth_valid] - direct_depth_crop[crop_depth_valid])))
                                if crop_depth_valid.any() else None
                            ),
                            "crop_depth_abs_p99_m_vs_full_frame_crop": (
                                float(np.percentile(np.abs(crop_depth[crop_depth_valid] - direct_depth_crop[crop_depth_valid]), 99))
                                if crop_depth_valid.any() else None
                            ),
                            "renderer_backend": roi_backend,
                        }
                        roi_frame_metrics.append(roi_metric)
                    # Any request outside the exact crop remains unknown and
                    # receives the original full-frame direct fallback.
                    if np.any(residual_request):
                        fallback_times.append(t)
                        full_frame_fallback = True
                        candidate = direct
                        candidate_final_luma = np.asarray(direct["radiance_luma"], dtype=np.float32).copy()
                    else:
                        candidate["radiance_luma"] = candidate_final_luma
                    if not np.isfinite(candidate_final_luma).all():
                        raise RuntimeError(f"candidate frame at {t} has non-finite radiance outside its render requests")
                candidate_frames.append(candidate)
                candidate_pre_fallback_frames.append(candidate_pre_fallback.copy())
                candidate_final_frames.append(candidate_final_luma.copy())
                candidate_request_masks.append(request_mask.copy())
                direct_luma = np.asarray(direct["radiance_luma"], dtype=np.float32)
                if diagnostics["mode"] != "exact_direct_endpoint":
                    evaluated = ~request_mask & np.isfinite(candidate_pre_fallback)
                    err = np.abs(np.log(np.maximum(candidate_pre_fallback, 1e-6))
                                 - np.log(np.maximum(direct_luma, 1e-6)))
                    target_mask = np.isin(direct["segmentation"], target_ids)
                    robot_mask = np.isin(direct["segmentation"], robot_ids)
                    candidate_labels = np.asarray(comp_diag["target_labels"], dtype=np.int64)
                    candidate_depth = np.asarray(comp_diag["target_depth"], dtype=np.float32)
                    direct_labels = np.asarray(direct["segmentation"], dtype=np.int64)
                    direct_depth = np.asarray(direct["depth"], dtype=np.float32)
                    depth_compare = evaluated & np.isfinite(candidate_depth) & np.isfinite(direct_depth)
                    same_label = depth_compare & (candidate_labels == direct_labels)
                    depth_error = np.abs(candidate_depth - direct_depth)
                    mismatch_mask = evaluated & (candidate_labels != direct_labels)
                    if mismatch_mask.any():
                        label_pairs, label_pair_counts = np.unique(
                            np.stack((candidate_labels[mismatch_mask], direct_labels[mismatch_mask]), axis=1),
                            axis=0, return_counts=True,
                        )
                        mismatch_pairs = [
                            {"candidate_label": int(pair[0]), "direct_label": int(pair[1]), "pixels": int(count)}
                            for pair, count in zip(label_pairs, label_pair_counts)
                        ]
                    else:
                        mismatch_pairs = []
                    direct_dynamic = np.isin(direct_labels, pose_ids)
                    candidate_dynamic = np.isin(candidate_labels, pose_ids)
                    frame_metrics.append({
                        "time_s": t, "source_lo_s": lower_time, "source_hi_s": upper_time,
                        "alpha": alpha, "render_request_fraction": float(request_mask.mean()),
                        "used_direct_fallback": bool(full_frame_fallback),
                        "used_roi_direct_render": bool(roi_metric is not None),
                        "roi_render_wall_s": float(roi_metric["render_wall_s"]) if roi_metric else 0.0,
                        "roi_request_pixels_patched": int(roi_metric["request_pixels_in_crop"]) if roi_metric else 0,
                        "roi_request_pixels_outside_crop": int(roi_metric["request_pixels_outside_crop"]) if roi_metric else int(request_mask.sum()),
                        "valid_pixels": int(evaluated.sum()),
                        "log_abs_error_mean_valid": float(err[evaluated].mean()) if evaluated.any() else None,
                        "log_abs_error_p95_valid": float(np.percentile(err[evaluated], 95)) if evaluated.any() else None,
                        "log_abs_error_p95_target": float(np.percentile(err[evaluated & target_mask], 95)) if np.any(evaluated & target_mask) else None,
                        "log_abs_error_p95_robot": float(np.percentile(err[evaluated & robot_mask], 95)) if np.any(evaluated & robot_mask) else None,
                        "log_abs_error_p95_background": float(np.percentile(err[evaluated & ~(target_mask | robot_mask)], 95)) if np.any(evaluated & ~(target_mask | robot_mask)) else None,
                        "supported_segmentation_agreement_fraction": float(np.mean(candidate_labels[evaluated] == direct_labels[evaluated])) if evaluated.any() else None,
                        "supported_segmentation_mismatch_pixels": int(np.count_nonzero(evaluated & (candidate_labels != direct_labels))),
                        "supported_depth_abs_p95_m": float(np.percentile(depth_error[depth_compare], 95)) if depth_compare.any() else None,
                        "supported_depth_abs_p99_m": float(np.percentile(depth_error[depth_compare], 99)) if depth_compare.any() else None,
                        "supported_depth_abs_p95_same_label_m": float(np.percentile(depth_error[same_label], 95)) if same_label.any() else None,
                        "supported_depth_compared_pixels": int(depth_compare.sum()),
                        "supported_depth_same_label_pixels": int(same_label.sum()),
                        "supported_dynamic_occlusion_misses": int(np.count_nonzero(
                            evaluated & direct_dynamic & ~candidate_dynamic)),
                        "supported_false_dynamic_occlusions": int(np.count_nonzero(
                            evaluated & ~direct_dynamic & candidate_dynamic)),
                        "supported_wrong_dynamic_link_pixels": int(np.count_nonzero(
                            evaluated & direct_dynamic & candidate_dynamic & (direct_labels != candidate_labels))),
                        "supported_label_mismatch_pairs": mismatch_pairs,
                        "valid_radiance_p01_p50_p99": np.percentile(candidate_pre_fallback[np.isfinite(candidate_pre_fallback)], [1, 50, 99]).tolist(),
                        "direct_radiance_p01_p50_p99": np.percentile(direct_luma, [1, 50, 99]).tolist(),
                        "diagnostics": diagnostics,
                    })
                if roi_metric is not None:
                    roi_frame_metrics[-1]["candidate_post_repair_log_abs_error_p95_requested"] = (
                        float(np.percentile(
                            np.abs(
                                np.log(np.maximum(candidate_final_luma[request_mask], 1e-6))
                                - np.log(np.maximum(direct_luma[request_mask], 1e-6))
                            ), 95,
                        )) if request_mask.any() else None
                    )
                query_source_meta.append({"time_s": t, "mode": diagnostics["mode"],
                                          "alpha": alpha,
                                          "fallback": bool(np.any(request_mask) and (
                                              roi_metric is None or roi_metric["request_pixels_outside_crop"] > 0
                                          )),
                                          "roi_rendered": bool(roi_metric is not None),
                                          "roi_repaired_pixels": int(roi_metric["request_pixels_in_crop"]) if roi_metric else 0})

            if main_scissor is not None:
                for idx, crop_xyxy, roi_metric in zip(roi_crop_row_indices, roi_crop_boxes, roi_frame_metrics):
                    _set_exact_logged_state(env, trace[idx])
                    x0, y0, x1, y1 = crop_xyxy
                    seg_started = time.perf_counter()
                    scissor_seg = main_scissor.render_segmentation_crop(float(times[idx]), crop_xyxy)
                    roi_segmentation_audit_wall_s += time.perf_counter() - seg_started
                    direct_seg_crop = np.asarray(direct_frames[idx]["segmentation"], dtype=np.int32)[y0:y1, x0:x1]
                    mismatch = int(np.count_nonzero(scissor_seg != direct_seg_crop))
                    roi_segmentation_mismatch_pixels += mismatch
                    roi_metric["crop_segmentation_mismatch_pixels_vs_full_frame_crop"] = mismatch
                    roi_metric["segmentation_audit_wall_s_excluded_from_candidate_cost"] = float(
                        roi_segmentation_audit_wall_s
                    )

            candidate_events, candidate_dvs_s = _events_from_frames(candidate_frames)
            direct_events, direct_dvs_s = _events_from_frames(direct_frames)
            request_mask_stack = np.stack(candidate_request_masks).astype(bool)
            event_match_by_support = _exact_event_match_by_support(
                candidate_events, direct_events, times, request_mask_stack, width,
            )
            unsupported_direct_events = 0
            supported_direct_events = 0
            unsupported_event_regions = {"target": 0, "robot": 0, "background": 0, "unknown": 0}
            for idx in range(1, len(times)):
                lo, hi = times[idx - 1], times[idx]
                interval_events = direct_events[(direct_events[:, 0] > lo) &
                                                (direct_events[:, 0] <= hi)] if len(direct_events) else direct_events
                if not len(interval_events):
                    continue
                ex = interval_events[:, 1].astype(np.int64)
                ey = interval_events[:, 2].astype(np.int64)
                unsupported = request_mask_stack[idx - 1, ey, ex] | request_mask_stack[idx, ey, ex]
                unsupported_direct_events += int(unsupported.sum())
                supported_direct_events += int((~unsupported).sum())
                event_labels = np.asarray(direct_frames[idx]["segmentation"])[ey, ex]
                event_target = np.isin(event_labels, target_ids)
                event_robot = np.isin(event_labels, robot_ids)
                event_unknown = event_labels < 0
                unsupported_event_regions["target"] += int(np.count_nonzero(unsupported & event_target))
                unsupported_event_regions["robot"] += int(np.count_nonzero(unsupported & event_robot))
                unsupported_event_regions["unknown"] += int(np.count_nonzero(unsupported & event_unknown))
                unsupported_event_regions["background"] += int(np.count_nonzero(
                    unsupported & ~(event_target | event_robot | event_unknown)))
            ep_output = output / f"episode_{episode_id:03d}"
            ep_output.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(ep_output / "candidate_surface_raw_events.npz", events=candidate_events)
            np.savez_compressed(ep_output / "direct_logged_state_raw_events.npz", events=direct_events)
            np.savez_compressed(
                ep_output / "candidate_roi_replay_arrays.npz",
                times_s=times,
                candidate_reconstruction_radiance_luma=np.stack(candidate_pre_fallback_frames).astype(np.float32),
                candidate_final_radiance_luma=np.stack(candidate_final_frames).astype(np.float32),
                request_masks=request_mask_stack.astype(np.uint8),
                roi_times_s=np.asarray(roi_crop_times, dtype=np.float64),
                roi_crop_boxes_xyxy=np.asarray(roi_crop_boxes, dtype=np.int32).reshape(-1, 4),
                roi_crop_shapes=np.asarray([frame.shape for frame in roi_crop_luma_frames], dtype=np.int32).reshape(-1, 2),
                roi_radiance_luma_flat=(np.concatenate([frame.reshape(-1) for frame in roi_crop_luma_frames]).astype(np.float32)
                                        if roi_crop_luma_frames else np.empty((0,), dtype=np.float32)),
                roi_radiance_luma=(
                    np.stack(roi_crop_luma_frames).astype(np.float32)
                    if roi_crop_luma_frames and len({frame.shape for frame in roi_crop_luma_frames}) == 1
                    else np.empty((0, 0, 0), dtype=np.float32)
                ),
                roi_depth=(np.stack(roi_crop_depth_frames).astype(np.float32)
                           if roi_crop_depth_frames else np.empty((0, 0, 0), dtype=np.float32)),
                roi_segmentation=(np.stack(roi_crop_seg_frames).astype(np.int32)
                                  if roi_crop_seg_frames else np.empty((0, 0, 0), dtype=np.int32)),
            )
            np.savez_compressed(ep_output / "candidate_render_request_masks.npz",
                                times_s=times, masks=request_mask_stack.astype(np.uint8))
            with (ep_output / "frame_metrics.jsonl").open("w", encoding="utf-8") as stream:
                for metric in frame_metrics:
                    stream.write(json.dumps(metric, separators=(",", ":")) + "\n")
            with (ep_output / "roi_frame_metrics.jsonl").open("w", encoding="utf-8") as stream:
                for metric in roi_frame_metrics:
                    stream.write(json.dumps(metric, separators=(",", ":")) + "\n")
            with (ep_output / "candidate_frames.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "time_s", "mode", "alpha", "fallback", "roi_rendered", "roi_repaired_pixels",
                ])
                writer.writeheader()
                writer.writerows(query_source_meta)

            candidate_region = _region_counts(candidate_events, direct_frames, target_ids, robot_ids)
            direct_region = _region_counts(direct_events, direct_frames, target_ids, robot_ids)
            interval_counts = []
            for idx in range(1, len(times)):
                lo, hi = times[idx - 1], times[idx]
                c = candidate_events[(candidate_events[:, 0] > lo) & (candidate_events[:, 0] <= hi)] if len(candidate_events) else candidate_events
                d = direct_events[(direct_events[:, 0] > lo) & (direct_events[:, 0] <= hi)] if len(direct_events) else direct_events
                interval_counts.append({
                    "t0_s": float(lo), "t1_s": float(hi), "dt_s": float(hi - lo),
                    "candidate_events": int(len(c)), "candidate_on": int(np.count_nonzero(c[:, 3] > 0)) if len(c) else 0,
                    "candidate_off": int(np.count_nonzero(c[:, 3] < 0)) if len(c) else 0,
                    "direct_events": int(len(d)), "direct_on": int(np.count_nonzero(d[:, 3] > 0)) if len(d) else 0,
                    "direct_off": int(np.count_nonzero(d[:, 3] < 0)) if len(d) else 0,
                })
            seam_rows = []
            for seam in source_times[1:-1]:
                candidate_near = int(np.count_nonzero(np.abs(candidate_events[:, 0] - seam) <= 0.00025)) if len(candidate_events) else 0
                direct_near = int(np.count_nonzero(np.abs(direct_events[:, 0] - seam) <= 0.00025)) if len(direct_events) else 0
                seam_rows.append({"source_keyframe_time_s": float(seam), "candidate_events_within_0p25ms": candidate_near,
                                  "direct_events_within_0p25ms": direct_near})

            query_rows = []
            for source_name, events in (("candidate_surface", candidate_events),
                                        ("direct_logged_state", direct_events)):
                query_rows.extend(_query_models(
                    events, model_records, query_times, all_trace, camera_pose,
                    intrinsics, width, height, fixed_height, source_name,
                ))
            inference_records.extend(query_rows)
            if frame_metrics:
                for event_name, stream in (("candidate_surface", candidate_events),
                                           ("direct_logged_state", direct_events)):
                    np.savez_compressed(ep_output / f"{event_name}_query_0p5855_to_0p5905_25ms_bins.npz",
                                        voxels=np.stack([five_bin_tensor(stream, t, width, height)
                                                         for t in query_times]))
            # Representative success/failure visual evidence; episode 004 is
            # centered, episode 005 is the +55 mm lateral failure diagnostic.
            visual_idx = min(range(len(times)), key=lambda i: abs(float(times[i]) - 0.5855))
            direct = direct_frames[visual_idx]
            candidate_radiance = candidate_final_frames[visual_idx]
            unknown = candidate_request_masks[visual_idx]
            _write_event_montage(ep_output / "candidate_vs_direct_0p5855.png",
                                 direct, candidate_events, candidate_radiance, unknown,
                                 f"episode {episode_id} t={times[visual_idx]:.4f}")
            intervals = np.diff(times)
            rec = {
                "episode_id": episode_id,
                "episode_role": "center_control" if episode_id == 4 else "plus55mm_development_failure",
                "trace_rows": int(len(trace)), "query_times_s": query_times,
                "actual_trace_dt_s": {"min": float(intervals.min()), "median": float(np.median(intervals)),
                                      "p95": float(np.percentile(intervals, 95)), "max": float(intervals.max()),
                                      "effective_mean_hz": float(1.0 / intervals.mean())},
                "direct_render": {"exact_trace_state_count": len(trace), "wall_s_all_for_audit": float(sum(render_per_frame)),
                                  "wall_s_source_100hz_endpoints": source_render_s,
                                  "wall_s_all_trace_plus_dvs": float(sum(render_per_frame) + direct_dvs_s),
                                  "source_endpoint_cache_rerender_checks": endpoint_rerender_checks,
                                  "dense_direct_cache_rerender_checks": dense_rerender_checks},
                "candidate": {
                    "source_endpoint_interval_s": float(args.source_interval),
                    "source_views_per_query": int(args.max_source_views),
                    "moving_source_views_per_query": int(args.moving_source_views),
                    "moving_edge_guard_px": moving_edge_guard_px,
                    "static_composition_backend": str(prepared_static.get("composition_backend", "numpy")),
                    "static_query_index_build_wall_s": float(
                        prepared_static.get("query_index_build_wall_s", 0.0)
                    ),
                    "static_query_index_numpy_audit": prepared_static.get(
                        "query_index_numpy_audit"
                    ),
                    "static_query_index_numpy_audit_wall_s_excluded_from_candidate_cost": float(
                        prepared_static.get(
                            "query_index_numpy_audit_wall_s_excluded_from_candidate_cost", 0.0
                        )
                    ),
                    "reconstructed_non_endpoint_frames": int(sum(x["mode"] != "exact_direct_endpoint" for x in query_source_meta)),
                    "full_frame_fallback_count": len(fallback_times),
                    "full_frame_fallback_times_s": fallback_times,
                    "fallback_render_wall_s": float(sum(render_time_by_time[t] for t in fallback_times)),
                    "roi_direct_render": {
                        "enabled": bool(roi_cfg),
                        "renderer_backend": roi_backend,
                        "camera_setup_wall_s": float(roi_camera_setup_wall_s),
                        "crop_render_wall_s": float(roi_render_wall_s),
                        "crop_render_count": int(len(roi_crop_times)),
                        "crop_render_pixels": int(roi_render_pixels),
                        "crop_fraction_of_full_frame_pixels": (
                            float((roi_render_pixels / (len(roi_crop_times) * width * height)))
                            if roi_crop_times else 0.0
                        ),
                        "requested_pixels_repaired": int(roi_repaired_pixels),
                        "remaining_full_frame_fallback_count": int(len(fallback_times)),
                        "camera_projection": roi_camera_info,
                        "main_camera_scissor": main_scissor_info,
                        "segmentation_audit_wall_s_excluded_from_candidate_cost": float(
                            roi_segmentation_audit_wall_s
                        ),
                        "segmentation_mismatch_pixels_full_crop_audit": int(
                            roi_segmentation_mismatch_pixels
                        ),
                        "per_frame_evidence_file": f"episode_{episode_id:03d}/roi_frame_metrics.jsonl",
                        "rendered_crop_arrays_file": f"episode_{episode_id:03d}/candidate_roi_replay_arrays.npz",
                    },
                    "source_preparation_wall_s": float(preparation_wall_s),
                    "moving_projection_source_prepare_wall_s": float(moving_projection_prepare_wall_s),
                    "surface_raster_source_prepare_wall_s": float(surface_raster_prepare_wall_s),
                    "static_visibility_preparation_wall_s": float(static_preparation_wall_s),
                    "reconstruction_wall_s": reconstruction_wall_s,
                    "reconstruction_components_wall_s": {
                        "projection": float(projection_wall_s),
                        "surface_raster": float(raster_wall_s),
                        "multireference_composition": float(composition_wall_s),
                        "moving_frontmost_composition": float(moving_composition_wall_s),
                        "static_frontmost_composition": float(static_composition_wall_s),
                        "moving_static_visibility_merge": float(visibility_merge_wall_s),
                        "unattributed_overhead": float(max(0.0, reconstruction_wall_s - projection_wall_s
                                                              - raster_wall_s - composition_wall_s)),
                    },
                    "moving_pair_numpy_audit_wall_s_excluded_from_candidate_cost": float(
                        moving_pair_numpy_audit_wall_s
                    ),
                    "dvs_wall_s": candidate_dvs_s,
                    "one_time_roi_setup_wall_s_amortized": float(
                        roi_camera_setup_wall_s / max(1, len(episodes))
                        if main_scissor_enabled else 0.0
                    ),
                    "total_production_wall_s_estimate": float(source_render_s + sum(render_time_by_time[t] for t in fallback_times)
                                                                 + roi_render_wall_s + preparation_wall_s
                                                                 + (roi_camera_setup_wall_s / max(1, len(episodes))
                                                                    if main_scissor_enabled else 0.0)
                                                                 + reconstruction_wall_s + candidate_dvs_s),
                    "ratio_to_all_state_direct_render_plus_dvs": float(
                        (source_render_s + sum(render_time_by_time[t] for t in fallback_times)
                         + roi_render_wall_s + preparation_wall_s
                         + (roi_camera_setup_wall_s / max(1, len(episodes))
                            if main_scissor_enabled else 0.0)
                         + reconstruction_wall_s + candidate_dvs_s)
                        / max(1e-12, sum(render_per_frame) + direct_dvs_s)
                    ),
                    "event_count": int(len(candidate_events)), "region_event_counts": candidate_region,
                    "exact_event_match_vs_direct_by_support": event_match_by_support,
                    "request_mask_event_sensitivity_only": {
                        "interpretation": "Pre-repair direct-reference event overlap on candidate request pixels, retained to show how many direct events fell in areas that needed exact ROI rendering.",
                        "unsupported_pixel_interval_fraction": float(np.mean(
                            request_mask_stack[:-1] | request_mask_stack[1:])),
                        "direct_events_on_unsupported_pixel_intervals": int(unsupported_direct_events),
                        "direct_events_on_supported_pixel_intervals": int(supported_direct_events),
                        "direct_event_fraction_on_unsupported_pixel_intervals": float(
                            unsupported_direct_events / max(1, len(direct_events))),
                        "unsupported_direct_event_regions": unsupported_event_regions,
                    },
                    "positive_events": int(np.count_nonzero(candidate_events[:, 3] > 0)) if len(candidate_events) else 0,
                    "negative_events": int(np.count_nonzero(candidate_events[:, 3] < 0)) if len(candidate_events) else 0,
                    "query_window_event_counts": {
                        str(t): int(np.count_nonzero((candidate_events[:, 0] >= t - 0.025) & (candidate_events[:, 0] < t)))
                        if len(candidate_events) else 0 for t in query_times},
                },
                "direct_logged_state_comparator": {
                    "dvs_wall_s": direct_dvs_s, "event_count": int(len(direct_events)),
                    "region_event_counts": direct_region,
                    "positive_events": int(np.count_nonzero(direct_events[:, 3] > 0)) if len(direct_events) else 0,
                    "negative_events": int(np.count_nonzero(direct_events[:, 3] < 0)) if len(direct_events) else 0,
                },
                "event_interval_counts": interval_counts,
                "source_seam_event_counts": seam_rows,
                "frame_metrics_count": len(frame_metrics),
                "frame_metrics_file": f"episode_{episode_id:03d}/frame_metrics.jsonl",
                "candidate_raw_events_file": f"episode_{episode_id:03d}/candidate_surface_raw_events.npz",
                "direct_raw_events_file": f"episode_{episode_id:03d}/direct_logged_state_raw_events.npz",
                "candidate_path_semantics": (
                    "independent diagnostic; exact selected source endpoints; moving labels are selectively rasterized from the configured temporally nearest source views into interpolated target poses; fixed-camera static labels reuse observed pixels; per-pixel depth chooses visibility; request pixels are repaired by same-state main-camera scissor HDR rendering; any residual request triggers the original full-frame direct fallback"
                    if main_scissor_enabled else
                    "independent diagnostic; exact selected source endpoints; moving labels are selectively rasterized from the configured temporally nearest source views into interpolated target poses; fixed-camera static labels reuse observed pixels; per-pixel depth chooses visibility; request pixels are repaired by exact-ray direct rendering from a cropped auxiliary camera; any request outside the fixed crop triggers the original full-frame direct fallback"
                    if roi_cfg else
                    "independent diagnostic; exact source endpoints; moving labels are selectively rasterized from all already-rendered 100 Hz views into interpolated target poses; fixed-camera static labels reuse observed pixels; per-pixel depth chooses visibility; unknown mask at each existing logged state triggers a charged full direct-render fallback"
                ),
                "direct_render_is_offline": True,
                "new_physics_rollout": False,
                "frozen_simulator_core_modified": False,
            }
            (ep_output / "episode_summary.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            episode_records[str(episode_id)] = rec
            print(json.dumps({"episode": episode_id, "queries": len(query_times),
                              "fallbacks": len(fallback_times), "candidate_events": len(candidate_events),
                              "direct_events": len(direct_events), "candidate_reconstruction_s": reconstruction_wall_s,
                              "candidate_total_estimate_s": rec["candidate"]["total_production_wall_s_estimate"]},
                             separators=(",", ":")), flush=True)
    finally:
        if main_scissor is not None:
            main_scissor.close()
        if roi_camera is not None:
            restore_intrinsics_crop_camera(roi_camera)
        gs.destroy()

    with (output / "all_query_inference.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(inference_records[0].keys()) if inference_records else []
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inference_records)
    batch_candidate_total_s = float(sum(
        record["candidate"]["total_production_wall_s_estimate"]
        for record in episode_records.values()
    ))
    batch_direct_total_s = float(sum(
        record["direct_render"]["wall_s_all_trace_plus_dvs"]
        for record in episode_records.values()
    ))
    summary = {
        "experiment_id": str(candidate_cfg["experiment_id"]),
        "code_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "frozen_core_base": "e79e656ac00f7847b98db9e12bf43b093f3afff1",
        "frozen_core_modified": False,
        "candidate_config_sha256": _sha256(args.candidate_config.resolve()),
        "roi_repair_config_sha256": (_sha256(args.roi_repair_config.resolve())
                                      if args.roi_repair_config is not None else None),
        "inputs_sha256": inputs,
        "checkpoint_sha256": checkpoint_hashes,
        "scene_and_rig": {"camera_resolution_wh": [width, height], "vertical_fov_deg": fov,
                           "camera_pose_source": "Genesis official robot scene; same fixed side camera as cached R1.5",
                           "source_endpoint_rate_hz": 1.0 / args.source_interval,
                           "source_views_per_query": int(args.max_source_views),
                           "moving_source_views_per_query": int(args.moving_source_views),
                           "scene_build_wall_s": float(scene_build_wall_s)},
        "direct_roi_repair_method": {
            "enabled": bool(roi_cfg),
            "renderer_backend": roi_backend,
            "config": roi_cfg,
            "camera_projection": roi_camera_info,
            "main_camera_scissor": main_scissor_info,
            "implementation_boundary": "diagnostic-only renderer hooks around the Genesis 1.2.2 offscreen rasterizer; ROI repair uses either a separate exact-intrinsics camera or the original camera's scissor rectangle; no simulator core file or normal camera path is changed",
        },
        "rendered_exact_trace_window_s": [args.start, args.end],
        "query_window_s": [args.query_start, args.end],
        "response": {"model": "physical_log", "threshold_on_off": 0.2,
                     "noise_bandwidth_refractory": "disabled", "radiometry": "Genesis native linear HDR luminance"},
        "episodes": episode_records,
        "batch_cost_comparison": {
            "episode_count": int(len(episode_records)),
            "candidate_total_wall_s_including_amortized_roi_setup": batch_candidate_total_s,
            "direct_total_wall_s_all_trace_render_plus_dvs": batch_direct_total_s,
            "candidate_to_direct_ratio": float(batch_candidate_total_s / max(1e-12, batch_direct_total_s)),
        },
        "inference_rows": "all_query_inference.csv",
        "old_checkpoint_interpretation": "diagnostic only; checkpoints were trained on old generated events and do not determine candidate radiometric correctness",
        "common_reference_policy": "no ranking; compare candidate reconstruction with direct renders at the same exact logged robot states; models receive event tensors only; GT is scored after inference",
        "cost_accounting": {
            "candidate_total_includes": [
                "exact selected source HDR render time",
                "moving projection, surface-raster, and static visibility preparation, counted once",
                "per-query projection, rasterization, composition, and exact scissored HDR repair",
                "candidate DVS event generation",
                "one-time ROI/scissor installation amortized evenly across this run's episodes",
            ],
            "direct_total_includes": [
                "all exact trace-state Direct-HDR frame render time",
                "Direct-HDR comparator DVS event generation",
            ],
            "excluded_from_both": [
                "shared Genesis scene construction",
                "offline segmentation audit renders and their analysis",
                "model inference and diagnostic artifact serialization",
            ],
            "preparation_subcomponents": "static_visibility_preparation_wall_s is a diagnostic subdivision of source_preparation_wall_s and is not added a second time.",
            "numba_jit": "JIT time is included in whichever timed source-preparation or reconstruction call first triggers compilation; the runtime log and per-component times preserve cold/warm differences.",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_hash_manifest(output)
    print(json.dumps({"output": str(output), "episodes": list(episode_records),
                      "query_rows": len(inference_records), "hash_manifest": str(output / "SHA256SUMS")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
