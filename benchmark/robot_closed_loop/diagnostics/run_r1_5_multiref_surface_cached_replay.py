#!/usr/bin/env python3
"""Reconstruct cached HDR intervals with selective multi-reference surfaces.

The runner re-renders only exact saved robot/target trace states on g1. It
does not advance physics or synthesize state rows. Those exact direct frames
are used both for the 100 Hz candidate endpoints and as references at logged
query rows. A render request at a logged time is serviced by the already
measured full-frame direct render and charged to the candidate path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
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
from benchmark.robot_closed_loop.diagnostics.r1_5_surface_raster_candidate import surface_raster_warp
from benchmark.robot_closed_loop.diagnostics.r1_5_multiref_surface_candidate import (
    compose_multireference_scene,
    interpolate_body_poses,
    project_moving_source_to_target,
)
from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.genesis_adapter import camera_to_world_cv
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_warp import interpolate_se3
from genesis_event_plugin.radiometry import linear_luminance
from benchmark.robot_closed_loop.event_state_model import five_bin_tensor


def _render_exact_logged_state(env, row: dict, pose_bodies: dict) -> dict:
    """Render only the HDR/depth/segmentation channels used by this method."""
    env.robot.set_qpos(np.asarray(row["robot_qpos"], dtype=np.float32), zero_velocity=True)
    obj = row["object"]
    target_qpos = np.asarray(list(obj["position_m"]) + list(obj["orientation_wxyz"]), dtype=np.float32)
    env.target.set_qpos(target_qpos, zero_velocity=True)
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
    parser.add_argument("--ours-checkpoint", type=Path, required=True)
    parser.add_argument("--v2e-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.5505)
    parser.add_argument("--end", type=float, default=0.5905)
    parser.add_argument("--query-start", type=float, default=0.5855)
    parser.add_argument("--source-interval", type=float, default=0.010)
    parser.add_argument("--max-source-views", type=int, default=5)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    scene_cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    scene_cfg["shared_control"] = json.loads(args.control_config.read_text(encoding="utf-8"))
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
    candidate_config = ROOT / "benchmark/robot_closed_loop/configs/r1_5_multiref_surface_candidate_20261021c.json"
    inputs[str(candidate_config.resolve())] = _sha256(candidate_config.resolve())
    for path in (args.ours_checkpoint, args.v2e_checkpoint):
        inputs[str(path.resolve())] = _sha256(path.resolve())

    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning")
    env = None
    episode_records = {}
    inference_records = []
    try:
        env = build_robot_scene(scene_cfg, dt_s=float(scene_cfg["physics"]["dt_s"]), segmentation_level="link")
        env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True, force_render=True)
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
            source_frames = [frame_by_time[float(t)] for t in source_times]
            if args.max_source_views < 2 or args.max_source_views > len(source_frames):
                raise ValueError(f"max-source-views must be in [2, {len(source_frames)}]")
            if args.max_source_views != len(source_frames):
                raise ValueError("this first multi-reference run requires all exact keyframes in the cached interval")
            source_render_s = float(sum(render_time_by_time[float(t)] for t in source_times))

            candidate_frames = []
            candidate_request_masks = []
            frame_metrics = []
            reconstruction_wall_s = 0.0
            fallback_times = []
            query_source_meta = []
            for row, direct in zip(trace, direct_frames):
                t = round(float(row["time_s"]), 9)
                exact_source_index = np.flatnonzero(np.isclose(source_times, t, atol=1e-8, rtol=0))
                request_mask = np.zeros((height, width), dtype=bool)
                candidate_pre_fallback = np.asarray(direct["radiance_luma"], dtype=np.float32).copy()
                diagnostics = {"mode": "exact_direct_endpoint"}
                if exact_source_index.size:
                    candidate = direct
                    lower_time = upper_time = t
                    alpha = 0.0
                else:
                    hi = int(np.searchsorted(source_times, t, side="right"))
                    lo = hi - 1
                    if lo < 0 or hi >= len(source_times):
                        raise RuntimeError(f"time {t} lies outside the recorded 100 Hz endpoints")
                    lower_time, upper_time = float(source_times[lo]), float(source_times[hi])
                    alpha = float((t - lower_time) / (upper_time - lower_time))
                    f0, f1 = frame_by_time[lower_time], frame_by_time[upper_time]
                    target_poses = interpolate_body_poses(f0["poses"], f1["poses"], alpha, pose_ids)
                    started = time.perf_counter()
                    Talpha = interpolate_se3(camera_pose, camera_pose, alpha)
                    moving_refs = []
                    raster_diagnostics = []
                    for source in source_frames:
                        uv, z, projected_valid = project_moving_source_to_target(
                            source["depth"], source["segmentation"], K,
                            camera_pose, Talpha, source["poses"], target_poses, pose_ids,
                        )
                        warped = surface_raster_warp(
                            source["radiance_luma"], source["depth"], source["segmentation"],
                            uv, z, projected_valid, K, raster_labels=set(pose_ids.tolist()),
                        )
                        moving_refs.append({
                            "time_s": float(source["time_s"]), "radiance": warped[0],
                            "depth": warped[1], "labels": warped[2], "visible": warped[3],
                        })
                        raster_diagnostics.append(warped[4])
                    static_refs = [{
                        "time_s": float(source["time_s"]), "radiance": source["radiance_luma"],
                        "depth": source["depth"], "labels": source["segmentation"],
                    } for source in source_frames]
                    candidate_pre_fallback, candidate_valid, request_mask, comp_diag = compose_multireference_scene(
                        moving_refs, static_refs, pose_ids.tolist(), t,
                    )
                    comp_summary = {key: value for key, value in comp_diag.items()
                                    if not isinstance(value, np.ndarray)}
                    reconstruction_wall_s += time.perf_counter() - started
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
                    # These are the exact logged physical rows already rendered
                    # for comparison. A request is charged as a full-frame
                    # render in the candidate runtime estimate; no synthetic
                    # intermediate state is created.
                    if np.any(request_mask):
                        fallback_times.append(t)
                        candidate = direct
                    else:
                        candidate = dict(direct)
                        candidate["radiance_luma"] = candidate_pre_fallback
                candidate_frames.append(candidate)
                candidate_request_masks.append(request_mask.copy())
                direct_luma = np.asarray(direct["radiance_luma"], dtype=np.float32)
                if diagnostics["mode"] != "exact_direct_endpoint":
                    evaluated = ~request_mask & np.isfinite(candidate_pre_fallback)
                    err = np.abs(np.log(np.maximum(candidate_pre_fallback, 1e-6))
                                 - np.log(np.maximum(direct_luma, 1e-6)))
                    target_mask = np.isin(direct["segmentation"], target_ids)
                    robot_mask = np.isin(direct["segmentation"], robot_ids)
                    frame_metrics.append({
                        "time_s": t, "source_lo_s": lower_time, "source_hi_s": upper_time,
                        "alpha": alpha, "render_request_fraction": float(request_mask.mean()),
                        "used_direct_fallback": bool(np.any(request_mask)),
                        "valid_pixels": int(evaluated.sum()),
                        "log_abs_error_mean_valid": float(err[evaluated].mean()) if evaluated.any() else None,
                        "log_abs_error_p95_valid": float(np.percentile(err[evaluated], 95)) if evaluated.any() else None,
                        "log_abs_error_p95_target": float(np.percentile(err[evaluated & target_mask], 95)) if np.any(evaluated & target_mask) else None,
                        "log_abs_error_p95_robot": float(np.percentile(err[evaluated & robot_mask], 95)) if np.any(evaluated & robot_mask) else None,
                        "log_abs_error_p95_background": float(np.percentile(err[evaluated & ~(target_mask | robot_mask)], 95)) if np.any(evaluated & ~(target_mask | robot_mask)) else None,
                        "valid_radiance_p01_p50_p99": np.percentile(candidate_pre_fallback[np.isfinite(candidate_pre_fallback)], [1, 50, 99]).tolist(),
                        "direct_radiance_p01_p50_p99": np.percentile(direct_luma, [1, 50, 99]).tolist(),
                        "diagnostics": diagnostics,
                    })
                query_source_meta.append({"time_s": t, "mode": diagnostics["mode"],
                                          "alpha": alpha, "fallback": bool(np.any(request_mask))})

            candidate_events, candidate_dvs_s = _events_from_frames(candidate_frames)
            direct_events, direct_dvs_s = _events_from_frames(direct_frames)
            ep_output = output / f"episode_{episode_id:03d}"
            ep_output.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(ep_output / "candidate_surface_raw_events.npz", events=candidate_events)
            np.savez_compressed(ep_output / "direct_logged_state_raw_events.npz", events=direct_events)
            with (ep_output / "frame_metrics.jsonl").open("w", encoding="utf-8") as stream:
                for metric in frame_metrics:
                    stream.write(json.dumps(metric, separators=(",", ":")) + "\n")
            with (ep_output / "candidate_frames.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["time_s", "mode", "alpha", "fallback"])
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
            candidate = candidate_frames[visual_idx]
            unknown = candidate_request_masks[visual_idx]
            _write_event_montage(ep_output / "candidate_vs_direct_0p5855.png",
                                 direct, candidate_events, candidate["radiance_luma"], unknown,
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
                                  "source_endpoint_cache_rerender_checks": endpoint_rerender_checks,
                                  "dense_direct_cache_rerender_checks": dense_rerender_checks},
                "candidate": {
                    "source_endpoint_interval_s": float(args.source_interval),
                    "source_views_per_query": int(args.max_source_views),
                    "reconstructed_non_endpoint_frames": int(sum(x["mode"] != "exact_direct_endpoint" for x in query_source_meta)),
                    "full_frame_fallback_count": len(fallback_times),
                    "full_frame_fallback_times_s": fallback_times,
                    "fallback_render_wall_s": float(sum(render_time_by_time[t] for t in fallback_times)),
                    "reconstruction_wall_s": reconstruction_wall_s,
                    "dvs_wall_s": candidate_dvs_s,
                    "total_production_wall_s_estimate": float(source_render_s + sum(render_time_by_time[t] for t in fallback_times)
                                                                 + reconstruction_wall_s + candidate_dvs_s),
                    "event_count": int(len(candidate_events)), "region_event_counts": candidate_region,
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
                "candidate_path_semantics": "independent diagnostic; exact source endpoints; moving labels are selectively rasterized from all already-rendered 100 Hz views into interpolated target poses; fixed-camera static labels reuse observed pixels; per-pixel depth chooses visibility; unknown mask at each existing logged state triggers a charged full direct-render fallback",
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
        gs.destroy()

    with (output / "all_query_inference.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(inference_records[0].keys()) if inference_records else []
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inference_records)
    summary = {
        "experiment_id": "r1_5_multiref_surface_cached_replay_20261021c",
        "code_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "frozen_core_base": "e79e656ac00f7847b98db9e12bf43b093f3afff1",
        "frozen_core_modified": False,
        "candidate_config_sha256": _sha256(ROOT / "benchmark/robot_closed_loop/configs/r1_5_multiref_surface_candidate_20261021c.json"),
        "inputs_sha256": inputs,
        "checkpoint_sha256": checkpoint_hashes,
        "scene_and_rig": {"camera_resolution_wh": [width, height], "vertical_fov_deg": fov,
                           "camera_pose_source": "Genesis official robot scene; same fixed side camera as cached R1.5",
                           "source_endpoint_rate_hz": 1.0 / args.source_interval,
                           "source_views_per_query": int(args.max_source_views)},
        "rendered_exact_trace_window_s": [args.start, args.end],
        "query_window_s": [args.query_start, args.end],
        "response": {"model": "physical_log", "threshold_on_off": 0.2,
                     "noise_bandwidth_refractory": "disabled", "radiometry": "Genesis native linear HDR luminance"},
        "episodes": episode_records,
        "inference_rows": "all_query_inference.csv",
        "old_checkpoint_interpretation": "diagnostic only; checkpoints were trained on old generated events and do not determine candidate radiometric correctness",
        "common_reference_policy": "no ranking; compare candidate reconstruction with direct renders at the same exact logged robot states; models receive event tensors only; GT is scored after inference",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_hash_manifest(output)
    print(json.dumps({"output": str(output), "episodes": list(episode_records),
                      "query_rows": len(inference_records), "hash_manifest": str(output / "SHA256SUMS")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
