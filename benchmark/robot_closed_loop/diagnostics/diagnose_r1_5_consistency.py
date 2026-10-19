#!/usr/bin/env python3
"""Compare diagnostic R1.5 image/event candidates on two cached episodes.

This is an offline replay of exact saved robot states. It does not advance
physics and it does not modify the frozen simulator core. The script separates
the endpoint projection candidate from temporal log-radiance handling, then
scores both against exact-state Genesis HDR renders. Cached direct renders are
used only to resolve explicitly requested uncertain pixels and as a reference;
they are never supplied to the learned model except as the direct-render event
source in a diagnostic comparison.
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
from typing import Any

import numpy as np
import h5py
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.diagnose_r1_native_hdr_replay import (  # noqa: E402
    infer_event_state,
    interpolate_trace_label,
    load_event_models,
    project_world,
    read_trace,
    render_state,
    truth_rows_at_times,
    verify_source_hashes,
)
from benchmark.robot_closed_loop.diagnostics.diagnose_r1_ours_localization_cadence import (  # noqa: E402
    generate_sparse_plugin,
    response_config,
)
from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import (  # noqa: E402
    _resolve_motion_bodies,
    _sha256,
    _write_hash_manifest,
)
from benchmark.robot_closed_loop.diagnostics.r1_5_consistency_candidates import (  # noqa: E402
    project_endpoint_candidate,
    reconstruct_pair_candidate,
)
from benchmark.robot_closed_loop.robot_env import build_robot_scene  # noqa: E402
from benchmark.robot_closed_loop.event_state_model import five_bin_tensor  # noqa: E402
from genesis_event_plugin.dvs_emulator import DvsEmulator  # noqa: E402
from genesis_event_plugin.radiometry import linear_luminance  # noqa: E402


def _events_from_frames(frames: list[np.ndarray], times: list[float]) -> tuple[np.ndarray, float]:
    if len(frames) != len(times) or len(frames) < 2:
        raise ValueError("event sequence requires at least two timestamped frames")
    height, width = frames[0].shape
    emulator = DvsEmulator(res=(height, width), config=response_config(), device="cpu")
    started = time.perf_counter()
    emulator.initialize(np.asarray(frames[0], dtype=np.float32), t=float(times[0]))
    chunks = []
    for frame, timestamp in zip(frames[1:], times[1:]):
        event = emulator.generate_events(np.asarray(frame, dtype=np.float32), float(timestamp))
        if len(event):
            chunks.append(np.asarray(event, dtype=np.float64))
    wall_s = time.perf_counter() - started
    events = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)
    if len(events):
        events = events[np.argsort(events[:, 0], kind="stable")]
    return events, wall_s


def _pair_for_time(time_s: float, source_times: np.ndarray) -> tuple[int, float]:
    exact = np.flatnonzero(np.isclose(source_times, time_s, atol=1e-9, rtol=0.0))
    if exact.size:
        idx = int(exact[0])
        if idx == 0:
            return 0, 0.0
        return idx - 1, 1.0
    right = int(np.searchsorted(source_times, time_s, side="right"))
    left = right - 1
    if left < 0 or right >= len(source_times):
        raise ValueError(f"time {time_s:.9f} is outside sparse endpoint support")
    alpha = (float(time_s) - float(source_times[left])) / (float(source_times[right]) - float(source_times[left]))
    return left, float(alpha)


def _image_errors(reference: np.ndarray, candidate: np.ndarray, target_mask: np.ndarray,
                  robot_mask: np.ndarray, floor: float) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    good = np.isfinite(ref) & np.isfinite(cand) & (ref >= 0.0) & (cand >= 0.0)
    diff = np.full_like(ref, np.nan, dtype=np.float32)
    diff[good] = np.abs(np.log(np.maximum(ref[good], floor)) - np.log(np.maximum(cand[good], floor)))

    def stats(mask: np.ndarray) -> dict[str, Any]:
        active = np.asarray(mask, dtype=bool) & good
        values = diff[active]
        if not values.size:
            return {"pixels": 0, "log_abs_mean": None, "log_abs_p95": None,
                    "log_abs_max": None, "pixels_over_0_2": 0}
        return {"pixels": int(values.size), "log_abs_mean": float(values.mean()),
                "log_abs_p95": float(np.percentile(values, 95)), "log_abs_max": float(values.max()),
                "pixels_over_0_2": int(np.count_nonzero(values > 0.2))}

    background = ~(np.asarray(target_mask, dtype=bool) | np.asarray(robot_mask, dtype=bool))
    return {"all": stats(np.ones(ref.shape, dtype=bool)), "target": stats(target_mask),
            "robot": stats(robot_mask), "background": stats(background),
            "invalid_pixels": int(np.count_nonzero(~good))}


def _save_montage(path: Path, direct: np.ndarray, named: dict[str, np.ndarray],
                  target_mask: np.ndarray, floor: float) -> None:
    values = [np.log(np.maximum(np.asarray(direct, dtype=np.float32), floor))]
    for image in named.values():
        values.append(np.log(np.maximum(np.asarray(image, dtype=np.float32), floor)))
    low = float(np.percentile(values[0], 1))
    high = float(np.percentile(values[0], 99.5))
    thumb_w, thumb_h = 320, 240
    labels = ["direct render", *named.keys()]
    images = [direct, *named.values()]
    canvas = Image.new("RGB", (thumb_w * len(images), thumb_h + 28), (20, 22, 26))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images)):
        log_image = np.log(np.maximum(np.asarray(image, dtype=np.float32), floor))
        gray = np.clip((log_image - low) / max(high - low, 1e-9), 0, 1)
        rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
        rgb[np.asarray(target_mask, dtype=bool)] = (255, 170, 0)
        thumb = Image.fromarray(rgb).resize((thumb_w, thumb_h), Image.Resampling.NEAREST)
        x = index * thumb_w
        canvas.paste(thumb, (x, 28))
        draw.text((x + 4, 6), label[:42], fill=(245, 245, 245))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True,
                        help="verified native HDR/geometry cache root from the prior center/+55 replay")
    parser.add_argument("--ours-checkpoint", type=Path, required=True)
    parser.add_argument("--v2e-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    scene_cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    scene_cfg["shared_control"] = json.loads(args.control_config.read_text(encoding="utf-8"))
    reference = args.reference_root.resolve()
    source_cache_root = args.source_cache_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output path already exists: {output}")

    trace_start, trace_end = (float(x) for x in config["trace_time_s"])
    query_start, query_end = (float(x) for x in config["query_time_s"])
    query_step = float(config["query_step_s"])
    endpoint_rate = float(config["sparse_endpoint_rate_hz"])
    endpoint_dt = 1.0 / endpoint_rate
    endpoint_times = np.arange(trace_start, trace_end + endpoint_dt * 0.25, endpoint_dt, dtype=np.float64)
    endpoint_times = np.asarray([round(float(value), 9) for value in endpoint_times], dtype=np.float64)
    if abs(float(endpoint_times[-1]) - trace_end) > 1e-8:
        raise ValueError("trace end must lie on the declared sparse endpoint grid")
    query_times = [round(float(value), 9) for value in np.arange(query_start, query_end + query_step / 2, query_step)]
    if not query_times or abs(query_times[-1] - query_end) > 1e-8:
        raise ValueError("query interval must divide evenly by query_step_s")

    episode_cfg = {int(ep_id): label for label, ep_id in config["episodes"].items()}
    preloaded: dict[int, dict[str, Any]] = {}
    required_names = ("direct_hdr_events_1khz_2khz.npz", "state_control_trace.jsonl", "episode.csv")
    for ep_id, label in sorted(episode_cfg.items()):
        ep_dir = reference / "episodes" / f"episode_{ep_id:03d}"
        paths = [ep_dir / name for name in required_names]
        verified = verify_source_hashes(reference, paths)
        source_cache_path = source_cache_root / f"episode_{ep_id:03d}" / "same_state_source_cache.h5"
        source_cache_hash = verify_source_hashes(source_cache_root, [source_cache_path])[
            f"episode_{ep_id:03d}/same_state_source_cache.h5"]
        trace = read_trace(paths[1])
        row_times = np.asarray([float(row["time_s"]) for row in trace], dtype=np.float64)
        rows = [row for row in trace if trace_start - 1e-9 <= float(row["time_s"]) <= trace_end + 1e-9]
        actual_times = np.asarray([round(float(row["time_s"]), 9) for row in rows], dtype=np.float64)
        if len(rows) < 3 or abs(actual_times[0] - trace_start) > 1e-8 or abs(actual_times[-1] - trace_end) > 1e-8:
            raise RuntimeError(f"episode_{ep_id:03d} lacks exact trace endpoint rows")
        sparse_rows = truth_rows_at_times(trace, endpoint_times.tolist())
        with np.load(paths[0], allow_pickle=False) as archive:
            if "events_2000hz" not in archive:
                raise RuntimeError(f"episode_{ep_id:03d} lacks archived 2 kHz events")
            archived_events = np.asarray(archive["events_2000hz"], dtype=np.float64)
        with paths[2].open("r", newline="", encoding="utf-8") as stream:
            episode = next(csv.DictReader(stream))
        # Every query's 25 ms history must be covered by the saved trace and by
        # archived event time. The longer trace start supplies that margin.
        if query_start - 0.025 < trace_start - 1e-9:
            raise RuntimeError("cached trace does not cover the earliest causal query window")
        if row_times[0] > query_start - 0.025 + 1e-8:
            raise RuntimeError("full trace starts too late for the requested causal input")
        preloaded[ep_id] = {
            "label": label, "episode_dir": ep_dir, "paths": paths, "verified": verified,
            "source_cache_path": source_cache_path, "source_cache_sha256": source_cache_hash,
            "trace": trace, "rows": rows, "times": actual_times, "sparse_rows": sparse_rows,
            "archived_events": archived_events, "episode": episode,
        }

    output.mkdir(parents=True, exist_ok=False)
    width, height = (int(v) for v in scene_cfg["camera"]["resolution_wh"])
    fov = float(scene_cfg["camera"]["vertical_fov_deg"])
    focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    K = np.asarray([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    intrinsics = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)

    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning")
    env = None
    episode_results: dict[str, Any] = {}
    query_rows: list[dict[str, Any]] = []
    try:
        env = build_robot_scene(scene_cfg, dt_s=float(scene_cfg["physics"]["dt_s"]), segmentation_level="link")
        env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True, force_render=True)
        bodies = _resolve_motion_bodies(env)
        from genesis_event_plugin.genesis_adapter import camera_to_world_cv
        camera_pose = np.asarray(camera_to_world_cv(env.camera), dtype=np.float64)
        target_seg_ids = [seg_id for seg_id, body in bodies.items() if body is env.target]
        robot_seg_ids = [seg_id for seg_id, body in bodies.items() if body is not env.target]
        fixed_height = float(scene_cfg["target"]["size_m"][2]) / 2.0
        models, checkpoint_hashes = load_event_models(
            {"ours_native_hdr": args.ours_checkpoint.resolve(), "v2e": args.v2e_checkpoint.resolve()},
            camera_pose, intrinsics, (width, height))

        for ep_id, pre in sorted(preloaded.items()):
            print(f"[render] episode_{ep_id:03d}: {len(pre['rows'])} exact cached states", flush=True)
            episode_out = output / f"episode_{ep_id:03d}"
            episode_out.mkdir()
            with h5py.File(pre["source_cache_path"], "r") as cache:
                cached_times = np.asarray(cache["timestamps_s"][:], dtype=np.float64)
                cached_ids = np.asarray(cache["segmentation_pose_ids"][:], dtype=np.int32)
                if not np.array_equal(cached_ids, np.asarray(sorted(bodies), dtype=np.int32)):
                    raise RuntimeError(f"episode_{ep_id:03d} source-cache segmentation pose IDs do not match the scene")
                if tuple(int(v) for v in cache.attrs["camera_resolution_wh"]) != (width, height):
                    raise RuntimeError(f"episode_{ep_id:03d} source-cache resolution does not match the scene")
                if not np.isclose(float(cache.attrs["vertical_fov_deg"]), fov, atol=1e-8, rtol=0):
                    raise RuntimeError(f"episode_{ep_id:03d} source-cache FOV does not match the scene")
                cached_camera = np.asarray(cache["camera_to_world_cv"][:], dtype=np.float64)
                if not np.allclose(cached_camera, camera_pose[None], atol=1e-5, rtol=0):
                    raise RuntimeError(f"episode_{ep_id:03d} source-cache camera pose does not match the scene")
                cache_frames = {}
                for cache_idx, cache_time in enumerate(cached_times):
                    time_key = round(float(cache_time), 9)
                    cache_frames[time_key] = {
                        "time_s": time_key,
                        "hdr_rgb": np.asarray(cache["radiance"][cache_idx], dtype=np.float32),
                        "radiance_luma": np.asarray(cache["radiance_luma"][cache_idx], dtype=np.float32),
                        "display_rgb": np.asarray(cache["display_rgb"][cache_idx], dtype=np.uint8),
                        "depth": np.asarray(cache["depth"][cache_idx], dtype=np.float32),
                        "segmentation": np.asarray(cache["segmentation"][cache_idx], dtype=np.int32),
                        "poses": {int(seg_id): np.asarray(cache["object_poses"][cache_idx, pose_idx], dtype=np.float64)
                                  for pose_idx, seg_id in enumerate(cached_ids)},
                    }
            missing_endpoints = [round(float(t), 9) for t in endpoint_times if round(float(t), 9) not in cache_frames]
            if missing_endpoints:
                raise RuntimeError(f"episode_{ep_id:03d} native source cache lacks endpoints: {missing_endpoints}")
            direct_frames = []
            render_started = time.perf_counter()
            rendered_count = 0
            endpoint_cache_comparison = []
            for row in pre["rows"]:
                key = round(float(row["time_s"]), 9)
                if key in cache_frames:
                    cached_frame = cache_frames[key]
                    direct = render_state(env, row, bodies)
                    rendered_count += 1
                    rgb_error = np.abs(np.asarray(direct["hdr_rgb"]) - cached_frame["hdr_rgb"])
                    endpoint_cache_comparison.append({
                        "time_s": key, "renderer_cache_max_abs_radiance_error": float(rgb_error.max()),
                        "renderer_cache_mean_abs_radiance_error": float(rgb_error.mean()),
                    })
                    direct_frames.append(cached_frame)
                else:
                    direct_frames.append(render_state(env, row, bodies))
                    rendered_count += 1
            render_wall = time.perf_counter() - render_started
            direct_by_time = {round(float(frame["time_s"]), 9): frame for frame in direct_frames}
            endpoint_frames = [direct_by_time[round(float(row["time_s"]), 9)] for row in pre["sparse_rows"]]
            endpoint_time_list = [float(frame["time_s"]) for frame in endpoint_frames]
            endpoint_time_array = np.asarray(endpoint_time_list, dtype=np.float64)

            original_events, original_intervals, original_plugin_s = generate_sparse_plugin(
                endpoint_frames, np.asarray(sorted(bodies), dtype=np.int32), camera_pose,
                width, height, fov)
            # The candidate image/event paths below share exact logged-state
            # timestamps. No missing physics state is synthesized.
            candidate_frames: dict[str, list[np.ndarray]] = {
                "frozen_reconstruction_exact_rows": [],
                "endpoint_projection_identity_exact_rows": [],
                "endpoint_exact_only_exact_rows": [],
                "log_radiance_with_cached_render_fallback": [],
            }
            frame_records: list[dict[str, Any]] = []
            total_render_requested_frames = 0
            requested_pixels_total = 0
            reconstructed_wall = {name: 0.0 for name in candidate_frames}
            exact_endpoint_rows: list[dict[str, Any]] = []
            endpoint_uv_rows: list[dict[str, Any]] = []

            for row_idx, frame in enumerate(direct_frames):
                time_s = round(float(frame["time_s"]), 9)
                direct_luma = np.asarray(frame["radiance_luma"], dtype=np.float32)
                if row_idx == 0:
                    for name in candidate_frames:
                        candidate_frames[name].append(direct_luma.copy())
                    continue
                pair_idx, alpha = _pair_for_time(time_s, endpoint_time_array)
                source0, source1 = endpoint_frames[pair_idx], endpoint_frames[pair_idx + 1]
                objects0, objects1 = source0["poses"], source1["poses"]
                target_seg = np.asarray(frame["segmentation"])
                target_mask = np.isin(target_seg, target_seg_ids)
                robot_mask = np.isin(target_seg, robot_seg_ids)
                pair_result: dict[str, Any] = {"time_s": time_s, "alpha": float(alpha),
                                                "pair_start_s": float(source0["time_s"]),
                                                "pair_end_s": float(source1["time_s"])}

                # Raw and corrected source-1 endpoint projections quantify the
                # tiny float error independently of photometric handling.
                if abs(alpha - 1.0) <= 1e-9:
                    raw_projection = project_endpoint_candidate(
                        source1["depth"], source1["segmentation"], K, camera_pose, camera_pose, 1.0, 1,
                        objects0, objects1, exact_endpoints=False)
                    fixed_projection = project_endpoint_candidate(
                        source1["depth"], source1["segmentation"], K, camera_pose, camera_pose, 1.0, 1,
                        objects0, objects1, exact_endpoints=True)
                    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
                    grid = np.stack((xx, yy), axis=0).astype(np.float32)
                    raw_err = np.abs(raw_projection[0] - grid)
                    fixed_err = np.abs(fixed_projection[0] - grid)
                    raw_err = raw_err[:, raw_projection[2]]
                    fixed_err = fixed_err[:, fixed_projection[2]]
                    endpoint_uv_rows.append({
                        "time_s": time_s, "raw_max_uv_error_px": float(raw_err.max()) if raw_err.size else None,
                        "raw_p99_uv_error_px": float(np.percentile(raw_err, 99)) if raw_err.size else None,
                        "exact_max_uv_error_px": float(fixed_err.max()) if fixed_err.size else None,
                    })

                for name, mode in (
                    ("frozen_reconstruction_exact_rows", "frozen"),
                    ("endpoint_projection_identity_exact_rows", "endpoint_projection_identity"),
                    ("endpoint_exact_only_exact_rows", "endpoint_exact"),
                    ("log_radiance_with_cached_render_fallback", "log_radiance_with_render_request"),
                ):
                    started = time.perf_counter()
                    result = reconstruct_pair_candidate(
                        source0["radiance_luma"], source0["depth"], source0["segmentation"], camera_pose,
                        source1["radiance_luma"], source1["depth"], source1["segmentation"], camera_pose,
                        K, objects0, objects1, float(alpha), mode=mode,
                    )
                    reconstructed_wall[name] += time.perf_counter() - started
                    reconstructed = np.asarray(result["frame"], dtype=np.float32).copy()
                    valid = np.asarray(result["valid"], dtype=bool)
                    request = np.asarray(result["request_render"], dtype=bool)
                    render_pixels = int(request.sum())
                    requested_pixels_total += render_pixels if name.startswith("log_radiance") else 0
                    if name.startswith("log_radiance") and render_pixels:
                        total_render_requested_frames += 1
                        # This diagnostic consumes the exact cached direct
                        # render at this state for requested pixels. This is a
                        # measured render fallback, not a free fill operation.
                        reconstructed[request] = direct_luma[request]
                    else:
                        invalid = ~valid | ~np.isfinite(reconstructed)
                        if invalid.any():
                            fallback = source0["radiance_luma"] if alpha < 0.5 else source1["radiance_luma"]
                            reconstructed[invalid] = np.asarray(fallback, dtype=np.float32)[invalid]
                    candidate_frames[name].append(reconstructed)
                    projected_visible_error = None
                    if abs(alpha - 1.0) <= 1e-9 and name == "endpoint_projection_identity_exact_rows":
                        visibility1 = np.asarray(result["diagnostics"].get("visibility1"), dtype=bool)
                        projected_visible_error = _image_errors(
                            direct_luma, reconstructed, visibility1, np.zeros_like(visibility1), 1e-6)["target"]
                    pair_result[name] = {
                        "image_error": _image_errors(direct_luma, reconstructed, target_mask, robot_mask, 1e-6),
                        "source1_projected_visible_error_at_endpoint": projected_visible_error,
                        "valid_fraction": float(valid.mean()), "request_render_fraction": float(request.mean()),
                        "request_render_pixels": render_pixels,
                        "diagnostics": {key: value for key, value in result["diagnostics"].items()
                                        if isinstance(value, (str, bool, int, float, type(None)))},
                    }
                    if name == "endpoint_exact_only_exact_rows" and abs(alpha - 1.0) <= 1e-9:
                        endpoint_error = np.abs(np.log(np.maximum(direct_luma, 1e-6)) -
                                                np.log(np.maximum(reconstructed, 1e-6)))
                        exact_endpoint_rows.append({
                            "time_s": time_s,
                            "pixels_over_log_error_0_2": int(np.count_nonzero(endpoint_error > 0.2)),
                            "max_log_error": float(endpoint_error.max()),
                            "source_pixels": int(direct_luma.size),
                        })
                frame_records.append(pair_result)

            times = [float(frame["time_s"]) for frame in direct_frames]
            direct_lumas = [np.asarray(frame["radiance_luma"], dtype=np.float32) for frame in direct_frames]
            direct_events, direct_event_s = _events_from_frames(direct_lumas, times)
            source_events: dict[str, np.ndarray] = {
                "original_sparse_100hz_plugin": original_events,
                "frozen_reconstruction_exact_rows": None,
                "endpoint_exact_only_exact_rows": None,
                "log_radiance_with_cached_render_fallback": None,
                "exact_state_direct_render": direct_events,
            }
            event_wall: dict[str, float] = {"original_sparse_100hz_plugin": original_plugin_s,
                                           "exact_state_direct_render": direct_event_s}
            for name, images in candidate_frames.items():
                source_events[name], event_wall[name] = _events_from_frames(images, times)
                np.savez_compressed(episode_out / f"{name}_events.npz", events=source_events[name])
            np.savez_compressed(episode_out / "original_sparse_100hz_plugin_events.npz", events=original_events)

            archived_all = pre["archived_events"]
            archived = archived_all[(archived_all[:, 0] >= trace_start) & (archived_all[:, 0] <= trace_end)]
            source_events["archived_direct_hdr_2khz_reference"] = archived
            np.savez_compressed(episode_out / "archived_direct_hdr_2khz_reference_events.npz", events=archived)

            episode_query_rows: list[dict[str, Any]] = []
            for query in query_times:
                label = interpolate_trace_label(pre["trace"], query)
                gt_pos = np.asarray(label["position_m"], dtype=np.float64)
                gt_vel = np.asarray(label["velocity_m_s"], dtype=np.float64)
                gt_uv = project_world(gt_pos, camera_pose, intrinsics)
                for source_name, stream in source_events.items():
                    voxel = five_bin_tensor(stream, query, width, height)
                    for model_name, model in models.items():
                        estimate, _ = infer_event_state(stream, model, query, width, height,
                                                        camera_pose, intrinsics, fixed_height)
                        row = {
                            "episode_id": ep_id, "episode": pre["label"], "query_s": query,
                            "model": model_name, "source": source_name,
                            "event_count_25ms": estimate["event_count_25ms"],
                            "nonempty_5ms_bins": estimate["nonempty_5ms_bins"],
                            "state_available": estimate["state_available"],
                            "position_error_xy_mm": None, "velocity_error_xy_m_s": None,
                            "heatmap_uv_error_px": None, "input_tensor_sha256": estimate["input_tensor_sha256"],
                        }
                        if estimate.get("state_available"):
                            pred_pos = np.asarray(estimate["position_xyz_m"], dtype=np.float64)
                            pred_vel = np.asarray(estimate["velocity_xy_m_s"], dtype=np.float64)
                            pred_uv = np.asarray(estimate["heatmap_uv_px"], dtype=np.float64)
                            row.update({
                                "position_error_xy_mm": float(np.linalg.norm(pred_pos[:2] - gt_pos[:2]) * 1000),
                                "velocity_error_xy_m_s": float(np.linalg.norm(pred_vel - gt_vel[:2])),
                                "heatmap_uv_error_px": float(np.linalg.norm(pred_uv - gt_uv)),
                            })
                        episode_query_rows.append(row)
                        query_rows.append(row)

            for source_name, stream in source_events.items():
                voxel = five_bin_tensor(stream, float(config["query_diagnostic_cutoff_s"]), width, height)
                np.savez_compressed(episode_out / f"{source_name}_diagnostic_voxel.npz", voxel=voxel)
            image_event_summary = {}
            cutoff = float(config["query_diagnostic_cutoff_s"])
            direct_voxel = five_bin_tensor(direct_events, cutoff, width, height).ravel()
            for name, stream in source_events.items():
                voxel = five_bin_tensor(stream, cutoff, width, height).ravel()
                denominator = float(np.linalg.norm(voxel) * np.linalg.norm(direct_voxel))
                image_event_summary[name] = {
                    "event_count_trace_interval": int(len(stream)),
                    "event_count_query_25ms": int(np.count_nonzero((stream[:, 0] >= cutoff - 0.025) &
                                                                     (stream[:, 0] < cutoff))),
                    "five_bin_cosine_vs_exact_direct": None if denominator == 0 else float(np.dot(voxel, direct_voxel) / denominator),
                    "event_emulation_wall_s": event_wall.get(name),
                }

            render_fallback_mean_s = render_wall / max(len(direct_frames), 1)
            summary = {
                "episode": pre["label"], "seed": int(pre["episode"]["seed"]),
                "speed_m_s": float(pre["episode"]["launch_speed_m_s"]),
                "start_y_m": float(pre["episode"]["start_y_m"]),
                "trace_times_s": times,
                "trace_interval_s": {"min": float(np.diff(times).min()), "median": float(np.median(np.diff(times))),
                                     "max": float(np.diff(times).max()),
                                     "effective_mean_rate_hz": float(1.0 / np.diff(times).mean())},
                "trace_sha256": pre["verified"][f"episodes/episode_{ep_id:03d}/state_control_trace.jsonl"],
                "reference_event_sha256": pre["verified"][f"episodes/episode_{ep_id:03d}/direct_hdr_events_1khz_2khz.npz"],
                "source_cache_sha256": pre["source_cache_sha256"],
                "direct_render_count": len(direct_frames), "renderer_calls_for_exact_trace_states": rendered_count,
                "direct_render_wall_s": render_wall,
                "source_cache_endpoint_replay_comparison": endpoint_cache_comparison,
                "estimated_full_frame_fallback_calls": total_render_requested_frames,
                "estimated_full_frame_fallback_wall_s": total_render_requested_frames * render_fallback_mean_s,
                "estimated_fallback_pixel_fraction_sum": float(requested_pixels_total / max(len(direct_frames) * height * width, 1)),
                "candidate_reconstruction_wall_s": reconstructed_wall,
                "original_sparse_plugin_wall_s": original_plugin_s,
                "original_sparse_plugin_interval_diagnostics": original_intervals,
                "event_sources": image_event_summary,
                "endpoint_projection_raw_vs_exact": endpoint_uv_rows,
                "endpoint_exact_composite": exact_endpoint_rows,
                "frame_reconstruction_diagnostics": frame_records,
                "queries": episode_query_rows,
                "boundaries": {
                    "exact_render": "saved physics trace rows replayed with set_qpos; renderer output is an offline reference, not a new physical rollout or event-camera ground truth",
                    "candidate_geometry": "same frozen projected warp and z-buffer; exact identity only at mathematical endpoints; intermediate motion remains unrounded",
                    "photometry": "candidate log-radiance interpolation only where both projected endpoints agree in label and depth; cached exact direct render fills explicit uncertain-pixel requests",
                    "learned_inputs": "causal five-bin event tensors only; GT labels are joined after inference for scoring",
                    "frozen_core_modified": False,
                },
            }
            episode_results[f"episode_{ep_id:03d}"] = summary
            (episode_out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
            # At the requested diagnostic cutoff, save a compact visual check.
            viz_idx = int(np.argmin(np.abs(np.asarray(times) - cutoff)))
            viz_time = times[viz_idx]
            direct = direct_lumas[viz_idx]
            target_mask = np.isin(direct_frames[viz_idx]["segmentation"], target_seg_ids)
            named = {name: imgs[viz_idx] for name, imgs in candidate_frames.items()}
            _save_montage(episode_out / "image_candidates_at_cutoff.png", direct, named, target_mask, 1e-6)
            print(f"[episode] {ep_id}: direct={len(direct_frames)} rows, events={image_event_summary}", flush=True)
    finally:
        gs.destroy()

    with (output / "query_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(query_rows[0].keys()))
        writer.writeheader()
        writer.writerows(query_rows)
    result = {
        "status": "PASS", "experiment_id": config["experiment_id"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": _sha256(args.config.resolve()),
        "scene_config_sha256": _sha256(args.scene_config.resolve()),
        "control_config_sha256": _sha256(args.control_config.resolve()),
        "reference_manifest_sha256": _sha256(reference / "SHA256SUMS"),
        "source_cache_manifest_sha256": _sha256(source_cache_root / "SHA256SUMS"),
        "checkpoint_sha256": checkpoint_hashes,
        "query_times_s": query_times,
        "episodes": episode_results,
        "scope": "bounded offline image/event/localization diagnostic on two previously inspected development episodes; no ranking, retraining, or new physics rollout",
        "frozen_core_modified": False,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    _write_hash_manifest(output)
    print(json.dumps({"output": str(output), "episodes": list(episode_results),
                      "query_rows": len(query_rows)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
