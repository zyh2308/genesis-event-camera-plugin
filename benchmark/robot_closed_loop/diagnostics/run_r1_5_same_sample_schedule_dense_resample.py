#!/usr/bin/env python3
"""Counterfactual: keep plugin timestamps, replace its intermediate frames.

Archived plugin intermediates are reproduced from the exact cached 100 Hz HDR
endpoints. A second DVS state receives log-luminance interpolation between
the already-rendered 0.5--1 ms Direct-HDR states, evaluated at those same
plugin timestamps. This holds the DVS implementation/history and timestamp
schedule fixed while changing only the sampled radiance images. The resampled
images are an offline approximation, never described as new direct renders.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from benchmark.robot_closed_loop.diagnostics.diagnose_r1_native_hdr_replay import (
    ReplayCamera, ReplayScene, infer_event_state, load_event_models,
)
from benchmark.robot_closed_loop.diagnostics.diagnose_r1_ours_localization_cadence import response_config
from genesis_event_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.dvs_emulator import DvsEmulator


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def interpolate_log_luma(direct_luma: np.ndarray, direct_times: np.ndarray,
                         query_times: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    floor = 1e-6
    out = np.empty((len(query_times), *direct_luma.shape[1:]), dtype=np.float32)
    bracket_dt = []
    nearest_offsets = []
    alpha_max = 0.0
    for i, t in enumerate(query_times):
        hi = int(np.searchsorted(direct_times, t, side="left"))
        candidates = [max(0, min(len(direct_times) - 1, hi - 1)),
                      max(0, min(len(direct_times) - 1, hi))]
        nearest_offsets.append(min(abs(float(t - direct_times[j])) for j in candidates))
        if hi < len(direct_times) and abs(float(direct_times[hi] - t)) <= 1e-10:
            out[i] = direct_luma[hi]
            continue
        lo = hi - 1
        if lo < 0 or hi >= len(direct_times):
            raise ValueError(f"plugin sample {t:.9f} outside Direct-HDR trace coverage")
        alpha = float((t - direct_times[lo]) / (direct_times[hi] - direct_times[lo]))
        a = np.log(np.maximum(direct_luma[lo], floor))
        b = np.log(np.maximum(direct_luma[hi], floor))
        out[i] = np.exp((1.0 - alpha) * a + alpha * b).astype(np.float32)
        bracket_dt.append(float(direct_times[hi] - direct_times[lo]))
        alpha_max = max(alpha_max, min(alpha, 1.0 - alpha))
    return out, {
        "nonexact_sample_count": int(len(bracket_dt)),
        "max_bracketing_direct_render_dt_ms": float(max(bracket_dt, default=0.0) * 1000.0),
        "max_nearest_state_time_offset_ms": float(max(nearest_offsets, default=0.0) * 1000.0),
        "max_log_interpolation_alpha_from_bracket_endpoint": alpha_max,
        "operation": "pixelwise linear interpolation in log(luminance), evaluated at plugin's exact intermediate timestamps",
    }


def run_plugin_and_capture(endpoints: list[dict[str, Any]], pose_ids: np.ndarray,
                           camera_pose: np.ndarray, fov_deg: float,
                           width: int, height: int):
    focal = height / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    intrinsics = np.asarray([[focal, 0, width / 2], [0, focal, height / 2],
                             [0, 0, 1]], dtype=np.float64)
    scene = ReplayScene()
    camera = ReplayCamera((width, height), intrinsics)
    state = {"index": 0}

    def frame_provider(_scene, _camera):
        frame = endpoints[state["index"]]
        return {"rgb": frame["display_rgb"], "radiance": frame["hdr_rgb"],
                "depth": frame["depth"], "seg": frame["segmentation"]}

    def motion_provider(_scene, _camera):
        frame = endpoints[state["index"]]
        return {"camera_to_world": camera_pose,
                "object_to_world": {int(seg): frame["poses"][int(seg)] for seg in pose_ids}}

    plugin = GenesisPhysicsEventPlugin(
        preset="clean", noise_config=response_config(), input_space="linear",
        radiometric_gain=1.0, radiometric_offset=0.0, radiometric_floor=1e-6,
        radiometric_calibration_status="uncalibrated",
        motion_state_provider=motion_provider, frame_provider=frame_provider,
        response_mode="physical_log", disocclusion_policy="hold", device="cpu",
    )
    plugin.attach(scene, camera)
    original_interpolate = plugin.interpolator.interpolate
    samples: list[tuple[float, np.ndarray]] = []

    def record_intermediates(*args, **kwargs):
        frames, timestamps = original_interpolate(*args, **kwargs)
        samples.extend((float(t), np.asarray(frame, dtype=np.float32).copy())
                       for frame, t in zip(frames, timestamps))
        return frames, timestamps

    plugin.interpolator.interpolate = record_intermediates
    plugin.start_episode()
    event_chunks = []
    start = time.perf_counter()
    for idx, frame in enumerate(endpoints):
        state["index"] = idx
        scene.cur_t = float(frame["time_s"])
        event_chunks.append(np.asarray(plugin.capture(), dtype=np.float64).reshape(-1, 4))
    runtime_s = time.perf_counter() - start
    plugin.end_episode()
    plugin.close()
    events = np.concatenate([e for e in event_chunks if len(e)], axis=0) if any(len(e) for e in event_chunks) else np.empty((0, 4), dtype=np.float64)
    if len(events):
        events = events[np.argsort(events[:, 0], kind="stable")]
    samples.sort(key=lambda item: item[0])
    sample_times = np.asarray([t for t, _ in samples], dtype=np.float64)
    sample_frames = np.stack([frame for _, frame in samples]).astype(np.float32)
    return events, sample_times, sample_frames, runtime_s


def summarize_events(events: np.ndarray) -> dict[str, Any]:
    if not len(events):
        return {"count": 0}
    return {
        "count": int(len(events)),
        "on": int(np.count_nonzero(events[:, 3] > 0)),
        "off": int(np.count_nonzero(events[:, 3] < 0)),
        "time_range_s": [float(events[:, 0].min()), float(events[:, 0].max())],
        "unique_pixels": int(np.unique(events[:, 1:3].astype(np.int64), axis=0).shape[0]),
        "unique_timestamps": int(np.unique(events[:, 0]).size),
    }


def event_region_counts(events: np.ndarray, direct_times: np.ndarray,
                        segmentations: np.ndarray, target_id: int,
                        robot_ids: set[int]) -> dict[str, Any]:
    if not len(events):
        return {"target": 0, "robot": 0, "static_or_scene": 0, "unlabeled": 0}
    t, x0, y0 = events[:, 0], events[:, 1], events[:, 2]
    x, y = np.rint(x0).astype(np.int64), np.rint(y0).astype(np.int64)
    h, w = segmentations.shape[1:]
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    idx = np.abs(t[:, None] - direct_times[None, :]).argmin(axis=1)
    ids = segmentations[idx, np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
    return {
        "target": int(np.count_nonzero(valid & (ids == target_id))),
        "robot": int(np.count_nonzero(valid & np.isin(ids, list(robot_ids)))),
        "static_or_scene": int(np.count_nonzero(valid & (ids != 0) & (ids != target_id) & ~np.isin(ids, list(robot_ids)))),
        "unlabeled": int(np.count_nonzero(valid & (ids == 0))),
        "out_of_bounds": int(np.count_nonzero(~valid)),
    }


def summarize_diff_regions(log_diff: np.ndarray, sample_times: np.ndarray,
                           direct_times: np.ndarray, segmentations: np.ndarray,
                           target_id: int, robot_ids: set[int]) -> dict[str, Any]:
    over = log_diff > 0.2
    frame_idx = np.abs(sample_times[:, None] - direct_times[None, :]).argmin(axis=1)
    labels = segmentations[frame_idx]
    regions = {
        "target": labels == target_id,
        "robot": np.isin(labels, list(robot_ids)),
    }
    regions["static_or_scene"] = (labels != 0) & ~regions["target"] & ~regions["robot"]
    regions["unlabeled"] = labels == 0
    class_stats = {}
    for name, mask in regions.items():
        total = int(np.count_nonzero(mask))
        affected = int(np.count_nonzero(mask & over))
        class_stats[name] = {
            "pixel_samples": total,
            "pixel_samples_log_luma_diff_over_0p2": affected,
            "fraction_over_0p2": float(affected / total) if total else 0.0,
            "max_log_luma_diff": float(np.max(np.where(mask, log_diff, 0.0))) if total else 0.0,
        }
    any_bad = np.any(over, axis=(1, 2))
    first_idx = int(np.flatnonzero(any_bad)[0]) if np.any(any_bad) else None
    max_flat = int(np.argmax(log_diff))
    ti, y, x = np.unravel_index(max_flat, log_diff.shape)
    region_id = int(labels[ti, y, x])
    region_name = ("target" if region_id == target_id else "robot" if region_id in robot_ids
                   else "unlabeled" if region_id == 0 else "static_or_scene")
    return {
        "threshold_log_luma_difference": 0.2,
        "affected_pixel_sample_count": int(np.count_nonzero(over)),
        "total_pixel_sample_count": int(over.size),
        "samples_with_any_affected_pixel": int(np.count_nonzero(any_bad)),
        "first_sample_with_affected_pixel_s": None if first_idx is None else float(sample_times[first_idx]),
        "affected_pixel_samples_by_posthoc_region": class_stats,
        "global_max_location": {
            "sample_time_s": float(sample_times[ti]), "u_px": int(x), "v_px": int(y),
            "segmentation_id": region_id, "region": region_name,
            "log_luma_absolute_difference": float(log_diff[ti, y, x]),
            "nearest_direct_state_time_s": float(direct_times[frame_idx[ti]]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint-ours", type=Path, required=True)
    parser.add_argument("--checkpoint-v2e", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    root = args.result_root.resolve()
    cache_root = args.cache_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    query_rows = read_csv(root / "all_four_group_queries.csv")
    scene_cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    fixed_height_m = float(scene_cfg["target"]["size_m"][2]) / 2.0
    direct_h5 = sorted((root / "episodes").glob("episode_*/direct_hdr_exact_state_frames.h5"))
    summary: dict[str, Any] = {
        "status": "completed",
        "method": "same plugin timestamp schedule; replace each plugin intermediate luma image with pixelwise log-luma interpolation between neighboring exact saved Direct-HDR frames",
        "interpretation_limit": "counterfactual uses interpolated saved renders, not new direct renders or physical truth; isolates image-sequence values from plugin timestamp partition approximately, within the available 0.5-1.0 ms Direct-HDR sample spacing",
        "physics_advanced": False,
        "frozen_core_modified": False,
        "runtime_device": "CPU only; same DvsEmulator class/response configuration in separate continuous histories",
        "code_commit": args.code_commit,
        "script_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(args.scene_config),
        "checkpoints": {"ours_sha256": sha256(args.checkpoint_ours), "v2e_sha256": sha256(args.checkpoint_v2e)},
        "episodes": {},
    }
    counterfactual_rows = []
    for frame_h5 in direct_h5:
        episode_dir = frame_h5.parent
        episode = episode_dir.name.split("_", 2)[2]
        ep_id = int(episode_dir.name.split("_")[1])
        source_cache = cache_root / f"episode_{ep_id:03d}" / "same_state_source_cache.h5"
        with h5py.File(frame_h5, "r") as h5:
            direct_times = np.asarray(h5["timestamps_s"][:], dtype=np.float64)
            direct_luma = np.asarray(h5["radiance_luma"][:], dtype=np.float32)
            direct_segmentation = np.asarray(h5["segmentation_post_inference_only"][:], dtype=np.int32)
            pose_ids = set(map(int, np.asarray(h5.attrs["segmentation_pose_ids"]).tolist()))
            camera_pose = np.asarray(h5.attrs["camera_to_world_cv"], dtype=np.float64)
            fov = float(h5.attrs["vertical_fov_deg"])
            width, height = map(int, h5.attrs["resolution_wh"])
        with h5py.File(source_cache, "r") as h5:
            cache_times = np.asarray(h5["timestamps_s"][:], dtype=np.float64)
            pose_ids = np.asarray(h5["segmentation_pose_ids"][:], dtype=np.int32)
            endpoint_idx = [int(np.argmin(np.abs(cache_times - t))) for t in
                            (direct_times[0], direct_times[-1])]
            endpoint_idx = [i for i, t in enumerate(cache_times)
                            if direct_times[0] - 1e-9 <= t <= direct_times[-1] + 1e-9]
            endpoints = []
            for i in endpoint_idx:
                endpoints.append({
                    "time_s": float(cache_times[i]),
                    "hdr_rgb": np.asarray(h5["radiance"][i], dtype=np.float32),
                    "radiance_luma": np.asarray(h5["radiance_luma"][i], dtype=np.float32),
                    "display_rgb": np.asarray(h5["display_rgb"][i], dtype=np.uint8),
                    "depth": np.asarray(h5["depth"][i], dtype=np.float32),
                    "segmentation": np.asarray(h5["segmentation"][i], dtype=np.int32),
                    "poses": {int(seg): np.asarray(h5["object_poses"][i, j], dtype=np.float64)
                              for j, seg in enumerate(pose_ids)},
                })
            camera_cache = np.asarray(h5["camera_to_world_cv"][:], dtype=np.float64)
            width_cache, height_cache = map(int, h5.attrs["camera_resolution_wh"])
            fov_cache = float(h5.attrs["vertical_fov_deg"])
        if not np.allclose(camera_cache, camera_pose, atol=1e-8, rtol=0):
            raise RuntimeError(f"{episode}: camera pose mismatch")
        if (width_cache, height_cache) != (width, height) or not np.isclose(fov_cache, fov):
            raise RuntimeError(f"{episode}: camera calibration mismatch")

        native_replay, sample_times, sample_frames, plugin_s = run_plugin_and_capture(
            endpoints, pose_ids, camera_pose, fov, width, height)
        archived_native = np.asarray(np.load(episode_dir / "raw_events_ours_native_hdr.npz")["events"], dtype=np.float64)
        native_reproduction = {
            "archived_count": int(len(archived_native)),
            "replayed_count": int(len(native_replay)),
            "count_delta": int(len(native_replay) - len(archived_native)),
            "exact_array_match": bool(native_replay.shape == archived_native.shape and np.array_equal(native_replay, archived_native)),
        }
        same_schedule_luma, resample_info = interpolate_log_luma(direct_luma, direct_times, sample_times)
        direct_dvs = DvsEmulator(res=(height, width), config=response_config(), device="cpu")
        direct_dvs.initialize(direct_luma[0], t=float(direct_times[0]))
        direct_chunks = [direct_dvs.generate_events(frame, float(t))
                         for frame, t in zip(same_schedule_luma, sample_times)]
        counterfactual_events = np.concatenate([np.asarray(e, dtype=np.float64).reshape(-1, 4)
                                                for e in direct_chunks if len(e)], axis=0) if any(len(e) for e in direct_chunks) else np.empty((0, 4), dtype=np.float64)
        if len(counterfactual_events):
            counterfactual_events = counterfactual_events[np.argsort(counterfactual_events[:, 0], kind="stable")]
        native_logs = np.log(np.maximum(sample_frames, 1e-6))
        direct_logs = np.log(np.maximum(same_schedule_luma, 1e-6))
        log_diff = np.abs(native_logs - direct_logs)
        target_query_rows = [r for r in query_rows if r["episode"] == episode and r["source"] == "direct_hdr" and r["checkpoint"] == "v2e"]
        target_votes = []
        for row in target_query_rows:
            qi = int(np.argmin(np.abs(direct_times - float(row["query_s"]))))
            u, v = int(round(float(row["gt_u_px"]))), int(round(float(row["gt_v_px"])))
            crop = direct_segmentation[qi, max(0, v - 7):min(height, v + 8), max(0, u - 7):min(width, u + 8)]
            values, counts = np.unique(crop[crop != 0], return_counts=True)
            if len(values):
                target_votes.append(int(values[np.argmax(counts)]))
        target_id = int(np.bincount(np.asarray(target_votes, dtype=np.int64)).argmax())
        robot_ids = set(map(int, pose_ids.tolist())) - {target_id}
        artifact_name = f"{episode}_same_schedule_dense_resampled_events.npz"
        np.savez_compressed(out / artifact_name, events=counterfactual_events)

        episode_queries = [r for r in query_rows if r["episode"] == episode and r["source"] == "direct_hdr" and r["checkpoint"] == "v2e"]
        episode_queries.sort(key=lambda r: float(r["query_s"]))
        with h5py.File(frame_h5, "r") as h5:
            intr_f = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
            intrinsics = np.asarray([intr_f, intr_f, width / 2.0, height / 2.0], dtype=np.float64)
        models, model_hashes = load_event_models(
            {"ours_native_hdr": args.checkpoint_ours, "v2e": args.checkpoint_v2e},
            camera_pose, intrinsics, (width, height))
        cf_group_rows = []
        for checkpoint, model in models.items():
            for row in episode_queries:
                cutoff = float(row["query_s"])
                state, _ = infer_event_state(counterfactual_events, model, cutoff, width, height,
                                             camera_pose, intrinsics,
                                             fixed_height_m)
                pred = state.get("position_xyz_m")
                error = None if pred is None else float(np.hypot(
                    float(pred[0]) * 1000.0 - float(row["gt_x_mm"]),
                    float(pred[1]) * 1000.0 - float(row["gt_y_mm"])))
                out_row = {
                    "episode": episode, "query_s": cutoff, "checkpoint": checkpoint,
                    "state_available": bool(state.get("state_available")),
                    "event_count_25ms": int(state.get("event_count_25ms", 0)),
                    "nonempty_5ms_bins": int(state.get("nonempty_5ms_bins", 0)),
                    "planar_error_xy_mm": error,
                }
                counterfactual_rows.append(out_row)
                cf_group_rows.append(out_row)

        summary["episodes"][episode] = {
            "episode_id": ep_id,
            "endpoint_count": len(endpoints),
            "direct_exact_state_count": len(direct_times),
            "plugin_intermediate_sample_count": len(sample_times),
            "sample_time_range_s": [float(sample_times.min()), float(sample_times.max())],
            "sample_schedule_sha256": hashlib.sha256(np.ascontiguousarray(sample_times).tobytes()).hexdigest(),
            "native_plugin_reproduction": native_reproduction,
            "native_plugin_runtime_s": plugin_s,
            "direct_resampling": resample_info,
            "native_vs_same_schedule_direct_resample_log_luma_error": {
                "median": float(np.median(log_diff)),
                "p95": float(np.percentile(log_diff, 95)),
                "max": float(log_diff.max()),
                "fraction_pixels_over_0p2": float(np.mean(log_diff > 0.2)),
                "posthoc_spatial_attribution": summarize_diff_regions(
                    log_diff, sample_times, direct_times, direct_segmentation, target_id, robot_ids),
            },
            "native_plugin_events": summarize_events(native_replay),
            "same_schedule_direct_resampled_events": summarize_events(counterfactual_events),
            "posthoc_counterfactual_event_region_counts": event_region_counts(
                counterfactual_events, direct_times, direct_segmentation, target_id, robot_ids),
            "checkpoint_metrics_same_schedule_direct_resampled": {
                checkpoint: {
                    "queries": len(rows),
                    "state_available_fraction": float(np.mean([r["state_available"] for r in rows])),
                    "planar_error_xy_mm_mean": float(np.mean([r["planar_error_xy_mm"] for r in rows if r["planar_error_xy_mm"] is not None])) if any(r["planar_error_xy_mm"] is not None for r in rows) else None,
                    "planar_error_xy_mm_p95": float(np.percentile([r["planar_error_xy_mm"] for r in rows if r["planar_error_xy_mm"] is not None], 95)) if any(r["planar_error_xy_mm"] is not None for r in rows) else None,
                    "per_query": rows,
                }
                for checkpoint in ("ours_native_hdr", "v2e")
                for rows in [[r for r in cf_group_rows if r["checkpoint"] == checkpoint]]
            },
            "input_hashes": {
                "direct_frames_h5": sha256(frame_h5),
                "native_source_cache_h5": sha256(source_cache),
                "archived_native_raw_events": sha256(episode_dir / "raw_events_ours_native_hdr.npz"),
                "archived_direct_raw_events": sha256(episode_dir / "raw_events_direct_hdr.npz"),
            },
            "counterfactual_events_file": artifact_name,
        }

    with (out / "counterfactual_queries.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(counterfactual_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(counterfactual_rows)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()}
    (out / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in manifest.items()), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "episodes": {
        ep: {"native_events": d["native_plugin_events"]["count"],
             "same_schedule_resampled_events": d["same_schedule_direct_resampled_events"]["count"],
             "luma_diff_p95": d["native_vs_same_schedule_direct_resample_log_luma_error"]["p95"],
             "ours_err_mean": d["checkpoint_metrics_same_schedule_direct_resampled"]["ours_native_hdr"]["planar_error_xy_mm_mean"],
             "v2e_err_mean": d["checkpoint_metrics_same_schedule_direct_resampled"]["v2e"]["planar_error_xy_mm_mean"]}
        for ep, d in summary["episodes"].items()}, "script_sha256": sha256(Path(__file__))}, indent=2))


if __name__ == "__main__":
    main()
