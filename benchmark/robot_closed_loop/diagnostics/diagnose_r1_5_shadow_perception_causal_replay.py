#!/usr/bin/env python3
"""Matched shadow-on/off causal replay for the old Ours state checkpoint.

The script restores exact saved Panda/target states from development traces,
direct-renders them with the same camera and fixed GL multisampling, and
changes only Genesis' diagnostic shadow flag. No physics is stepped. A single
ideal physical-log DVS response is applied per condition, then the same old
Ours checkpoint consumes only causal five-bin event tensors. GT is joined
after each network forward pass for evaluator-only error metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.diagnose_r1_5_shadow_multisample_cached_states import (  # noqa: E402
    render_condition,
    set_cached_state,
)
from benchmark.robot_closed_loop.diagnostics.diagnose_r1_native_hdr_replay import (  # noqa: E402
    infer_event_state,
    interpolate_trace_label,
    load_event_models,
    project_world,
    read_trace,
)
from benchmark.robot_closed_loop.diagnostics.diagnose_r1_ours_localization_cadence import (  # noqa: E402
    generate_dense_direct,
)
from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import (  # noqa: E402
    _sha256,
    _write_hash_manifest,
)
from benchmark.robot_closed_loop.event_state_model import five_bin_tensor  # noqa: E402
from genesis_event_plugin.radiometry import linear_luminance  # noqa: E402
from benchmark.robot_closed_loop.diagnostics.diagnose_r1_5_shadow_on_off_cached_states import (  # noqa: E402
    log_luma,
)


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def camera_to_world_cv_from_config(config: dict[str, Any]) -> np.ndarray:
    """Recreate the fixed Genesis look-at camera in the benchmark CV convention."""
    camera = config["camera"]
    position = np.asarray(camera["position_m"], dtype=np.float64)
    forward = np.asarray(camera["lookat_m"], dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    up = np.asarray(camera["up_m"], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((right, -camera_up, forward))
    transform[:3, 3] = position
    return transform


def exact_causal_rows(trace: list[dict[str, Any]], query_start: float,
                      query_end: float, window_s: float) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    times = np.asarray([float(row["time_s"]) for row in trace], dtype=np.float64)
    queries = times[(times >= query_start - 1e-9) & (times <= query_end + 1e-9)]
    if not len(queries):
        raise RuntimeError("trace contains no exact logged query times in the requested interval")
    first_render_time = float(queries[0] - window_s)
    start_idx = int(np.argmin(np.abs(times - first_render_time)))
    if abs(float(times[start_idx]) - first_render_time) > 1e-8:
        raise RuntimeError(
            f"no exact logged state at earliest causal input time {first_render_time:.9f}s; "
            f"nearest is {times[start_idx]:.9f}s"
        )
    end_idx = int(np.argmin(np.abs(times - float(queries[-1]))))
    if abs(float(times[end_idx]) - float(queries[-1])) > 1e-9:
        raise RuntimeError("query cutoff is not an exact state row")
    rows = trace[start_idx:end_idx + 1]
    row_times = [float(row["time_s"]) for row in rows]
    for query in queries:
        if not np.any(np.isclose(np.asarray(row_times), query, rtol=0, atol=1e-9)):
            raise RuntimeError(f"query cutoff {query:.9f}s missing from rendered exact rows")
    return rows, row_times, [float(t) for t in queries]


def preview(path: Path, hdr: np.ndarray, events: np.ndarray,
            gt_uv: np.ndarray, pred_uv: np.ndarray | None,
            label: str, query_time: float) -> None:
    h, w = hdr.shape[:2]
    luma = np.log(np.maximum(np.asarray(linear_luminance(hdr), dtype=np.float64), 1e-6))
    lo, hi = np.percentile(luma, [2, 98])
    gray = np.uint8(np.clip((luma - lo) / max(float(hi - lo), 1e-8), 0, 1) * 255)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    canvas = Image.new("RGB", (w * 2, h + 30), (18, 20, 24))
    canvas.paste(Image.fromarray(rgb), (0, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 7), f"{label}: direct HDR at {query_time:.4f}s (display-normalized)", fill="white")
    event_rgb = np.zeros((h, w, 3), dtype=np.uint32)
    if len(events):
        x = events[:, 1].astype(np.int64)
        y = events[:, 2].astype(np.int64)
        p = events[:, 3] > 0
        good = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        np.add.at(event_rgb[..., 1], (y[good & p], x[good & p]), 32)
        np.add.at(event_rgb[..., 0], (y[good & ~p], x[good & ~p]), 32)
    event_rgb = np.clip(event_rgb, 0, 255).astype(np.uint8)
    canvas.paste(Image.fromarray(event_rgb), (w, 30))
    draw.text((w + 5, 7), "raw ideal-DVS events in rendered interval; ON green / OFF red", fill="white")
    for xoff in (0, w):
        draw.ellipse((xoff + gt_uv[0] - 4, 30 + gt_uv[1] - 4,
                      xoff + gt_uv[0] + 4, 30 + gt_uv[1] + 4), outline=(255, 240, 0), width=2)
        if pred_uv is not None:
            draw.rectangle((xoff + pred_uv[0] - 4, 30 + pred_uv[1] - 4,
                            xoff + pred_uv[0] + 4, 30 + pred_uv[1] + 4), outline=(0, 255, 255), width=2)
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    experiment_config = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    episode_values = args.episodes or experiment_config["episode_ids"]
    query_start_s = float(experiment_config["query_interval_s"][0])
    query_end_s = float(experiment_config["query_interval_s"][1])
    window_s = float(experiment_config["causal_window_s"])
    if window_s != 0.025:
        raise ValueError("this diagnostic is fixed to the causal five-bin 25 ms input contract")
    episode_ids = [str(value).zfill(3) for value in episode_values]
    if not episode_ids or any(value not in {"004", "005"} for value in episode_ids):
        raise ValueError("this diagnostic is restricted to cached development episodes 004 and 005")

    scene_config = json.loads(args.scene_config.read_text(encoding="utf-8"))
    width, height = (int(v) for v in scene_config["camera"]["resolution_wh"])
    fov_deg = float(scene_config["camera"]["vertical_fov_deg"])
    focal = height / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    intrinsics = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)
    fixed_height_m = float(scene_config["target"]["size_m"][2]) / 2.0

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    import genesis as gs

    episode_records: dict[str, Any] = {}
    rows_out: list[dict[str, Any]] = []
    for ep in episode_ids:
        cache_dir = args.cache_root / f"episode_{ep}"
        trace_path = cache_dir / "trace_window.jsonl"
        trace = read_trace(trace_path)
        rows, row_times, cutoffs = exact_causal_rows(
            trace, query_start_s, query_end_s, window_s,
        )
        camera_pose = camera_to_world_cv_from_config(scene_config)

        model_records, checkpoint_hashes = load_event_models(
            {"ours_old_checkpoint": args.checkpoint.resolve()}, camera_pose,
            intrinsics, (width, height),
        )
        conditions: dict[str, dict[str, Any]] = {}
        condition_started = time.perf_counter()
        for shadow in (True, False):
            gs.init(backend=gs.cpu, logging_level="warning")
            key = "shadow_on" if shadow else "shadow_off"
            conditions[key] = render_condition(
                gs, scene_config, rows, shadow=shadow, multisample=True,
                capture_repeats=2, warmup_passes=1,
            )
            print(f"[render] episode_{ep} {key}: {len(rows)} exact states; "
                  f"build={conditions[key]['scene_build_wall_s']:.1f}s; "
                  f"capture={conditions[key]['captured_render_wall_s']:.2f}s", flush=True)
        render_wall_s = time.perf_counter() - condition_started

        segmentation_on = conditions["shadow_on"]["segmentation"]
        segmentation_off = conditions["shadow_off"]["segmentation"]
        if not np.array_equal(segmentation_on, segmentation_off):
            raise RuntimeError(f"episode {ep}: shadow toggle changed segmentation")
        depths_on = conditions["shadow_on"]["depth"]
        depths_off = conditions["shadow_off"]["depth"]
        depth_delta = float(np.nanmax(np.abs(depths_on - depths_off)))
        hdr_on = conditions["shadow_on"]["hdr_rgb"]
        hdr_off = conditions["shadow_off"]["hdr_rgb"]
        frames_by_condition = {
            key: [{"time_s": t, "hdr_rgb": frame}
                  for t, frame in zip(row_times, conditions[key]["hdr_rgb"])]
            for key in ("shadow_on", "shadow_off")
        }

        event_streams: dict[str, np.ndarray] = {}
        dvs_times: dict[str, float] = {}
        for key in ("shadow_on", "shadow_off"):
            event_streams[key], dvs_times[key] = generate_dense_direct(frames_by_condition[key])
        inference_started = time.perf_counter()
        episode_rows: list[dict[str, Any]] = []
        query_voxels: dict[str, list[np.ndarray]] = {key: [] for key in event_streams}
        for cutoff in cutoffs:
            gt = interpolate_trace_label(trace, cutoff)
            gt_position = np.asarray(gt["position_m"], dtype=np.float64)
            gt_uv = project_world(gt_position, camera_pose, intrinsics)
            estimates: dict[str, dict[str, Any]] = {}
            for key, events in event_streams.items():
                estimate, voxel = infer_event_state(
                    events, model_records["ours_old_checkpoint"], cutoff,
                    width, height, camera_pose, intrinsics, fixed_height_m,
                )
                query_voxels[key].append(voxel.astype(np.uint32, copy=False))
                if estimate.get("state_available"):
                    pred_xy = np.asarray(estimate["position_xyz_m"], dtype=np.float64)[:2]
                    pred_uv = np.asarray(estimate["heatmap_uv_px"], dtype=np.float64)
                    estimate["position_error_xy_mm"] = float(np.linalg.norm(pred_xy - gt_position[:2]) * 1000.0)
                    estimate["heatmap_uv_error_px"] = float(np.linalg.norm(pred_uv - gt_uv))
                    estimate["position_error_y_mm"] = float((pred_xy[1] - gt_position[1]) * 1000.0)
                estimates[key] = estimate
            on_voxel = query_voxels["shadow_on"][-1].astype(np.float64).ravel()
            off_voxel = query_voxels["shadow_off"][-1].astype(np.float64).ravel()
            denom = float(np.linalg.norm(on_voxel) * np.linalg.norm(off_voxel))
            cosine = None if denom == 0 else float(np.dot(on_voxel, off_voxel) / denom)
            pair = {
                "episode_id": int(ep), "query_cutoff_s": cutoff,
                "label_method": gt["label_method"],
                "gt_position_xy_m": gt_position[:2].tolist(),
                "gt_velocity_xy_m_s": np.asarray(gt["velocity_m_s"], dtype=np.float64)[:2].tolist(),
                "gt_uv_px": gt_uv.tolist(),
                "shadow_event_tensor_cosine": cosine,
                "shadow_on": estimates["shadow_on"],
                "shadow_off": estimates["shadow_off"],
                "input_contract": "events only; five causal 5 ms bins x 2 polarity x 64x64; no GT/depth/segmentation supplied",
            }
            episode_rows.append(pair)
            for key, estimate in estimates.items():
                rows_out.append({
                    "episode_id": int(ep), "query_cutoff_s": cutoff,
                    "condition": key, "event_count_25ms": estimate["event_count_25ms"],
                    "nonempty_5ms_bins": estimate["nonempty_5ms_bins"],
                    "state_available": estimate["state_available"],
                    "position_error_xy_mm": estimate.get("position_error_xy_mm"),
                    "position_error_y_mm": estimate.get("position_error_y_mm"),
                    "heatmap_uv_error_px": estimate.get("heatmap_uv_error_px"),
                    "input_tensor_sha256": estimate["input_tensor_sha256"],
                })
        inference_wall_s = time.perf_counter() - inference_started
        event_files = {}
        for key, events in event_streams.items():
            event_path = args.output / f"episode_{ep}_{key}_raw_events.npz"
            np.savez_compressed(event_path, events=events)
            event_files[key] = file_record(event_path)
            voxel_path = args.output / f"episode_{ep}_{key}_five_bin_inputs.npz"
            np.savez_compressed(voxel_path, cutoffs_s=np.asarray(cutoffs),
                                tensors=np.stack(query_voxels[key]))
            event_files[f"{key}_five_bin_inputs"] = file_record(voxel_path)

        first_cutoff = cutoffs[0]
        first_label = interpolate_trace_label(trace, first_cutoff)
        first_gt_uv = project_world(np.asarray(first_label["position_m"]), camera_pose, intrinsics)
        first_frame_index = int(np.argmin(np.abs(np.asarray(row_times) - first_cutoff)))
        for key in ("shadow_on", "shadow_off"):
            first_estimate = episode_rows[0][key]
            pred_uv = first_estimate.get("heatmap_uv_px") if first_estimate.get("state_available") else None
            first_events = event_streams[key][
                (event_streams[key][:, 0] >= first_cutoff - window_s)
                & (event_streams[key][:, 0] < first_cutoff)
            ]
            preview(args.output / f"episode_{ep}_{key}_causal_preview.png",
                    conditions[key]["hdr_rgb"][first_frame_index], first_events, first_gt_uv,
                    None if pred_uv is None else np.asarray(pred_uv), key, first_cutoff)

        log_delta = np.abs(log_luma(hdr_on) - log_luma(hdr_off))
        summary_path = args.output / f"episode_{ep}_summary.json"
        record = {
            "episode_id": int(ep),
            "query_cutoffs_actual_trace_rows_s": cutoffs,
            "exact_render_state_times_s": row_times,
            "actual_render_intervals_s": np.diff(np.asarray(row_times, dtype=np.float64)).tolist(),
            "state_trace_interval_summary_s": {
                "min": float(np.min(np.diff(row_times))),
                "median": float(np.median(np.diff(row_times))),
                "max": float(np.max(np.diff(row_times))),
            },
            "physics_advanced": False,
            "shadow_is_only_scene_factor": True,
            "fixed_gl_multisampling": True,
            "segmentation_identical": True,
            "depth_abs_max_shadow_on_off": depth_delta,
            "scene_linear_hdr_log_luma_abs_delta": {
                "mean": float(log_delta.mean()),
                "p95": float(np.percentile(log_delta, 95)),
                "fraction_gt_0_02": float(np.mean(log_delta > 0.02)),
                "fraction_gt_0_20": float(np.mean(log_delta > 0.20)),
            },
            "conditions": {
                key: {
                    "scene_build_wall_s": conditions[key]["scene_build_wall_s"],
                    "captured_render_wall_s": conditions[key]["captured_render_wall_s"],
                    "condition_wall_s": conditions[key]["condition_wall_s_including_scene_build_warmup_and_destroy"],
                    "gl_samples_per_hdr_pass": [item["samples"] for item in conditions[key]["hdr_pass_gl_audit"]],
                    "actual_gl_multisample_enabled": [item["actual_gl_multisample_enabled"] for item in conditions[key]["hdr_pass_gl_audit"]],
                    "render_repeat_metrics": conditions[key]["same_state_repeat"],
                    "raw_event_count": int(len(event_streams[key])),
                    "dvs_cpu_wall_s": dvs_times[key],
                    "event_file": event_files[key],
                    "voxel_file": event_files[f"{key}_five_bin_inputs"],
                }
                for key in ("shadow_on", "shadow_off")
            },
            "per_query_model_metrics": episode_rows,
            "render_total_wall_s": render_wall_s,
            "inference_total_wall_s": inference_wall_s,
            "source_hashes": {
                "trace_window_jsonl": file_record(trace_path),
                "scene_config": file_record(args.scene_config),
                "checkpoint": file_record(args.checkpoint),
            },
            "camera_calibration_contract": "CV extrinsics reconstructed from the fixed scene config and verified against the checkpoint; intrinsics derived from the same resolution/FOV and checked by the checkpoint loader",
            "checkpoint_loader_sha256": checkpoint_hashes,
            "limitations": [
                "old Ours checkpoint is a diagnostic probe, not a model retrained for this shadow ablation",
                "cached-state direct-render replay is not a new closed-loop dynamics rollout",
                "diagnostic shadow toggle isolates only the scene shadow setting; it does not prove shadows are the sole failure cause",
            ],
        }
        summary_path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        episode_records[str(ep)] = {"summary_file": file_record(summary_path), **record}
        print(f"[episode] {ep}: queries={len(cutoffs)} events="
              f"{len(event_streams['shadow_on'])}/{len(event_streams['shadow_off'])}; "
              f"valid={sum(bool(x['shadow_on']['state_available']) for x in episode_rows)}/"
              f"{sum(bool(x['shadow_off']['state_available']) for x in episode_rows)}", flush=True)

    import csv
    with (args.output / "per_query_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    summary = {
        "status": "PASS",
        "experiment_id": experiment_config["experiment_id"],
        "repository_checkout_commit": code_commit,
        "diagnostic_source_commit": os.environ.get("DIAGNOSTIC_SCRIPT_SOURCE_COMMIT"),
        "diagnostic_script_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(args.scene_config.resolve()),
        "experiment_config_sha256": _sha256(args.experiment_config.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "device_contract": "Genesis CPU backend; direct HDR rendering on the explicitly selected EGL GPU; ideal DVS and inference on CPU",
        "condition_contract": "shadow on/off only; GL multisampling kept enabled in both conditions; same cached states, camera, DVS response, checkpoint, and causal query times",
        "input_boundary": "the learned model receives only raw ideal-DVS events encoded as five causal 5 ms bins; target GT joins after inference; depth/segmentation are evaluator-only",
        "episodes": episode_records,
        "wall_s": round(float(time.perf_counter() - started), 3),
        "frozen_core_modified": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    _write_hash_manifest(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "wall_s": summary["wall_s"],
                      "per_query_metrics": str((args.output / "per_query_metrics.csv").resolve())}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
