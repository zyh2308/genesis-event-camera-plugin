#!/usr/bin/env python3
"""Measure exact main-camera scissor repair on archived R1.5 traces.

The old K reconstruction/request masks are treated as fixed development
inputs. This probe re-renders only the same saved robot states, compares every
requested scissor pixel with a full-frame scene-linear HDR render at that
state, and measures the HDR-only direct baseline on the same rows. It does
not advance physics, train models, or modify the frozen simulator core.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.run_r1_5_multiref_surface_cached_replay import (
    _events_from_frames,
    _response_config,
    _rows,
    _set_exact_logged_state,
)
from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import _write_hash_manifest
from benchmark.robot_closed_loop.diagnostics.r1_5_scissor_roi_render import (
    MainCameraScissorROI,
    roi_from_mask,
)
from benchmark.robot_closed_loop.robot_env import build_robot_scene
from genesis_event_plugin.radiometry import linear_luminance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--probe-config", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True,
                        help="completed K run containing reconstruction arrays and support masks")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def events_from_luma_frames(luma_frames: list[np.ndarray], times: np.ndarray) -> tuple[np.ndarray, float]:
    from genesis_event_plugin.dvs_emulator import DvsEmulator

    h, w = luma_frames[0].shape
    emulator = DvsEmulator(res=(h, w), config=_response_config(), device="cpu")
    started = time.perf_counter()
    emulator.initialize(luma_frames[0], float(times[0]))
    chunks = []
    for luma, t in zip(luma_frames[1:], times[1:]):
        events = emulator.generate_events(luma, float(t))
        if len(events):
            chunks.append(np.asarray(events, dtype=np.float64))
    elapsed = time.perf_counter() - started
    result = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)
    if len(result):
        result = result[np.argsort(result[:, 0], kind="stable")]
    return result, elapsed


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prior = args.prior_run.resolve()
    prior_summary = json.loads((prior / "summary.json").read_text(encoding="utf-8"))
    scene_cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    scene_cfg["shared_control"] = json.loads(args.control_config.read_text(encoding="utf-8"))
    probe_cfg = json.loads(args.probe_config.read_text(encoding="utf-8"))
    width, height = (int(x) for x in scene_cfg["camera"]["resolution_wh"])
    inputs: dict[str, str] = {}
    for path in (args.scene_config, args.control_config, args.probe_config, Path(__file__),
                 Path(__file__).with_name("r1_5_scissor_roi_render.py")):
        inputs[str(path.resolve())] = file_sha(path.resolve())
    inputs[str(prior / "summary.json")] = file_sha(prior / "summary.json")

    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning")
    env = None
    scissor = None
    episode_summaries = {}
    try:
        env = build_robot_scene(
            scene_cfg, dt_s=float(scene_cfg["physics"]["dt_s"]), segmentation_level="link",
        )
        # Warm Genesis scene and allocate its full-resolution HDR framebuffer.
        env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True, force_render=True)
        scissor = MainCameraScissorROI(env.camera)
        method = scissor.install()

        for episode_id in (4, 5):
            ep_cache = args.cache_root.resolve() / f"episode_{episode_id:03d}"
            trace_path = ep_cache / "trace_window.jsonl"
            masks_path = prior / f"episode_{episode_id:03d}" / "candidate_render_request_masks.npz"
            arrays_path = prior / f"episode_{episode_id:03d}" / "candidate_roi_replay_arrays.npz"
            prior_direct_events_path = prior / f"episode_{episode_id:03d}" / "direct_logged_state_raw_events.npz"
            prior_candidate_events_path = prior / f"episode_{episode_id:03d}" / "candidate_surface_raw_events.npz"
            for path in (trace_path, masks_path, arrays_path, prior_direct_events_path, prior_candidate_events_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
                inputs[str(path)] = file_sha(path)
            trace_all = _rows(trace_path)
            trace_window = probe_cfg.get("trace_window_s")
            if not isinstance(trace_window, list) or len(trace_window) != 2:
                raise ValueError("probe config trace_window_s must be [start_s, end_s]")
            t_start, t_end = (float(x) for x in trace_window)
            trace = [row for row in trace_all
                     if t_start - 1e-9 <= float(row["time_s"]) <= t_end + 1e-9]
            with np.load(masks_path) as z:
                mask_times = np.asarray(z["times_s"], dtype=np.float64)
                request_masks = np.asarray(z["masks"], dtype=bool)
            with np.load(arrays_path) as z:
                candidate_base = np.asarray(z["candidate_reconstruction_radiance_luma"], dtype=np.float32)
                saved_times = np.asarray(z["times_s"], dtype=np.float64)
            if len(trace) != len(mask_times) or len(trace) != len(saved_times):
                raise RuntimeError(f"episode {episode_id}: trace/mask/candidate row counts differ")
            trace_times = np.asarray([round(float(row["time_s"]), 9) for row in trace], dtype=np.float64)
            if not np.array_equal(trace_times, np.round(mask_times, 9)) or not np.array_equal(trace_times, np.round(saved_times, 9)):
                raise RuntimeError(f"episode {episode_id}: prior masks do not align with exact trace rows")
            if candidate_base.shape != (len(trace), height, width) or request_masks.shape != candidate_base.shape:
                raise RuntimeError(f"episode {episode_id}: candidate arrays have unexpected shape")

            full_luma_frames: list[np.ndarray] = []
            full_hdr_frames: list[np.ndarray] = []
            repaired_luma_frames = [x.copy() for x in candidate_base]
            direct_render_s = []
            scissor_render_s = []
            rows_out = []
            requested_pixel_total = 0
            bbox_pixel_total = 0
            requested_supported_by_roi = 0
            exact_requested_mismatch_pixels = 0
            requested_log_errors = []
            roi_times = []
            roi_crops = []
            roi_boxes = []
            roi_row_indices = []
            seg_audit_wall_s = 0.0
            seg_mismatch_pixels = 0

            for idx, row in enumerate(trace):
                t = trace_times[idx]
                _set_exact_logged_state(env, row)
                started = time.perf_counter()
                hdr, _, _, _ = env.camera.render(
                    rgb=True, depth=False, segmentation=False, radiance=True, force_render=True,
                )
                hdr_render_s = time.perf_counter() - started
                hdr = np.asarray(hdr, dtype=np.float32)
                if hdr.shape != (height, width, 3) or not np.isfinite(hdr).all():
                    raise RuntimeError(f"episode {episode_id} t={t}: invalid full HDR frame {hdr.shape}")
                radiometry_started = time.perf_counter()
                full_luma = linear_luminance(hdr).astype(np.float32, copy=False)
                radiometry_s = time.perf_counter() - radiometry_started
                direct_render_s.append(hdr_render_s + radiometry_s)
                full_luma_frames.append(full_luma.copy())
                full_hdr_frames.append(hdr.copy())

                mask = request_masks[idx]
                crop = roi_from_mask(mask)
                rows_out.append({
                    "time_s": float(t), "request_pixels": int(mask.sum()),
                    "crop_xyxy": list(crop) if crop else None,
                    "crop_area_pixels": 0, "full_hdr_only_render_s": float(hdr_render_s),
                    "full_hdr_to_luma_s": float(radiometry_s),
                    "scissor_hdr_render_s": 0.0, "request_exact_mismatch_pixels": None,
                    "request_log_abs_p95": None,
                })

            # Measure the production HDR repair path in a separate pass. This
            # avoids alternating HDR and segmentation framebuffer formats,
            # which would add segmentation allocation cost to the HDR timings.
            for idx, row in enumerate(trace):
                t = trace_times[idx]
                mask = request_masks[idx]
                crop = roi_from_mask(mask)
                if crop is not None:
                    x0, y0, x1, y1 = crop
                    started = time.perf_counter()
                    crop_hdr = scissor.render_hdr_crop(float(t), crop)
                    crop_render_only_s = time.perf_counter() - started
                    requested_pixel_total += int(mask.sum())
                    bbox_pixel_total += (x1 - x0) * (y1 - y0)
                    local_mask = mask[y0:y1, x0:x1]
                    direct_crop = full_hdr_frames[idx][y0:y1, x0:x1]
                    diff = np.any(crop_hdr != direct_crop, axis=-1)
                    radiometry_started = time.perf_counter()
                    local_luma = linear_luminance(crop_hdr).astype(np.float32, copy=False)
                    local_radiometry_s = time.perf_counter() - radiometry_started
                    elapsed = crop_render_only_s + local_radiometry_s
                    scissor_render_s.append(elapsed)
                    direct_crop_luma = full_luma_frames[idx][y0:y1, x0:x1]
                    exact_requested_mismatch_pixels += int(np.count_nonzero(diff & local_mask))
                    local_log = np.abs(
                        np.log(np.maximum(local_luma[local_mask], 1e-6))
                        - np.log(np.maximum(direct_crop_luma[local_mask], 1e-6))
                    ) if local_mask.any() else np.empty((0,), np.float32)
                    if len(local_log):
                        requested_log_errors.extend(local_log.tolist())
                    repaired = repaired_luma_frames[idx]
                    repaired_region = repaired[y0:y1, x0:x1]
                    repaired_region[local_mask] = local_luma[local_mask]
                    repaired_luma_frames[idx][y0:y1, x0:x1] = repaired_region
                    requested_supported_by_roi += int(local_mask.sum())
                    roi_times.append(float(t))
                    roi_crops.append(crop_hdr.copy())
                    roi_boxes.append(crop)
                    roi_row_indices.append(idx)
                    rows_out[idx].update({
                        "crop_area_pixels": int((x1 - x0) * (y1 - y0)),
                        "scissor_hdr_render_s": float(crop_render_only_s),
                        "scissor_hdr_to_luma_s": float(local_radiometry_s),
                        "scissor_hdr_plus_luma_s": float(elapsed),
                        "request_exact_mismatch_pixels": int(np.count_nonzero(diff & local_mask)),
                        "request_log_abs_p95": float(np.percentile(local_log, 95)) if len(local_log) else 0.0,
                        "crop_hdr_exact_equal_full_frame": bool(np.array_equal(crop_hdr, direct_crop)),
                    })

            full_events, full_dvs_s = events_from_luma_frames(full_luma_frames, trace_times)
            repaired_events, repaired_dvs_s = events_from_luma_frames(repaired_luma_frames, trace_times)
            with np.load(prior_direct_events_path) as z:
                prior_direct_events = np.asarray(z["events"], dtype=np.float64)
            with np.load(prior_candidate_events_path) as z:
                prior_candidate_events = np.asarray(z["events"], dtype=np.float64)

            # Separate label audit: the scissored ID buffer must match the
            # same-state full-camera ID buffer across each entire request ROI.
            for idx, crop in zip(roi_row_indices, roi_boxes):
                row = trace[idx]
                t = trace_times[idx]
                x0, y0, x1, y1 = crop
                _set_exact_logged_state(env, row)
                seg_started = time.perf_counter()
                scissor_seg = scissor.render_segmentation_crop(float(t), crop)
                scissor_seg_elapsed = time.perf_counter() - seg_started
                seg_audit_wall_s += scissor_seg_elapsed
                _, _, direct_seg, _ = env.camera.render(
                    rgb=False, depth=False, segmentation=True, force_render=True,
                )
                direct_seg = np.asarray(direct_seg, dtype=np.int32)
                direct_seg_crop = direct_seg[y0:y1, x0:x1]
                seg_mismatch = int(np.count_nonzero(scissor_seg != direct_seg_crop))
                seg_mismatch_pixels += seg_mismatch
                rows_out[idx]["scissor_segmentation_audit_s"] = float(scissor_seg_elapsed)
                rows_out[idx]["scissor_segmentation_mismatch_pixels_full_crop"] = seg_mismatch
                rows_out[idx]["scissor_segmentation_exact_equal_full_frame_crop"] = bool(
                    np.array_equal(scissor_seg, direct_seg_crop)
                )
            prev_episode = prior_summary["episodes"][str(episode_id)]
            prev_candidate = prev_episode["candidate"]
            prev_roi_s = float(prev_candidate["roi_direct_render"]["crop_render_wall_s"])
            prev_reconstruction_s = float(prev_candidate["reconstruction_wall_s"])
            prev_prep_s = float(prev_candidate["source_preparation_wall_s"])
            prev_source_s = float(prev_episode["direct_render"]["wall_s_source_100hz_endpoints"])
            prev_fallback_s = float(prev_candidate.get("fallback_render_wall_s", 0.0))
            new_crop_s = float(sum(scissor_render_s))
            candidate_mixed_estimate_s = (
                prev_source_s + prev_prep_s + prev_reconstruction_s + prev_fallback_s
                + new_crop_s + repaired_dvs_s
            )
            direct_hdr_path_s = float(sum(direct_render_s) + full_dvs_s)
            prior_candidate_total = float(prev_candidate["total_production_wall_s_estimate"])
            direct_events = full_events
            rows_path = output / f"episode_{episode_id:03d}_per_frame.csv"
            with rows_path.open("w", newline="", encoding="utf-8") as stream:
                fieldnames = list(dict.fromkeys(key for row in rows_out for key in row))
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_out)
            event_match = {
                "candidate_events": int(len(repaired_events)),
                "direct_events": int(len(direct_events)),
                "candidate_positive": int(np.count_nonzero(repaired_events[:, 3] > 0)) if len(repaired_events) else 0,
                "candidate_negative": int(np.count_nonzero(repaired_events[:, 3] < 0)) if len(repaired_events) else 0,
                "direct_events_exact_match_vs_prior_k": bool(np.array_equal(direct_events, prior_direct_events)),
                "candidate_events_exact_match_vs_prior_k": bool(np.array_equal(repaired_events, prior_candidate_events)),
                "candidate_event_count_delta_vs_prior_k": int(len(repaired_events) - len(prior_candidate_events)),
            }
            np.savez_compressed(
                output / f"episode_{episode_id:03d}_scissor_patch.npz",
                times_s=np.asarray(roi_times, dtype=np.float64),
                crops_xyxy=np.asarray(roi_boxes, dtype=np.int32).reshape(-1, 4),
                crop_shapes=np.asarray([a.shape for a in roi_crops], dtype=np.int32).reshape(-1, 3),
                crop_rgb_flat=(np.concatenate([a.reshape(-1) for a in roi_crops]).astype(np.float32)
                               if roi_crops else np.empty((0,), np.float32)),
                direct_hdr_only_events=direct_events,
                scissor_repaired_candidate_events=repaired_events,
            )
            episode_summaries[str(episode_id)] = {
                "episode_role": "center_control" if episode_id == 4 else "plus55mm_development_failure",
                "trace_window_s": [t_start, t_end],
                "trace_rows": int(len(trace)),
                "unsupported_requested_pixels": int(requested_pixel_total),
                "unsupported_fraction_over_trace": float(requested_pixel_total / max(1, len(trace) * width * height)),
                "frames_with_requests": int(len(scissor_render_s)),
                "request_bbox_pixel_fraction_mean": float(bbox_pixel_total / max(1, len(scissor_render_s) * width * height)),
                "request_repair_coverage_fraction": float(requested_supported_by_roi / max(1, requested_pixel_total)),
                "request_exact_rgb_mismatch_pixels": int(exact_requested_mismatch_pixels),
                "request_log_abs_p95": float(np.percentile(requested_log_errors, 95)) if requested_log_errors else 0.0,
                "scissor_link_segmentation_mismatch_pixels_full_crop_audit": int(seg_mismatch_pixels),
                "scissor_link_segmentation_audit_wall_s_excluded_from_candidate_cost": float(seg_audit_wall_s),
                "scissor_crop_render_wall_s": new_crop_s,
                "scissor_crop_render_p50_s": float(np.median(scissor_render_s)) if scissor_render_s else 0.0,
                "scissor_crop_render_first_call_s_including_numba_compile": float(scissor_render_s[0]) if scissor_render_s else 0.0,
                "direct_hdr_only_render_wall_s": float(sum(direct_render_s)),
                "direct_hdr_only_render_p50_s": float(np.median(direct_render_s)) if direct_render_s else 0.0,
                "direct_dvs_wall_s": full_dvs_s,
                "scissor_repaired_candidate_dvs_wall_s": repaired_dvs_s,
                "direct_hdr_only_plus_dvs_wall_s": direct_hdr_path_s,
                "candidate_mixed_estimate_wall_s": candidate_mixed_estimate_s,
                "candidate_mixed_estimate_over_direct_hdr_only": candidate_mixed_estimate_s / max(1e-12, direct_hdr_path_s),
                "candidate_prior_k_total_wall_s": prior_candidate_total,
                "candidate_prior_k_roi_wall_s_replaced": prev_roi_s,
                "candidate_mixed_estimate_semantics": "prior K source-render/preparation/reconstruction timings plus fresh main-camera scissor HDR-only repair and fresh DVS, compared with fresh full-frame HDR-only Direct-HDR plus fresh DVS; estimate is not a single uninterrupted end-to-end timing",
                "event_counts": event_match,
                "per_frame_file": rows_path.name,
                "arrays_file": f"episode_{episode_id:03d}_scissor_patch.npz",
            }
            print(json.dumps({"episode": episode_id,
                              "request_fraction": episode_summaries[str(episode_id)]["unsupported_fraction_over_trace"],
                              "bbox_fraction": episode_summaries[str(episode_id)]["request_bbox_pixel_fraction_mean"],
                              "scissor_exact_mismatch": exact_requested_mismatch_pixels,
                              "scissor_s": round(new_crop_s, 4),
                              "direct_hdr_only_plus_dvs_s": round(direct_hdr_path_s, 4),
                              "candidate_mixed_ratio": round(episode_summaries[str(episode_id)]["candidate_mixed_estimate_over_direct_hdr_only"], 3)},
                             separators=(",", ":")), flush=True)
        summary = {
            "experiment_id": str(probe_cfg["experiment_id"]),
            "code_commit": __import__("subprocess").check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            ).strip(),
            "frozen_core_modified": False,
            "method": method,
            "scene_resolution_wh": [width, height],
            "probe_config_sha256": file_sha(args.probe_config.resolve()),
            "prior_k_run_manifest_sha256": file_sha(prior / "SHA256SUMS"),
            "prior_candidate_summary_sha256": file_sha(prior / "summary.json"),
            "input_sha256": inputs,
            "episodes": episode_summaries,
            "interpretation": "The dynamic crop is computed only from the candidate's current unknown/request mask. Cropped values come from the unchanged main camera and are checked against same-state full-frame scene-linear HDR, preserving the renderer's full-scene depth test. This is a local renderer-path probe; candidate total is assembled from prior K non-ROI timings plus fresh scissor/DVS timings, and is not presented as one uninterrupted end-to-end execution.",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_hash_manifest(output)
    finally:
        if scissor is not None:
            scissor.close()
        gs.destroy()
    print(json.dumps({"output": str(output), "episodes": list(episode_summaries),
                      "hash_manifest": str(output / "SHA256SUMS")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
