#!/usr/bin/env python3
"""Build post-inference scene overlays from a completed R1.5 crossover run.

This utility reads saved direct-HDR frames, saved heatmaps, and saved labels.
It does not render, run inference, or alter any model/controller inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from benchmark.robot_closed_loop.diagnostics.run_r1_5_same_trajectory_source_crossover import (
    _write_contact_sheet,
    _write_manifest,
)
from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import _sha256


def _episode_id_from_dir(path: Path) -> int:
    name = path.name
    prefix = "episode_"
    if not name.startswith(prefix):
        raise ValueError(f"unexpected episode directory: {name}")
    return int(name[len(prefix):].split("_", 1)[0])


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            row["query_s"] = float(row["query_s"])
            row["planar_error_xy_mm"] = float(row["planar_error_xy_mm"])
            rows.append(row)
    return rows


def _area_weights(source_size: int, target_size: int) -> np.ndarray:
    """Map source pixel areas to target-grid cells without point sampling."""
    edges = np.linspace(0.0, float(source_size), target_size + 1)
    weights = np.zeros((target_size, source_size), dtype=np.float32)
    for out_idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        first = int(np.floor(left))
        last = int(np.ceil(right))
        for src_idx in range(first, min(last, source_size)):
            overlap = max(0.0, min(right, src_idx + 1.0) - max(left, float(src_idx)))
            weights[out_idx, src_idx] = overlap / (right - left)
    return weights


def _region_coverage(segmentation: np.ndarray, target_ids: list[int],
                     robot_ids: list[int], output_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    height, width = segmentation.shape
    out_h, out_w = output_hw
    wy = _area_weights(height, out_h)
    wx = _area_weights(width, out_w)
    target = np.isin(segmentation, target_ids)
    robot = np.isin(segmentation, robot_ids)
    other = (segmentation != 0) & ~target & ~robot
    masks = {
        "target": target,
        "robot": robot,
        "other_scene": other,
        "zero_seg_background": segmentation == 0,
    }
    return {name: wy @ mask.astype(np.float32) @ wx.T for name, mask in masks.items()}


def _class_at_pixel(segmentation: np.ndarray, u: float, v: float,
                    target_ids: list[int], robot_ids: list[int]) -> str:
    height, width = segmentation.shape
    x = int(np.clip(np.floor(u), 0, width - 1))
    y = int(np.clip(np.floor(v), 0, height - 1))
    value = int(segmentation[y, x])
    if value in target_ids:
        return "target"
    if value in robot_ids:
        return "robot"
    return "other_scene" if value != 0 else "zero_seg_background"


def _region_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("target", "robot", "other_scene", "zero_seg_background")
    result: dict[str, Any] = {"query_count": len(rows)}
    for name in names:
        mass = np.asarray([float(row[f"heatmap_mass_{name}"]) for row in rows], dtype=np.float64)
        result[f"heatmap_mass_{name}"] = {
            "mean": float(mass.mean()), "median": float(np.median(mass)),
            "p95": float(np.percentile(mass, 95)),
        }
    for name in ("peak_region", "prediction_point_region"):
        counts: dict[str, int] = {}
        for row in rows:
            key = str(row[name])
            counts[key] = counts.get(key, 0) + 1
        result[f"{name}_counts"] = counts
    errors = np.asarray([float(row["planar_error_xy_mm"]) for row in rows], dtype=np.float64)
    result["planar_error_xy_mm"] = {
        "mean": float(errors.mean()), "p50": float(np.percentile(errors, 50)),
        "p95": float(np.percentile(errors, 95)), "max": float(errors.max()),
    }
    first_crossings: dict[str, Any] = {}
    ordered = sorted(rows, key=lambda row: float(row["query_s"]))
    for threshold in (25.0, 50.0, 100.0):
        crossing = next((row for row in ordered if float(row["planar_error_xy_mm"]) > threshold), None)
        if crossing is None:
            first_crossings[str(int(threshold))] = None
        else:
            first_crossings[str(int(threshold))] = {
                "query_s": float(crossing["query_s"]),
                "planar_error_xy_mm": float(crossing["planar_error_xy_mm"]),
                "prediction_point_region": crossing["prediction_point_region"],
                "peak_region": crossing["peak_region"],
                "heatmap_mass_target": float(crossing["heatmap_mass_target"]),
                "heatmap_mass_robot": float(crossing["heatmap_mass_robot"]),
                "heatmap_mass_other_scene": float(crossing["heatmap_mass_other_scene"]),
                "heatmap_mass_zero_seg_background": float(crossing["heatmap_mass_zero_seg_background"]),
                "pred_u_px": float(crossing["pred_u_px"]),
                "pred_v_px": float(crossing["pred_v_px"]),
                "peak_u_px": float(crossing["peak_u_px"]),
                "peak_v_px": float(crossing["peak_v_px"]),
            }
    result["first_error_threshold_crossings"] = first_crossings
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--query-times", type=float, nargs="+",
                        default=[0.5755, 0.5825, 0.5855, 0.5905])
    args = parser.parse_args()

    root = args.result_root.resolve()
    if not (root / "summary.json").is_file() or not (root / "run_metadata.json").is_file():
        raise FileNotFoundError(f"not a completed crossover result: {root}")
    visualization_root = root / "visualizations"
    visualization_root.mkdir(exist_ok=True)
    postprocess: dict[str, Any] = {
        "purpose": "post-inference overlays only; no rendering, inference, checkpoint changes, or model-input changes",
        "region_analysis": "post-inference heatmap probability mass is area-weighted against the exact direct-frame segmentation; segmentation is not fed to either checkpoint",
        "region_definitions": {
            "target": "segmentation ID inferred from saved target pose displacement",
            "robot": "saved segmentation pose IDs excluding the target",
            "other_scene": "nonzero segmentation IDs outside target and robot lists",
            "zero_seg_background": "segmentation ID zero; may contain floor/background pixels and is not a shadow mask",
            "peak_region": "region with greatest area coverage in the heatmap argmax cell; mixed_boundary if no region covers at least half the cell",
            "prediction_point_region": "post-inference segmentation class at the decoded predicted pixel",
        },
        "query_times_s": [round(float(t), 9) for t in args.query_times],
        "episodes": {},
        "group_region_summary": {},
    }
    all_region_rows: list[dict[str, Any]] = []

    for episode_dir in sorted((root / "episodes").iterdir()):
        if not episode_dir.is_dir() or not episode_dir.name.startswith("episode_"):
            continue
        ep_id = _episode_id_from_dir(episode_dir)
        direct_path = episode_dir / "direct_hdr_exact_state_frames.h5"
        arrays_path = episode_dir / "four_group_inputs_heatmaps.npz"
        rows_path = episode_dir / "four_group_queries.csv"
        cache_path = root / "source_inputs" / "native_hdr_cutoff059" / f"episode_{ep_id:03d}" / "same_state_source_cache.h5"
        if not all(p.is_file() for p in (direct_path, arrays_path, rows_path, cache_path)):
            raise FileNotFoundError(f"episode_{ep_id:03d} lacks a saved overlay input")

        with h5py.File(direct_path, "r") as direct, h5py.File(cache_path, "r") as source:
            frame_times = np.asarray(direct["timestamps_s"][:], dtype=np.float64)
            frame_lookup = {round(float(t), 9): i for i, t in enumerate(frame_times)}
            saved_ids = np.asarray(source["segmentation_pose_ids"][:], dtype=np.int32)
            poses = np.asarray(source["object_poses"][:], dtype=np.float64)
            displacement = np.linalg.norm(poses[-1, :, :3, 3] - poses[0, :, :3, 3], axis=1)
            target_idx = int(np.argmax(displacement))
            target_id = int(saved_ids[target_idx])
            if float(displacement[target_idx]) < 0.015:
                raise RuntimeError(f"episode_{ep_id:03d}: target body is not separable by saved motion")
            robot_ids = [int(v) for v in saved_ids if int(v) != target_id]
            frames_by_time: dict[float, dict[str, np.ndarray]] = {}
            for time_s in args.query_times:
                key = round(float(time_s), 9)
                if key not in frame_lookup:
                    raise ValueError(f"episode_{ep_id:03d}: no exact saved direct frame at {key:.9f}")
                i = frame_lookup[key]
                frames_by_time[key] = {
                    "display_rgb": np.asarray(direct["display_rgb_post_inference_only"][i], dtype=np.uint8),
                    "segmentation": np.asarray(direct["segmentation_post_inference_only"][i], dtype=np.int32),
                }
            segmentation_by_time = {
                round(float(t), 9): np.asarray(direct["segmentation_post_inference_only"][i], dtype=np.int32)
                for i, t in enumerate(frame_times)
            }

        arrays = np.load(arrays_path)
        query_times = np.asarray(arrays["query_times_s"], dtype=np.float64)
        query_lookup = {round(float(t), 9): i for i, t in enumerate(query_times)}
        rows = _load_rows(rows_path)
        probs: dict[tuple[str, str, float], np.ndarray] = {}
        for time_s in args.query_times:
            key = round(float(time_s), 9)
            if key not in query_lookup:
                raise ValueError(f"episode_{ep_id:03d}: query {key:.9f} is absent from saved inputs")
            qi = query_lookup[key]
            for source_name in ("ours_native_hdr", "direct_hdr"):
                for model_name in ("ours_native_hdr", "v2e"):
                    array_name = f"{source_name}__{model_name}__heatmap_probability"
                    probs[(source_name, model_name, key)] = np.asarray(arrays[array_name][qi], dtype=np.float32)

        # Score every saved query. These labels are strictly post-inference diagnostics.
        episode_region_rows = []
        coverage_by_time: dict[float, dict[str, np.ndarray]] = {}
        for source_name in ("ours_native_hdr", "direct_hdr"):
            for model_name in ("ours_native_hdr", "v2e"):
                group_rows = [r for r in rows if r["source"] == source_name and r["checkpoint"] == model_name]
                for row in group_rows:
                    key = round(float(row["query_s"]), 9)
                    if key not in query_lookup or key not in segmentation_by_time:
                        raise ValueError(f"episode_{ep_id:03d}: query {key:.9f} lacks an exact saved frame")
                    qi = query_lookup[key]
                    probability = np.asarray(arrays[f"{source_name}__{model_name}__heatmap_probability"][qi], dtype=np.float32)
                    seg = segmentation_by_time[key]
                    if key not in coverage_by_time:
                        coverage_by_time[key] = _region_coverage(seg, [target_id], robot_ids, probability.shape)
                    coverage = coverage_by_time[key]
                    mass = {name: float(np.sum(probability * fraction)) for name, fraction in coverage.items()}
                    peak_v, peak_u = np.unravel_index(int(np.argmax(probability)), probability.shape)
                    peak_coverages = {name: float(fraction[peak_v, peak_u]) for name, fraction in coverage.items()}
                    peak_region = max(peak_coverages, key=peak_coverages.get)
                    if peak_coverages[peak_region] < 0.5:
                        peak_region = "mixed_boundary"
                    assoc = {
                        "episode_id": ep_id,
                        "source": source_name,
                        "checkpoint": model_name,
                        "query_s": float(row["query_s"]),
                        "planar_error_xy_mm": float(row["planar_error_xy_mm"]),
                        "pixel_error_px": float(row["pixel_error_px"]),
                        **{f"heatmap_mass_{name}": value for name, value in mass.items()},
                        "peak_region": peak_region,
                        "peak_region_coverage_fraction": peak_coverages[peak_region] if peak_region != "mixed_boundary" else max(peak_coverages.values()),
                        "peak_u_px": float(row["peak_u_px"]),
                        "peak_v_px": float(row["peak_v_px"]),
                        "prediction_point_region": _class_at_pixel(seg, float(row["pred_u_px"]), float(row["pred_v_px"]), [target_id], robot_ids),
                        "pred_u_px": float(row["pred_u_px"]),
                        "pred_v_px": float(row["pred_v_px"]),
                        "gt_u_px": float(row["gt_u_px"]),
                        "gt_v_px": float(row["gt_v_px"]),
                    }
                    episode_region_rows.append(assoc)
                    all_region_rows.append(assoc)

        for source_name in ("ours_native_hdr", "direct_hdr"):
            for model_name in ("ours_native_hdr", "v2e"):
                group_key = f"episode_{ep_id:03d}/{source_name}/{model_name}"
                selected = [r for r in episode_region_rows if r["source"] == source_name and r["checkpoint"] == model_name]
                postprocess["group_region_summary"][group_key] = _region_summary(selected)

        label = "center_control" if ep_id == 4 else "plus55_failure" if ep_id == 5 else f"episode_{ep_id:03d}"
        image_path = visualization_root / f"episode_{ep_id:03d}_{label}_spatial_overlay.png"
        _write_contact_sheet(image_path, label, frames_by_time, rows, probs,
                             [round(float(t), 9) for t in args.query_times],
                             [target_id], robot_ids)
        postprocess["episodes"][str(ep_id)] = {
            "label": label,
            "target_segmentation_id_inferred_post_inference_from_saved_pose_displacement": target_id,
            "target_pose_displacement_m": float(displacement[target_idx]),
            "robot_segmentation_ids": robot_ids,
            "visualization": image_path.relative_to(root).as_posix(),
            "visualization_sha256": _sha256(image_path),
            "inputs": {p.name: _sha256(p) for p in (direct_path, arrays_path, rows_path, cache_path)},
        }

    for source_name in ("ours_native_hdr", "direct_hdr"):
        for model_name in ("ours_native_hdr", "v2e"):
            selected = [r for r in all_region_rows
                        if r["source"] == source_name and r["checkpoint"] == model_name]
            postprocess["group_region_summary"][f"all_episodes/{source_name}/{model_name}"] = _region_summary(selected)

    assoc_path = root / "post_inference_region_association.csv"
    if all_region_rows:
        with assoc_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_region_rows[0]))
            writer.writeheader()
            writer.writerows(all_region_rows)

    postprocess["code_commit"] = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"], text=True
    ).strip()
    postprocess["script_sha256"] = _sha256(Path(__file__).resolve())
    postprocess_path = root / "postprocess_visualization_metadata.json"
    postprocess_path.write_text(json.dumps(postprocess, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_manifest(root)
    print(json.dumps(postprocess, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
