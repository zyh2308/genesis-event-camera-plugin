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
        "query_times_s": [round(float(t), 9) for t in args.query_times],
        "episodes": {},
    }

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
