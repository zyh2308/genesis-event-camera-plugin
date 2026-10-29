#!/usr/bin/env python3
"""Score archived 2 kHz Direct-HDR events at matched shadow-replay cutoffs.

This is a CPU-only post-analysis. The archived event stream and the matched
shadow-on/off streams are passed through the same old Ours checkpoint and
the same causal five-bin encoder. GT is joined only after each forward pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.diagnose_r1_native_hdr_replay import (  # noqa: E402
    infer_event_state,
    interpolate_trace_label,
    load_event_models,
    project_world,
    read_trace,
)
from benchmark.robot_closed_loop.diagnostics.replay_event_state import _camera_to_world_cv  # noqa: E402
from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import (  # noqa: E402
    _sha256,
    _write_hash_manifest,
)
from benchmark.robot_closed_loop.event_state_model import five_bin_tensor  # noqa: E402


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denom == 0 else float(np.dot(a, b) / denom)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p95_linear": float(np.percentile(values, 95)),
        "max": float(max(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--archived-event-root", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    scene = json.loads(args.scene_config.read_text(encoding="utf-8"))
    camera_pose = _camera_to_world_cv(scene)
    width, height = (int(x) for x in scene["camera"]["resolution_wh"])
    focal = height / (2.0 * np.tan(np.deg2rad(float(scene["camera"]["vertical_fov_deg"])) / 2.0))
    intrinsics = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)
    fixed_height = float(scene["target"]["size_m"][2]) / 2.0
    model_records, checkpoint_hashes = load_event_models(
        {"ours_old_checkpoint": args.checkpoint.resolve()}, camera_pose, intrinsics,
        (width, height),
    )

    csv_rows: list[dict[str, Any]] = []
    episodes: dict[str, Any] = {}
    for ep in ("004", "005"):
        replay_summary_path = args.replay_root / f"episode_{ep}_summary.json"
        replay_summary = json.loads(replay_summary_path.read_text(encoding="utf-8"))
        trace_path = args.cache_root / f"episode_{ep}" / "trace_window.jsonl"
        trace = read_trace(trace_path)
        archived_path = args.archived_event_root / f"episode_{ep}" / "archived_direct_hdr_2khz_reference_events.npz"
        with np.load(archived_path, allow_pickle=False) as archive:
            archived_events = np.asarray(archive["events"], dtype=np.float64)
        if archived_events.ndim != 2 or archived_events.shape[1] != 4:
            raise ValueError(f"{archived_path}: expected event array with shape (N,4)")

        shadow_voxels: dict[str, np.ndarray] = {}
        shadow_input_paths: dict[str, Path] = {}
        for condition in ("shadow_on", "shadow_off"):
            path = args.replay_root / f"episode_{ep}_{condition}_five_bin_inputs.npz"
            shadow_input_paths[condition] = path
            with np.load(path, allow_pickle=False) as archive:
                cutoffs = np.asarray(archive["cutoffs_s"], dtype=np.float64)
                voxels = np.asarray(archive["tensors"], dtype=np.float32)
            if not np.allclose(cutoffs, replay_summary["query_cutoffs_actual_trace_rows_s"], rtol=0, atol=1e-9):
                raise RuntimeError(f"{path}: query times differ from episode summary")
            shadow_voxels[condition] = voxels

        query_rows: list[dict[str, Any]] = []
        for query_index, cutoff in enumerate(replay_summary["query_cutoffs_actual_trace_rows_s"]):
            gt = interpolate_trace_label(trace, float(cutoff))
            gt_position = np.asarray(gt["position_m"], dtype=np.float64)
            gt_uv = project_world(gt_position, camera_pose, intrinsics)
            reference_estimate, reference_voxel = infer_event_state(
                archived_events, model_records["ours_old_checkpoint"], float(cutoff),
                width, height, camera_pose, intrinsics, fixed_height,
            )
            if reference_estimate.get("state_available"):
                ref_xy = np.asarray(reference_estimate["position_xyz_m"], dtype=np.float64)[:2]
                ref_uv = np.asarray(reference_estimate["heatmap_uv_px"], dtype=np.float64)
                reference_estimate["position_error_xy_mm"] = float(np.linalg.norm(ref_xy - gt_position[:2]) * 1000.0)
                reference_estimate["heatmap_uv_error_px"] = float(np.linalg.norm(ref_uv - gt_uv))

            conditions: dict[str, Any] = {"archived_direct_hdr_2khz": reference_estimate}
            ref_flat = reference_voxel.astype(np.float64).ravel()
            ref_bin_counts = reference_voxel.sum(axis=(1, 2, 3)).astype(np.int64)
            bin_comparison: dict[str, Any] = {}
            for condition in ("shadow_on", "shadow_off"):
                estimate = replay_summary["per_query_model_metrics"][query_index][condition]
                condition_voxel = shadow_voxels[condition][query_index].astype(np.float64)
                condition_flat = condition_voxel.ravel()
                estimate = dict(estimate)
                estimate["five_bin_cosine_vs_archived_direct_hdr_2khz"] = cosine(condition_flat, ref_flat)
                conditions[condition] = estimate
                bin_cosines = [
                    cosine(reference_voxel[bin_idx], condition_voxel[bin_idx])
                    for bin_idx in range(5)
                ]
                bin_counts = condition_voxel.sum(axis=(1, 2, 3)).astype(np.int64)
                bin_l1_relative = [
                    float(np.abs(condition_voxel[bin_idx] - reference_voxel[bin_idx]).sum()
                          / max(float(reference_voxel[bin_idx].sum()), 1.0))
                    for bin_idx in range(5)
                ]
                bin_comparison[condition] = {
                    "cosine_by_5ms_bin": bin_cosines,
                    "relative_l1_by_5ms_bin": bin_l1_relative,
                    "event_count_archived_by_5ms_bin": ref_bin_counts.tolist(),
                    "event_count_condition_by_5ms_bin": bin_counts.tolist(),
                }
            record = {
                "episode_id": int(ep),
                "query_cutoff_s": float(cutoff),
                "gt_position_xy_m": gt_position[:2].tolist(),
                "gt_label_method": gt["label_method"],
                "archived_direct_hdr_2khz": conditions["archived_direct_hdr_2khz"],
                "shadow_on": conditions["shadow_on"],
                "shadow_off": conditions["shadow_off"],
                "five_ms_bin_comparison_vs_archived": bin_comparison,
            }
            query_rows.append(record)
            for condition, estimate in conditions.items():
                csv_rows.append({
                    "episode_id": int(ep), "query_cutoff_s": float(cutoff),
                    "condition": condition,
                    "state_available": estimate["state_available"],
                    "event_count_25ms": estimate["event_count_25ms"],
                    "nonempty_5ms_bins": estimate["nonempty_5ms_bins"],
                    "position_error_xy_mm": estimate.get("position_error_xy_mm"),
                    "heatmap_uv_error_px": estimate.get("heatmap_uv_error_px"),
                    "five_bin_cosine_vs_archived_direct_hdr_2khz": estimate.get("five_bin_cosine_vs_archived_direct_hdr_2khz"),
                    "per_5ms_bin_cosines_vs_archived": json.dumps(
                        bin_comparison.get(condition, {}).get("cosine_by_5ms_bin")
                    ),
                    "relative_l1_by_5ms_bin_vs_archived": json.dumps(
                        bin_comparison.get(condition, {}).get("relative_l1_by_5ms_bin")
                    ),
                    "event_counts_archived_by_5ms_bin": json.dumps(ref_bin_counts.tolist()),
                    "event_counts_condition_by_5ms_bin": json.dumps(
                        bin_comparison.get(condition, {}).get("event_count_condition_by_5ms_bin")
                    ),
                    "input_tensor_sha256": estimate.get("input_tensor_sha256"),
                })

        per_condition: dict[str, Any] = {}
        for condition in ("archived_direct_hdr_2khz", "shadow_on", "shadow_off"):
            errors = [float(row[condition]["position_error_xy_mm"]) for row in query_rows
                      if row[condition].get("state_available") and row[condition].get("position_error_xy_mm") is not None]
            cosines = [float(row[condition]["five_bin_cosine_vs_archived_direct_hdr_2khz"])
                       for row in query_rows if row[condition].get("five_bin_cosine_vs_archived_direct_hdr_2khz") is not None]
            per_condition[condition] = {
                "state_available_n": sum(bool(row[condition]["state_available"]) for row in query_rows),
                "query_count": len(query_rows),
                "position_error_xy_mm": summarize(errors) if errors else None,
                "mean_tensor_cosine_vs_archived": None if not cosines else float(statistics.mean(cosines)),
            }
        episodes[ep] = {
            "query_times_s": [row["query_cutoff_s"] for row in query_rows],
            "per_condition_summary": per_condition,
            "queries": query_rows,
            "source_hashes": {
                "archived_events_npz": _sha256(archived_path),
                "trace_window_jsonl": _sha256(trace_path),
                "replay_episode_summary_json": _sha256(replay_summary_path),
                "shadow_on_five_bin_npz": _sha256(shadow_input_paths["shadow_on"]),
                "shadow_off_five_bin_npz": _sha256(shadow_input_paths["shadow_off"]),
            },
        }

    fields = list(csv_rows[0])
    csv_path = args.output / "archived_reference_same_query_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    result = {
        "status": "PASS",
        "purpose": "same old Ours checkpoint, same exact causal query times, compare archived Direct-HDR against matched shadow-on/off direct renders",
        "analysis_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "checkpoint_sha256": checkpoint_hashes,
        "scene_config_sha256": _sha256(args.scene_config.resolve()),
        "archived_event_root": str(args.archived_event_root.resolve()),
        "episodes": episodes,
        "wall_s": round(float(time.perf_counter() - started), 3),
        "gt_boundary": "GT is joined only after each causal event inference; no GT/depth/segmentation is given to the network",
    }
    json_path = args.output / "archived_reference_same_query_comparison.json"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    _write_hash_manifest(args.output)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "wall_s": result["wall_s"],
                      "summary": {ep: item["per_condition_summary"] for ep, item in episodes.items()}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
