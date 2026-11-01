#!/usr/bin/env python3
"""Evaluate both frozen heatmap checkpoints on paired strict/direct events.

The input archives are generated from one exact fine-clock physical rollout.
Events are filtered by both event time and availability time at every query.
GT is used only for post-inference scoring. This is a two-episode development
diagnostic, not a ranking or independent test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.replay_event_state import _camera_to_world_cv, _read_trace
from benchmark.robot_closed_loop.event_state_model import (
    HeatmapEventStateNet, decode_heatmap_state, five_bin_tensor, load_checkpoint,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_at_exact_time(trace: list[dict[str, Any]], time_s: float) -> dict[str, Any]:
    times = np.asarray([float(row["time_s"]) for row in trace], dtype=np.float64)
    index = int(np.argmin(np.abs(times - time_s)))
    if abs(float(times[index]) - time_s) > 1e-8:
        raise RuntimeError(f"query {time_s:.9f}s is absent from exact physical trace")
    return trace[index]


def summarize(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if not x.size:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": int(x.size), "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }


def project_world(point_world: np.ndarray, camera_to_world: np.ndarray,
                  intrinsics: np.ndarray) -> np.ndarray:
    camera = np.linalg.inv(camera_to_world) @ np.append(point_world, 1.0)
    fx, fy, cx, cy = intrinsics
    return np.asarray([fx * camera[0] / camera[2] + cx,
                       fy * camera[1] / camera[2] + cy], dtype=np.float64)


def load_models(paths: dict[str, Path], camera_pose: np.ndarray,
                intrinsics: np.ndarray, resolution_wh: tuple[int, int]):
    import torch
    torch.set_num_threads(1)
    models, hashes = {}, {}
    for name, path in paths.items():
        model, mean, scale, checkpoint = load_checkpoint(str(path), device="cpu")
        if model.__class__ is not HeatmapEventStateNet:
            raise ValueError(f"{name}: expected heatmap_v2, got {checkpoint.get('architecture')}")
        calibration = checkpoint["spatial_calibration"]
        if not np.allclose(np.asarray(calibration["camera_to_world_cv"]), camera_pose, atol=1e-5, rtol=0):
            raise ValueError(f"{name}: checkpoint camera calibration differs")
        if list(calibration["resolution_wh"]) != list(resolution_wh):
            raise ValueError(f"{name}: checkpoint resolution differs")
        models[name] = (model, mean, scale, checkpoint)
        hashes[name] = sha256(path)
    return models, hashes


def infer_causal(events: np.ndarray, model_record, cutoff: float, width: int, height: int,
                 camera_pose: np.ndarray, intrinsics: np.ndarray,
                 fixed_height: float) -> tuple[dict[str, Any], np.ndarray]:
    import torch
    model, mean, scale, checkpoint = model_record
    voxel = five_bin_tensor(events, cutoff, width, height)
    event_count = int(voxel.sum())
    nonempty = int(np.count_nonzero(voxel.sum(axis=(1, 2, 3))))
    policy = checkpoint.get("validity_policy", {})
    available = (event_count >= int(policy.get("minimum_events", 12))
                 and nonempty >= int(policy.get("minimum_nonempty_5ms_bins", 3)))
    result = {
        "state_available": available,
        "state_status": "insufficient_events_or_temporal_coverage" if not available else "fresh",
        "event_count_25ms": event_count, "nonempty_5ms_bins": nonempty,
        "input_tensor_sha256": hashlib.sha256(np.ascontiguousarray(voxel).tobytes()).hexdigest(),
    }
    if not available:
        return result, voxel
    tensor = torch.as_tensor(np.log1p(voxel).reshape(1, 10, 64, 64), dtype=torch.float32)
    with torch.no_grad():
        raw = model(tensor)
        state = decode_heatmap_state(
            raw, mean, scale,
            torch.as_tensor(camera_pose, dtype=torch.float32),
            torch.as_tensor(intrinsics, dtype=torch.float32), (width, height),
            fixed_height_m=fixed_height,
        )[0]
        probability = torch.softmax(raw[0][0].reshape(-1) / 0.5, dim=0)
        grid_x = (torch.arange(64, dtype=torch.float32) + 0.5) * (width / 64.0)
        grid_y = (torch.arange(64, dtype=torch.float32) + 0.5) * (height / 64.0)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        uv = [float((probability * gx.flatten()).sum()), float((probability * gy.flatten()).sum())]
    state = state.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(state).all():
        result.update({"state_available": False, "state_status": "nonfinite"})
        return result, voxel
    result.update({"position_xyz_m": state[:3].tolist(), "velocity_xy_m_s": state[3:5].tolist(),
                   "heatmap_uv_px": uv})
    return result, voxel


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True,
                   help="render output directory containing episodes/episode_NNN")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--scene-config", type=Path, required=True)
    p.add_argument("--ours-checkpoint", type=Path, required=True)
    p.add_argument("--v2e-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    source_root = args.source_root.resolve()
    config_path = args.config.resolve()
    scene_path = args.scene_config.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    exp = json.loads(config_path.read_text(encoding="utf-8"))
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    width, height = (int(v) for v in scene["camera"]["resolution_wh"])
    fov_deg = float(scene["camera"]["vertical_fov_deg"])
    focal = height / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    intrinsics = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)
    camera_pose = _camera_to_world_cv(scene)
    models, model_hashes = load_models(
        {"ours_native_hdr": args.ours_checkpoint.resolve(), "v2e": args.v2e_checkpoint.resolve()},
        camera_pose, intrinsics, (width, height),
    )
    for model, *_ in models.values():
        model.eval()

    fixed_height = float(scene["target"]["size_m"][2]) / 2.0
    thresholds = (25.0, 50.0, 100.0)
    output.mkdir(parents=True)
    all_rows: list[dict[str, Any]] = []
    episode_summaries: dict[str, Any] = {}
    queries = [float(t) for t in exp["evaluation"]["query_times_s"]]

    for episode_id in exp["training_rollouts"]["episode_ids"]:
        ep = source_root / "episodes" / f"episode_{int(episode_id):03d}"
        direct_path = ep / "direct_hdr_events_exact_trace.npz"
        strict_path = ep / "strict_sampled_events.npz"
        trace_path = ep / "state_control_trace.jsonl"
        for path in (direct_path, strict_path, trace_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        trace = _read_trace(trace_path)
        with np.load(direct_path) as archive:
            direct_events = np.asarray(archive["events"], dtype=np.float64)
            direct_available = np.asarray(archive["available_at_s"], dtype=np.float64)
        with np.load(strict_path) as archive:
            strict_events = np.asarray(archive["events"], dtype=np.float64)
            strict_available = np.asarray(archive["available_at_s"], dtype=np.float64)
        for name, events, available in (("direct_hdr", direct_events, direct_available),
                                        ("strict_sampled", strict_events, strict_available)):
            if len(events) != len(available):
                raise RuntimeError(f"{name}: event/availability lengths differ")
            if len(events) and (not np.isfinite(events).all() or not np.isfinite(available).all()):
                raise RuntimeError(f"{name}: non-finite event or availability value")

        episode_rows = []
        for cutoff in queries:
            state = state_at_exact_time(trace, cutoff)
            gt_position = np.asarray(state["object"]["position_m"], dtype=np.float64)
            gt_velocity = np.asarray(state["object"]["velocity_m_s"], dtype=np.float64)
            gt_uv = project_world(gt_position, camera_pose, intrinsics)
            for source_name, events, available in (
                ("strict_sampled", strict_events, strict_available),
                ("direct_hdr", direct_events, direct_available),
            ):
                causal = (events[:, 0] <= cutoff + 1e-12) & (available <= cutoff + 1e-12)
                visible = events[causal]
                for model_name, model_record in models.items():
                    estimate, voxel = infer_causal(
                        visible, model_record, cutoff, width, height,
                        camera_pose, intrinsics, fixed_height,
                    )
                    row = {
                        "episode_id": int(episode_id), "query_s": cutoff,
                        "source": source_name, "checkpoint": model_name,
                        "state_available": bool(estimate["state_available"]),
                        "state_status": estimate["state_status"],
                        "source_events_available": int(len(visible)),
                        "input_event_count_25ms": int(estimate["event_count_25ms"]),
                        "nonempty_5ms_bins": int(estimate["nonempty_5ms_bins"]),
                        "input_tensor_sha256": estimate["input_tensor_sha256"],
                        "gt_u_px": float(gt_uv[0]), "gt_v_px": float(gt_uv[1]),
                        "gt_x_m": float(gt_position[0]), "gt_y_m": float(gt_position[1]),
                        "gt_vx_m_s": float(gt_velocity[0]), "gt_vy_m_s": float(gt_velocity[1]),
                        "available_at_filter": "event timestamp <= cutoff and source availability <= cutoff",
                    }
                    if estimate["state_available"]:
                        pred = np.asarray(estimate["position_xyz_m"], dtype=np.float64)
                        pred_uv = np.asarray(estimate["heatmap_uv_px"], dtype=np.float64)
                        pred_vel = np.asarray(estimate["velocity_xy_m_s"], dtype=np.float64)
                        row.update({
                            "pred_x_m": float(pred[0]), "pred_y_m": float(pred[1]),
                            "pred_u_px": float(pred_uv[0]), "pred_v_px": float(pred_uv[1]),
                            "pixel_error_px": float(np.linalg.norm(pred_uv - gt_uv)),
                            "position_xy_error_mm": float(np.linalg.norm(pred[:2] - gt_position[:2]) * 1000.0),
                            "pred_vx_m_s": float(pred_vel[0]), "pred_vy_m_s": float(pred_vel[1]),
                            "velocity_xy_error_m_s": float(np.linalg.norm(pred_vel - gt_velocity[:2])),
                        })
                    else:
                        row.update({
                            "pred_x_m": None, "pred_y_m": None, "pred_u_px": None, "pred_v_px": None,
                            "pixel_error_px": None, "position_xy_error_mm": None,
                            "pred_vx_m_s": None, "pred_vy_m_s": None, "velocity_xy_error_m_s": None,
                        })
                    episode_rows.append(row)
                    all_rows.append(row)

        episode_summaries[str(episode_id)] = {}
        for source_name in ("strict_sampled", "direct_hdr"):
            for model_name in ("ours_native_hdr", "v2e"):
                group = [r for r in episode_rows if r["source"] == source_name and r["checkpoint"] == model_name]
                valid = [r for r in group if r["state_available"]]
                errs = [float(r["position_xy_error_mm"]) for r in valid]
                episode_summaries[str(episode_id)][f"{source_name}__{model_name}"] = {
                    "queries": len(group),
                    "state_coverage": len(valid) / len(group) if group else 0.0,
                    "position_xy_error_mm": summarize(errs),
                    "first_over_threshold_s": {
                        str(t): next((r["query_s"] for r in valid if r["position_xy_error_mm"] > t), None)
                        for t in thresholds
                    },
                }

    fields = list(all_rows[0]) if all_rows else []
    with (output / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    groups: dict[str, Any] = {}
    for source_name in ("strict_sampled", "direct_hdr"):
        for model_name in ("ours_native_hdr", "v2e"):
            group = [r for r in all_rows if r["source"] == source_name and r["checkpoint"] == model_name]
            valid = [r for r in group if r["state_available"]]
            groups[f"{source_name}__{model_name}"] = {
                "queries": len(group), "state_coverage": len(valid) / len(group) if group else 0.0,
                "position_xy_error_mm": summarize([float(r["position_xy_error_mm"]) for r in valid]),
                "velocity_xy_error_m_s": summarize([float(r["velocity_xy_error_m_s"]) for r in valid]),
                "first_over_threshold_s": {
                    str(t): next((r["query_s"] for r in valid if r["position_xy_error_mm"] > t), None)
                    for t in thresholds
                },
            }
    summary = {
        "schema": "r1_5_fineclock_localization_v1",
        "status": "completed_two_episode_development_diagnostic",
        "config_sha256": sha256(config_path), "scene_config_sha256": sha256(scene_path),
        "checkpoint_sha256": model_hashes,
        "query_count_per_group_per_episode": len(queries),
        "episodes": episode_summaries, "pooled_correlated_query_summary": groups,
        "causal_input_contract": "only events with event timestamp <= cutoff and availability timestamp <= cutoff enter five_bin_tensor; GT used after inference only",
        "limits": [
            "two previously inspected development trajectories only; no ranking or independent test claim",
            "the physical integration clock is 1/6000 s and therefore differs numerically from the legacy 0.5 ms rollouts",
            "strict sampled and direct streams share the same new physical rollout but have different online availability timestamps",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = []
    for path in sorted(p for p in source_root.rglob("*") if p.is_file() and p.name in {
        "direct_hdr_events_exact_trace.npz", "strict_sampled_events.npz", "state_control_trace.jsonl"
    }):
        manifest.append({"path": str(path), "sha256": sha256(path)})
    (output / "input_manifest.json").write_text(json.dumps({
        "source_root": str(source_root), "source_files": manifest,
        "checkpoint_sha256": model_hashes,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"groups": groups, "output": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
