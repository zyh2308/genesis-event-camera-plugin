#!/usr/bin/env python3
"""Inspect spatial heatmap outputs and target-region event support offline.

This diagnostic uses saved Direct-HDR events and saved query labels. GT is used
only after inference to project an approximate cube ROI for event-support
measurement; no GT-derived quantity is fed to either checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.event_state_model import (  # noqa: E402
    decode_heatmap_state,
    five_bin_tensor,
    load_checkpoint,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_to_world_cv(camera_cfg: dict[str, Any]) -> np.ndarray:
    position = np.asarray(camera_cfg["position_m"], dtype=np.float64)
    forward = np.asarray(camera_cfg["lookat_m"], dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    up = np.asarray(camera_cfg["up_m"], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, -camera_up, forward))
    matrix[:3, 3] = position
    return matrix


def project_world(points: np.ndarray, world_from_camera: np.ndarray,
                  intrinsics: np.ndarray) -> np.ndarray:
    camera_points = (points - world_from_camera[:3, 3]) @ world_from_camera[:3, :3]
    fx, fy, cx, cy = intrinsics
    z = camera_points[:, 2]
    uv = np.stack((fx * camera_points[:, 0] / z + cx,
                   fy * camera_points[:, 1] / z + cy), axis=1)
    uv[z <= 1e-6] = np.nan
    return uv


def unproject_to_plane(uv: np.ndarray, plane_z: float,
                       world_from_camera: np.ndarray,
                       intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    ray_camera = np.asarray([(uv[0] - cx) / fx, (uv[1] - cy) / fy, 1.0], dtype=np.float64)
    ray_world = world_from_camera[:3, :3] @ ray_camera
    scale = (plane_z - world_from_camera[2, 3]) / ray_world[2]
    return world_from_camera[:3, 3] + scale * ray_world


def cube_bounds(position: np.ndarray, size: np.ndarray,
                world_from_camera: np.ndarray, intrinsics: np.ndarray,
                pad_px: float, width: int, height: int) -> tuple[float, float, float, float]:
    offsets = np.asarray([
        [sx * size[0] / 2, sy * size[1] / 2, sz * size[2] / 2]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ], dtype=np.float64)
    uv = project_world(position[None, :] + offsets, world_from_camera, intrinsics)
    uv = uv[np.all(np.isfinite(uv), axis=1)]
    if not len(uv):
        raise ValueError("cube projects behind the camera")
    x0 = max(0.0, float(uv[:, 0].min()) - pad_px)
    x1 = min(float(width), float(uv[:, 0].max()) + pad_px)
    y0 = max(0.0, float(uv[:, 1].min()) - pad_px)
    y1 = min(float(height), float(uv[:, 1].max()) + pad_px)
    return x0, y0, x1, y1


def region_mass(probability: np.ndarray, bounds: tuple[float, float, float, float],
                width: int, height: int) -> float:
    x0, y0, x1, y1 = bounds
    heat_h, heat_w = probability.shape
    xs = (np.arange(heat_w, dtype=np.float64) + 0.5) * width / heat_w
    ys = (np.arange(heat_h, dtype=np.float64) + 0.5) * height / heat_h
    mask = ((ys[:, None] >= y0) & (ys[:, None] < y1) &
            (xs[None, :] >= x0) & (xs[None, :] < x1))
    return float(probability[mask].sum())


def checkpoint_record(path: Path) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
    record = load_checkpoint(str(path.resolve()), device="cpu")
    if not isinstance(record[0](torch.zeros((1, 10, 64, 64)))[0], torch.Tensor):
        raise TypeError(f"checkpoint is not a heatmap model: {path}")
    if record[0].architecture != "event-heatmap64-aux-z-velocity-v2":
        raise ValueError(f"unexpected architecture {record[0].architecture}: {path}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--query-source-field", default="source")
    parser.add_argument("--query-source-value", default="archive_and_exact_rerun")
    parser.add_argument("--query-time-field", default="query_s")
    parser.add_argument("--query-position-field", default="gt_position_xy_m")
    parser.add_argument("--query-x-field")
    parser.add_argument("--query-y-field")
    parser.add_argument("--query-z-field")
    parser.add_argument("--query-phase-field")
    parser.add_argument("--query-phase-value")
    parser.add_argument("--event-key", default="auto",
                        help="event array key in NPZ; auto accepts events or events_2000hz")
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="NAME=PATH; repeat for paired checkpoint comparison")
    parser.add_argument("--roi-padding-px", type=float, default=6.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--local-radius-cells", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    scene = json.loads(args.scene_config.read_text(encoding="utf-8"))
    width, height = map(int, scene["camera"]["resolution_wh"])
    focal = height / (2.0 * np.tan(np.deg2rad(float(scene["camera"]["vertical_fov_deg"])) / 2.0))
    intrinsics = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)
    world_from_camera = camera_to_world_cv(scene["camera"])
    target_size = np.asarray(scene["target"]["size_m"], dtype=np.float64)
    fixed_height = float(target_size[2] / 2.0)
    with np.load(args.events, allow_pickle=False) as archive:
        event_key = args.event_key
        if event_key == "auto":
            event_key = next((key for key in ("events", "events_2000hz") if key in archive), "")
        if not event_key or event_key not in archive:
            raise KeyError(f"event key {event_key!r} missing in {args.events}; found {archive.files}")
        events = np.asarray(archive[event_key], dtype=np.float64).reshape(-1, 4)
    with args.queries.open(newline="", encoding="utf-8") as stream:
        all_queries = list(csv.DictReader(stream))
    queries = [row for row in all_queries
               if row.get(args.query_source_field) == args.query_source_value]
    if args.query_phase_field and args.query_phase_value is not None:
        queries = [row for row in queries
                   if row.get(args.query_phase_field) == args.query_phase_value]
    if not queries:
        raise ValueError(
            f"query CSV has no rows where {args.query_source_field}={args.query_source_value!r}")

    checkpoints: list[tuple[str, Path, Any, torch.Tensor, torch.Tensor, dict[str, Any]]] = []
    for spec in args.checkpoint:
        if "=" not in spec:
            raise ValueError("--checkpoint must be NAME=PATH")
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        model, label_mean, label_scale, metadata = checkpoint_record(path)
        checkpoints.append((name, path, model, label_mean, label_scale, metadata))

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for query in queries:
        cutoff = float(query[args.query_time_field])
        if args.query_position_field in query:
            raw_position = np.asarray(json.loads(query[args.query_position_field]), dtype=np.float64)
            gt_position = np.asarray([raw_position[0], raw_position[1],
                                      raw_position[2] if len(raw_position) > 2 else fixed_height],
                                     dtype=np.float64)
        elif args.query_x_field and args.query_y_field:
            gt_position = np.asarray([
                float(query[args.query_x_field]),
                float(query[args.query_y_field]),
                float(query[args.query_z_field]) if args.query_z_field else fixed_height,
            ], dtype=np.float64)
        else:
            raise KeyError(
                f"query position requires {args.query_position_field} or --query-x-field/--query-y-field")
        gt_uv = project_world(gt_position[None, :], world_from_camera, intrinsics)[0]
        roi = cube_bounds(gt_position, target_size, world_from_camera, intrinsics,
                          args.roi_padding_px, width, height)
        voxel = five_bin_tensor(events, cutoff, width, height)
        bin_counts = voxel.sum(axis=(1, 2, 3)).astype(np.int64)
        keep = ((events[:, 0] >= cutoff - 0.025) & (events[:, 0] < cutoff))
        window_events = events[keep]
        x0, y0, x1, y1 = roi
        in_roi = ((window_events[:, 1] >= x0) & (window_events[:, 1] < x1) &
                  (window_events[:, 2] >= y0) & (window_events[:, 2] < y1))
        bin_index = np.floor((window_events[:, 0] - (cutoff - 0.025)) / 0.005).astype(np.int64)
        roi_bin_counts = np.bincount(np.clip(bin_index[in_roi], 0, 4), minlength=5).astype(int)
        common = {
            "query_s": cutoff,
            "query_source_label": query.get(args.query_source_field, ""),
            "episode_id": query.get("episode_id", ""),
            "gt_position_xy_m": gt_position[:2].tolist(),
            "gt_pixel_uv": gt_uv.tolist(),
            "target_roi_xyxy_px": list(roi),
            "event_count_25ms": int(len(window_events)),
            "event_count_by_5ms_bin": bin_counts.tolist(),
            "target_roi_event_count_25ms": int(in_roi.sum()),
            "target_roi_event_count_by_5ms_bin": roi_bin_counts.tolist(),
            "target_roi_fraction_of_events": float(in_roi.mean()) if len(in_roi) else 0.0,
        }
        tensor = torch.as_tensor(np.log1p(voxel).reshape(1, 10, 64, 64), dtype=torch.float32)
        for name, path, model, label_mean, label_scale, metadata in checkpoints:
            with torch.no_grad():
                output_pair = model(tensor)
                logits = output_pair[0][0]
                probability = torch.softmax(logits.flatten() / args.temperature, dim=0).reshape(64, 64)
                decoded = decode_heatmap_state(
                    output_pair, label_mean, label_scale,
                    torch.as_tensor(world_from_camera, dtype=torch.float32),
                    torch.as_tensor(intrinsics, dtype=torch.float32), (width, height),
                    fixed_height_m=fixed_height,
                )[0].cpu().numpy().astype(np.float64)
            prob = probability.cpu().numpy().astype(np.float64)
            flat = np.argsort(prob.reshape(-1))[::-1][:5]
            peaks = []
            for idx in flat:
                py, px = np.unravel_index(int(idx), prob.shape)
                peaks.append({"u_px": float((px + 0.5) * width / 64),
                              "v_px": float((py + 0.5) * height / 64),
                              "probability": float(prob[py, px])})
            entropy = float(-(prob * np.log(np.maximum(prob, 1e-30))).sum())
            heatmap_argmax = peaks[0]
            peak_position = unproject_to_plane(
                np.asarray([heatmap_argmax["u_px"], heatmap_argmax["v_px"]]),
                fixed_height, world_from_camera, intrinsics)
            peak_distance_px = float(np.linalg.norm(
                np.asarray([heatmap_argmax["u_px"], heatmap_argmax["v_px"]]) - gt_uv))
            peak_y, peak_x = np.unravel_index(int(flat[0]), prob.shape)
            radius = max(0, int(args.local_radius_cells))
            local_y0, local_y1 = max(0, peak_y - radius), min(64, peak_y + radius + 1)
            local_x0, local_x1 = max(0, peak_x - radius), min(64, peak_x + radius + 1)
            local_patch = prob[local_y0:local_y1, local_x0:local_x1]
            local_patch_mass = float(local_patch.sum())
            local_patch = local_patch / max(local_patch_mass, 1e-30)
            local_xs = (np.arange(local_x0, local_x1, dtype=np.float64) + 0.5) * width / 64
            local_ys = (np.arange(local_y0, local_y1, dtype=np.float64) + 0.5) * height / 64
            local_uv = np.asarray([
                float(local_patch.sum(axis=0) @ local_xs),
                float(local_patch.sum(axis=1) @ local_ys),
            ], dtype=np.float64)
            local_position = unproject_to_plane(local_uv, fixed_height, world_from_camera, intrinsics)
            rows.append({
                **common,
                "checkpoint": name,
                "checkpoint_sha256": sha256(path),
                "pred_position_xy_m": decoded[:2].tolist(),
                "position_xy_error_mm": float(np.linalg.norm(decoded[:2] - gt_position[:2]) * 1000),
                "softargmax_pixel_uv": [float(v) for v in project_world(decoded[None, :3], world_from_camera, intrinsics)[0]],
                "heatmap_peak_pixel_uv": [heatmap_argmax["u_px"], heatmap_argmax["v_px"]],
                "heatmap_peak_pixel_distance_to_gt_px": peak_distance_px,
                "heatmap_argmax_position_xy_m": peak_position[:2].tolist(),
                "heatmap_argmax_position_xy_error_mm": float(
                    np.linalg.norm(peak_position[:2] - gt_position[:2]) * 1000),
                "local_softargmax_pixel_uv": local_uv.tolist(),
                "local_softargmax_position_xy_m": local_position[:2].tolist(),
                "local_softargmax_position_xy_error_mm": float(
                    np.linalg.norm(local_position[:2] - gt_position[:2]) * 1000),
                "local_patch_probability_mass": local_patch_mass,
                "heatmap_peak_probability": heatmap_argmax["probability"],
                "heatmap_entropy_normalized": entropy / np.log(prob.size),
                "heatmap_effective_cells": float(np.exp(entropy)),
                "heatmap_probability_in_target_roi": region_mass(prob, roi, width, height),
                "top5_heatmap_peaks": peaks,
            })

    csv_path = output / "heatmap_spatial_signature.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_commit = "unknown"
    summary = {
        "status": "completed",
        "purpose": "offline heatmap and target-event support diagnosis; no new training or rendering",
        "git_commit_at_run": git_commit,
        "diagnostic_script_sha256": sha256(Path(__file__).resolve()),
        "runtime_s": time.perf_counter() - started,
        "device": "cpu",
        "query_selection": {
            "source_field": args.query_source_field,
            "source_value": args.query_source_value,
            "time_field": args.query_time_field,
            "position_field": args.query_position_field,
            "x_field": args.query_x_field,
            "y_field": args.query_y_field,
            "z_field": args.query_z_field,
            "phase_field": args.query_phase_field,
            "phase_value": args.query_phase_value,
        },
        "roi_padding_px": args.roi_padding_px,
        "heatmap_temperature": args.temperature,
        "local_radius_cells": args.local_radius_cells,
        "inputs": {
            "events_sha256": sha256(args.events),
            "event_key": event_key,
            "queries_sha256": sha256(args.queries),
            "scene_config_sha256": sha256(args.scene_config),
            "event_count_in_archive": int(len(events)),
            "event_time_range_s": [float(events[:, 0].min()), float(events[:, 0].max())],
        },
        "checkpoints": [{
            "name": name,
            "path": str(path),
            "sha256": sha256(path),
            "architecture": metadata.get("architecture"),
        } for name, path, _, _, _, metadata in checkpoints],
        "rows": len(rows),
        "output_csv": csv_path.name,
        "diagnostic_boundary": "GT projects a post-inference cube ROI for counting only; the network input is the same causal five-bin event tensor.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in (csv_path, output / "summary.json")),
        encoding="utf-8")
    print(json.dumps({"status": summary["status"], "runtime_s": summary["runtime_s"],
                      "rows": summary["rows"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
