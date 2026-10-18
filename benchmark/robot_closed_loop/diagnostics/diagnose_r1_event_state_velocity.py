#!/usr/bin/env python3
"""Diagnose localization and velocity on two existing common Direct-HDR episodes.

The two checkpoints receive byte-identical raw event slices and query cutoffs.
GT is read only after inference for offline scoring. This is a development
diagnostic, not a formal test or generator ranking.
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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.event_state_model import (  # noqa: E402
    HeatmapEventStateNet,
    decode_heatmap_state,
    five_bin_tensor,
    load_checkpoint,
)
from benchmark.robot_closed_loop.predictor import FixedDecelerationPredictor, MotionState  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_hashes(root: Path) -> dict[str, str]:
    result = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        result[relative.lstrip("* ")] = digest
    return result


def read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def truth_at(trace: list[dict[str, Any]], time_s: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([float(row["time_s"]) for row in trace], dtype=np.float64)
    position = np.asarray([row["object"]["position_m"] for row in trace], dtype=np.float64)
    velocity = np.asarray([row["object"]["velocity_m_s"] for row in trace], dtype=np.float64)
    return tuple(np.asarray([np.interp(time_s, times, values[:, axis]) for axis in range(3)])
                 for values in (position, velocity))  # type: ignore[return-value]


def robust_pairwise_velocity(history: list[tuple[float, np.ndarray]], window_s: float,
                             min_samples: int, min_pair_dt_s: float) -> np.ndarray | None:
    if not history:
        return None
    newest = history[-1][0]
    samples = [(t, xy) for t, xy in history if t >= newest - window_s - 1e-12]
    if len(samples) < min_samples:
        return None
    slopes = [
        (samples[j][1] - samples[i][1]) / (samples[j][0] - samples[i][0])
        for i in range(len(samples)) for j in range(i + 1, len(samples))
        if samples[j][0] - samples[i][0] >= min_pair_dt_s - 1e-12
    ]
    if not slopes:
        return None
    return np.median(np.stack(slopes), axis=0)


def ols_velocity(history: list[tuple[float, np.ndarray]], window_s: float,
                 min_samples: int) -> np.ndarray | None:
    if not history:
        return None
    newest = history[-1][0]
    samples = [(t, xy) for t, xy in history if t >= newest - window_s - 1e-12]
    if len(samples) < min_samples:
        return None
    times = np.asarray([item[0] for item in samples], dtype=np.float64)
    positions = np.stack([item[1] for item in samples])
    centered_t = times - times.mean()
    denominator = float(centered_t @ centered_t)
    if denominator <= 1e-12:
        return None
    return centered_t @ positions / denominator


def image_uv(point_world: np.ndarray, camera_to_world: np.ndarray,
             intrinsics: list[float]) -> np.ndarray:
    fx, fy, cx, cy = (float(v) for v in intrinsics)
    camera = np.linalg.inv(camera_to_world) @ np.append(point_world, 1.0)
    return np.asarray([fx * camera[0] / camera[2] + cx,
                       fy * camera[1] / camera[2] + cy])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_diagnostic_image(path: Path, panels: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    scale = 5
    label_h = 38
    panel_w = 64 * scale
    panel_h = 64 * scale
    canvas = Image.new("RGB", (panel_w * 2, (panel_h + label_h) * len(panels)), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    for row_id, panel in enumerate(panels):
        y = row_id * (panel_h + label_h)
        event = np.log1p(panel["voxel"].sum(axis=(0, 1)))
        vmax = max(float(np.percentile(event, 99.5)), 1e-6)
        gray = np.clip(event / vmax * 255.0, 0, 255).astype(np.uint8)
        event_rgb = np.stack((gray, gray, gray), axis=-1)
        event_image = Image.fromarray(event_rgb).resize((panel_w, panel_h), Image.Resampling.NEAREST)
        canvas.paste(event_image, (0, y + label_h))
        heat = panel["heatmap"]
        heat = heat - float(np.max(heat))
        heat = np.exp(np.clip(heat, -40, 0))
        heat = heat / max(float(heat.max()), 1e-12)
        heat_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        heat_rgb[..., 0] = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
        heat_rgb[..., 1] = np.clip(np.sqrt(heat) * 150.0, 0, 255).astype(np.uint8)
        heat_image = Image.fromarray(heat_rgb).resize((panel_w, panel_h), Image.Resampling.NEAREST)
        canvas.paste(heat_image, (panel_w, y + label_h))
        draw.text((2, y + 4), f"{panel['label']} | events (left) / heatmap (right)", fill=(235, 238, 245))
        for xoff, point, color in ((panel_w, panel["gt_uv64"], (40, 220, 80)),
                                   (panel_w, panel["pred_uv64"], (255, 230, 30))):
            x = int(np.clip(point[0] * scale, 0, panel_w - 1))
            py = int(np.clip(point[1] * scale, 0, panel_h - 1)) + y + label_h
            draw.ellipse((xoff + x - 7, py - 7, xoff + x + 7, py + 7), outline=color, width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--ours-checkpoint", type=Path, required=True)
    parser.add_argument("--v2e-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    scene = json.loads(args.scene_config.read_text(encoding="utf-8"))
    reference = args.reference_root.resolve()
    expected_hashes = reference_hashes(reference)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(1)

    width, height = (int(v) for v in scene["camera"]["resolution_wh"])
    focal = height / (2.0 * np.tan(np.deg2rad(float(scene["camera"]["vertical_fov_deg"])) / 2.0))
    intrinsics = [focal, focal, width / 2.0, height / 2.0]
    camera = scene["camera"]
    forward = np.asarray(camera["lookat_m"], dtype=np.float64) - np.asarray(camera["position_m"], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = np.asarray(camera["up_m"], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = np.column_stack((right, -camera_up, forward))
    camera_to_world[:3, 3] = np.asarray(camera["position_m"], dtype=np.float64)
    fixed_height = float(scene["target"]["size_m"][2]) / 2.0

    checkpoint_paths = {"ours": args.ours_checkpoint.resolve(), "v2e": args.v2e_checkpoint.resolve()}
    models = {}
    checkpoints = {}
    for source, path in checkpoint_paths.items():
        model, label_mean, label_scale, checkpoint = load_checkpoint(str(path), device="cpu")
        if model.__class__ is not HeatmapEventStateNet:
            raise ValueError(f"expected heatmap_v2 checkpoint, got {checkpoint.get('architecture')}")
        calibration = checkpoint["spatial_calibration"]
        if not np.allclose(np.asarray(calibration["camera_to_world_cv"]), camera_to_world, atol=1e-5, rtol=0):
            raise ValueError(f"{source} checkpoint camera extrinsics do not match the common reference")
        if list(calibration["resolution_wh"]) != [width, height]:
            raise ValueError(f"{source} checkpoint resolution differs from the common reference")
        models[source] = (model, label_mean, label_scale, checkpoint)
        checkpoints[source] = {"sha256": sha256(path), "path": str(path)}

    predictor = FixedDecelerationPredictor(
        float(scene["controller"]["intercept_x_m"]),
        float(scene["controller"]["initial_deceleration_assumption_m_s2"]),
    )
    query = cfg["query"]
    first_poll = float(query["first_poll_s"])
    poll_period = float(query["poll_period_s"])
    latency = float(query["observation_latency_s"])
    strategy_cfg = cfg["velocity_strategies"]
    ols_window = float(strategy_cfg["existing_history_ols"]["window_s"])
    robust_window = float(strategy_cfg["robust_pairwise_median"]["window_s"])
    min_samples = int(strategy_cfg["existing_history_ols"]["minimum_samples"])
    min_pair_dt = float(strategy_cfg["robust_pairwise_median"]["minimum_pair_separation_s"])
    selected_id_to_name = {int(v): k for k, v in cfg["episodes"].items()}
    all_rows: list[dict[str, Any]] = []
    heatmap_panels: list[dict[str, Any]] = []
    selected_voxels: dict[str, np.ndarray] = {}
    provenance = []
    started = time.perf_counter()

    for episode_id, label in sorted(selected_id_to_name.items()):
        episode_dir = reference / "episodes" / f"episode_{episode_id:03d}"
        event_path = episode_dir / "direct_hdr_events_1khz_2khz.npz"
        trace_path = episode_dir / "state_control_trace.jsonl"
        csv_path = episode_dir / "episode.csv"
        for path in (event_path, trace_path, csv_path):
            relative = str(path.relative_to(reference))
            if sha256(path) != expected_hashes.get(relative):
                raise RuntimeError(f"source Direct-HDR artifact failed manifest verification: {relative}")
        events = np.asarray(np.load(event_path)["events_2000hz"], dtype=np.float64)
        events = events[np.argsort(events[:, 0], kind="stable")]
        trace = read_trace(trace_path)
        with csv_path.open("r", newline="", encoding="utf-8") as stream:
            episode = next(csv.DictReader(stream))
        contact_s = float(episode["first_contact_s"])
        polls = np.arange(first_poll, contact_s - 1e-9, poll_period)
        provenance.append({
            "episode_id": episode_id, "label": label, "seed": int(episode["seed"]),
            "speed_m_s": float(episode["launch_speed_m_s"]), "launch_y_m": float(episode["start_y_m"]),
            "contact_s": contact_s, "event_sha256": sha256(event_path), "trace_sha256": sha256(trace_path),
            "query_count": int(len(polls)),
        })
        histories = {source: [] for source in models}
        for now_s in polls:
            cutoff = round(float(now_s) - latency, 10)
            stop = int(np.searchsorted(events[:, 0], now_s, side="left"))
            truth_position, truth_velocity = truth_at(trace, cutoff)
            gt_prediction = predictor.predict(MotionState(cutoff, truth_position, truth_velocity), latency)
            for source, (model, label_mean, label_scale, checkpoint) in models.items():
                voxel = five_bin_tensor(events[:stop], cutoff, width, height)
                # The event tensor is reconstructed from the full causal file at each cutoff;
                # events after cutoff are excluded by five_bin_tensor itself.
                counts = voxel.sum(axis=(1, 2, 3))
                event_count = int(voxel.sum())
                bin_count = int(np.count_nonzero(counts))
                minimum_events = int(checkpoint.get("validity_policy", {}).get("minimum_events", 12))
                minimum_bins = int(checkpoint.get("validity_policy", {}).get("minimum_nonempty_5ms_bins", 3))
                fresh = event_count >= minimum_events and bin_count >= minimum_bins and latency <= 0.030
                row: dict[str, Any] = {
                    "episode_id": episode_id, "episode_label": label, "source": source,
                    "seed": int(episode["seed"]), "speed_m_s": float(episode["launch_speed_m_s"]),
                    "launch_y_m": float(episode["start_y_m"]), "poll_s": float(now_s), "cutoff_s": cutoff,
                    "state_available": fresh, "event_count_25ms": event_count, "nonempty_bins": bin_count,
                    "gt_predictor_reachable": bool(gt_prediction.reachable),
                    "gt_x_m": float(truth_position[0]), "gt_y_m": float(truth_position[1]),
                    "gt_vx_m_s": float(truth_velocity[0]), "gt_vy_m_s": float(truth_velocity[1]),
                }
                if not fresh:
                    row["state_status"] = "no_event" if event_count == 0 else "tracking_lost"
                    all_rows.append(row)
                    continue
                input_tensor = torch.as_tensor(np.log1p(voxel).reshape(1, 10, 64, 64), dtype=torch.float32)
                with torch.no_grad():
                    raw = model(input_tensor)
                    decoded = decode_heatmap_state(
                        raw, label_mean, label_scale,
                        torch.as_tensor(camera_to_world, dtype=torch.float32),
                        torch.as_tensor(intrinsics, dtype=torch.float32), (width, height), fixed_height_m=fixed_height,
                    )[0]
                state = decoded.cpu().numpy().astype(np.float64)
                if not np.isfinite(state).all():
                    row["state_available"] = False
                    row["state_status"] = "nonfinite"
                    all_rows.append(row)
                    continue
                position_xy = state[:2].copy()
                history = histories[source]
                history.append((cutoff, position_xy.copy()))
                history[:] = [item for item in history if item[0] >= cutoff - max(ols_window, robust_window) - 1e-12]
                velocities = {
                    "direct_network_head": state[3:5].copy(),
                    "existing_history_ols": ols_velocity(history, ols_window, min_samples),
                    "robust_pairwise_median": robust_pairwise_velocity(history, robust_window, min_samples, min_pair_dt),
                }
                row.update({
                    "state_status": "fresh", "pred_x_m": float(state[0]), "pred_y_m": float(state[1]),
                    "pred_z_m": float(state[2]), "position_error_xy_m": float(np.linalg.norm(position_xy - truth_position[:2])),
                    "image_uv_px": None,
                    "gt_image_uv_px": json.dumps(image_uv(truth_position, camera_to_world, intrinsics).tolist()),
                })
                for strategy, velocity in velocities.items():
                    row[f"{strategy}_vx_m_s"] = None if velocity is None else float(velocity[0])
                    row[f"{strategy}_vy_m_s"] = None if velocity is None else float(velocity[1])
                    row[f"{strategy}_available"] = velocity is not None
                    if velocity is None:
                        row[f"{strategy}_velocity_error_m_s"] = None
                        row[f"{strategy}_ttc_abs_error_s"] = None
                        row[f"{strategy}_prediction_reachable"] = None
                        continue
                    row[f"{strategy}_velocity_error_m_s"] = float(np.linalg.norm(velocity - truth_velocity[:2]))
                    pred_state = MotionState(cutoff, np.asarray([state[0], state[1], fixed_height]),
                                             np.asarray([velocity[0], velocity[1], 0.0]))
                    prediction = predictor.predict(pred_state, latency)
                    row[f"{strategy}_prediction_reachable"] = bool(prediction.reachable)
                    row[f"{strategy}_ttc_abs_error_s"] = (
                        abs(float(prediction.ttc_s) - float(gt_prediction.ttc_s))
                        if prediction.reachable and gt_prediction.reachable else None
                    )
                    if prediction.reachable and gt_prediction.reachable:
                        row[f"{strategy}_intercept_y_error_m"] = float(prediction.position_m[1] - gt_prediction.position_m[1])
                    else:
                        row[f"{strategy}_intercept_y_error_m"] = None
                all_rows.append(row)

                cutoff_ref = float(query["decision_cutoff_s"])
                if abs(cutoff - cutoff_ref) < 1e-8:
                    selected_voxels[f"episode_{episode_id:03d}_{source}"] = voxel
                    heatmap = raw[0][0].detach().cpu().numpy().astype(np.float32)
                    prob = torch.softmax(raw[0][0].reshape(-1) / 0.5, dim=0)
                    grid_x = (torch.arange(64, dtype=torch.float32) + 0.5) * (width / 64.0)
                    grid_y = (torch.arange(64, dtype=torch.float32) + 0.5) * (height / 64.0)
                    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
                    pred_uv = np.asarray([float((prob * gx.flatten()).sum()), float((prob * gy.flatten()).sum())])
                    gt_uv = image_uv(truth_position, camera_to_world, intrinsics)
                    heatmap_panels.append({
                        "label": f"{label} episode {episode_id}, {source}, cutoff {cutoff:.2f}s",
                        "voxel": voxel, "heatmap": heatmap,
                        "gt_uv64": [float(gt_uv[0] * 64 / width), float(gt_uv[1] * 64 / height)],
                        "pred_uv64": [float(pred_uv[0] * 64 / width), float(pred_uv[1] * 64 / height)],
                    })

    _write_csv(output / "per_query_velocity_comparison.csv", all_rows)
    np.savez_compressed(output / "selected_causal_five_bins.npz", **selected_voxels)
    _make_diagnostic_image(output / "localization_inputs_and_heatmaps.png", heatmap_panels)

    summary: dict[str, Any] = {
        "status": "PASS", "purpose": cfg["purpose"], "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(args.config.resolve()), "scene_config_sha256": sha256(args.scene_config.resolve()),
        "reference_manifest_sha256": sha256(reference / "SHA256SUMS"),
        "checkpoint": checkpoints, "episodes": provenance,
        "causal_input_contract": cfg["inference_contract"],
        "runtime_s": time.perf_counter() - started,
        "summary_by_episode_source": {},
        "velocity_strategies": strategy_cfg,
        "interpretation": "Velocity-only strategy comparison holds each model's position output fixed. It cannot repair a wrong heatmap localization and is not a grasp-success result.",
    }
    for episode_id, label in sorted(selected_id_to_name.items()):
        for source in models:
            rows = [r for r in all_rows if r["episode_id"] == episode_id and r["source"] == source]
            valid = [r for r in rows if r.get("state_available")]
            entry: dict[str, Any] = {"label": label, "polls": len(rows), "fresh_count": len(valid),
                                     "fresh_fraction": len(valid) / len(rows) if rows else None,
                                     "position_error_xy_mm": _percentiles([r["position_error_xy_m"] * 1000 for r in valid])}
            for strategy in strategy_cfg:
                available = [r for r in valid if r.get(f"{strategy}_available")]
                entry[strategy] = {
                    "available_count": len(available), "coverage_over_all_polls": len(available) / len(rows) if rows else None,
                    "velocity_error_m_s": _percentiles([r[f"{strategy}_velocity_error_m_s"] for r in available]),
                    "ttc_abs_error_ms": _percentiles([r[f"{strategy}_ttc_abs_error_s"] * 1000 for r in available if r.get(f"{strategy}_ttc_abs_error_s") is not None]),
                    "intercept_y_abs_error_mm": _percentiles([abs(r[f"{strategy}_intercept_y_error_m"]) * 1000 for r in available if r.get(f"{strategy}_intercept_y_error_m") is not None]),
                    "reachable_fraction_all_polls": sum(r.get(f"{strategy}_prediction_reachable") is True for r in available) / len(rows) if rows else None,
                }
            summary["summary_by_episode_source"][f"episode_{episode_id:03d}_{source}"] = entry

    summary_path = output / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    files = [p for p in sorted(output.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (output / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(output)}\n" for p in files), encoding="utf-8")
    print(json.dumps({"output": str(output), "episodes": list(selected_id_to_name.items()),
                      "summary_by_episode_source": summary["summary_by_episode_source"],
                      "runtime_s": summary["runtime_s"]}, indent=2))
    return 0


def _percentiles(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {"n": int(len(array)), "mean": float(np.mean(array)),
            "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95))}


if __name__ == "__main__":
    raise SystemExit(main())
