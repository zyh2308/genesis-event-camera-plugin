#!/usr/bin/env python3
"""Run a two-source, two-checkpoint crossover on exact saved robot states.

The native-HDR source is the frozen Genesis plugin fed its saved scene-linear
100 Hz endpoints. The Direct-HDR source is freshly rendered from every exact
logged state in the diagnostic interval. Each source has one DVS initialization
at the common first frame, then runs continuously; query windows never reset
the DVS. Physics is not advanced. GT and segmentation are used only after
inference for scoring and visualization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.robot_closed_loop.diagnostics.diagnose_r1_5_consistency import (
    _events_from_frames,
)
from benchmark.robot_closed_loop.diagnostics.diagnose_r1_native_hdr_replay import (
    infer_event_state,
    load_event_models,
    project_world,
    read_trace,
    render_state,
    verify_source_hashes,
)
from benchmark.robot_closed_loop.diagnostics.diagnose_r1_ours_localization_cadence import (
    generate_sparse_plugin,
    response_config,
)
from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import (
    _resolve_motion_bodies,
    _sha256,
)
from benchmark.robot_closed_loop.event_state_model import five_bin_tensor
from benchmark.robot_closed_loop.robot_env import build_robot_scene
from genesis_event_plugin.genesis_adapter import camera_to_world_cv


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _trace_index(trace: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    result: dict[float, dict[str, Any]] = {}
    for row in trace:
        key = round(float(row["time_s"]), 9)
        if key in result:
            raise ValueError(f"duplicate state timestamp {key}")
        result[key] = row
    return result


def _same_state_rows(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if not np.allclose(a["robot_qpos"], b["robot_qpos"], atol=1e-9, rtol=0):
        return False
    for field in ("position_m", "orientation_wxyz", "velocity_m_s"):
        if not np.allclose(a["object"][field], b["object"][field], atol=1e-9, rtol=0):
            return False
    return True


def _write_h5(path: Path, frames: list[dict[str, Any]], pose_ids: np.ndarray,
              camera_pose: np.ndarray, fov_deg: float, resolution_wh: tuple[int, int],
              provenance: str) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["provenance"] = provenance
        handle.attrs["camera_to_world_cv"] = camera_pose
        handle.attrs["vertical_fov_deg"] = float(fov_deg)
        handle.attrs["resolution_wh"] = np.asarray(resolution_wh, dtype=np.int32)
        handle.attrs["segmentation_pose_ids"] = pose_ids.astype(np.int32)
        for name, array in (
            ("timestamps_s", np.asarray([f["time_s"] for f in frames], dtype=np.float64)),
            ("radiance_rgb", np.stack([f["hdr_rgb"] for f in frames]).astype(np.float32)),
            ("radiance_luma", np.stack([f["radiance_luma"] for f in frames]).astype(np.float32)),
            ("display_rgb_post_inference_only", np.stack([f["display_rgb"] for f in frames]).astype(np.uint8)),
            ("depth_post_inference_only", np.stack([f["depth"] for f in frames]).astype(np.float32)),
            ("segmentation_post_inference_only", np.stack([f["segmentation"] for f in frames]).astype(np.int32)),
        ):
            handle.create_dataset(name, data=array, compression="gzip", compression_opts=1, shuffle=True)


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _heatmap(model_record: Any, voxel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model = model_record[0]
    model.eval()
    net_input = np.log1p(np.asarray(voxel, dtype=np.float32)).reshape(1, 10, 64, 64)
    with torch.no_grad():
        raw = model(torch.as_tensor(net_input, dtype=torch.float32))
        logits = raw[0][0].detach().cpu().numpy().astype(np.float32)
        probability = torch.softmax(raw[0][0].reshape(-1) / 0.5, dim=0).reshape(64, 64)
        probability = probability.detach().cpu().numpy().astype(np.float32)
    return logits, probability


def _target_roi_probability(probability: np.ndarray, center_m: np.ndarray,
                            target_size_m: np.ndarray, z_m: float,
                            camera_pose: np.ndarray, intrinsics: np.ndarray,
                            image_wh: tuple[int, int]) -> float:
    corners = []
    for dx in (-0.5, 0.5):
        for dy in (-0.5, 0.5):
            point = np.asarray([center_m[0] + dx * target_size_m[0],
                                center_m[1] + dy * target_size_m[1], z_m], dtype=np.float64)
            corners.append(project_world(point, camera_pose, intrinsics))
    corners_px = np.asarray(corners)
    width, height = image_wh
    u0, v0 = corners_px.min(axis=0)
    u1, v1 = corners_px.max(axis=0)
    grid_u = (np.arange(64, dtype=np.float64) + 0.5) * width / 64.0
    grid_v = (np.arange(64, dtype=np.float64) + 0.5) * height / 64.0
    yy, xx = np.meshgrid(grid_v, grid_u, indexing="ij")
    mask = (xx >= u0) & (xx <= u1) & (yy >= v0) & (yy <= v1)
    return float(probability[mask].sum()) if mask.any() else 0.0


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    interior = value.copy()
    interior[1:] &= value[:-1]
    interior[:-1] &= value[1:]
    interior[:, 1:] &= value[:, :-1]
    interior[:, :-1] &= value[:, 1:]
    return value & ~interior


def _context_panel(frame: dict[str, Any], probability: np.ndarray, row: dict[str, Any],
                   title: str, target_ids: list[int], robot_ids: list[int]) -> Image.Image:
    display = Image.fromarray(np.asarray(frame["display_rgb"], dtype=np.uint8), mode="RGB")
    width, height = display.size
    heat = Image.fromarray(np.uint8(np.clip(probability / max(float(probability.max()), 1e-12), 0, 1) * 255))
    heat = heat.resize((width, height), Image.Resampling.BILINEAR)
    heat_a = np.asarray(heat, dtype=np.float32) / 255.0
    heat_rgb = np.stack((heat_a, 0.1 * heat_a, 0.85 * heat_a), axis=-1)
    base = np.asarray(display, dtype=np.float32) / 255.0
    composite = np.uint8(np.clip((0.72 * base + 0.28 * heat_rgb) * 255, 0, 255))
    canvas = Image.fromarray(composite, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    segmentation = np.asarray(frame["segmentation"], dtype=np.int32)
    target_boundary = _mask_boundary(np.isin(segmentation, target_ids))
    robot_boundary = _mask_boundary(np.isin(segmentation, robot_ids))
    yy, xx = np.nonzero(target_boundary)
    draw.point(list(zip(xx.tolist(), yy.tolist())), fill=(255, 230, 0))
    yy, xx = np.nonzero(robot_boundary)
    draw.point(list(zip(xx.tolist(), yy.tolist())), fill=(0, 230, 255))
    gu, gv = float(row["gt_u_px"]), float(row["gt_v_px"])
    pu, pv = float(row["pred_u_px"]), float(row["pred_v_px"])
    ku, kv = float(row["peak_u_px"]), float(row["peak_v_px"])
    draw.ellipse((gu - 4, gv - 4, gu + 4, gv + 4), outline=(255, 255, 255), width=2)
    draw.line((pu - 5, pv, pu + 5, pv), fill=(255, 50, 255), width=2)
    draw.line((pu, pv - 5, pu, pv + 5), fill=(255, 50, 255), width=2)
    draw.ellipse((ku - 3, kv - 3, ku + 3, kv + 3), outline=(255, 50, 50), width=2)
    panel = Image.new("RGB", (width, height + 28), (18, 20, 24))
    panel.paste(canvas, (0, 28))
    d = ImageDraw.Draw(panel)
    d.text((4, 7), f"{title} | {row['query_s']:.4f}s | XY {row['position_error_xy_mm']:.1f}mm", fill=(245, 245, 245))
    return panel


def _write_contact_sheet(path: Path, episode: str, frames_by_time: dict[float, dict[str, Any]],
                         context_rows: list[dict[str, Any]], probs: dict[tuple[str, str, float], np.ndarray],
                         config_times: list[float], target_ids: list[int], robot_ids: list[int]) -> None:
    columns = [
        ("ours_native_hdr", "ours_native_hdr"),
        ("ours_native_hdr", "v2e"),
        ("direct_hdr", "ours_native_hdr"),
        ("direct_hdr", "v2e"),
    ]
    panel_w, panel_h = 320, 268
    canvas = Image.new("RGB", (panel_w * len(columns), 24 + panel_h * len(config_times)), (12, 14, 17))
    draw = ImageDraw.Draw(canvas)
    for col, (source, model) in enumerate(columns):
        draw.text((col * panel_w + 4, 4), f"{source} → {model}", fill=(250, 250, 250))
    for ridx, query_time in enumerate(config_times):
        frame_key = round(float(query_time), 9)
        frame = frames_by_time[frame_key]
        for col, (source, model) in enumerate(columns):
            row = next(r for r in context_rows if r["source"] == source and r["checkpoint"] == model
                       and round(float(r["query_s"]), 9) == frame_key)
            panel = _context_panel(frame, probs[(source, model, frame_key)], row,
                                   f"{episode} {source}/{model}", target_ids, robot_ids)
            canvas.paste(panel, (col * panel_w, 24 + ridx * panel_h))
    canvas.save(path)


def _metric(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(array.size), "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
            "max": float(array.max())}


def _write_manifest(root: Path) -> None:
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        entries.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path,
                        help="optional full-episode reference root; otherwise use the exact saved trace_window rows")
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--ours-checkpoint", type=Path, required=True)
    parser.add_argument("--v2e-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()

    config = _json(args.config)
    scene_cfg = _json(args.scene_config)
    scene_cfg["shared_control"] = _json(args.control_config)
    reference_root = args.reference_root.resolve() if args.reference_root else None
    cache_root = args.source_cache_root.resolve()
    output = args.output.resolve()
    code_root = args.code_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    start_s, end_s = map(float, config["trace_time_s"])
    query_start_s, query_end_s = map(float, config["query_time_s"])
    dt_endpoint = 0.01
    endpoint_count = int(round((end_s - start_s) / dt_endpoint)) + 1
    endpoint_times = np.asarray([round(start_s + i * dt_endpoint, 9) for i in range(endpoint_count)])
    width, height = map(int, scene_cfg["camera"]["resolution_wh"])
    fov = float(scene_cfg["camera"]["vertical_fov_deg"])
    focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    intrinsics = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)

    prepared: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, Any] = {
        "config_sha256": _sha256(args.config),
        "scene_config_sha256": _sha256(args.scene_config),
        "control_config_sha256": _sha256(args.control_config),
        "episodes": {},
        "checkpoints": {
            "ours_native_hdr": _sha256(args.ours_checkpoint),
            "v2e": _sha256(args.v2e_checkpoint),
        },
    }
    for label, ep_spec in config["episodes"].items():
        ep_id = int(ep_spec["episode_id"])
        source_ep = cache_root / f"episode_{ep_id:03d}"
        cache_path = source_ep / "same_state_source_cache.h5"
        trace_window_path = source_ep / "trace_window.jsonl"
        diagnostic_path = source_ep / "episode_diagnostic.json"
        source_hashes = verify_source_hashes(cache_root, [cache_path, trace_window_path, diagnostic_path])
        source_trace = read_trace(trace_window_path)
        source_index = _trace_index(source_trace)
        diagnostic = _json(diagnostic_path)
        expected_hashes = ep_spec["expected_source_sha256"]
        actual_window_hash = source_hashes[f"episode_{ep_id:03d}/trace_window.jsonl"]
        actual_cache_hash = source_hashes[f"episode_{ep_id:03d}/same_state_source_cache.h5"]
        declared_trace_hash = diagnostic["reference_inputs_sha256"][
            f"episodes/episode_{ep_id:03d}/state_control_trace.jsonl"]
        declared_episode_hash = diagnostic["reference_inputs_sha256"][
            f"episodes/episode_{ep_id:03d}/episode.csv"]
        if actual_window_hash != expected_hashes["trace_window.jsonl"]:
            raise RuntimeError(f"episode_{ep_id:03d}: saved state-window hash differs from the locked config")
        if actual_cache_hash != expected_hashes["same_state_source_cache.h5"]:
            raise RuntimeError(f"episode_{ep_id:03d}: native-HDR cache hash differs from the locked config")
        if declared_trace_hash != expected_hashes["declared_full_state_trace.jsonl"]:
            raise RuntimeError(f"episode_{ep_id:03d}: diagnostic metadata declares an unexpected full-trace hash")
        if declared_episode_hash != expected_hashes["declared_episode.csv"]:
            raise RuntimeError(f"episode_{ep_id:03d}: diagnostic metadata declares unexpected episode metadata")

        ref_ep = reference_root / "episodes" / f"episode_{ep_id:03d}" if reference_root else None
        trace_path = ref_ep / "state_control_trace.jsonl" if ref_ep else None
        episode_path = ref_ep / "episode.csv" if ref_ep else None
        has_full_trace = bool(trace_path and trace_path.is_file())
        has_episode_csv = bool(episode_path and episode_path.is_file())
        if has_full_trace:
            full_trace = read_trace(trace_path)
            actual_full_trace_hash = _sha256(trace_path)
            if actual_full_trace_hash != declared_trace_hash:
                raise RuntimeError(f"episode_{ep_id:03d}: full trace hash differs from diagnostic metadata")
            trace_provenance = "full state_control_trace.jsonl; selected rows cross-checked against trace_window.jsonl"
        else:
            # This exact 70 ms state excerpt contains all real rows needed by this diagnostic.
            # Row indices stay explicitly local to the excerpt; the full episode is not implied.
            full_trace = source_trace
            actual_full_trace_hash = None
            trace_provenance = (
                "exact saved trace_window.jsonl rows only; full episode trace not staged; "
                "declared full-trace hash retained from episode_diagnostic.json"
            )
        full_index = _trace_index(full_trace)
        selected = [row for row in full_trace if start_s - 1e-9 <= float(row["time_s"]) <= end_s + 1e-9]
        selected_times = [round(float(row["time_s"]), 9) for row in selected]
        if not selected or selected_times[0] != round(start_s, 9) or selected_times[-1] != round(end_s, 9):
            raise RuntimeError(f"episode_{ep_id:03d}: saved trace rows do not contain exact interval endpoints")
        if has_full_trace and any(t not in source_index or not _same_state_rows(row, source_index[t]) for t, row in
                                  ((round(float(r["time_s"]), 9), r) for r in selected)):
            raise RuntimeError(f"episode_{ep_id:03d}: cropped rows do not exactly match saved diagnostic trace")
        query_rows = [r for r in selected if query_start_s - 1e-9 <= float(r["time_s"]) <= query_end_s + 1e-9]
        query_times = [round(float(r["time_s"]), 9) for r in query_rows]
        full_trace_row_index = {round(float(r["time_s"]), 9): i for i, r in enumerate(full_trace)}
        if not query_times or query_times[0] != round(query_start_s, 9) or query_times[-1] != round(query_end_s, 9):
            raise RuntimeError(f"episode_{ep_id:03d}: exact query interval endpoints are absent")
        if any(round(t - 0.025, 9) < round(start_s, 9) - 1e-9 for t in query_times):
            raise RuntimeError("a causal query window extends before the shared source initialization")
        if has_episode_csv:
            actual_episode_hash = _sha256(episode_path)
            if actual_episode_hash != declared_episode_hash:
                raise RuntimeError(f"episode_{ep_id:03d}: episode.csv hash differs from diagnostic metadata")
            with episode_path.open(newline="", encoding="utf-8") as stream:
                episode_row = next(csv.DictReader(stream))
        else:
            actual_episode_hash = None
            episode_row = {
                "seed": diagnostic["seed"],
                "launch_speed_m_s": diagnostic["speed_m_s"],
                "start_y_m": diagnostic["launch_y_m"],
                "first_contact_s": diagnostic["first_contact_s"],
            }
        if int(episode_row["seed"]) != int(ep_spec["seed"]):
            raise RuntimeError(f"episode_{ep_id:03d}: seed does not match selected trajectory")
        if not np.isclose(float(episode_row["launch_speed_m_s"]), float(ep_spec["speed_m_s"]), atol=1e-8):
            raise RuntimeError(f"episode_{ep_id:03d}: speed does not match selected trajectory")
        if not np.isclose(float(episode_row["start_y_m"]), float(ep_spec["launch_y_m"]), atol=1e-8):
            raise RuntimeError(f"episode_{ep_id:03d}: lateral start does not match selected trajectory")
        with h5py.File(cache_path, "r") as cache:
            cache_times = np.asarray(cache["timestamps_s"][:], dtype=np.float64)
            missing = [float(t) for t in endpoint_times if not np.any(np.isclose(cache_times, t, atol=1e-9, rtol=0))]
            if missing:
                raise RuntimeError(f"episode_{ep_id:03d}: native HDR cache lacks exact 100 Hz endpoints {missing}")
            cache_by_time: dict[float, dict[str, Any]] = {}
            pose_ids = np.asarray(cache["segmentation_pose_ids"][:], dtype=np.int32)
            for i, t in enumerate(cache_times):
                key = round(float(t), 9)
                if key < start_s - 1e-9 or key > end_s + 1e-9:
                    continue
                cache_by_time[key] = {
                    "time_s": key,
                    "hdr_rgb": np.asarray(cache["radiance"][i], dtype=np.float32),
                    "radiance_luma": np.asarray(cache["radiance_luma"][i], dtype=np.float32),
                    "display_rgb": np.asarray(cache["display_rgb"][i], dtype=np.uint8),
                    "depth": np.asarray(cache["depth"][i], dtype=np.float32),
                    "segmentation": np.asarray(cache["segmentation"][i], dtype=np.int32),
                    "poses": {int(seg_id): np.asarray(cache["object_poses"][i, j], dtype=np.float64)
                              for j, seg_id in enumerate(pose_ids)},
                }
            cached_camera = np.asarray(cache["camera_to_world_cv"][:], dtype=np.float64)
            cache_attrs = {
                "camera_resolution_wh": tuple(int(x) for x in cache.attrs["camera_resolution_wh"]),
                "vertical_fov_deg": float(cache.attrs["vertical_fov_deg"]),
            }
        input_hashes["episodes"][label] = {
            "episode_id": ep_id,
            "seed": int(ep_spec["seed"]),
            "speed_m_s": float(ep_spec["speed_m_s"]),
            "launch_y_m": float(ep_spec["launch_y_m"]),
            "declared_full_trace_sha256": declared_trace_hash,
            "full_trace_sha256_actual_file_present": actual_full_trace_hash,
            "declared_episode_csv_sha256": declared_episode_hash,
            "episode_csv_sha256_actual_file_present": actual_episode_hash,
            "trace_provenance": trace_provenance,
            "trace_window_sha256": actual_window_hash,
            "native_hdr_source_cache_sha256": actual_cache_hash,
            "episode_diagnostic_sha256": source_hashes[f"episode_{ep_id:03d}/episode_diagnostic.json"],
            "query_times_s": query_times,
            "trace_window_row_count": len(source_trace),
            "trace_rows_in_render_interval": len(selected),
            "trace_rows_verified_equal_to_full_trace": len(selected) if has_full_trace else None,
        }
        prepared[label] = {
            "episode_id": ep_id, "full_trace": full_trace, "trace_rows": selected,
            "query_rows": query_rows, "query_times": query_times, "cache_frames": cache_by_time,
            "pose_ids": pose_ids, "cached_camera": cached_camera, "cache_attrs": cache_attrs,
            "episode_row": episode_row, "full_trace_row_index": full_trace_row_index,
            "trace_provenance": trace_provenance,
        }

    output.mkdir(parents=True, exist_ok=False)
    (output / "visualizations").mkdir()
    (output / "episodes").mkdir()
    config_copy = output / "effective_config.json"
    config_copy.write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")

    import genesis as gs

    gs.init(backend=gs.cpu, logging_level="warning")
    env = None
    all_query_rows: list[dict[str, Any]] = []
    episode_summary: dict[str, Any] = {}
    render_started_all = time.perf_counter()
    try:
        env = build_robot_scene(scene_cfg, dt_s=float(scene_cfg["physics"]["dt_s"]), segmentation_level="link")
        env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True, force_render=True)
        bodies = _resolve_motion_bodies(env)
        camera_pose = np.asarray(camera_to_world_cv(env.camera), dtype=np.float64)
        target_ids = [seg_id for seg_id, body in bodies.items() if body is env.target]
        robot_ids = [seg_id for seg_id, body in bodies.items() if body is not env.target]
        target_size = np.asarray(scene_cfg["target"]["size_m"], dtype=np.float64)
        fixed_height = float(target_size[2]) / 2.0
        models, checkpoint_hashes = load_event_models(
            {"ours_native_hdr": args.ours_checkpoint.resolve(), "v2e": args.v2e_checkpoint.resolve()},
            camera_pose, intrinsics, (width, height),
        )
        input_hashes["checkpoint_sha256_verified_by_model_loader"] = checkpoint_hashes
        for model_record in models.values():
            model_record[0].eval()

        for label, pre in prepared.items():
            ep_id = int(pre["episode_id"])
            episode_out = output / "episodes" / f"episode_{ep_id:03d}_{label}"
            episode_out.mkdir()
            if tuple(pre["cache_attrs"]["camera_resolution_wh"]) != (width, height):
                raise RuntimeError(f"episode_{ep_id:03d}: cache resolution differs from scene config")
            if not np.isclose(pre["cache_attrs"]["vertical_fov_deg"], fov, atol=1e-8, rtol=0):
                raise RuntimeError(f"episode_{ep_id:03d}: cache FOV differs from scene config")
            if not np.allclose(pre["cached_camera"], camera_pose[None], atol=1e-5, rtol=0):
                raise RuntimeError(f"episode_{ep_id:03d}: cached camera pose differs from rendered scene")
            if not np.array_equal(pre["pose_ids"], np.asarray(sorted(bodies), dtype=np.int32)):
                raise RuntimeError(f"episode_{ep_id:03d}: cached scene body IDs differ from rendered scene")

            direct_frames: list[dict[str, Any]] = []
            endpoint_checks: list[dict[str, Any]] = []
            for row in pre["trace_rows"]:
                frame = render_state(env, row, bodies)
                key = round(float(row["time_s"]), 9)
                if key in pre["cache_frames"]:
                    cached = pre["cache_frames"][key]
                    rgb_diff = np.abs(frame["hdr_rgb"] - cached["hdr_rgb"])
                    depth = np.asarray(frame["depth"], dtype=np.float32)
                    cache_depth = np.asarray(cached["depth"], dtype=np.float32)
                    common = np.isfinite(depth) & np.isfinite(cache_depth)
                    endpoint_checks.append({
                        "time_s": key,
                        "radiance_max_abs_error": float(rgb_diff.max()),
                        "radiance_mean_abs_error": float(rgb_diff.mean()),
                        "depth_finite_mask_mismatch": int(np.count_nonzero(np.isfinite(depth) != np.isfinite(cache_depth))),
                        "segmentation_mismatch_pixels": int(np.count_nonzero(frame["segmentation"] != cached["segmentation"])),
                    })
                direct_frames.append(frame)
            direct_times = [round(float(f["time_s"]), 9) for f in direct_frames]
            if direct_times != [round(float(r["time_s"]), 9) for r in pre["trace_rows"]]:
                raise RuntimeError("direct frame timestamps are not the exact logged state rows")
            direct_by_time = {round(float(f["time_s"]), 9): f for f in direct_frames}
            endpoint_frames = [pre["cache_frames"][round(float(t), 9)] for t in endpoint_times]
            direct_initial_luma = direct_frames[0]["radiance_luma"]
            native_initial_luma = endpoint_frames[0]["radiance_luma"]
            initial_frame_error = float(np.max(np.abs(direct_initial_luma - native_initial_luma)))
            if initial_frame_error > 1e-6:
                raise RuntimeError(f"the two sources do not share the same first-frame initialization: {initial_frame_error}")
            bad_endpoint_checks = [r for r in endpoint_checks
                                   if r["radiance_max_abs_error"] > 1e-6
                                   or r["depth_finite_mask_mismatch"] != 0
                                   or r["segmentation_mismatch_pixels"] != 0]
            if bad_endpoint_checks:
                raise RuntimeError(f"saved HDR endpoints do not replay exactly in the fixed scene: {bad_endpoint_checks}")

            pose_ids = pre["pose_ids"]
            native_events, native_intervals, native_event_s = generate_sparse_plugin(
                endpoint_frames, pose_ids, camera_pose, width, height, fov,
            )
            direct_lumas = [np.asarray(frame["radiance_luma"], dtype=np.float32) for frame in direct_frames]
            direct_events, direct_event_s = _events_from_frames(direct_lumas, direct_times)
            native_events = np.asarray(native_events, dtype=np.float64)
            direct_events = np.asarray(direct_events, dtype=np.float64)
            np.savez_compressed(episode_out / "raw_events_ours_native_hdr.npz", events=native_events)
            np.savez_compressed(episode_out / "raw_events_direct_hdr.npz", events=direct_events)
            _write_h5(
                episode_out / "native_hdr_endpoints.h5", endpoint_frames, pose_ids,
                camera_pose, fov, (width, height),
                "exact cached native Genesis HDR 100 Hz endpoints used by the frozen plugin; GT geometry is generator-internal only",
            )
            _write_h5(
                episode_out / "direct_hdr_exact_state_frames.h5", direct_frames, pose_ids,
                camera_pose, fov, (width, height),
                "fresh Genesis HDR renders from exact logged robot and target states; display/depth/segmentation are post-inference diagnostics only",
            )

            arrays: dict[str, np.ndarray] = {"query_times_s": np.asarray(pre["query_times"], dtype=np.float64)}
            heatmap_probs: dict[tuple[str, str, float], np.ndarray] = {}
            episode_rows: list[dict[str, Any]] = []
            stream_by_name = {"ours_native_hdr": native_events, "direct_hdr": direct_events}
            start_infer = time.perf_counter()
            for source_name, stream in stream_by_name.items():
                for query_idx, query_row in enumerate(pre["query_rows"]):
                    query_time = round(float(query_row["time_s"]), 9)
                    gt_position = np.asarray(query_row["object"]["position_m"], dtype=np.float64)
                    gt_velocity = np.asarray(query_row["object"]["velocity_m_s"], dtype=np.float64)
                    gt_uv = project_world(gt_position, camera_pose, intrinsics)
                    for model_name, model_record in models.items():
                        estimate, voxel = infer_event_state(
                            stream, model_record, query_time, width, height,
                            camera_pose, intrinsics, fixed_height,
                        )
                        if not estimate.get("state_available"):
                            raise RuntimeError(f"invalid model state for {label}/{source_name}/{model_name}/{query_time}")
                        model_input = np.log1p(np.asarray(voxel, dtype=np.float32)).reshape(10, 64, 64)
                        logits, probability = _heatmap(model_record, voxel)
                        pred_xyz = np.asarray(estimate["position_xyz_m"], dtype=np.float64)
                        pred_uv = np.asarray(estimate["heatmap_uv_px"], dtype=np.float64)
                        grid_u = (np.arange(64, dtype=np.float64) + 0.5) * width / 64.0
                        grid_v = (np.arange(64, dtype=np.float64) + 0.5) * height / 64.0
                        peak_v_idx, peak_u_idx = np.unravel_index(int(np.argmax(probability)), probability.shape)
                        peak_uv = np.asarray([grid_u[peak_u_idx], grid_v[peak_v_idx]], dtype=np.float64)
                        err_px = float(np.linalg.norm(pred_uv - gt_uv))
                        err_mm = float(np.linalg.norm(pred_xyz[:2] - gt_position[:2]) * 1000.0)
                        entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-12)))) / float(np.log(probability.size))
                        roi_mass = _target_roi_probability(
                            probability, gt_position, target_size, fixed_height,
                            camera_pose, intrinsics, (width, height),
                        )
                        row_out = {
                            "episode": label, "episode_id": ep_id, "query_state_index": query_idx,
                            "trace_window_row_index": int(pre["full_trace_row_index"][query_time]),
                            "query_s": query_time, "query_state_exact": True,
                            "source": source_name, "checkpoint": model_name,
                            "state_available": True,
                            "event_count_25ms": int(estimate["event_count_25ms"]),
                            "nonempty_5ms_bins": int(estimate["nonempty_5ms_bins"]),
                            "input_counts_sha256": estimate["input_tensor_sha256"],
                            "model_input_log1p_sha256": _array_sha(model_input),
                            "gt_u_px": float(gt_uv[0]), "gt_v_px": float(gt_uv[1]),
                            "pred_u_px": float(pred_uv[0]), "pred_v_px": float(pred_uv[1]),
                            "peak_u_px": float(peak_uv[0]), "peak_v_px": float(peak_uv[1]),
                            "pixel_error_px": err_px,
                            "planar_error_xy_mm": err_mm,
                            "gt_x_mm": float(gt_position[0] * 1000), "gt_y_mm": float(gt_position[1] * 1000),
                            "pred_x_mm": float(pred_xyz[0] * 1000), "pred_y_mm": float(pred_xyz[1] * 1000),
                            "gt_vx_m_s": float(gt_velocity[0]), "gt_vy_m_s": float(gt_velocity[1]),
                            "pred_vx_m_s": float(estimate["velocity_xy_m_s"][0]),
                            "pred_vy_m_s": float(estimate["velocity_xy_m_s"][1]),
                            "velocity_error_xy_m_s": float(np.linalg.norm(np.asarray(estimate["velocity_xy_m_s"]) - gt_velocity[:2])),
                            "heatmap_peak_distance_to_gt_px": float(np.linalg.norm(peak_uv - gt_uv)),
                            "heatmap_entropy_normalized": entropy,
                            "heatmap_peak_probability": float(probability.max()),
                            "heatmap_probability_in_gt_target_roi": roi_mass,
                        }
                        episode_rows.append(row_out)
                        all_query_rows.append(row_out)
                        key = f"{source_name}__{model_name}"
                        arrays.setdefault(f"{key}__voxel_counts", np.zeros(
                            (len(pre["query_rows"]), 5, 2, 64, 64), dtype=np.float32))
                        arrays.setdefault(f"{key}__model_input_log1p", np.zeros(
                            (len(pre["query_rows"]), 10, 64, 64), dtype=np.float32))
                        arrays.setdefault(f"{key}__heatmap_logits", np.zeros(
                            (len(pre["query_rows"]), 64, 64), dtype=np.float32))
                        arrays.setdefault(f"{key}__heatmap_probability", np.zeros(
                            (len(pre["query_rows"]), 64, 64), dtype=np.float32))
                        arrays[f"{key}__voxel_counts"][query_idx] = voxel
                        arrays[f"{key}__model_input_log1p"][query_idx] = model_input
                        arrays[f"{key}__heatmap_logits"][query_idx] = logits
                        arrays[f"{key}__heatmap_probability"][query_idx] = probability
                        heatmap_probs[(source_name, model_name, query_time)] = probability
            inference_s = time.perf_counter() - start_infer
            np.savez_compressed(episode_out / "four_group_inputs_heatmaps.npz", **arrays)
            csv_path = episode_out / "four_group_queries.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
                writer.writeheader()
                writer.writerows(episode_rows)

            context_times = [float(t) for t in config.get("visualization_query_times_s", [])]
            context_times = [round(t, 9) for t in context_times]
            if context_times:
                if any(t not in direct_by_time or t not in pre["query_times"] for t in context_times):
                    raise RuntimeError("visualization cutoffs must be exact rendered/query state rows")
                _write_contact_sheet(
                    output / "visualizations" / f"episode_{ep_id:03d}_{label}_spatial_overlay.png",
                    label, direct_by_time, episode_rows, heatmap_probs,
                    context_times, target_ids, robot_ids,
                )

            groups = {}
            for source_name in stream_by_name:
                for model_name in models:
                    subset = [r for r in episode_rows if r["source"] == source_name and r["checkpoint"] == model_name]
                    errors = [float(r["planar_error_xy_mm"]) for r in subset]
                    groups[f"{source_name}__{model_name}"] = {
                        "queries": len(subset), "state_valid_fraction": float(np.mean([r["state_available"] for r in subset])),
                        "planar_error_xy_mm": _metric(errors),
                        "pixel_error_px": _metric([float(r["pixel_error_px"]) for r in subset]),
                        "velocity_error_xy_m_s": _metric([float(r["velocity_error_xy_m_s"]) for r in subset]),
                        "first_query_over_25mm_s": next((r["query_s"] for r in subset if r["planar_error_xy_mm"] > 25), None),
                        "first_query_over_50mm_s": next((r["query_s"] for r in subset if r["planar_error_xy_mm"] > 50), None),
                        "first_query_over_100mm_s": next((r["query_s"] for r in subset if r["planar_error_xy_mm"] > 100), None),
                        "error_start_state": next(({
                            "query_s": r["query_s"], "gt_uv_px": [r["gt_u_px"], r["gt_v_px"]],
                            "pred_uv_px": [r["pred_u_px"], r["pred_v_px"]],
                            "peak_uv_px": [r["peak_u_px"], r["peak_v_px"]],
                            "planar_error_xy_mm": r["planar_error_xy_mm"],
                        } for r in subset if r["planar_error_xy_mm"] > 50), None),
                    }
            source_deltas = {}
            for model_name in models:
                native = {r["query_s"]: float(r["planar_error_xy_mm"]) for r in episode_rows
                          if r["source"] == "ours_native_hdr" and r["checkpoint"] == model_name}
                direct = {r["query_s"]: float(r["planar_error_xy_mm"]) for r in episode_rows
                          if r["source"] == "direct_hdr" and r["checkpoint"] == model_name}
                delta = np.asarray([direct[t] - native[t] for t in pre["query_times"]], dtype=np.float64)
                source_deltas[model_name] = {
                    "paired_queries": int(delta.size),
                    "direct_minus_ours_native_error_mm_mean": float(delta.mean()),
                    "direct_minus_ours_native_error_mm_p50": float(np.percentile(delta, 50)),
                    "direct_worse_query_count": int(np.count_nonzero(delta > 0)),
                    "direct_better_query_count": int(np.count_nonzero(delta < 0)),
                }
            episode_summary[label] = {
                "episode_id": ep_id,
                "seed": int(pre["episode_row"]["seed"]),
                "speed_m_s": float(pre["episode_row"]["launch_speed_m_s"]),
                "launch_y_m": float(pre["episode_row"]["start_y_m"]),
                "first_contact_s": float(pre["episode_row"]["first_contact_s"]),
                "rendered_state_rows": len(direct_frames),
                "render_state_time_interval_s": {"min": float(np.diff(direct_times).min()),
                                                  "median": float(np.median(np.diff(direct_times))),
                                                  "max": float(np.diff(direct_times).max())},
                "query_times_s": pre["query_times"],
                "query_count_per_group": len(pre["query_times"]),
                "trace_provenance": pre["trace_provenance"],
                "native_source_event_count": int(len(native_events)),
                "direct_source_event_count": int(len(direct_events)),
                "native_source_event_time_range_s": [float(native_events[:, 0].min()), float(native_events[:, 0].max())] if len(native_events) else None,
                "direct_source_event_time_range_s": [float(direct_events[:, 0].min()), float(direct_events[:, 0].max())] if len(direct_events) else None,
                "source_initial_frame_max_abs_luma_difference": initial_frame_error,
                "source_cache_exact_render_checks": endpoint_checks,
                "native_plugin_event_runtime_s": native_event_s,
                "direct_dvs_event_runtime_s": direct_event_s,
                "inference_wall_s": inference_s,
                "groups": groups,
                "paired_source_error_deltas": source_deltas,
                "diagnostic_history": "each source initialized once at 0.5505 s and consumed its whole source sequence continuously; first scored query is at least 25 ms later",
            }
            print(f"[done] episode_{ep_id:03d} {label}: {len(pre['query_times'])} exact cutoffs, "
                  f"events native/direct={len(native_events)}/{len(direct_events)}", flush=True)
    finally:
        if env is not None:
            env.close()

    total_render_s = time.perf_counter() - render_started_all
    query_csv = output / "all_four_group_queries.csv"
    with query_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_query_rows[0]))
        writer.writeheader()
        writer.writerows(all_query_rows)

    metadata = {
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "code_commit": __import__("subprocess").check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "runtime": {"device_for_physics": "Genesis CPU (no physics steps)",
                    "renderer_gpu_uuid": args.gpu_uuid,
                    "inference_device": "CPU", "torch_threads": 1,
                    "render_and_inference_wall_s": total_render_s},
        "input_hashes": input_hashes,
        "response_config": {
            "pos_thres": 0.2, "neg_thres": 0.2, "sigma_thres": 0.0,
            "cutoff_hz": 0.0, "leak_rate_hz": 0.0, "shot_noise_rate_hz": 0.0,
            "refractory_period_s": 0.0, "response_mode": "physical_log",
            "radiometric_gain": 1.0, "radiometric_offset": 0.0,
            "radiometric_floor": 1e-6, "radiometric_calibration_status": "uncalibrated",
        },
        "source_initialization": "both paths begin at the same exact native HDR frame and timestamp; separate DVS states are initialized once and carried continuously across the full 50 ms diagnostic segment",
        "causal_input": "existing five_bin_tensor implementation; [cutoff-25ms, cutoff); five 5ms bins, two polarities, 64x64, log1p counts",
        "scoring": "exact saved target state at each query row; pixel projection and fixed-plane XY error reported separately",
        "gt_boundary": "GT, full-link segmentation, depth and display RGB do not enter either network input; they are used after inference for scoring, overlays, and Ours generator geometry only",
        "physics_advanced": False,
        "training_started": False,
        "controller_modified": False,
        "frozen_simulator_core_modified": False,
        "episodes": episode_summary,
    }
    _write_json(output / "run_metadata.json", metadata)
    _write_json(output / "summary.json", {
        "status": "completed", "purpose": "paired source crossover on fixed physical trajectories; development diagnosis only",
        "episode_count": len(episode_summary), "four_group_query_rows": len(all_query_rows),
        "episodes": episode_summary,
        "limits": [
            "only exact cached state rows in the 70 ms trace_window are used; the full-episode raw trace is not staged, although its declared SHA256 is retained from episode_diagnostic.json",
            "source sequence begins at the shared saved time 0.5505 s; comparator history before this segment is not reconstructed",
            "Direct-HDR is an offline renderer reference, not physical sensor ground truth",
            "query rows are temporally correlated and are not independent episodes",
            "no generator ranking, retraining, or closed-loop result",
        ],
    })
    _write_manifest(output)
    print(json.dumps({"status": "completed", "output": str(output),
                      "queries": len(all_query_rows), "episodes": len(episode_summary),
                      "render_and_inference_wall_s": total_render_s}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
