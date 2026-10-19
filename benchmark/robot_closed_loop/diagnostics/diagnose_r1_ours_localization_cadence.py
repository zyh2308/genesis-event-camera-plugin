#!/usr/bin/env python3
"""Replay exact cached robot states to isolate the Ours 100 Hz input cadence.

The two episode traces are immutable development data.  This script directly
renders the same 0.5 ms states and compares (1) the existing 100 Hz HDR
endpoint + geometric reconstruction path, (2) direct 2 kHz HDR frames passed
through the same ideal physical-log DVS response, and (3) the already archived
2 kHz Direct-HDR event stream.  Learned models receive causal event bins only.
No physics is advanced and no frozen simulator-core file is edited.
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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.diagnose_r1_native_hdr_replay import (  # noqa: E402
    ReplayCamera,
    ReplayScene,
    event_boundary_counts,
    infer_event_state,
    interpolate_trace_label,
    load_event_models,
    project_world,
    read_trace,
    render_state,
    truth_rows_at_times,
    verify_source_hashes,
)
from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import (  # noqa: E402
    _resolve_motion_bodies,
    _sha256,
    _write_hash_manifest,
)
from benchmark.robot_closed_loop.robot_env import build_robot_scene  # noqa: E402
from genesis_event_plugin import GenesisPhysicsEventPlugin  # noqa: E402
from genesis_event_plugin.dvs_emulator import DvsEmulator  # noqa: E402
from genesis_event_plugin.noise_model import DvsNoiseConfig  # noqa: E402
from genesis_event_plugin.radiometry import linear_luminance  # noqa: E402
from benchmark.robot_closed_loop.event_state_model import five_bin_tensor  # noqa: E402


def response_config() -> DvsNoiseConfig:
    return DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
        cutoff_hz=0.0, leak_rate_hz=0.0, leak_jitter_fraction=0.0,
        shot_noise_rate_hz=0.0, noise_rate_cov_decades=0.0,
        refractory_period_s=0.0, log_input=False, radiance_input=True,
        response_mode="physical_log", radiometric_gain=1.0,
        radiometric_offset=0.0, radiometric_floor=1e-6,
        radiometric_calibration_status="uncalibrated",
        readout_model="ideal_readout", seed=42,
    )


def generate_sparse_plugin(frames: list[dict[str, Any]], pose_ids: np.ndarray,
                           camera_pose: np.ndarray, width: int, height: int,
                           fov_deg: float) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    scene = ReplayScene()
    focal = height / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    camera = ReplayCamera((width, height), np.asarray(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64))
    state = {"i": 0}

    def frame_provider(_scene, _camera):
        frame = frames[state["i"]]
        return {"rgb": frame["display_rgb"], "radiance": frame["hdr_rgb"],
                "depth": frame["depth"], "seg": frame["segmentation"]}

    def motion_provider(_scene, _camera):
        frame = frames[state["i"]]
        return {"camera_to_world": camera_pose,
                "object_to_world": {int(seg_id): frame["poses"][int(seg_id)] for seg_id in pose_ids}}

    plugin = GenesisPhysicsEventPlugin(
        preset="clean", noise_config=response_config(), input_space="linear",
        radiometric_gain=1.0, radiometric_offset=0.0, radiometric_floor=1e-6,
        radiometric_calibration_status="uncalibrated",
        motion_state_provider=motion_provider, frame_provider=frame_provider,
        response_mode="physical_log", disocclusion_policy="hold", device="cpu",
    )
    plugin.attach(scene, camera)
    plugin.start_episode()
    chunks: list[np.ndarray] = []
    intervals: list[dict[str, Any]] = []
    started = time.perf_counter()
    for idx, frame in enumerate(frames):
        state["i"] = idx
        scene.cur_t = float(frame["time_s"])
        events = np.asarray(plugin.capture(), dtype=np.float64).reshape(-1, 4)
        if len(events):
            chunks.append(events)
        if idx:
            sampling = plugin.interpolator.last_sampling
            unknown = np.asarray(plugin.interpolator.last_visibility.get(
                "intermediate_unknown", np.empty((0,))), dtype=bool)
            intervals.append({
                "time_s": float(frame["time_s"]), "event_count": int(len(events)),
                "adaptive_samples": None if sampling is None else int(sampling.samples),
                "unknown_hold_fraction": None if unknown.size == 0 else float(unknown.mean()),
            })
    plugin.end_episode()
    plugin.close()
    elapsed = time.perf_counter() - started
    events = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)
    if len(events):
        events = events[np.argsort(events[:, 0], kind="stable")]
    return events, intervals, elapsed


def generate_dense_direct(frames: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    height, width = frames[0]["hdr_rgb"].shape[:2]
    emulator = DvsEmulator(res=(height, width), config=response_config(), device="cpu")
    start = time.perf_counter()
    first_luma = linear_luminance(frames[0]["hdr_rgb"])
    emulator.initialize(first_luma, t=float(frames[0]["time_s"]))
    chunks: list[np.ndarray] = []
    for frame in frames[1:]:
        luma = linear_luminance(frame["hdr_rgb"])
        events = emulator.generate_events(luma, float(frame["time_s"]))
        if len(events):
            chunks.append(np.asarray(events, dtype=np.float64))
    elapsed = time.perf_counter() - start
    events = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)
    if len(events):
        events = events[np.argsort(events[:, 0], kind="stable")]
    return events, elapsed


def event_viz(voxel: np.ndarray, path: Path, title: str, gt_uv: np.ndarray,
              pred_uv: np.ndarray | None, error_mm: float | None) -> None:
    pos = np.asarray(voxel[:, 0].sum(axis=0), dtype=np.float32)
    neg = np.asarray(voxel[:, 1].sum(axis=0), dtype=np.float32)
    scale = max(float(np.percentile(np.concatenate([pos.ravel(), neg.ravel()]), 99.5)), 1.0)
    rgb = np.stack((np.clip(pos / scale, 0, 1), np.zeros_like(pos),
                    np.clip(neg / scale, 0, 1)), axis=-1)
    event_image = Image.fromarray(np.uint8(rgb * 255)).resize((256, 192), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (280, 236), (24, 26, 30))
    canvas.paste(event_image, (12, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title[:38], fill=(245, 245, 245))
    gx, gy = float(gt_uv[0] * 0.8 + 12), float(gt_uv[1] * 0.8 + 30)
    draw.ellipse((gx - 5, gy - 5, gx + 5, gy + 5), outline=(255, 255, 255), width=2)
    if pred_uv is not None:
        px, py = float(pred_uv[0] * 0.8 + 12), float(pred_uv[1] * 0.8 + 30)
        draw.line((px - 5, py, px + 5, py), fill=(255, 230, 0), width=2)
        draw.line((px, py - 5, px, py + 5), fill=(255, 230, 0), width=2)
    error_text = "unavailable" if error_mm is None else f"{error_mm:.1f} mm"
    draw.text((8, 220), f"white=GT yellow=pred error={error_text}", fill=(220, 220, 220))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--ours-checkpoint", type=Path, required=True)
    parser.add_argument("--v2e-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scene_cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    scene_cfg["shared_control"] = json.loads(args.control_config.read_text(encoding="utf-8"))
    reference = args.reference_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output path already exists: {output}")
    start_s, end_s = (float(x) for x in config["trace_time_s"])
    cutoff = float(config["decision_cutoff_s"])
    dense_dt = 1.0 / float(config["dense_direct_rate_hz"])
    sparse_stride = int(round(float(config["dense_direct_rate_hz"]) / float(config["sparse_endpoint_rate_hz"])))
    render_times = [round(float(t), 9) for t in np.arange(start_s, end_s + dense_dt / 2, dense_dt)]
    if len(render_times) < 3 or abs(render_times[-1] - cutoff) > 1e-8:
        raise ValueError("dense render grid must end exactly at the causal query cutoff")

    episode_cfg = {int(ep_id): label for label, ep_id in config["episodes"].items()}
    preloaded: dict[int, dict[str, Any]] = {}
    for ep_id, label in sorted(episode_cfg.items()):
        ep_dir = reference / "episodes" / f"episode_{ep_id:03d}"
        paths = [ep_dir / name for name in (
            "direct_hdr_events_1khz_2khz.npz", "state_control_trace.jsonl",
            "episode.csv", "direct_hdr_key_snapshots.npz")]
        verified = verify_source_hashes(reference, paths)
        trace = read_trace(paths[1])
        rows = truth_rows_at_times(trace, render_times)
        label = interpolate_trace_label(trace, cutoff)
        with np.load(paths[0], allow_pickle=False) as archive:
            reference_events = np.asarray(archive["events_2000hz"], dtype=np.float64)
        with paths[2].open("r", newline="", encoding="utf-8") as stream:
            episode = next(csv.DictReader(stream))
        preloaded[ep_id] = {"label": episode_cfg[ep_id], "rows": rows, "trace": trace,
                            "scoring_label": label, "reference_events": reference_events,
                            "verified": verified, "episode": episode}

    output.mkdir(parents=True, exist_ok=False)
    width, height = (int(x) for x in scene_cfg["camera"]["resolution_wh"])
    fov = float(scene_cfg["camera"]["vertical_fov_deg"])
    focal = height / (2 * np.tan(np.deg2rad(fov) / 2))
    intrinsics = np.asarray([focal, focal, width / 2, height / 2], dtype=np.float64)
    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning")
    env = None
    rows_out: list[dict[str, Any]] = []
    episode_records: dict[str, Any] = {}
    try:
        env = build_robot_scene(scene_cfg, dt_s=float(scene_cfg["physics"]["dt_s"]), segmentation_level="link")
        env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True, force_render=True)
        bodies = _resolve_motion_bodies(env)
        pose_ids = np.asarray(sorted(bodies), dtype=np.int32)
        from genesis_event_plugin.genesis_adapter import camera_to_world_cv
        camera_pose = np.asarray(camera_to_world_cv(env.camera), dtype=np.float64)
        fixed_height = float(scene_cfg["target"]["size_m"][2]) / 2.0
        model_records, checkpoint_hashes = load_event_models(
            {"ours_native_hdr": args.ours_checkpoint.resolve(), "v2e": args.v2e_checkpoint.resolve()},
            camera_pose, intrinsics, (width, height))

        for ep_id, pre in sorted(preloaded.items()):
            print(f"[render] episode_{ep_id:03d}: {len(render_times)} exact cached-state frames", flush=True)
            render_started = time.perf_counter()
            frames = [render_state(env, row, bodies) for row in pre["rows"]]
            render_wall_s = time.perf_counter() - render_started
            sparse_frames = frames[::sparse_stride]
            if sparse_frames[-1]["time_s"] != frames[-1]["time_s"]:
                sparse_frames.append(frames[-1])
            sparse_events, interval_diagnostics, sparse_wall_s = generate_sparse_plugin(
                sparse_frames, pose_ids, camera_pose, width, height, fov)
            dense_events, dense_wall_s = generate_dense_direct(frames)
            reference_events = pre["reference_events"]
            sources = {"sparse_100hz_ours": sparse_events, "direct_2khz_ours": dense_events,
                       "archived_2khz_reference": reference_events}
            gt_pos = np.asarray(pre["scoring_label"]["position_m"], dtype=np.float64)
            gt_vel = np.asarray(pre["scoring_label"]["velocity_m_s"], dtype=np.float64)
            gt_uv = project_world(gt_pos, camera_pose, intrinsics)
            results: dict[str, dict[str, Any]] = {}
            tensors: dict[str, np.ndarray] = {}
            for source_name, stream in sources.items():
                tensor = five_bin_tensor(stream, cutoff, width, height)
                tensors[source_name] = tensor
                keep = (stream[:, 0] >= cutoff - float(config["causal_input_window_s"])) & (stream[:, 0] < cutoff)
                raw_window = stream[keep]
                np.savez_compressed(output / f"episode_{ep_id:03d}_{source_name}_window.npz", events=raw_window)
                results[source_name] = {}
                for model_name, model_record in model_records.items():
                    estimate, _ = infer_event_state(stream, model_record, cutoff, width, height,
                                                     camera_pose, intrinsics, fixed_height)
                    if estimate.get("state_available"):
                        pred_pos = np.asarray(estimate["position_xyz_m"], dtype=np.float64)
                        pred_vel = np.asarray(estimate["velocity_xy_m_s"], dtype=np.float64)
                        pred_uv = np.asarray(estimate["heatmap_uv_px"], dtype=np.float64)
                        estimate["position_error_xy_mm"] = float(np.linalg.norm(pred_pos[:2] - gt_pos[:2]) * 1000)
                        estimate["velocity_error_xy_m_s"] = float(np.linalg.norm(pred_vel - gt_vel[:2]))
                        estimate["heatmap_uv_error_px"] = float(np.linalg.norm(pred_uv - gt_uv))
                    else:
                        pred_uv = None
                    estimate["input_contract"] = "causal 5 x 5 ms x 2 polarity x 64x64; events only"
                    estimate["label_used_only_after_inference"] = pre["scoring_label"]["label_method"]
                    results[source_name][model_name] = estimate
                    rows_out.append({
                        "episode_id": ep_id, "episode": pre["label"], "speed_m_s": float(pre["episode"]["launch_speed_m_s"]),
                        "start_y_m": float(pre["episode"]["start_y_m"]), "cutoff_s": cutoff,
                        "source": source_name, "model": model_name,
                        "event_count_25ms": estimate["event_count_25ms"],
                        "nonempty_5ms_bins": estimate["nonempty_5ms_bins"],
                        "state_available": estimate["state_available"],
                        "position_error_xy_mm": estimate.get("position_error_xy_mm"),
                        "velocity_error_xy_m_s": estimate.get("velocity_error_xy_m_s"),
                        "heatmap_uv_error_px": estimate.get("heatmap_uv_error_px"),
                    })
                    event_viz(tensor, output / f"episode_{ep_id:03d}_{source_name}_{model_name}.png",
                              f"{source_name} / {model_name}", gt_uv, pred_uv,
                              estimate.get("position_error_xy_mm"))
            tensor_cosines = {}
            base = tensors["archived_2khz_reference"].ravel()
            for name, tensor in tensors.items():
                value = tensor.ravel()
                denom = float(np.linalg.norm(value) * np.linalg.norm(base))
                tensor_cosines[name] = None if denom == 0 else float(np.dot(value, base) / denom)
            seams = [round(start_s + i * 0.01, 9) for i in range(1, 5)]
            record = {
                "episode": pre["label"], "seed": int(pre["episode"]["seed"]),
                "speed_m_s": float(pre["episode"]["launch_speed_m_s"]),
                "start_y_m": float(pre["episode"]["start_y_m"]), "cutoff_s": cutoff,
                "cached_state_trace_sha256": pre["verified"][f"episodes/episode_{ep_id:03d}/state_control_trace.jsonl"],
                "reference_raw_event_sha256": pre["verified"][f"episodes/episode_{ep_id:03d}/direct_hdr_events_1khz_2khz.npz"],
                "direct_render_count": len(frames), "direct_render_rate_hz": 2000,
                "render_wall_s": render_wall_s, "sparse_100hz_plugin_wall_s": sparse_wall_s,
                "direct_2khz_emulator_wall_s": dense_wall_s,
                "generation_response": "same physical_log gain=1 floor=1e-6 threshold=0.2; no noise/bandwidth/refractory; uncalibrated",
                "sparse_plugin_intervals": interval_diagnostics,
                "raw_event_and_five_bin_counts": {
                    name: {"raw_total_in_rendered_interval": int(len(stream)),
                           "raw_in_query_window": int(np.count_nonzero((stream[:, 0] >= cutoff - 0.025) & (stream[:, 0] < cutoff))),
                           "voxel_sum": int(tensors[name].sum()),
                           "nonempty_bins": int(np.count_nonzero(tensors[name].sum(axis=(1, 2, 3))))}
                    for name, stream in sources.items()},
                "tensor_cosine_against_archived_reference": tensor_cosines,
                "seam_counts_pm_1ms": {name: event_boundary_counts(stream, seams) for name, stream in sources.items()},
                "checkpoint_sha256": checkpoint_hashes,
                "inference_and_posthoc_metrics": results,
                "ground_truth_boundary": "target state and projection are joined only after inference; never supplied to either model",
                "render_boundary": "exact logged state rows direct-rendered offline; no physics advance; not a new rollout or real event-camera truth",
                "frozen_core_modified": False,
            }
            episode_records[f"episode_{ep_id:03d}"] = record
            (output / f"episode_{ep_id:03d}_diagnostic.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
            print(f"[episode] {ep_id}: render={render_wall_s:.2f}s sparse_events={len(sparse_events)} direct_events={len(dense_events)}", flush=True)

    finally:
        gs.destroy()

    with (output / "per_source_model.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    summary = {
        "status": "PASS", "experiment_id": config["experiment_id"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": _sha256(args.config.resolve()), "scene_config_sha256": _sha256(args.scene_config.resolve()),
        "control_config_sha256": _sha256(args.control_config.resolve()),
        "reference_manifest_sha256": _sha256(reference / "SHA256SUMS"),
        "checkpoint_sha256": checkpoint_hashes, "episodes": episode_records,
        "wall_s": None,
        "scope": "two selected cached development episodes; same 0.5 ms physical trajectory, camera, timestamps, response and query; no new rollout or ranking",
    }
    # Measure only the overall script wall clock using a stable run-start value.
    summary["wall_s"] = round(float(time.perf_counter() - SCRIPT_START), 3)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    _write_hash_manifest(output)
    print(json.dumps({"output": str(output), "wall_s": summary["wall_s"],
                      "position_errors": {f"ep{row['episode_id']:03d}/{row['source']}/{row['model']}": row["position_error_xy_mm"] for row in rows_out}}, indent=2), flush=True)
    return 0


SCRIPT_START = time.perf_counter()

if __name__ == "__main__":
    raise SystemExit(main())
