#!/usr/bin/env python3
"""Replay two immutable Direct-HDR traces and verify the native-HDR route.

The script renders only selected logged states for offline visual diagnosis.
It does not advance physics or feed GT/depth/segmentation to a learned model.
Ours uses native scene-linear HDR and its geometric interpolation path; V2E
uses the same timestamps' display PNGs and the official SuperSloMo pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import (  # noqa: E402
    _display_rgb_uint8,
    _pose,
    _resolve_motion_bodies,
    _sha256,
    _write_hash_manifest,
)
from benchmark.robot_closed_loop.robot_env import build_robot_scene  # noqa: E402
from genesis_event_plugin import GenesisPhysicsEventPlugin  # noqa: E402
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig  # noqa: E402
from genesis_event_plugin.noise_model import DvsNoiseConfig  # noqa: E402
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator  # noqa: E402
from genesis_event_plugin.radiometry import linear_luminance  # noqa: E402


class ReplayScene:
    def __init__(self):
        self.cur_t = 0.0


class ReplayCamera:
    def __init__(self, resolution_wh: tuple[int, int], intrinsics: np.ndarray):
        self.res = resolution_wh
        self.intrinsics = intrinsics


def read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def truth_rows_at_times(trace: list[dict[str, Any]], times: list[float]) -> list[dict[str, Any]]:
    trace_times = np.asarray([float(row["time_s"]) for row in trace], dtype=np.float64)
    rows = []
    for time_s in times:
        idx = int(np.argmin(np.abs(trace_times - time_s)))
        row = trace[idx]
        if abs(float(row["time_s"]) - time_s) > 1e-7:
            raise RuntimeError(f"requested time {time_s:.9f}s is absent from the physical trace")
        rows.append(row)
    return rows


def verify_source_hashes(root: Path, paths: list[Path]) -> dict[str, str]:
    expected = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        expected[relative.lstrip("* ")] = digest
    verified = {}
    for path in paths:
        relative = str(path.relative_to(root))
        actual = _sha256(path)
        if actual != expected.get(relative):
            raise RuntimeError(f"reference manifest mismatch for {relative}")
        verified[relative] = actual
    return verified


def render_state(env, row: dict[str, Any], pose_bodies: dict[int, Any]) -> dict[str, Any]:
    env.robot.set_qpos(np.asarray(row["robot_qpos"], dtype=np.float32), zero_velocity=True)
    obj = row["object"]
    target_qpos = np.asarray(list(obj["position_m"]) + list(obj["orientation_wxyz"]), dtype=np.float32)
    env.target.set_qpos(target_qpos, zero_velocity=True)
    hdr, depth, seg, _ = env.camera.render(
        rgb=True, depth=True, segmentation=True, radiance=True, force_render=True,
    )
    hdr = np.asarray(hdr, dtype=np.float32).copy()
    depth = np.asarray(depth, dtype=np.float32).copy()
    seg = np.asarray(seg, dtype=np.int32).copy()
    if not np.isfinite(hdr).all() or np.nanmin(hdr) < 0:
        raise RuntimeError("native renderer returned invalid HDR values")
    display, _, _, _ = env.camera.render(
        rgb=True, depth=False, segmentation=False, radiance=False, force_render=True,
    )
    display = _display_rgb_uint8(np.asarray(display))
    poses = {int(seg_id): _pose(body) for seg_id, body in pose_bodies.items()}
    return {
        "time_s": round(float(row["time_s"]), 9), "hdr_rgb": hdr,
        "radiance_luma": linear_luminance(hdr), "display_rgb": display,
        "depth": depth, "segmentation": seg, "poses": poses,
    }


def write_source_cache(path: Path, frames: list[dict[str, Any]], pose_ids: np.ndarray,
                       camera_to_world: np.ndarray, width: int, height: int, fov_deg: float) -> None:
    selected = [frame for frame in frames if frame["time_s"] in set(np.round(SOURCE_TIMES, 9))]
    if len(selected) != len(SOURCE_TIMES):
        raise RuntimeError(f"source frame count {len(selected)} != {len(SOURCE_TIMES)}")
    with h5py.File(path, "w") as handle:
        handle.attrs["camera_resolution_wh"] = np.asarray([width, height], dtype=np.int32)
        handle.attrs["vertical_fov_deg"] = float(fov_deg)
        handle.attrs["frame_timestamp_contract"] = "offline exact-state replay at original 100 Hz source keyframe times"
        handle.attrs["radiance_source"] = "Genesis HDR overlay camera.render(radiance=True), native float32 linear RGB"
        handle.attrs["rollout_id"] = path.parent.name
        handle.create_dataset("timestamps_s", data=np.asarray([f["time_s"] for f in selected], dtype=np.float64))
        handle.create_dataset("radiance", data=np.stack([f["hdr_rgb"] for f in selected]), compression="gzip", compression_opts=1)
        handle.create_dataset("radiance_luma", data=np.stack([f["radiance_luma"] for f in selected]), compression="gzip", compression_opts=1)
        handle.create_dataset("display_rgb", data=np.stack([f["display_rgb"] for f in selected]), compression="gzip", compression_opts=1)
        handle.create_dataset("depth", data=np.stack([f["depth"] for f in selected]), compression="gzip", compression_opts=1)
        handle.create_dataset("segmentation", data=np.stack([f["segmentation"] for f in selected]), compression="gzip", compression_opts=1)
        handle.create_dataset("segmentation_pose_ids", data=pose_ids.astype(np.int32))
        handle.create_dataset("object_poses", data=np.stack([
            np.stack([f["poses"][int(seg_id)] for seg_id in pose_ids]) for f in selected
        ]), compression="gzip", compression_opts=1)
        handle.create_dataset("camera_to_world_cv", data=np.repeat(camera_to_world[None], len(selected), axis=0))


def generate_ours(cache_path: Path, output_dir: Path, response_mode: str,
                  display_gray: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with h5py.File(cache_path, "r") as h5:
        timestamps = np.asarray(h5["timestamps_s"][:], dtype=np.float64)
        hdr = h5["radiance"]
        gray = []
        for image in h5["display_rgb"][:]:
            Image.fromarray(image).save(output_dir / f"gray_src_{len(gray):03d}.png")
            reread = cv2.imread(str(output_dir / f"gray_src_{len(gray):03d}.png"), cv2.IMREAD_GRAYSCALE)
            if reread is None:
                raise RuntimeError("cannot recover displayed PNG grayscale")
            gray.append(reread.astype(np.float32))
        gray = np.stack(gray)
        depth, seg = h5["depth"], h5["segmentation"]
        ids = h5["segmentation_pose_ids"][:].astype(int)
        poses = h5["object_poses"]
        camera_poses = h5["camera_to_world_cv"][:]
        width, height = (int(v) for v in h5.attrs["camera_resolution_wh"])
        fov = float(h5.attrs["vertical_fov_deg"])
        focal = height / (2 * np.tan(np.deg2rad(fov) / 2))
        K = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
        fake_scene = ReplayScene()
        fake_camera = ReplayCamera((width, height), K)
        state = {"i": 0}

        def frame_provider(_scene, _camera):
            i = state["i"]
            intensity = gray[i] if display_gray else hdr[i]
            return {"rgb": h5["display_rgb"][i], "radiance": intensity,
                    "depth": depth[i], "seg": seg[i]}

        def motion_provider(_scene, _camera):
            i = state["i"]
            return {"camera_to_world": camera_poses[i],
                    "object_to_world": {int(seg_id): poses[i, j] for j, seg_id in enumerate(ids)}}

        response = DvsNoiseConfig(
            pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
            cutoff_hz=0.0, leak_rate_hz=0.0, leak_jitter_fraction=0.0,
            shot_noise_rate_hz=0.0, noise_rate_cov_decades=0.0,
            refractory_period_s=0.0, log_input=False, radiance_input=False,
            response_mode=response_mode, readout_model="ideal_readout", seed=42,
        )
        plugin = GenesisPhysicsEventPlugin(
            preset="clean", noise_config=response, input_space="linear",
            radiometric_gain=1.0, radiometric_floor=1e-6,
            radiometric_calibration_status="uncalibrated",
            motion_state_provider=motion_provider, frame_provider=frame_provider,
            response_mode=response_mode, disocclusion_policy="hold", device="cpu",
        )
        plugin.attach(fake_scene, fake_camera)
        plugin.start_episode()
        chunks = []
        start = time.perf_counter()
        for i, t in enumerate(timestamps):
            state["i"] = i
            fake_scene.cur_t = float(t)
            result = plugin.capture()
            if len(result):
                chunks.append(np.asarray(result, dtype=np.float64))
        plugin.end_episode()
        plugin.close()
        elapsed_s = time.perf_counter() - start
    events = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)
    if len(events):
        events = events[np.argsort(events[:, 0], kind="stable")]
    np.savez_compressed(output_dir / "raw_events.npz", events=events)
    info = {
        "response_mode": response_mode,
        "input_endpoint": "display PNG read by OpenCV grayscale [0,255]" if display_gray else "native Genesis HDR linear RGB converted to linear luminance by plugin",
        "radiance_input_forced_by_frozen_plugin": True,
        "radiance_luma_range": None if display_gray else [float(np.min(hdr)), float(np.max(hdr))],
        "gray_png_range": [float(np.min(gray)), float(np.max(gray))],
        "event_count": int(len(events)),
        "runtime_s": elapsed_s,
        "response": {"threshold_on_off": 0.2, "mismatch_noise_bandwidth_refractory": "disabled"},
    }
    (output_dir / "generation.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return events, info


def run_v2e(frames_dir: Path, output_dir: Path, source: Path, python: str, slomo: Path) -> tuple[np.ndarray, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [
        python, str(ROOT / "benchmark/scripts/v2e_official_pipeline_smoke.py"),
        "--frames", str(frames_dir), "--output-dir", str(output_dir),
        "--source", str(source), "--python", python, "--slomo-model", str(slomo),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    (output_dir / "driver.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"official V2E pipeline failed: {completed.stdout[-2000:]}")
    with h5py.File(output_dir / "events.h5", "r") as h5:
        events = np.asarray(h5["events"][:], dtype=np.float64)
    events[:, 0] = events[:, 0] / 1e6 + SOURCE_TIMES[0]
    events[:, 3] = np.where(events[:, 3] > 0, 1.0, -1.0)
    events = events[np.argsort(events[:, 0], kind="stable")]
    np.savez_compressed(output_dir / "raw_events_shifted_to_episode_clock.npz", events=events)
    info = {"event_count": int(len(events)), "runtime_s": time.perf_counter() - started,
            "time_offset_s": float(SOURCE_TIMES[0]), "pixel_model": "official V2E defaults",
            "super_slomo": str(slomo), "official_config": str(ROOT / "benchmark/configs/v2e.yaml"),
            "official_config_sha256": _sha256(ROOT / "benchmark/configs/v2e.yaml")}
    (output_dir / "generation_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return events, info


def event_boundary_counts(events: np.ndarray, seams: list[float], width_s: float = 0.001) -> dict[str, Any]:
    result = {"event_count": int(len(events)), "seams": {}}
    if len(events):
        result["event_rate_hz_equiv"] = float(len(events) / max(events[:, 0].max() - events[:, 0].min(), 1e-9))
        result["time_range_s"] = [float(events[:, 0].min()), float(events[:, 0].max())]
    else:
        result["event_rate_hz_equiv"] = None
        result["time_range_s"] = None
    for seam in seams:
        result["seams"][f"{seam:.3f}"] = int(np.count_nonzero(np.abs(events[:, 0] - seam) <= width_s)) if len(events) else 0
    return result


def render_diagnostic_montage(path: Path, data: dict[str, np.ndarray], interp: np.ndarray,
                              unknown: np.ndarray, event_voxels: dict[str, np.ndarray],
                              cutoff: float) -> None:
    width, height = data["display_rgb"].shape[2], data["display_rgb"].shape[1]
    source_idx = int(np.argmin(np.abs(data["source_times"] - cutoff)))
    dense_idx = int(np.argmin(np.abs(data["dense_times"] - (cutoff - 0.005))))
    dense = data["dense_luma"][dense_idx]
    rec = interp[dense_idx]
    log_diff = np.abs(np.log(np.maximum(dense, 1e-6)) - np.log(np.maximum(rec, 1e-6)))
    def gray(array, low=None, high=None):
        arr = np.asarray(array, dtype=np.float32)
        if low is None: low = float(np.percentile(arr, 1))
        if high is None: high = float(np.percentile(arr, 99))
        normalized = np.clip((arr - low) / max(high - low, 1e-9), 0, 1)
        return Image.fromarray(np.uint8(normalized * 255)).resize((160, 120), Image.Resampling.NEAREST).convert("RGB")
    hdr = data["source_hdr_luma"][source_idx]
    hdr_luma = gray(hdr, 0.0, max(float(np.percentile(hdr, 99.5)), 1e-6))
    display = Image.fromarray(cv2.cvtColor(data["display_rgb"][source_idx], cv2.COLOR_RGB2GRAY)).resize((160, 120), Image.Resampling.NEAREST).convert("RGB")
    direct_image = gray(dense, 0.0, max(float(np.percentile(dense, 99.5)), 1e-6))
    interp_image = gray(rec, 0.0, max(float(np.percentile(dense, 99.5)), 1e-6))
    error_image = gray(log_diff, 0.0, max(float(np.percentile(log_diff, 99.5)), 1e-6))
    invalid = Image.fromarray(np.uint8(unknown[dense_idx]) * 255).resize((160, 120), Image.Resampling.NEAREST).convert("RGB")
    labels = ["display gray", "native HDR luma (display only)", "direct replay @0.595s",
              "Ours warp @0.595s", "abs log residual", "hold/invalid mask"]
    images = [display, hdr_luma, direct_image, interp_image, error_image, invalid]
    canvas = Image.new("RGB", (480, 360), (20, 22, 25))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(zip(labels, images)):
        x = (idx % 3) * 160
        y = (idx // 3) * 150
        draw.text((x + 3, y + 3), label, fill=(245, 245, 245))
        canvas.paste(image, (x, y + 20))
    row3_y = 300
    for col, (label, voxel) in enumerate(event_voxels.items()):
        counts = np.log1p(np.asarray(voxel).sum(axis=(0, 1)))
        counts = np.clip(counts / max(float(np.percentile(counts, 99.5)), 1e-6), 0, 1)
        event_img = Image.fromarray(np.uint8(counts * 255)).resize((120, 42), Image.Resampling.NEAREST).convert("RGB")
        x = col * 160
        draw.text((x + 3, row3_y), f"five-bin {label}", fill=(245, 245, 245))
        canvas.paste(event_img, (x + 20, row3_y + 18))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    global SOURCE_TIMES
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v2e-source", type=Path, required=True)
    parser.add_argument("--v2e-python", required=True)
    parser.add_argument("--slomo-model", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scene_cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    reference = args.reference_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cutoff = float(config["decision_cutoff_s"])
    start = float(config["source_time_start_s"])
    end = float(config["source_time_end_s"])
    period = 1.0 / float(config["keyframe_source_rate_hz"])
    SOURCE_TIMES = [round(value, 9) for value in np.arange(start, end + period / 2, period)]
    dense_start, dense_end = (float(v) for v in config["dense_direct_interval_s"])
    dense_period = 1.0 / float(config["dense_direct_rate_hz"])
    # PhysicsInterpolator emits alpha=k/N (k=1..N); the direct source
    # keyframe at dense_start is not an interpolated sample.
    dense_times = [round(value, 9) for value in np.arange(dense_start + dense_period, dense_end + dense_period / 2, dense_period)]
    episode_cfg = {int(ep_id): name for name, ep_id in config["episodes"].items()}
    scene_cfg["shared_control"] = json.loads((ROOT / "benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json").read_text())
    width, height = (int(v) for v in scene_cfg["camera"]["resolution_wh"])
    fov = float(scene_cfg["camera"]["vertical_fov_deg"])
    fx = height / (2 * np.tan(np.deg2rad(fov) / 2))
    K = np.asarray([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=np.float64)
    started = time.perf_counter()
    records = {}

    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning")
    env = None
    try:
        env = build_robot_scene(scene_cfg, dt_s=float(scene_cfg["physics"]["dt_s"]), segmentation_level="link")
        env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True, force_render=True)
        pose_bodies = _resolve_motion_bodies(env)
        pose_ids = np.asarray(sorted(pose_bodies), dtype=np.int32)
        from genesis_event_plugin.genesis_adapter import camera_to_world_cv
        camera_pose = np.asarray(camera_to_world_cv(env.camera), dtype=np.float64)
        target_seg_ids = [seg_id for seg_id, body in pose_bodies.items() if body is env.target]
        robot_seg_ids = [seg_id for seg_id, body in pose_bodies.items() if body is not env.target]

        for episode_id, label in sorted(episode_cfg.items()):
            ep_dir = reference / "episodes" / f"episode_{episode_id:03d}"
            event_path = ep_dir / "direct_hdr_events_1khz_2khz.npz"
            trace_path = ep_dir / "state_control_trace.jsonl"
            episode_path = ep_dir / "episode.csv"
            snapshot_path = ep_dir / "direct_hdr_key_snapshots.npz"
            verified = verify_source_hashes(reference, [event_path, trace_path, episode_path, snapshot_path])
            trace = read_trace(trace_path)
            source_rows = truth_rows_at_times(trace, SOURCE_TIMES)
            dense_rows = truth_rows_at_times(trace, dense_times)
            with episode_path.open("r", newline="", encoding="utf-8") as stream:
                episode = next(csv.DictReader(stream))
            snapshot_archive = np.load(snapshot_path)
            snapshot_times = np.asarray(snapshot_archive["times_s"], dtype=np.float64)
            extra_snapshot_rows = truth_rows_at_times(trace, snapshot_times.tolist())
            all_rows: dict[float, dict[str, Any]] = {}
            for row in source_rows + dense_rows + extra_snapshot_rows:
                all_rows[round(float(row["time_s"]), 9)] = row
            rendered = {}
            for time_s, row in sorted(all_rows.items()):
                rendered[time_s] = render_state(env, row, pose_bodies)

            # Direct key-frame rerender is checked against saved Genesis HDR snapshots.
            snapshot_errors = []
            for idx, time_s in enumerate(snapshot_times):
                frame = rendered[round(float(time_s), 9)]
                current = frame["hdr_rgb"]
                original = np.asarray(snapshot_archive["hdr"][idx], dtype=np.float32)
                diff = np.abs(current - original)
                snapshot_errors.append({
                    "label": str(snapshot_archive["labels"][idx]), "time_s": float(time_s),
                    "mae": float(diff.mean()), "p95_abs": float(np.percentile(diff, 95)),
                    "max_abs": float(diff.max()),
                })
            cache_dir = output / f"episode_{episode_id:03d}"
            cache_dir.mkdir()
            frames_dir = cache_dir / "source_frames"
            frames_dir.mkdir()
            source_frames = [rendered[t] for t in SOURCE_TIMES]
            for idx, frame in enumerate(source_frames):
                Image.fromarray(frame["display_rgb"]).save(frames_dir / f"frame_{idx:06d}.png")
            cache_path = cache_dir / "same_state_source_cache.h5"
            write_source_cache(cache_path, source_frames, pose_ids, camera_pose, width, height, fov)

            # A dense direct-render window uses logged 0.5 ms states exactly. It is not
            # interpolation and does not advance or alter the episode's physics.
            dense_frames = [rendered[t] for t in dense_times]
            source_ids = {round(t, 9): i for i, t in enumerate(SOURCE_TIMES)}
            pair_idx0, pair_idx1 = source_ids[round(dense_start, 9)], source_ids[round(dense_end, 9)]
            source0, source1 = source_frames[pair_idx0], source_frames[pair_idx1]
            object0, object1 = source0["poses"], source1["poses"]
            delta = {int(seg_id): object1[int(seg_id)] @ np.linalg.inv(object0[int(seg_id)]) for seg_id in pose_ids}
            interpolator = PhysicsInterpolator(
                K, sampler_config=AdaptiveSamplingConfig(
                    min_samples=int(config["debug_reconstruction_samples"]),
                    max_samples=int(config["debug_reconstruction_samples"]),
                ), disocclusion_policy="hold", device="cpu", warp_backend="numpy",
            )
            recon_frames, recon_times = interpolator.interpolate(
                source0["hdr_rgb"], source0["depth"], source0["segmentation"],
                source1["hdr_rgb"], source1["depth"], source1["segmentation"],
                camera_pose, camera_pose, delta, dense_start, dense_end, input_space="linear",
                object_world0=object0, object_world1=object1,
            )
            recon_luma = np.stack(recon_frames).astype(np.float32)
            if len(recon_times) != len(dense_times) or not np.allclose(recon_times, dense_times, atol=1e-8, rtol=0):
                raise RuntimeError(f"debug reconstruction samples do not align to direct renderer times: {recon_times}")
            direct_luma = np.stack([frame["radiance_luma"] for frame in dense_frames])
            direct_seg = np.stack([frame["segmentation"] for frame in dense_frames])
            unknown = np.asarray(interpolator.last_visibility["intermediate_unknown"], dtype=bool)
            frame_metrics = []
            for i, time_s in enumerate(dense_times):
                direct = direct_luma[i]
                recon = recon_luma[i]
                diff = np.abs(np.log(np.maximum(direct, 1e-6)) - np.log(np.maximum(recon, 1e-6)))
                target_mask = np.isin(direct_seg[i], target_seg_ids)
                robot_mask = np.isin(direct_seg[i], robot_seg_ids)
                valid = ~unknown[i]
                frame_metrics.append({
                    "time_s": time_s,
                    "target_visible_pixels": int(target_mask.sum()),
                    "robot_pixels": int(robot_mask.sum()),
                    "unknown_hold_pixels": int(unknown[i].sum()),
                    "unknown_hold_fraction": float(unknown[i].mean()),
                    "log_abs_error_mean_all": float(diff.mean()),
                    "log_abs_error_p95_all": float(np.percentile(diff, 95)),
                    "log_abs_error_mean_valid": None if not valid.any() else float(diff[valid].mean()),
                    "log_abs_error_mean_target": None if not target_mask.any() else float(diff[target_mask].mean()),
                    "log_abs_error_mean_robot": None if not robot_mask.any() else float(diff[robot_mask].mean()),
                    "log_abs_error_mean_background": None if not (~(target_mask | robot_mask)).any() else float(diff[~(target_mask | robot_mask)].mean()),
                })

            # Produce the two Ours routes on these identical 100 Hz keyframes.
            native_events, native_info = generate_ours(
                cache_path, cache_dir / "ours_native_hdr", "physical_log", display_gray=False,
            )
            legacy_events, legacy_info = generate_ours(
                cache_path, cache_dir / "ours_legacy_display_gray", "video_linlog", display_gray=True,
            )

            # Official V2E consumes the very same Genesis display frames/keyframe times.
            v2e_events, v2e_info = run_v2e(
                frames_dir, cache_dir / "v2e_official", args.v2e_source.resolve(),
                str(Path(args.v2e_python).resolve()), args.slomo_model.resolve(),
            )
            from benchmark.robot_closed_loop.event_state_model import five_bin_tensor
            voxel_inputs = {
                "ours_native_hdr": five_bin_tensor(native_events, cutoff, width, height),
                "ours_legacy_display_gray": five_bin_tensor(legacy_events, cutoff, width, height),
                "v2e_official": five_bin_tensor(v2e_events, cutoff, width, height),
            }
            np.savez_compressed(cache_dir / "five_bin_inputs_at_cutoff.npz", **voxel_inputs)

            # Preserve the exact shared-reference model input as a compact raw-event slice.
            reference_events = np.asarray(np.load(event_path)["events_2000hz"], dtype=np.float64)
            keep = ((reference_events[:, 0] >= cutoff - 0.025) & (reference_events[:, 0] < cutoff))
            np.savez_compressed(cache_dir / "common_reference_event_window.npz", events=reference_events[keep])

            # Store only the measured narrow replay region plus source frames and labels.
            with h5py.File(cache_dir / "rendered_frame_evidence.h5", "w") as handle:
                handle.attrs["purpose"] = "offline direct camera render from logged robot/target state; no physics advanced"
                handle.attrs["direct_hdr_render_matches_saved_snapshots_exactly"] = all(item["mae"] == 0 for item in snapshot_errors)
                handle.attrs["source_state_trace_sha256"] = verified[str(trace_path.relative_to(reference))]
                handle.create_dataset("source_times_s", data=SOURCE_TIMES)
                handle.create_dataset("source_hdr_rgb", data=np.stack([f["hdr_rgb"] for f in source_frames]), compression="gzip", compression_opts=1)
                handle.create_dataset("source_display_rgb", data=np.stack([f["display_rgb"] for f in source_frames]), compression="gzip", compression_opts=1)
                handle.create_dataset("dense_times_s", data=dense_times)
                handle.create_dataset("dense_direct_hdr_luminance", data=direct_luma, compression="gzip", compression_opts=1)
                handle.create_dataset("dense_direct_segmentation", data=direct_seg, compression="gzip", compression_opts=1)
                handle.create_dataset("ours_intermediate_luminance", data=recon_luma, compression="gzip", compression_opts=1)
                handle.create_dataset("ours_intermediate_unknown_mask", data=unknown.astype(np.uint8), compression="gzip", compression_opts=1)
                handle.create_dataset("dense_direct_hdr_rgb", data=np.stack([f["hdr_rgb"] for f in dense_frames]), compression="gzip", compression_opts=1)
                handle.create_dataset("source_state_trace_rows_json", data=np.asarray([json.dumps(r, separators=(",", ":")) for r in source_rows], dtype=h5py.string_dtype("utf-8")))
                handle.create_dataset("dense_state_trace_rows_json", data=np.asarray([json.dumps(r, separators=(",", ":")) for r in dense_rows], dtype=h5py.string_dtype("utf-8")))
            trace_window = [row for row in trace if start - 1e-9 <= float(row["time_s"]) <= end + 1e-9]
            (cache_dir / "trace_window.jsonl").write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in trace_window),
                encoding="utf-8",
            )
            montage_data = {
                "source_times": np.asarray(SOURCE_TIMES),
                "display_rgb": np.stack([f["display_rgb"] for f in source_frames]),
                "source_hdr_luma": np.stack([f["radiance_luma"] for f in source_frames]),
                "dense_times": np.asarray(dense_times), "dense_luma": direct_luma,
            }
            render_diagnostic_montage(cache_dir / "failure_chain_montage.png", montage_data,
                                      recon_luma, unknown, voxel_inputs, cutoff)
            seam_times = [t for t in SOURCE_TIMES[1:-1]]
            record = {
                "label": label, "episode_id": episode_id, "seed": int(episode["seed"]),
                "speed_m_s": float(episode["speed_m_s"]), "launch_y_m": float(episode["launch_y_m"]),
                "first_contact_s": float(episode["first_contact_s"]),
                "reference_inputs_sha256": verified,
                "renderer_snapshot_replay": snapshot_errors,
                "native_hdr_source_range_rgb": [float(min(f["hdr_rgb"].min() for f in source_frames)), float(max(f["hdr_rgb"].max() for f in source_frames))],
                "native_hdr_luminance_percentiles": np.percentile(np.stack([f["radiance_luma"] for f in source_frames]), [1, 50, 99]).tolist(),
                "dense_render_frame_count": len(dense_times), "dense_render_wall_scope": "scene-state setters plus HDR/depth/seg camera renders for selected source/dense timestamps; includes startup/JIT amortized across both episodes",
                "reconstruction": {"sampler_config": config["debug_reconstruction_samples"],
                                   "sample_count": len(recon_times), "metrics": frame_metrics,
                                   "recommend_render": bool(interpolator.last_uncertainty.get("recommend_render")),
                                   "uncertainty": interpolator.last_uncertainty,
                                   "disocclusion_policy": "hold"},
                "generator_info": {"ours_native_hdr": native_info, "ours_legacy_display_gray": legacy_info, "v2e_official": v2e_info},
                "raw_event_temporal_counts": {
                    "ours_native_hdr": event_boundary_counts(native_events, seam_times),
                    "ours_legacy_display_gray": event_boundary_counts(legacy_events, seam_times),
                    "v2e_official": event_boundary_counts(v2e_events, seam_times),
                },
                "five_bin_event_counts": {name: int(vox.sum()) for name, vox in voxel_inputs.items()},
                "five_bin_nonempty_bins": {name: int(np.count_nonzero(vox.sum(axis=(1, 2, 3)))) for name, vox in voxel_inputs.items()},
                "frozen_core_modified": False,
                "dense_direct_render_scope": "offline visual replay from the exact logged 2 kHz state trace; not a new dynamics rollout and not real event-camera ground truth",
                "display_and_hdr_are_not_claimed_as_shared_pixel_response": True,
            }
            (cache_dir / "episode_diagnostic.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            records[f"episode_{episode_id:03d}"] = record
    finally:
        gs.destroy()

    summary = {
        "status": "PASS", "purpose": config["purpose"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": _sha256(args.config.resolve()), "scene_config_sha256": _sha256(args.scene_config.resolve()),
        "shared_control_config_sha256": _sha256(ROOT / "benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json"),
        "reference_manifest_sha256": _sha256(reference / "SHA256SUMS"),
        "v2e_wrapper_sha256": _sha256(ROOT / "benchmark/scripts/v2e_official_pipeline_smoke.py"),
        "v2e_default_config_sha256": _sha256(ROOT / "benchmark/configs/v2e.yaml"),
        "v2e_source": str(args.v2e_source.resolve()), "slomo_checkpoint_sha256": _sha256(args.slomo_model.resolve()),
        "renderer_overlay": "installed Genesis 1.2.2 HDR overlay in .venv-robot; radiance=True returns linear float32 RGB",
        "renderer_replay_is_offline": True, "frozen_core_modified": False,
        "episodes": records, "wall_s": time.perf_counter() - started,
        "interpretation": "The old model weights are not retrained on native HDR. Their output on native-HDR-generated bins is only a diagnostic transfer probe, not the downstream utility of a complete HDR-trained method.",
    }
    (output / "native_hdr_replay_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    _write_hash_manifest(output)
    print(json.dumps({"output": str(output), "wall_s": summary["wall_s"],
                      "episodes": {key: {"snapshot_exact": all(row["mae"] == 0 for row in value["renderer_snapshot_replay"]),
                                         "events": value["five_bin_event_counts"],
                                         "interpolation_unknown_fraction_mean": float(np.mean([r["unknown_hold_fraction"] for r in value["reconstruction"]["metrics"]]))}
                                 for key, value in records.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    SOURCE_TIMES = []
    raise SystemExit(main())
