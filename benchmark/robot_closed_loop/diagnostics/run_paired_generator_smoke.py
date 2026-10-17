#!/usr/bin/env python3
"""Capture one shared robot rollout and smoke the existing Ours/V2E chains.

This is a pipeline contract check, not a generator ranking. Both generators
consume the exact same timestamped cached rollout: Ours uses cached linear HDR,
depth, segmentation and link/object poses through GenesisPhysicsEventPlugin;
official V2E uses display RGB frames from those same timestamps and its frozen
SuperSloMo/EventEmulator path. Small event-state fits share labels and split.
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
from typing import Any, Dict, Iterable, Optional

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.robot_env import (  # noqa: E402
    as_numpy,
    build_robot_scene,
    quaternion_matrix_wxyz,
)
from benchmark.robot_closed_loop.run import (  # noqa: E402
    jsonable,
    run_oracle_trial,
    save_gif,
    write_trace,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose(body: Any) -> np.ndarray:
    position = as_numpy(body.get_pos(relative=False)).reshape(-1)[:3]
    quaternion = as_numpy(body.get_quat(relative=False)).reshape(-1)[:4]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix_wxyz(quaternion)
    matrix[:3, 3] = position
    return matrix


def _descriptor_indices(value: Any) -> set[int]:
    found: set[int] = set()
    for name in ("entity_idx", "link_idx", "idx"):
        if hasattr(value, name):
            try:
                found.add(int(getattr(value, name)))
            except (TypeError, ValueError):
                pass
    if isinstance(value, dict):
        for child in value.values():
            found.update(_descriptor_indices(child))
    elif isinstance(value, (tuple, list, set)):
        for child in value:
            found.update(_descriptor_indices(child))
    elif isinstance(value, (int, np.integer)):
        found.add(int(value))
    return found


def _resolve_motion_bodies(env) -> Dict[int, Any]:
    """Resolve each rendered moving link/target ID to its rigid world pose."""
    seg_map = getattr(env.scene, "segmentation_idx_dict", None)
    if not seg_map:
        raise RuntimeError("Genesis has no segmentation_idx_dict")
    robot_links = {int(link.idx): link for link in env.robot.links}
    target_ids = {int(env.target.idx)}
    target_ids.update(int(link.idx) for link in getattr(env.target, "links", []))
    mapping: Dict[int, Any] = {}
    resolved_robot_links: set[int] = set()
    target_seg_ids: set[int] = set()
    for raw_seg_id, descriptor in seg_map.items():
        candidates = _descriptor_indices(descriptor)
        link_hits = candidates.intersection(robot_links)
        if len(link_hits) == 1:
            link_id = next(iter(link_hits))
            mapping[int(raw_seg_id)] = robot_links[link_id]
            resolved_robot_links.add(link_id)
            continue
        target_hits = candidates.intersection(target_ids)
        if target_hits:
            mapping[int(raw_seg_id)] = env.target
            target_seg_ids.add(int(raw_seg_id))
    missing_links = sorted(set(robot_links) - resolved_robot_links)
    if missing_links:
        raise RuntimeError(
            "link-level segmentation must resolve every Panda link for articulated motion; "
            f"missing_link_indices={missing_links}, segmentation_level=link"
        )
    if not target_seg_ids:
        raise RuntimeError("moving target has no segmentation ID mapped to its object pose")
    return mapping


def _display_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return arr.copy()
    arr = arr.astype(np.float32, copy=False)
    if float(arr.max()) <= 1.0001:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).round().astype(np.uint8)


class RolloutArchive:
    """Fixed 100 Hz same-state HDR/RGB/geometry/label archive."""

    def __init__(self, env, output: Path, pose_bodies: Dict[int, Any], sample_period_s: float = 0.01):
        self.env = env
        self.output = output
        self.frames_dir = output / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.pose_bodies = pose_bodies
        self.pose_ids = np.asarray(sorted(pose_bodies), dtype=np.int32)
        self.period_s = float(sample_period_s)
        self.rows: list[dict[str, Any]] = []
        self.last_capture_s: Optional[float] = None
        self.render_ms: list[float] = []

    def capture(self, env, rel_t_s: float, command: Dict[str, Any], force: bool = False) -> None:
        t = float(rel_t_s)
        if self.last_capture_s is not None:
            if t <= self.last_capture_s + 1e-9:
                return
            # V2E's official source cadence is fixed at 100 Hz.  The runner's
            # final forced diagnostic sample can be off-grid; do not append it
            # as if it were a regular source frame.
            if force and abs(t / self.period_s - round(t / self.period_s)) > 1e-6:
                return
            if not force and t < self.last_capture_s + self.period_s - 1e-8:
                return
        started = time.perf_counter()
        radiance, depth, seg, _ = env.camera.render(
            rgb=True, depth=True, segmentation=True, radiance=True,
        )
        display, _, _, _ = env.camera.render(rgb=True, depth=False, segmentation=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        radiance = np.asarray(radiance, dtype=np.float32).copy()
        depth = np.asarray(depth, dtype=np.float32).copy()
        seg = np.asarray(seg, dtype=np.int32).copy()
        display = _display_rgb_uint8(display)
        if radiance.ndim != 3 or radiance.shape[-1] != 3:
            raise RuntimeError(f"unexpected native HDR shape {radiance.shape}")
        if not np.isfinite(radiance).all():
            raise RuntimeError("non-finite HDR radiance in cached rollout")
        poses = np.stack([_pose(self.pose_bodies[int(seg_id)]) for seg_id in self.pose_ids]).astype(np.float64)
        obj = env.target
        qpos = as_numpy(obj.get_qpos()).reshape(-1)
        qvel = as_numpy(obj.get_dofs_velocity()).reshape(-1)
        object_state = {
            "time_s": t,
            "position_m": as_numpy(obj.get_pos(relative=False)).reshape(-1)[:3],
            "orientation_wxyz": qpos[3:7],
            "velocity_m_s": np.pad(qvel[:3], (0, max(0, 3 - len(qvel[:3]))))[:3],
        }
        self.rows.append({
            "time_s": t,
            "camera_to_world_cv": _camera_to_world(env.camera),
            "radiance": radiance,
            "display_rgb": display,
            "depth": depth,
            "segmentation": seg,
            "object_poses": poses,
            "object": object_state,
            "robot_qpos": as_numpy(env.robot.get_dofs_position()).reshape(-1),
            "robot_qvel": as_numpy(env.robot.get_dofs_velocity()).reshape(-1),
            "jaw_opening_m": env.jaw_opening_m(),
            "command": jsonable(command),
        })
        from PIL import Image
        Image.fromarray(display).save(self.frames_dir / f"frame_{len(self.rows) - 1:06d}.png")
        self.render_ms.append(elapsed_ms)
        self.last_capture_s = t

    def write(self, config: Dict[str, Any]) -> Path:
        path = self.output / "trajectory_cache.h5"
        rows = self.rows
        if len(rows) < 3:
            raise RuntimeError(f"trajectory cache needs at least 3 frames, got {len(rows)}")
        with h5py.File(path, "w") as handle:
            handle.attrs["rollout_id"] = self.output.name
            handle.attrs["sample_period_s"] = self.period_s
            handle.attrs["camera_resolution_wh"] = np.asarray(config["camera"]["resolution_wh"], dtype=np.int32)
            handle.attrs["vertical_fov_deg"] = float(config["camera"]["vertical_fov_deg"])
            handle.attrs["segmentation_pose_mapping"] = "every Panda link plus the moving target; static plane omitted"
            handle.attrs["same_scene_state_rgb_hdr"] = True
            handle.attrs["frame_timestamp_contract"] = "camera render captured at stored scene time, 100 Hz controller cadence"
            handle.create_dataset("segmentation_pose_ids", data=self.pose_ids)
            handle.create_dataset("timestamps_s", data=np.asarray([row["time_s"] for row in rows], dtype=np.float64))
            for name, dtype in (("radiance", "f4"), ("display_rgb", "u1"), ("depth", "f4"), ("segmentation", "i4"), ("object_poses", "f8")):
                stack = np.stack([row[name] for row in rows])
                handle.create_dataset(
                    name, data=stack, chunks=(1, *stack.shape[1:]),
                    compression="gzip", compression_opts=1,
                )
            handle.create_dataset("camera_to_world_cv", data=np.stack([r["camera_to_world_cv"] for r in rows]))
            for name in ("position_m", "orientation_wxyz", "velocity_m_s"):
                handle.create_dataset(f"target_{name}", data=np.stack([r["object"][name] for r in rows]))
            handle.create_dataset("robot_qpos", data=np.stack([r["robot_qpos"] for r in rows]))
            handle.create_dataset("robot_qvel", data=np.stack([r["robot_qvel"] for r in rows]))
            handle.create_dataset("jaw_opening_m", data=np.asarray([r["jaw_opening_m"] for r in rows]))
            handle.create_dataset("control_json", data=np.asarray([json.dumps(r["command"], ensure_ascii=False) for r in rows], dtype=h5py.string_dtype("utf-8")))
        frame_manifest = {
            "rollout_id": self.output.name,
            "frame_count": len(rows),
            "source_cadence_hz": 1.0 / self.period_s,
            "timestamps_s": [float(row["time_s"]) for row in rows],
            "timestamp_sha256": hashlib.sha256(np.asarray([r["time_s"] for r in rows], dtype=np.float64).tobytes()).hexdigest(),
            "frame_png_sha256": {
                path.name: _sha256(path) for path in sorted(self.frames_dir.glob("*.png"))
            },
        }
        (self.output / "frame_manifest.json").write_text(json.dumps(frame_manifest, indent=2), encoding="utf-8")
        return path


def _camera_to_world(camera) -> np.ndarray:
    from genesis_event_plugin.genesis_adapter import camera_to_world_cv
    return as_numpy(camera_to_world_cv(camera), dtype=np.float64).reshape(4, 4)


class _ReplayScene:
    def __init__(self):
        self.cur_t = 0.0


class _ReplayCamera:
    def __init__(self, res, intrinsics):
        self.res = tuple(int(v) for v in res)
        self.intrinsics = np.asarray(intrinsics, dtype=np.float64)


def _make_ours_events(cache_path: Path, output_dir: Path) -> np.ndarray:
    from genesis_event_plugin import GenesisPhysicsEventPlugin
    from genesis_event_plugin.noise_model import DvsNoiseConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, "r") as h5:
        timestamps = h5["timestamps_s"][:]
        radiance = h5["radiance"]
        display = h5["display_rgb"]
        depth = h5["depth"]
        seg = h5["segmentation"]
        ids = h5["segmentation_pose_ids"][:].astype(int)
        poses = h5["object_poses"]
        camera_poses = h5["camera_to_world_cv"][:]
        width, height = (int(v) for v in h5.attrs["camera_resolution_wh"])
        fov = float(h5.attrs["vertical_fov_deg"])
        # Genesis pinhole K uses continuous raster center and square pixels.
        focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
        K = np.asarray([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
        scene = _ReplayScene()
        camera = _ReplayCamera((width, height), K)
        state = {"index": 0}

        def frame_provider(_scene, _camera):
            i = state["index"]
            return {"rgb": display[i], "radiance": radiance[i], "depth": depth[i], "seg": seg[i]}

        def motion_provider(_scene, _camera):
            i = state["index"]
            return {"camera_to_world": camera_poses[i], "object_to_world": {int(sid): poses[i, j] for j, sid in enumerate(ids)}}

        ideal_dvs = DvsNoiseConfig(
            pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
            cutoff_hz=0.0, leak_rate_hz=0.0, leak_jitter_fraction=0.0,
            shot_noise_rate_hz=0.0, refractory_period_s=0.0,
            log_input=False, readout_model="ideal_readout",
        )
        plugin = GenesisPhysicsEventPlugin(
            preset="clean", noise_config=ideal_dvs, input_space="linear",
            motion_state_provider=motion_provider, frame_provider=frame_provider,
            disocclusion_policy="hold", device="cpu",
        )
        plugin.attach(scene, camera)
        plugin.start_episode()
        chunks = []
        start = time.perf_counter()
        for i, t in enumerate(timestamps):
            state["index"] = i
            scene.cur_t = float(t)
            events = plugin.capture()
            if len(events):
                chunks.append(np.asarray(events, dtype=np.float64))
        plugin.end_episode()
        plugin.close()
        plugin_ms = (time.perf_counter() - start) * 1000.0
        if not chunks:
            raise RuntimeError("Ours plugin generated zero events from shared cache")
        result = np.concatenate(chunks, axis=0)
        result = result[np.argsort(result[:, 0], kind="stable")]
    np.savez_compressed(output_dir / "events.npz", events=result)
    (output_dir / "ours_manifest.json").write_text(json.dumps({
        "generator": "GenesisPhysicsEventPlugin", "core": "frozen plugin core",
        "input_cache": str(cache_path), "input_cache_sha256": _sha256(cache_path),
        "input_frames": len(timestamps), "input_timestamps_sha256": hashlib.sha256(timestamps.tobytes()).hexdigest(),
        "event_count": len(result), "event_schema": "[t_seconds_from_launch, x_px, y_px, polarity]",
        "pixel_response": "shared ideal DvsEmulator thresholds 0.2, zero noise, infinite bandwidth",
        "plugin_wall_ms": plugin_ms,
    }, indent=2), encoding="utf-8")
    return result


def _write_source_episode(source_root: Path, episode_id: int, result: Dict[str, Any], trace: list[dict], events: np.ndarray) -> None:
    episode_dir = source_root / "episodes" / f"episode_{episode_id:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    fields = ["episode_id", "seed", "launch_speed_m_s", "start_y_m", "first_contact_s", "lift_start_s", "sim_duration_s"]
    with (episode_dir / "episode.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: result.get(key) for key in fields})
    with (episode_dir / "state_control_trace.jsonl").open("w", encoding="utf-8") as stream:
        for row in trace:
            stream.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
    np.savez_compressed(episode_dir / "direct_hdr_events_1khz_2khz.npz", events_2000hz=events)


def _hash_tree(root: Path) -> dict[str, dict[str, Any]]:
    return {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(root.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"}


def _write_hash_manifest(root: Path) -> dict[str, dict[str, Any]]:
    files = _hash_tree(root)
    (root / "SHA256SUMS").write_text("".join(f"{item['sha256']}  {name}\n" for name, item in files.items()), encoding="utf-8")
    return files


def _load_v2e_events(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        ds = handle["events"]
        events = np.asarray(ds[:], dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != 4:
        raise RuntimeError(f"unexpected V2E events shape: {events.shape}")
    # The official v2e smoke adapter records timestamps in microseconds; x/y
    # are pixels and polarity is converted to the shared +/-1 convention.
    events[:, 0] /= 1e6
    events[:, 3] = np.where(events[:, 3] > 0, 1.0, -1.0)
    return events[np.argsort(events[:, 0], kind="stable")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json")
    parser.add_argument("--control-config", type=Path, default=ROOT / "benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--y-values", type=float, nargs=2, default=[-0.055, 0.055])
    parser.add_argument("--seed", type=int, default=2026104130)
    parser.add_argument("--v2e-source", type=Path, required=True)
    parser.add_argument("--slomo-model", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-epochs", type=int, default=3)
    args = parser.parse_args()

    import torch
    import genesis as gs

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    scene_path, control_path = args.config.resolve(), args.control_config.resolve()
    config = json.loads(scene_path.read_text(encoding="utf-8"))
    config["shared_control"] = json.loads(control_path.read_text(encoding="utf-8"))
    gs.init(backend=gs.cpu, logging_level="warning")
    env = build_robot_scene(config, segmentation_level="link")
    # Materialize Genesis' link segmentation descriptors before resolving IDs.
    env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True)
    pose_bodies = _resolve_motion_bodies(env)
    cache_root = out / "trajectory_cache"
    ours_root, v2e_root = out / "ours_training", out / "v2e_training"
    common_episode_rows: list[tuple[dict, list[dict], Path]] = []
    capture_started = time.perf_counter()
    try:
        for episode_id, y in enumerate(args.y_values):
            episode_seed = args.seed + episode_id
            cache_dir = cache_root / f"episode_{episode_id:03d}"
            cache_dir.mkdir(parents=True, exist_ok=False)
            archive = RolloutArchive(env, cache_dir, pose_bodies)
            result, trace, frames = run_oracle_trial(
                env, episode_id, episode_seed, float(args.speed),
                float(config["controller"]["initial_deceleration_assumption_m_s2"]),
                initial_y_m=float(y), trajectory_cache=archive,
            )
            archive.write(config)
            if frames:
                save_gif(cache_dir / "oracle_rollout.gif", frames)
            write_trace(cache_dir / "oracle_trace.jsonl", trace)
            (cache_dir / "episode_result.json").write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
            common_episode_rows.append((result, trace, cache_dir))
        gs.destroy()
    except Exception:
        gs.destroy()
        raise
    capture_wall_s = time.perf_counter() - capture_started

    # Derive Ours and official V2E training events only after the shared source
    # rollout archives have closed, proving neither generator drove physics.
    ours_events_by_episode = {}
    generation_started = time.perf_counter()
    v2e_python = Path(args.python).resolve()
    for episode_id, (_result, _trace, cache_dir) in enumerate(common_episode_rows):
        cache_file = cache_dir / "trajectory_cache.h5"
        ours_events = _make_ours_events(cache_file, cache_dir / "ours_replay")
        ours_events_by_episode[episode_id] = ours_events
        v2e_out = cache_dir / "v2e_replay"
        command = [str(v2e_python), str(ROOT / "benchmark/scripts/v2e_official_pipeline_smoke.py"),
                   "--frames", str(cache_dir / "frames"), "--output-dir", str(v2e_out),
                   "--source", str(args.v2e_source.resolve()), "--python", str(v2e_python),
                   "--slomo-model", str(args.slomo_model.resolve())]
        proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (cache_dir / "v2e_driver.log").write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"official V2E pipeline failed for episode {episode_id}: {proc.stdout[-2000:]}")
        v2e_events = _load_v2e_events(v2e_out / "events.h5")
        _write_source_episode(ours_root, episode_id, _result, _trace, ours_events)
        _write_source_episode(v2e_root, episode_id, _result, _trace, v2e_events)
        frame_manifest_hash = _sha256(cache_dir / "frame_manifest.json")
        label_hashes = {}
        for relative in ("episode.csv", "state_control_trace.jsonl"):
            ours_label = ours_root / "episodes" / f"episode_{episode_id:03d}" / relative
            v2e_label = v2e_root / "episodes" / f"episode_{episode_id:03d}" / relative
            label_hashes[relative] = {"ours": _sha256(ours_label), "v2e": _sha256(v2e_label)}
            if label_hashes[relative]["ours"] != label_hashes[relative]["v2e"]:
                raise RuntimeError(f"paired label hash mismatch for episode {episode_id}: {relative}")
        with h5py.File(cache_file, "r") as cache_h5:
            cached_frame_count = int(len(cache_h5["timestamps_s"]))
        pair_provenance = {
            "episode_id": episode_id,
            "rollout_cache_path": str(cache_file),
            "rollout_cache_sha256": _sha256(cache_file),
            "frame_manifest_sha256": frame_manifest_hash,
            "source_frame_count": cached_frame_count,
            "timestamp_sha256": json.loads((cache_dir / "frame_manifest.json").read_text())["timestamp_sha256"],
            "shared_training_label_hashes": label_hashes,
            "split_unit": "episode",
            "assignment": "episode 0 train / episode 1 validation" if episode_id == 0 else "episode 1 validation",
        }
        (ours_root / "episodes" / f"episode_{episode_id:03d}" / "pair_provenance.json").write_text(
            json.dumps(pair_provenance, indent=2), encoding="utf-8")
        (v2e_root / "episodes" / f"episode_{episode_id:03d}" / "pair_provenance.json").write_text(
            json.dumps(pair_provenance, indent=2), encoding="utf-8")
        (cache_dir / "paired_event_counts.json").write_text(json.dumps({
            "episode_id": episode_id,
            "rollout_cache_sha256": _sha256(cache_file),
            "frame_manifest_sha256": _sha256(cache_dir / "frame_manifest.json") if (cache_dir / "frame_manifest.json").exists() else None,
            "ours_event_count": len(ours_events), "v2e_event_count": len(v2e_events),
            "input_timestamp_contract": "both generators use the same cached 100 Hz timestamps; only Ours uses cached HDR/depth/seg/poses, V2E uses display RGB + official SuperSloMo",
        }, indent=2), encoding="utf-8")
    generation_wall_s = time.perf_counter() - generation_started
    ours_hashes = _write_hash_manifest(ours_root)
    v2e_hashes = _write_hash_manifest(v2e_root)

    # Same architecture, episode split, labels, optimization budget and code.
    fit_started = time.perf_counter()
    train_outputs = {}
    for name, source in (("ours", ours_root), ("v2e", v2e_root)):
        train_out = out / f"{name}_fit"
        command = [str(v2e_python), "-m", "benchmark.robot_closed_loop.diagnostics.train_event_state_development",
                   "--config", str(scene_path), "--source-results", str(source), "--out", str(train_out),
                   "--train-episodes", "0", "--validation-episodes", "1",
                   "--epochs", str(args.train_epochs), "--patience", str(args.train_epochs),
                   "--batch-size", "32", "--seed", "606", "--threads", "2", "--architecture", "heatmap_v2",
                   "--source-label", "Ours cached Direct-HDR plugin events" if name == "ours" else "official V2E cached-display-RGB events"]
        proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONPATH": str(ROOT)})
        (out / f"{name}_fit.log").write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"{name} tiny training smoke failed: {proc.stdout[-2000:]}")
        train_outputs[name] = {
            "source_root": str(source), "source_manifest_sha256": _sha256(source / "SHA256SUMS"),
            "training_summary_sha256": _sha256(train_out / "development_summary.json"),
            "checkpoint_sha256": _sha256(train_out / "spatial_event_state.pt"),
            "validation_metrics": json.loads((train_out / "development_summary.json").read_text())["validation_metrics"],
        }
    fit_wall_s = time.perf_counter() - fit_started

    # Compare source cache/frame/label/split hashes and make cost boundaries explicit.
    cache_files = {str(path.relative_to(cache_root)): _sha256(path)
                   for path in sorted(cache_root.rglob("*")) if path.is_file() and path.name not in {"ours_manifest.json"}}
    rollout_rows = []
    for episode_id, row in enumerate(common_episode_rows):
        with h5py.File(row[2] / "trajectory_cache.h5", "r") as cache:
            timestamps = cache["timestamps_s"][:]
        rollout_rows.append({
            "episode_id": episode_id, "seed": row[0]["seed"], "speed_m_s": row[0]["launch_speed_m_s"],
            "start_y_m": row[0]["start_y_m"], "strict_success": row[0]["success"],
            "cache_h5_sha256": _sha256(row[2] / "trajectory_cache.h5"),
            "timestamp_sha256": hashlib.sha256(timestamps.tobytes()).hexdigest(),
        })
    result = {
        "status": "PASS",
        "purpose": "minimum paired pipeline smoke only; no ranking, protocol freeze, or generalization claim",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scene_config_sha256": _sha256(scene_path), "shared_control_config_sha256": _sha256(control_path),
        "rollouts": rollout_rows,
        "paired_inputs": {
            "ours_source_manifest_sha256": _sha256(ours_root / "SHA256SUMS"),
            "v2e_source_manifest_sha256": _sha256(v2e_root / "SHA256SUMS"),
            "same_trajectory_cache_hashes": True,
            "same_frame_timestamps": True,
            "same_GT_labels_and_episode_split": True,
            "episode_split": {"train": [0], "validation": [1], "split_unit": "episode", "smoke_only": True},
            "ours_source_files": ours_hashes, "v2e_source_files": v2e_hashes,
        },
        "training_smoke": train_outputs,
        "fairness_boundary": {
            "same_scene_rollouts_frames_timestamps_labels_and_split": True,
            "ours_temporal_path": "GenesisPhysicsEventPlugin physics/depth/segmentation warp over cached scene-linear HDR endpoints",
            "v2e_temporal_path": "official V2E + SuperSloMo from cached display RGB at 100 Hz; frozen V2E sensor parameters",
            "pixel_response_is_not_shared_for_training_generators": "Ours smoke forces ideal DvsEmulator thresholds 0.2, no noise/infinite bandwidth; official V2E uses benchmark/configs/v2e.yaml with sigma 0.03, cutoff 300 Hz, leak/shot noise and refractory 0.5 ms. Do not attribute a later performance difference solely to interpolation until the response-model comparison is separately aligned/frozen.",
            "direct_hdr_r2_reference": "run.py reference mode uses common DirectHdrEventSource and its shared ideal DvsEmulator for both learned checkpoints; this is a common evaluation event source, distinct from each training generator.",
            "no_GT_in_generator_or_model_inputs": True,
        },
        "cost_s": {"shared_physics_robot_rollout_and_cache": capture_wall_s,
                   "ours_and_official_v2e_generation": generation_wall_s,
                   "two_tiny_fit_calls": fit_wall_s,
                   "wall_scope": "full per-stage Python wall time; excludes subsequent two R2 live rollouts"},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "physics_backend": "CPU", "frame_cadence_hz": 100, "physics_dt_s": float(config["physics"]["dt_s"])},
    }
    (out / "paired_pipeline_smoke.json").write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
    (out / "cache_sha256_manifest.json").write_text(json.dumps(cache_files, indent=2), encoding="utf-8")
    _write_hash_manifest(out)
    print(json.dumps({"status": "PASS", "output": str(out), "rollouts": len(common_episode_rows),
                      "training_smoke": {name: value["checkpoint_sha256"] for name, value in train_outputs.items()},
                      "cost_s": result["cost_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
