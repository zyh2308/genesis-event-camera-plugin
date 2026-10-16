#!/usr/bin/env python3
"""Generate the fixed Stage-4 shared 32-rollout physical bank.

The camera/object trajectories are deterministic scripted kinematic paths
inside Genesis. Genesis advances at 1 ms; radiance, depth, segmentation, poses
and labels are captured at the registered 100 Hz endpoints. EventGenesis uses
the captured linear radiance and physical endpoint poses. V2E input frames are
written from the same fixed display mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np


CORE = "e79e656ac00f7847b98db9e12bf43b093f3afff1"
STRATA = (
    "camera_motion_static_scene",
    "independent_rigid_motion_static_camera",
    "camera_and_independent_rigid_motion",
    "combined_motion_with_occlusion",
)
WIDTH, HEIGHT = 346, 260
FPS, DT, DURATION = 100, 0.001, 2.0
FRAMES = int(FPS * DURATION) + 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def radiance_to_v2e_luma_u8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    y = np.maximum(0.0, 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
    tone = y / (1.0 + y)
    srgb = np.where(tone <= 0.0031308, 12.92 * tone, 1.055 * np.power(tone, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(255.0 * srgb), 0, 255).astype(np.uint8)


def registered_episodes() -> list[dict]:
    rows = []
    for stratum_index, stratum in enumerate(STRATA):
        for local in range(8):
            rows.append({
                "id": f"s{stratum_index}_{local:02d}",
                "stratum": stratum,
                "stratum_index": stratum_index,
                "local_index": local,
                "split": "train" if local < 6 else "validation",
                "scene_seed": 20261016 + 100 * stratum_index + local,
                "noise_seed": 42,
            })
    return rows


def verify_flow_direction() -> dict:
    from genesis_event_plugin.physics_warp import compute_se3_flow
    depth = np.ones((3, 4), dtype=np.float32)
    seg = np.ones((3, 4), dtype=np.int32)
    K = np.asarray([[2., 0., 1.5], [0., 2., 1.], [0., 0., 1.]])
    T0 = np.eye(4)
    delta = np.eye(4); delta[0, 3] = .25
    flow, valid = compute_se3_flow(depth, seg, K, T0, T0, {1: delta})
    observed = flow[:, 1, 1]
    if not valid[1, 1] or not np.allclose(observed, (.5, 0.), atol=1e-6):
        raise RuntimeError(f"forward flow direction fixture failed: {observed}")
    return {"status": "pass", "expected_uv_px": [0.5, 0.0], "observed_uv_px": observed.tolist()}


def trajectory_pose(t: float, episode: dict, object_index: int, base: np.ndarray):
    """Registered smooth position path; all amplitudes are in world meters."""
    stratum = episode["stratum"]
    moving = (
        stratum == "independent_rigid_motion_static_camera" and object_index == 0
    ) or (
        stratum in ("camera_and_independent_rigid_motion", "combined_motion_with_occlusion")
        and object_index == 0
    ) or (stratum == "combined_motion_with_occlusion" and object_index == 1)
    if not moving:
        return base.copy()
    phase = 2.0 * math.pi * (0.42 + 0.07 * object_index) * t + 0.37 * episode["local_index"]
    amp = 0.11 if object_index == 0 else 0.15
    pos = base.copy()
    pos[0] += amp * math.sin(phase)
    pos[1] += 0.72 * amp * math.sin(phase * 0.73 + 0.8)
    pos[2] += 0.035 * math.sin(phase * 1.21)
    return pos


def camera_pose(t: float, episode: dict):
    base = np.asarray([1.05, 1.12, 0.78], dtype=np.float64)
    look = np.asarray([0.0, 0.0, 0.24], dtype=np.float64)
    if episode["stratum"] in ("camera_motion_static_scene", "camera_and_independent_rigid_motion", "combined_motion_with_occlusion"):
        phase = 2.0 * math.pi * 0.31 * t + 0.19 * episode["local_index"]
        base += np.asarray([0.12 * math.sin(phase), 0.10 * math.sin(phase * .81 + .6), 0.035 * math.sin(phase * 1.17)])
        look += np.asarray([0.025 * math.sin(phase * .65), 0.02 * math.cos(phase * .7), 0.0])
    return base, look


def run_episode(gs, episode: dict, root: Path, h5: h5py.File, completed_before: int) -> dict:
    from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
    from genesis_event_plugin.genesis_adapter import camera_to_world_cv, link_to_world_cv
    from genesis_event_plugin.noise_model import DvsNoiseConfig
    from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
    from genesis_event_plugin.physics_warp import project_endpoint_to_alpha

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, -9.81)),
        vis_options=gs.options.VisOptions(ambient_light=(0.38, 0.38, 0.38), segmentation_level="link"),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.38, 0.40, 0.43, 1.0)))
    rng = np.random.default_rng(episode["scene_seed"])
    palette = ((.88, .16, .09, 1.), (.08, .68, .22, 1.), (.10, .28, .92, 1.), (.92, .68, .08, 1.))
    objects = []
    bases = []
    sizes = []
    for oi in range(3):
        size = float(rng.uniform(.10, .16))
        base = np.asarray([rng.uniform(-.25, .25), rng.uniform(-.20, .20), size * .5 + .025])
        if episode["stratum"] == "combined_motion_with_occlusion" and oi == 1:
            # Cross the center at a nearer diagonal depth to create occlusion.
            base = np.asarray([-.26, -.16, .16])
            size = .16
        body = scene.add_entity(
            gs.morphs.Box(pos=tuple(base), size=(size, size * rng.uniform(.75, 1.25), size * rng.uniform(.75, 1.2))),
            surface=gs.surfaces.Default(color=palette[(oi + episode["local_index"]) % len(palette)], emissive=(.015, .015, .015)),
        )
        objects.append(body); bases.append(base); sizes.append(size)
    initial_cam, initial_look = camera_pose(0.0, episode)
    camera = scene.add_camera(
        res=(WIDTH, HEIGHT), pos=tuple(initial_cam), lookat=tuple(initial_look),
        fov=58.0, GUI=False, near=.03, far=8.0,
    )
    scene.build()

    links = {int(link.idx): link for link in scene.rigid_solver.links}
    object_ids = sorted(int(body.links[0].idx) for body in objects)
    if any(object_id not in links for object_id in object_ids):
        raise RuntimeError("Genesis object link/segmentation contract failed")
    frame_cache, state_cache = {}, {}

    def frame_provider(active_scene, active_camera):
        rgb, depth, seg, _ = active_camera.render(rgb=True, depth=True, segmentation=True, radiance=True)
        payload = {"rgb": as_numpy(rgb).astype(np.float32), "radiance": as_numpy(rgb).astype(np.float32),
                   "depth": as_numpy(depth).astype(np.float32), "seg": as_numpy(seg).astype(np.int32)}
        frame_cache["last"] = payload
        return payload

    def state_provider(active_scene, active_camera):
        state = {"camera_to_world": camera_to_world_cv(active_camera),
                 "object_to_world": {sid: link_to_world_cv(link) for sid, link in links.items()}}
        state_cache["last"] = state
        return state

    sensor = DvsNoiseConfig(
        pos_thres=.2, neg_thres=.2, sigma_thres=.03, cutoff_hz=300., leak_rate_hz=.01,
        leak_jitter_fraction=.1, noise_rate_cov_decades=.1, shot_noise_rate_hz=.001,
        refractory_period_s=.0005, seed=42, radiance_input=True, response_mode="physical_log",
        radiometric_calibration_status="uncalibrated",
    )
    plugin = GenesisPhysicsEventPlugin(
        input_space="linear", response_mode="physical_log", radiometric_calibration_status="uncalibrated",
        noise_config=sensor, sampler_config=AdaptiveSamplingConfig(), frame_provider=frame_provider,
        motion_state_provider=state_provider, device="cpu", warp_backend="numpy",
    )
    plugin.attach(scene, camera).start_episode()

    times, radiance, depths, segmentations, camera_poses, object_poses, display, event_chunks = [], [], [], [], [], [], [], []
    started = time.monotonic()
    for endpoint in range(FRAMES):
        for substep in range(10 if endpoint else 0):
            t = (endpoint - 1 + (substep + 1) / 10.0) / FPS
            for oi, body in enumerate(objects):
                body.set_pos(trajectory_pose(t, episode, oi, bases[oi]))
            cam_pos, look = camera_pose(t, episode)
            camera.set_pose(pos=cam_pos, lookat=look)
            scene.step()
        chunk = np.asarray(plugin.capture(), dtype=np.float64).reshape(-1, 4)
        payload, state = frame_cache["last"], state_cache["last"]
        t = float(scene.cur_t)
        if not np.isfinite(t) or payload["radiance"].shape != (HEIGHT, WIDTH, 3):
            raise RuntimeError(f"invalid endpoint render/time at {episode['id']} frame {endpoint}: t={t}")
        if payload["depth"].shape != (HEIGHT, WIDTH) or payload["seg"].shape != (HEIGHT, WIDTH):
            raise RuntimeError("Genesis depth/segmentation shape mismatch")
        if not np.isfinite(payload["radiance"]).all():
            raise RuntimeError("non-finite linear radiance")
        object_pixels = sum(int(np.count_nonzero(payload["seg"] == sid)) for sid in object_ids)
        if object_pixels == 0:
            raise RuntimeError(f"no moving-scene object pixels visible in {episode['id']} frame {endpoint}")
        times.append(t); radiance.append(payload["radiance"]); depths.append(payload["depth"])
        segmentations.append(payload["seg"]); camera_poses.append(state["camera_to_world"])
        object_poses.append(state["object_to_world"]); display.append(radiance_to_v2e_luma_u8(payload["radiance"]))
        if endpoint:
            event_chunks.append(chunk)
        if endpoint % 20 == 0:
            elapsed = time.monotonic() - started
            atomic_json(root / "generation_progress.json", {
                "completed_rollouts": completed_before, "total_rollouts": 32,
                "active_rollout": episode["id"], "active_endpoint": endpoint,
                "active_total_endpoints": FRAMES, "active_rollout_elapsed_s": round(elapsed, 2),
                "active_endpoint_rate_hz": round(endpoint / elapsed, 3) if elapsed > 0 else None,
                "updated_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
            })
            print(json.dumps({"event": "render_progress", "rollout": episode["id"], "endpoint": endpoint,
                              "total_endpoints": FRAMES, "elapsed_s": round(time.monotonic() - started, 2)}), flush=True)

    times = np.asarray(times, dtype=np.float64)
    if not np.allclose(np.diff(times), .01, atol=1e-7, rtol=0.0):
        raise RuntimeError(f"nonuniform endpoint timestamps in {episode['id']}")
    K = as_numpy(camera.intrinsics).astype(np.float64)
    object_pose_array = np.stack([[state[sid] for sid in sorted(state)] for state in object_poses]).astype(np.float64)
    all_seg_ids = sorted(object_poses[0])
    pose_col = {sid: i for i, sid in enumerate(all_seg_ids)}
    H, W = HEIGHT, WIDTH
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    flow30, valid30 = [], []
    for i in range(0, FRAMES - 3, 3):
        uv, projected_z, geometric = project_endpoint_to_alpha(
            depths[i], segmentations[i], K, camera_poses[i], camera_poses[i + 3], 1.0,
            source_index=0,
            object_world0=object_poses[i], object_world1=object_poses[i + 3],
        )
        target_x, target_y = uv[0], uv[1]
        inside = geometric & (target_x >= 0) & (target_x <= W - 1) & (target_y >= 0) & (target_y <= H - 1)
        target_depth = cv2.remap(
            depths[i + 3].astype(np.float32), np.where(inside, target_x, -1).astype(np.float32),
            np.where(inside, target_y, -1).astype(np.float32), cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        tolerance = np.maximum(.02, .02 * projected_z)
        visible = inside & np.isfinite(target_depth) & (target_depth > 1e-6) & (np.abs(target_depth - projected_z) <= tolerance)
        flow = np.stack((target_x - xx, target_y - yy), axis=0).astype(np.float32)
        flow[:, ~visible] = 0.0
        flow30.append(flow); valid30.append(visible.astype(np.uint8))
    flow30, valid30 = np.stack(flow30), np.stack(valid30)
    if flow30.shape != (66, 2, H, W) or valid30.shape != (66, H, W):
        raise RuntimeError(f"30 ms label count/shape mismatch in {episode['id']}: {flow30.shape}")

    events = np.concatenate([c for c in event_chunks if len(c)], axis=0) if any(len(c) for c in event_chunks) else np.empty((0, 4), dtype=np.float64)
    if len(events) and (not np.isfinite(events).all() or not np.all(np.diff(events[:, 0]) >= 0)):
        raise RuntimeError(f"EventGenesis returned invalid events in {episode['id']}")
    event_counts_10ms = np.asarray([len(c) for c in event_chunks], dtype=np.int64)

    v2e_dir = root / "v2e_input" / episode["id"]
    v2e_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(display):
        path = v2e_dir / f"{i:06d}.png"
        if not cv2.imwrite(str(path), image):
            raise IOError(f"failed to write {path}")

    group = h5.create_group(f"episodes/{episode['id']}")
    for key in ("split", "stratum", "scene_seed", "noise_seed"):
        group.attrs[key] = episode[key]
    group.attrs["physical_trajectory_kind"] = "Genesis scripted kinematic camera/object path, scene stepped at 1 ms"
    group.attrs["camera_frame_convention"] = "OpenCV x-right/y-down/z-forward"
    group.attrs["world_frame_convention"] = "Genesis z-up; metric translations"
    group.attrs["timestamp_unit"] = "second, relative to rollout start"
    group.attrs["flow_contract"] = "forward source-pixel displacement from endpoint i to i+3; px per 30 ms interval"
    group.attrs["validity_contract"] = "source geometry, in-bounds target projection, and nearest-target-depth agreement within max(2 cm, 2%); common to both generators"
    group.attrs["v2e_display_mapping"] = "Rec.709 linear Y, exposure 1, Reinhard, sRGB transfer, uint8 round-to-nearest; fixed across endpoints"
    group.create_dataset("frame_timestamps_s", data=times)
    group.create_dataset("radiance_linear_rgb", data=np.stack(radiance).astype(np.float32), compression="gzip", compression_opts=1)
    group.create_dataset("depth_m", data=np.stack(depths).astype(np.float32), compression="gzip", compression_opts=1)
    group.create_dataset("segmentation_link_idx", data=np.stack(segmentations).astype(np.int32), compression="gzip", compression_opts=1)
    group.create_dataset("v2e_display_luma_u8", data=np.stack(display), compression="gzip", compression_opts=1)
    group.create_dataset("camera_intrinsics_K_px", data=K)
    group.create_dataset("camera_to_world_cv", data=np.stack(camera_poses).astype(np.float64))
    group.create_dataset("object_segmentation_link_ids", data=np.asarray(all_seg_ids, dtype=np.int32))
    group.create_dataset("object_to_world", data=object_pose_array)
    group.create_dataset("flow_forward_30ms_px", data=flow30, compression="gzip", compression_opts=1)
    group.create_dataset("flow_valid_30ms", data=valid30, compression="gzip", compression_opts=1)
    group.create_dataset("eventgenesis_events_txyp", data=events, compression="gzip", compression_opts=1)
    group.create_dataset("eventgenesis_interval_offsets_10ms", data=np.concatenate(([0], np.cumsum(event_counts_10ms))))
    group.create_dataset("eventgenesis_interval_counts_10ms", data=event_counts_10ms)
    group.attrs["target_object_segmentation_link_ids"] = json.dumps(object_ids)
    h5.flush()
    return {
        "id": episode["id"], "split": episode["split"], "stratum": episode["stratum"],
        "scene_seed": episode["scene_seed"], "frames": FRAMES, "time_s": [float(times[0]), float(times[-1])],
        "eventgenesis_events": int(len(events)), "valid_fraction_30ms_mean": float(valid30.mean()),
        "flow30_shape": list(flow30.shape), "K": K.tolist(),
        "v2e_input_files": len(display), "v2e_input_sha256": sha256(v2e_dir / "000000.png"),
        "rollout_seconds": round(time.monotonic() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    launch = json.loads(config_path.read_text())
    bank_cfg = launch["shared_bank"]
    if (bank_cfg["rollouts"], bank_cfg["duration_s"], bank_cfg["sensor_wh"],
            bank_cfg["physics_dt_s"], bank_cfg["radiance_endpoint_hz"], bank_cfg["expected_frames_per_rollout"]) != (
            32, 2.0, [346, 260], .001, 100, 201):
        raise RuntimeError("Stage4 configuration changed from registered bank size/duration")
    if bank_cfg["train_rollouts"] != 24 or bank_cfg["validation_rollouts"] != 8 or launch["flow_dt1"]["window_duration_s"] != .03:
        raise RuntimeError("Stage4 resolution/window contract mismatch")
    event_cfg = launch["generators"]["eventgenesis"]
    required_event = {"input_space": "linear", "response_mode": "physical_log", "radiometric_calibration_status": "uncalibrated",
                      "warp_backend": "numpy", "device": "cpu", "pos_thres": .2, "neg_thres": .2,
                      "sigma_thres": .03, "cutoff_hz": 300., "leak_rate_hz": .01,
                      "leak_jitter_fraction": .1, "noise_rate_cov_decades": .1,
                      "shot_noise_rate_hz": .001, "refractory_period_s": .0005, "seed": 42}
    if any(event_cfg.get(k) != value for k, value in required_event.items()):
        raise RuntimeError("EventGenesis generator settings differ from the locked Stage4 config")
    if launch["generators"]["v2e"]["timestamp_resolution_s"] != 2.5e-5 or launch["generators"]["forbid_V2E_1ms_fallback"] is not True:
        raise RuntimeError("V2E timestamp configuration is not the registered 25-us mode")
    root = args.output_root.resolve(); root.mkdir(parents=True, exist_ok=False)
    episodes = registered_episodes()
    (root / "registered_split.json").write_text(json.dumps(episodes, indent=2) + "\n")
    (root / "stage4_training_launch.json").write_bytes(config_path.read_bytes())

    import genesis as gs
    if getattr(gs, "__version__", None) != "1.2.2":
        raise RuntimeError(f"expected Genesis 1.2.2, found {getattr(gs, '__version__', None)}")
    gs.init(backend=gs.gpu, logging_level="warning")
    repo_root = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema": "stage4_shared_flow_bank_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "repo_commit": commit, "frozen_simulator_core": CORE,
        "config_sha256": sha256(config_path), "split_seed": launch["shared_bank"]["split_seed"],
        "flow_direction_check": verify_flow_direction(), "episodes": [],
    }
    h5_path = root / "shared_trajectory.h5.partial"
    try:
        with h5py.File(h5_path, "w") as h5:
            h5.attrs["schema"] = "stage4_shared_flow_bank_v1"
            h5.attrs["config_sha256"] = manifest["config_sha256"]
            h5.attrs["frozen_simulator_core"] = CORE
            h5.attrs["split_unit"] = "whole independent rollout; validation groups have distinct scene_seed and trajectory ID"
            h5.attrs["endpoint_hz"] = FPS; h5.attrs["physics_dt_s"] = DT; h5.attrs["duration_s"] = DURATION
            h5.attrs["resolution_wh"] = (WIDTH, HEIGHT)
            h5.create_group("episodes")
            for idx, episode in enumerate(episodes, 1):
                start = time.monotonic()
                manifest["episodes"].append(run_episode(gs, episode, root, h5, idx - 1))
                atomic_json(root / "generation_progress.json", {
                    "completed_rollouts": idx, "total_rollouts": len(episodes),
                    "latest_rollout": episode["id"], "elapsed_s": round(time.monotonic() - start, 2),
                    "updated_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                })
                print(json.dumps({"event": "rollout_complete", **manifest["episodes"][-1],
                                  "completed": idx, "total": len(episodes)}), flush=True)
    finally:
        gs.destroy()
    manifest["episodes"] = sorted(manifest["episodes"], key=lambda e: e["id"])
    if len(manifest["episodes"]) != 32 or sum(e["split"] == "train" for e in manifest["episodes"]) != 24:
        raise RuntimeError("incomplete registered bank")
    os.replace(h5_path, root / "shared_trajectory.h5")
    manifest["bank_sha256"] = sha256(root / "shared_trajectory.h5")
    manifest["split_sha256"] = sha256(root / "registered_split.json")
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(root / "bank_manifest.json", manifest)
    atomic_json(root / "generation_progress.json", {
        "completed_rollouts": 32, "total_rollouts": 32, "status": "complete",
        "bank_sha256": manifest["bank_sha256"], "updated_utc": manifest["finished_utc"],
    })
    print(json.dumps({"status": "complete", "bank_sha256": manifest["bank_sha256"], "rollouts": 32}, indent=2), flush=True)


if __name__ == "__main__":
    main()
