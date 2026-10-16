#!/usr/bin/env python3
"""Render a tiny server-side physics fixture shared by EventGenesis and V2E."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def radiance_to_v2e_luma_u8(rgb: np.ndarray, exposure: float) -> np.ndarray:
    """Fixed global scene-linear Y -> Reinhard -> sRGB mapping; no frame fit."""
    rgb = np.asarray(rgb, dtype=np.float32)
    y = np.maximum(0.0, 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
    tone = (exposure * y) / (1.0 + exposure * y)
    srgb = np.where(tone <= 0.0031308, 12.92 * tone, 1.055 * np.power(tone, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(255.0 * srgb), 0, 255).astype(np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_forward_flow_direction() -> dict:
    """Check that positive object-x motion produces positive source-to-target u."""
    from genesis_event_plugin.physics_warp import compute_se3_flow

    depth = np.ones((3, 4), dtype=np.float32)
    segmentation = np.ones((3, 4), dtype=np.int32)
    intrinsics = np.asarray([[2.0, 0.0, 1.5], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    camera_to_world = np.eye(4, dtype=np.float64)
    object_delta = np.eye(4, dtype=np.float64)
    object_delta[0, 3] = 0.25
    flow, valid = compute_se3_flow(
        depth, segmentation, intrinsics, camera_to_world, camera_to_world,
        object_world_deltas={1: object_delta},
    )
    center = (1, 1)
    center_flow = flow[:, center[0], center[1]]
    if not valid[center] or not np.isclose(center_flow[0], 0.5, atol=1e-6):
        raise RuntimeError(f"forward flow sign/scale fixture failed: {center_flow}")
    if not np.isclose(center_flow[1], 0.0, atol=1e-6):
        raise RuntimeError(f"forward flow vertical component fixture failed: {center_flow}")
    return {
        "status": "pass",
        "object_x_translation_m": 0.25,
        "depth_m": 1.0,
        "fx_px": 2.0,
        "expected_forward_u_displacement_px": 0.5,
        "observed_flow_xy_px": center_flow.astype(float).tolist(),
        "valid": bool(valid[center]),
    }


def run_episode(gs, config: dict, episode: dict, output_root: Path, h5: h5py.File) -> dict:
    from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
    from genesis_event_plugin.genesis_adapter import camera_to_world_cv, link_to_world_cv
    from genesis_event_plugin.noise_model import DvsNoiseConfig
    from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
    from genesis_event_plugin.physics_warp import compute_se3_flow

    sim = config["simulation"]
    width, height = map(int, sim["image_size_wh"])
    dt = float(sim["physics_dt_s"])
    rate = float(sim["endpoint_rate_hz"])
    steps_per_frame = int(round(1.0 / (rate * dt)))
    interval_count = int(round(float(sim["duration_s"]) * rate))
    if steps_per_frame < 1 or not np.isclose(steps_per_frame * dt, 1.0 / rate):
        raise ValueError("physics dt must divide the endpoint interval exactly")
    if interval_count < 1:
        raise ValueError("duration must contain at least one endpoint interval")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=tuple(sim["gravity_m_s2"])),
        vis_options=gs.options.VisOptions(ambient_light=(0.3, 0.3, 0.3), segmentation_level="link"),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Default(color=(0.48, 0.48, 0.48, 1.0)),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            pos=(
                float(episode["initial_x_m"]),
                float(episode.get("initial_y_m", 0.0)),
                float(episode.get("initial_z_m", sim["target_box"]["initial_z_m"])),
            ),
            size=tuple(sim["target_box"]["size_m"]),
        ),
        surface=gs.surfaces.Default(color=(0.9, 0.16, 0.08, 1.0), emissive=(0.05, 0.0, 0.0)),
    )
    cam_cfg = sim["camera"]
    camera = scene.add_camera(
        res=(width, height),
        pos=tuple(cam_cfg["position_m"]),
        lookat=tuple(cam_cfg["look_at_m"]),
        fov=float(cam_cfg["vertical_fov_deg"]),
        GUI=False,
        near=float(cam_cfg["near_m"]),
        far=float(cam_cfg["far_m"]),
    )
    scene.build()

    links = {int(link.idx): link for link in scene.rigid_solver.links}
    target_seg_id = int(box.links[0].idx)
    if target_seg_id not in links:
        raise RuntimeError(f"target link {target_seg_id} is absent from scene.rigid_solver.links")

    frame_cache: dict = {}
    state_cache: dict = {}

    def frame_provider(active_scene, active_camera):
        rgb, depth, seg, _ = active_camera.render(
            rgb=True, depth=True, segmentation=True, radiance=True
        )
        payload = {
            "rgb": as_numpy(rgb).astype(np.float32, copy=False),
            "radiance": as_numpy(rgb).astype(np.float32, copy=False),
            "depth": as_numpy(depth).astype(np.float32, copy=False),
            "seg": as_numpy(seg).astype(np.int64, copy=False),
        }
        frame_cache["last"] = payload
        return payload

    def motion_state_provider(active_scene, active_camera):
        state = {
            "camera_to_world": camera_to_world_cv(active_camera),
            "object_to_world": {
                seg_id: link_to_world_cv(link) for seg_id, link in links.items()
            },
        }
        state_cache["last"] = state
        return state

    sensor = config["eventgenesis"]
    episode_seed = int(sensor["seed"]) + int(episode.get("seed_offset", 0))
    noise = DvsNoiseConfig(
        pos_thres=float(sensor["pos_thres"]),
        neg_thres=float(sensor["neg_thres"]),
        sigma_thres=float(sensor["sigma_thres"]),
        cutoff_hz=float(sensor["cutoff_hz"]),
        leak_rate_hz=float(sensor["leak_rate_hz"]),
        leak_jitter_fraction=float(sensor["leak_jitter_fraction"]),
        noise_rate_cov_decades=float(sensor["noise_rate_cov_decades"]),
        shot_noise_rate_hz=float(sensor["shot_noise_rate_hz"]),
        refractory_period_s=float(sensor["refractory_period_s"]),
        seed=episode_seed,
        radiance_input=True,
        response_mode=str(sensor["response_mode"]),
        radiometric_calibration_status=str(sensor["radiometric_calibration_status"]),
    )
    plugin = GenesisPhysicsEventPlugin(
        input_space="linear",
        response_mode=str(sensor["response_mode"]),
        radiometric_calibration_status=str(sensor["radiometric_calibration_status"]),
        noise_config=noise,
        sampler_config=AdaptiveSamplingConfig(),
        frame_provider=frame_provider,
        motion_state_provider=motion_state_provider,
        device=str(sensor["device"]),
        warp_backend=str(sensor["warp_backend"]),
    )
    plugin.attach(scene, camera).start_episode()

    timestamps = []
    radiance_frames = []
    depths = []
    segmentations = []
    camera_poses = []
    object_poses = []
    display_frames = []
    event_chunks = []
    event_counts_per_interval = []
    for frame_index in range(interval_count + 1):
        if frame_index:
            for _ in range(steps_per_frame):
                scene.step()
        chunk = np.asarray(plugin.capture(), dtype=np.float64).reshape(-1, 4)
        payload = frame_cache["last"]
        state = state_cache["last"]
        t = float(scene.cur_t)
        if not np.isfinite(t):
            raise RuntimeError("scene.cur_t is not finite")
        if payload["radiance"].shape != (height, width, 3):
            raise RuntimeError(f"unexpected radiance shape {payload['radiance'].shape}")
        if payload["depth"].shape != (height, width) or payload["seg"].shape != (height, width):
            raise RuntimeError("rendered depth/segmentation shape does not match configured size")
        if not np.isfinite(payload["radiance"]).all():
            raise RuntimeError("radiance contains non-finite values")
        if int((payload["seg"] == target_seg_id).sum()) == 0:
            raise RuntimeError(f"target segmentation ID {target_seg_id} is not visible")
        timestamps.append(t)
        radiance_frames.append(payload["radiance"])
        depths.append(payload["depth"])
        segmentations.append(payload["seg"])
        camera_poses.append(state["camera_to_world"])
        object_poses.append(state["object_to_world"])
        display_frames.append(
            radiance_to_v2e_luma_u8(payload["radiance"], float(config["v2e_input_mapping"]["exposure"]))
        )
        if frame_index:
            event_chunks.append(chunk)
            event_counts_per_interval.append(int(len(chunk)))

    timestamps_a = np.asarray(timestamps, dtype=np.float64)
    if not np.isclose(timestamps_a[0], 0.0, atol=1e-9, rtol=0.0):
        raise RuntimeError(f"rollout timestamp does not start at zero: {timestamps_a[0]}")
    if not np.allclose(np.diff(timestamps_a), 1.0 / rate, atol=1e-7, rtol=0.0):
        raise RuntimeError(f"simulator endpoint timestamps are irregular: {np.diff(timestamps_a)}")

    object_ids = sorted(object_poses[0])
    if any(sorted(state) != object_ids for state in object_poses):
        raise RuntimeError("segmentation-link pose set changed within the rollout")
    object_pose_array = np.stack(
        [[state[seg_id] for seg_id in object_ids] for state in object_poses]
    ).astype(np.float64)
    target_column = object_ids.index(target_seg_id)
    target_pose_array = object_pose_array[:, target_column]

    def compute_interval_flow(i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        state0, state1 = object_poses[i], object_poses[j]
        deltas = {
            seg_id: state1[seg_id] @ np.linalg.inv(state0[seg_id])
            for seg_id in object_ids
        }
        flow, valid = compute_se3_flow(
            depths[i], segmentations[i], as_numpy(camera.intrinsics),
            camera_poses[i], camera_poses[j], object_world_deltas=deltas,
        )
        flow = np.asarray(flow, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        target_x = xx + flow[0]
        target_y = yy + flow[1]
        valid &= (
            np.isfinite(flow).all(axis=0)
            & (target_x >= 0.0) & (target_x <= width - 1.0)
            & (target_y >= 0.0) & (target_y <= height - 1.0)
        )
        flow[:, ~valid] = 0.0
        return flow, valid.astype(np.uint8)

    flow_frames, flow_valid = zip(
        *(compute_interval_flow(i, i + 1) for i in range(interval_count))
    )
    flow_dt4_frames, flow_dt4_valid = zip(
        *(
            compute_interval_flow(i, i + 4)
            for i in range(max(0, interval_count - 3))
        )
    ) if interval_count >= 4 else ((), ())

    events = (
        np.concatenate([chunk for chunk in event_chunks if len(chunk)], axis=0)
        if any(len(chunk) for chunk in event_chunks)
        else np.empty((0, 4), dtype=np.float64)
    )
    if len(events) and (not np.isfinite(events).all() or not np.all(np.diff(events[:, 0]) >= 0)):
        raise RuntimeError("EventGenesis returned non-finite or unsorted raw events")
    event_offsets = np.concatenate(([0], np.cumsum(event_counts_per_interval, dtype=np.int64)))

    input_dir = output_root / "v2e_input" / episode["id"]
    input_dir.mkdir(parents=True, exist_ok=False)
    frame_hashes = []
    for index, gray in enumerate(display_frames):
        frame_path = input_dir / f"{index:06d}.png"
        if not cv2.imwrite(str(frame_path), gray):
            raise IOError(f"failed to write V2E input frame {frame_path}")
        frame_hashes.append({"path": str(frame_path), "sha256": sha256(frame_path)})

    group = h5.create_group(f"episodes/{episode['id']}")
    group.attrs["split"] = episode["split"]
    group.attrs["event_window"] = "(t_i,t_i_plus_1]"
    group.attrs["pose_label_time"] = "right_endpoint"
    group.attrs["camera_frame_convention"] = "OpenCV: x right, y down, z forward"
    group.attrs["world_frame_convention"] = "Genesis world frame: z up; object_to_world translations are meters"
    group.attrs["length_unit"] = "meter"
    group.attrs["time_unit"] = "second"
    group.attrs["flow_label_contract"] = "forward source-pixel displacement to target endpoint, pixels per interval; dt1=one endpoint, dt4=four endpoints"
    group.attrs["flow_valid_contract"] = "shared source-depth and positive-target-depth geometry validity plus target-image bounds; generator independent; no target-occlusion z-buffer"
    group.attrs["event_timestamp_unit"] = "second"
    group.attrs["event_coordinate_order"] = "x,y"
    group.attrs["event_polarity"] = "signed -1/+1"
    group.create_dataset("camera_intrinsics_K_px", data=as_numpy(camera.intrinsics).astype(np.float64))
    group.attrs["target_segmentation_link_id"] = target_seg_id
    group.create_dataset("frame_timestamps_s", data=timestamps_a)
    group.create_dataset("radiance_linear_rgb", data=np.stack(radiance_frames).astype(np.float32), compression="gzip")
    group.create_dataset("depth_m", data=np.stack(depths).astype(np.float32), compression="gzip")
    group.create_dataset("segmentation_link_idx", data=np.stack(segmentations).astype(np.int32), compression="gzip")
    group.create_dataset("v2e_display_luma_u8", data=np.stack(display_frames), compression="gzip")
    group.create_dataset("camera_to_world_cv", data=np.stack(camera_poses).astype(np.float64))
    group.create_dataset("object_segmentation_link_ids", data=np.asarray(object_ids, dtype=np.int32))
    group.create_dataset("object_to_world", data=object_pose_array)
    group.create_dataset("target_pose", data=target_pose_array)
    group.create_dataset("flow_forward_px_per_interval", data=np.stack(flow_frames), compression="gzip")
    group.create_dataset("flow_valid", data=np.stack(flow_valid), compression="gzip")
    group.create_dataset("flow_forward_dt4_px_per_interval", data=np.stack(flow_dt4_frames), compression="gzip")
    group.create_dataset("flow_valid_dt4", data=np.stack(flow_dt4_valid), compression="gzip")
    group.create_dataset("eventgenesis_events_txyp", data=events)
    group.create_dataset("eventgenesis_interval_offsets", data=event_offsets)
    group.create_dataset("eventgenesis_interval_event_counts", data=np.asarray(event_counts_per_interval, dtype=np.int64))
    group.attrs["eventgenesis_polarities"] = json.dumps(np.unique(events[:, 3]).tolist() if len(events) else [])

    target_z = target_pose_array[:, 2, 3]
    return {
        "episode_id": episode["id"],
        "split": episode["split"],
        "seed_offset": int(episode.get("seed_offset", 0)),
        "frame_count": len(timestamps_a),
        "interval_count": interval_count,
        "camera_intrinsics_K_px": as_numpy(camera.intrinsics).astype(np.float64).tolist(),
        "camera_frame_convention": "OpenCV x-right/y-down/z-forward; camera_to_world_cv translation in meters",
        "world_frame_convention": "Genesis world z-up; object_to_world translation in meters",
        "timestamp_unit": "seconds relative to rollout start",
        "flow_label_contract": group.attrs["flow_label_contract"],
        "target_segmentation_link_id": target_seg_id,
        "target_visible_pixel_counts": [int((seg == target_seg_id).sum()) for seg in segmentations],
        "target_z_start_end_m": [float(target_z[0]), float(target_z[-1])],
        "eventgenesis_event_count": int(len(events)),
        "eventgenesis_events_per_interval": event_counts_per_interval,
        "eventgenesis_polarities": np.unique(events[:, 3]).tolist() if len(events) else [],
        "flow_valid_fraction_per_interval": [float(valid.mean()) for valid in flow_valid],
        "v2e_input_dir": str(input_dir),
        "v2e_frame_sha256": frame_hashes,
        "linear_radiance_min_max": [float(np.min(radiance_frames)), float(np.max(radiance_frames))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    import genesis as gs

    if getattr(gs, "__version__", None) != config["simulation"]["genesis_version"]:
        raise SystemExit(f"Genesis version mismatch: {getattr(gs, '__version__', None)}")
    gs.init(backend=gs.gpu, logging_level="warning")
    repo_root = Path(__file__).resolve().parents[3]
    core_sha = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "repo_commit": core_sha,
        "frozen_simulator_core_commit": "e79e656ac00f7847b98db9e12bf43b093f3afff1",
        "genesis_world_version": importlib.metadata.version("genesis-world"),
        "torch_version": importlib.metadata.version("torch"),
        "genesis_module_path": str(Path(gs.__file__).resolve()),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "endpoint_rate_hz": config["simulation"]["endpoint_rate_hz"],
        "flow_direction_fixture": verify_forward_flow_direction(),
        "episodes": [],
    }
    (output_root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    try:
        with h5py.File(output_root / "shared_trajectory.h5", "w") as h5:
            h5.attrs["schema_version"] = 1
            h5.attrs["trajectory_config_sha256"] = summary["config_sha256"]
            h5.attrs["frozen_simulator_core_commit"] = "e79e656ac00f7847b98db9e12bf43b093f3afff1"
            h5.attrs["split_unit"] = "whole physical rollout"
            h5.attrs["shared_time_unit"] = "seconds"
            h5.attrs["flow_direction"] = "forward source endpoint to target endpoint"
            for episode in config["simulation"]["episodes"]:
                summary["episodes"].append(run_episode(gs, config, episode, output_root, h5))
    finally:
        gs.destroy()
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
