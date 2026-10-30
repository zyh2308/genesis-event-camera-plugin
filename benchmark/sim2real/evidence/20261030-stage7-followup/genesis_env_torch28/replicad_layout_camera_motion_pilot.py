#!/usr/bin/env python3
"""Import a registered ReplicaCAD layout into Genesis and render a camera-motion pilot."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

import genesis as gs
import numpy as np
from PIL import Image

from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.genesis_adapter import camera_to_world_cv
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.physics_warp import compute_se3_flow


class PixelIndexCameraProxy:
    """Pass integer-index pixel intrinsics to the geometry interpolator."""

    def __init__(self, camera):
        self._camera = camera
        self.intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).copy()
        self.intrinsics[0, 2] -= 0.5
        self.intrinsics[1, 2] -= 0.5

    def __getattr__(self, name):
        return getattr(self._camera, name)


def capture_camera_motion_probe(
    scene, camera, output_dir: Path, translation_habitat_xyz: tuple[float, float, float]
) -> dict:
    """Render a fixed 30 ms camera translation, events, and independent flow check."""
    motion_dir = output_dir / "camera_motion_30ms"
    motion_dir.mkdir()
    width, height = 320, 240
    start_pos_h = np.array([1.5, 1.55, 7.1], dtype=np.float64)
    start_look_h = np.array([1.5, 1.1, 2.0], dtype=np.float64)
    translation_h = np.asarray(translation_habitat_xyz, dtype=np.float64)
    if translation_h.shape != (3,) or not np.isfinite(translation_h).all() or np.linalg.norm(translation_h) <= 0:
        raise ValueError(f"translation must be a finite nonzero Habitat xyz vector, got {translation_h}")
    duration_s = 0.03
    endpoints = 3
    steps_per_endpoint = 10  # scene dt is 1 ms, so each interval is 10 ms
    frame_cache: dict[str, np.ndarray] = {}
    state_rows: list[np.ndarray] = []

    def frame_provider(_scene, active_camera):
        rad, _, _, _ = active_camera.render(
            rgb=True, depth=False, segmentation=False, radiance=True, force_render=True
        )
        _, depth, seg, _ = active_camera.render(
            rgb=False, depth=True, segmentation=True, force_render=True
        )
        payload = {
            "radiance": np.asarray(rad, dtype=np.float32),
            "depth": np.asarray(depth, dtype=np.float32),
            "seg": np.asarray(seg, dtype=np.int32),
        }
        frame_cache.clear()
        frame_cache.update(payload)
        return payload

    def motion_state_provider(_scene, active_camera):
        T = camera_to_world_cv(active_camera)
        state_rows.append(T.copy())
        return {"camera_to_world": T, "object_to_world": {}}

    # Reuse the Stage-5 pilot sensor values. These remain explicitly
    # uncalibrated and are used only to exercise the event path.
    sensor = DvsNoiseConfig(
        pos_thres=.2, neg_thres=.2, sigma_thres=.03, cutoff_hz=300.0,
        leak_rate_hz=.01, leak_jitter_fraction=.1, noise_rate_cov_decades=.1,
        shot_noise_rate_hz=.001, refractory_period_s=.0005, seed=42,
        radiance_input=True, response_mode="physical_log",
        radiometric_calibration_status="uncalibrated",
    )
    plugin = GenesisPhysicsEventPlugin(
        input_space="linear", response_mode="physical_log",
        radiometric_calibration_status="uncalibrated", noise_config=sensor,
        sampler_config=AdaptiveSamplingConfig(), frame_provider=frame_provider,
        motion_state_provider=motion_state_provider, device="cpu", warp_backend="numpy",
    )
    K_renderer = np.asarray(camera.intrinsics, dtype=np.float64)
    K_pixel = K_renderer.copy()
    K_pixel[0, 2] -= 0.5
    K_pixel[1, 2] -= 0.5
    plugin.attach(scene, PixelIndexCameraProxy(camera)).start_episode()

    event_chunks: list[np.ndarray] = []
    radiance_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    segmentation_frames: list[np.ndarray] = []
    times_s = [float(scene.cur_t)]
    first = np.asarray(plugin.capture(), dtype=np.float64).reshape(-1, 4)
    event_chunks.append(first)
    radiance_frames.append(frame_cache["radiance"].copy())
    depth_frames.append(frame_cache["depth"].copy())
    segmentation_frames.append(frame_cache["seg"].copy())

    for endpoint in range(1, endpoints + 1):
        for substep in range(1, steps_per_endpoint + 1):
            elapsed_s = ((endpoint - 1) * steps_per_endpoint + substep) * 0.001
            alpha = elapsed_s / duration_s
            delta_h = translation_h * alpha
            pos_h = start_pos_h + delta_h
            look_h = start_look_h + delta_h
            camera.set_pose(
                pos=tuple(habitat_xyz_to_genesis_xyz(pos_h)),
                lookat=tuple(habitat_xyz_to_genesis_xyz(look_h)),
            )
            scene.step()
        chunk = np.asarray(plugin.capture(), dtype=np.float64).reshape(-1, 4)
        event_chunks.append(chunk)
        times_s.append(float(scene.cur_t))
        radiance_frames.append(frame_cache["radiance"].copy())
        depth_frames.append(frame_cache["depth"].copy())
        segmentation_frames.append(frame_cache["seg"].copy())

    if len(state_rows) != endpoints + 1:
        raise RuntimeError(f"expected {endpoints + 1} captured camera states, got {len(state_rows)}")
    times = np.asarray(times_s, dtype=np.float64)
    if not np.allclose(np.diff(times), 0.01, atol=1e-7, rtol=0):
        raise RuntimeError(f"expected three 10 ms camera intervals, got {np.diff(times).tolist()}")
    poses = np.stack(state_rows)
    R0, R1 = poses[0, :3, :3], poses[-1, :3, :3]
    translation = poses[-1, :3, 3] - poses[0, :3, 3]
    observed_delta_camera = R0.T @ translation
    dx_camera, dy_camera, dz_camera = map(float, observed_delta_camera)
    expected_delta_genesis = COORD_H_TO_G @ translation_h
    expected_delta_camera = R0.T @ expected_delta_genesis
    rotation_delta_max = float(np.max(np.abs(R1 - R0)))
    if not np.allclose(observed_delta_camera, expected_delta_camera, atol=2e-4, rtol=0):
        raise RuntimeError(
            "camera translation contract failed: "
            f"observed={observed_delta_camera.tolist()} expected={expected_delta_camera.tolist()}"
        )
    if rotation_delta_max > 2e-4:
        raise RuntimeError(f"camera orientation changed during translation: max_delta={rotation_delta_max}")

    flow, valid = compute_se3_flow(
        depth_frames[0], segmentation_frames[0], K_pixel, poses[0], poses[-1]
    )
    z = depth_frames[0]
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    source_points = np.stack(
        ((xx - K_pixel[0, 2]) * z / K_pixel[0, 0],
         (yy - K_pixel[1, 2]) * z / K_pixel[1, 1], z), axis=-1
    )
    relative_cam = np.linalg.inv(poses[-1]) @ poses[0]
    target_points = source_points @ relative_cam[:3, :3].T + relative_cam[:3, 3]
    target_z = target_points[..., 2]
    projected_x = K_pixel[0, 0] * target_points[..., 0] / np.maximum(target_z, 1e-12) + K_pixel[0, 2]
    projected_y = K_pixel[1, 1] * target_points[..., 1] / np.maximum(target_z, 1e-12) + K_pixel[1, 2]
    expected_x = projected_x - xx
    expected_y = projected_y - yy
    end_x = xx + expected_x
    end_y = yy + expected_y
    comparison = (
        valid & np.isfinite(z) & (z > 1e-4)
        & np.isfinite(target_points).all(axis=-1) & (target_z > 1e-4)
        & (end_x >= 1) & (end_x < width - 1)
        & (end_y >= 1) & (end_y < height - 1)
    )
    if comparison.sum() < 100:
        raise RuntimeError(f"too few in-frame valid flow pixels: {int(comparison.sum())}")
    error = np.sqrt((flow[0] - expected_x) ** 2 + (flow[1] - expected_y) ** 2)[comparison]
    if float(np.percentile(error, 99)) > 0.08:
        raise RuntimeError(f"analytic camera-translation flow mismatch p99={np.percentile(error,99)} px")

    interval_flows, interval_valids = [], []
    for i in range(endpoints):
        f_i, v_i = compute_se3_flow(
            depth_frames[i], segmentation_frames[i], K_pixel, poses[i], poses[i + 1]
        )
        interval_flows.append(f_i)
        interval_valids.append(v_i)
    interval_flows = np.stack(interval_flows).astype(np.float32)
    interval_valids = np.stack(interval_valids).astype(np.uint8)
    events = np.concatenate([e for e in event_chunks if len(e)], axis=0) if any(map(len, event_chunks)) else np.empty((0, 4), dtype=np.float64)
    if events.size and (not np.isfinite(events).all() or np.any(np.diff(events[:, 0]) < 0)):
        raise RuntimeError("EventGenesis pilot returned non-finite or unsorted events")
    if events.size and (events[:, 1].min() < 0 or events[:, 1].max() >= width or events[:, 2].min() < 0 or events[:, 2].max() >= height):
        raise RuntimeError("EventGenesis pilot coordinates exceed registered frame")

    np.save(motion_dir / "camera_poses_cv.npy", poses.astype(np.float64))
    np.save(motion_dir / "radiance_frames.npy", np.stack(radiance_frames).astype(np.float32))
    np.save(motion_dir / "depth_frames.npy", np.stack(depth_frames).astype(np.float32))
    np.save(motion_dir / "renderer_segmentation_raw.npy", np.stack(segmentation_frames).astype(np.int32))
    np.save(motion_dir / "gt_flow_px_t0_to_t30ms.npy", flow.astype(np.float32))
    np.save(motion_dir / "gt_flow_valid.npy", valid.astype(np.uint8))
    np.save(motion_dir / "gt_flow_px_each_10ms_interval.npy", interval_flows)
    np.save(motion_dir / "gt_flow_valid_each_10ms_interval.npy", interval_valids)
    np.save(motion_dir / "eventgenesis_events_seconds_xy_p.npy", events.astype(np.float64))
    rgb_last = np.asarray(camera.render(rgb=True, depth=False, segmentation=False, force_render=True)[0])
    rgb_last_u8 = np.rint(np.clip(rgb_last, 0, 255)).astype(np.uint8) if rgb_last.dtype == np.uint8 or float(rgb_last.max()) > 1.5 else np.rint(np.clip(rgb_last, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb_last_u8, mode="RGB").save(motion_dir / "camera_motion_endpoint_rgb.png")
    event_frame = np.zeros((height, width, 3), dtype=np.uint8)
    for ev in events:
        x, y, p = int(round(ev[1])), int(round(ev[2])), int(ev[3])
        if 0 <= x < width and 0 <= y < height:
            event_frame[y, x] = (255, 64, 64) if p > 0 else (64, 128, 255)
    Image.fromarray(event_frame, mode="RGB").save(motion_dir / "eventgenesis_support.png")
    result = {
        "schema": "replicacad_layout_camera_translation_flow_event_probe_v1",
        "status": "passed_camera_translation_flow_analytic_check",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trajectory": {
            "camera_only": True, "start_habitat_position_m": start_pos_h.tolist(),
            "start_habitat_lookat_m": start_look_h.tolist(),
            "registered_translation_habitat_xyz_m": translation_h.tolist(),
            "registered_translation_habitat_x_m": float(translation_h[0]),
            "speed_m_s": float(np.linalg.norm(translation_h) / duration_s),
            "duration_s": duration_s, "endpoint_count": endpoints,
            "interval_s": np.diff(times).tolist(), "scene_timestep_s": 0.001,
        },
        "camera_pose_check": {
            "observed_delta_camera_xyz_m": [dx_camera, dy_camera, dz_camera],
            "expected_delta_camera_xyz_m": expected_delta_camera.tolist(),
            "max_rotation_matrix_delta": rotation_delta_max,
            "K_renderer": K_renderer.tolist(), "K_pixel_index": K_pixel.tolist(),
        },
        "flow": {
            "units": "pixel displacement over t0 to t30ms",
            "valid_pixels": int(valid.sum()), "analytic_comparison_pixels": int(comparison.sum()),
            "mean_vector_px": np.mean(flow[:, valid], axis=1).tolist(),
            "p99_analytic_endpoint_error_px": float(np.percentile(error, 99)),
            "median_analytic_endpoint_error_px": float(np.median(error)),
            "independent_check": "project source 3-D pixels through inv(Tcw1)@Tcw0 and compare with compute_se3_flow",
            "expected_relation": "camera-only pinhole reprojection from relative camera pose; orientation held fixed",
            "depth_contract": "Genesis renderer depth supplied as z-depth; checked by independent pinhole reprojection",
        },
        "per_10ms_ground_truth": {
            "flow_path": "gt_flow_px_each_10ms_interval.npy",
            "valid_path": "gt_flow_valid_each_10ms_interval.npy",
            "shape": list(interval_flows.shape),
            "units": "pixel displacement for the corresponding adjacent 10 ms endpoints",
            "valid_pixels_per_interval": [int(x.sum()) for x in interval_valids],
        },
        "events": {
            "generator": "repository GenesisPhysicsEventPlugin; CPU, NumPy warp backend",
            "counts_by_10ms_interval": [int(len(x)) for x in event_chunks[1:]],
            "total_count": int(len(events)), "timestamp_unit": "seconds",
            "input": "same scene-linear float32 radiance frames used by the physical interpolation plugin",
            "sensor_calibration": "uncalibrated; reused Stage-5 pilot values only",
        },
        "signal": {
            "radiance_shape": list(np.stack(radiance_frames).shape),
            "radiance_range": [float(np.min(radiance_frames)), float(np.max(radiance_frames))],
            "interframe_absdiff_mean": [float(np.mean(np.abs(radiance_frames[i + 1] - radiance_frames[i]))) for i in range(endpoints)],
            "ordinary_rgb_endpoint_matches_static_path": "not checked in this run",
        },
        "limitations": [
            "camera translation only; no independently moving objects or visibility change trajectory",
            "EventGenesis pilot only; V2E has not yet consumed these exact frames",
            "physics flow is checked against an analytic camera-only pinhole relation, but scene conversion validity is not comprehensively certified",
            "raw renderer segmentation IDs retained and not mapped to physical link IDs; no moving-object labels used",
            "sensor radiometric calibration and actual camera characterization remain unverified",
        ],
    }
    (motion_dir / "motion_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


COORD_H_TO_G = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
Q_BASIS_WXYZ = np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def habitat_xyzw_to_genesis_wxyz(value: list[float]) -> list[float]:
    if len(value) != 4 or not np.isfinite(value).all():
        raise ValueError(f"invalid Habitat quaternion: {value}")
    x, y, z, w = map(float, value)
    q_h = np.array([w, x, y, z], dtype=np.float64)
    q_h /= np.linalg.norm(q_h)
    q_basis_inv = Q_BASIS_WXYZ * np.array([1.0, -1.0, -1.0, -1.0])
    q_g = qmul(qmul(Q_BASIS_WXYZ, q_h), q_basis_inv)
    q_g /= np.linalg.norm(q_g)
    return q_g.tolist()


def habitat_xyz_to_genesis_xyz(value: list[float] | tuple[float, ...]) -> list[float]:
    p = np.asarray(value, dtype=np.float64)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError(f"invalid Habitat vector: {value}")
    return (COORD_H_TO_G @ p).tolist()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_coordinate_contract() -> None:
    if not np.allclose(COORD_H_TO_G @ COORD_H_TO_G.T, np.eye(3), atol=1e-12):
        raise RuntimeError("coordinate basis is not orthonormal")
    if not np.isclose(np.linalg.det(COORD_H_TO_G), 1.0, atol=1e-12):
        raise RuntimeError("coordinate basis changes handedness")
    if not np.allclose(habitat_xyz_to_genesis_xyz([0, 1, 0]), [0, 0, 1]):
        raise RuntimeError("Habitat +Y up does not map to Genesis +Z up")
    if not np.allclose(habitat_xyz_to_genesis_xyz([0, 0, -1]), [0, 1, 0]):
        raise RuntimeError("Habitat -Z forward basis does not map as declared")
    half = math.sqrt(0.5)
    # Habitat +Y 90 degree yaw, stored [x,y,z,w], becomes Genesis +Z yaw.
    q_h_yaw = [0.0, half, 0.0, half]
    expected = [half, 0.0, 0.0, half]
    if not np.allclose(habitat_xyzw_to_genesis_wxyz(q_h_yaw), expected, atol=1e-12):
        raise RuntimeError("quaternion basis conversion failed the independent yaw case")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--scene-config", choices=["apt_1.scene_instance.json", "v3_sc0_staging_00.scene_instance.json", "v3_sc1_staging_00.scene_instance.json", "v3_sc2_staging_00.scene_instance.json"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--translation-habitat-xyz", type=float, nargs=3, default=[0.06, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="registered 30 ms camera translation in Habitat xyz meters; camera and lookat move together",
    )
    args = parser.parse_args()
    scene_id = args.scene_config.removesuffix(".scene_instance.json")
    started = time.monotonic()
    asset_root = args.asset_root.resolve()
    scene_path = asset_root / "configs/scenes" / args.scene_config
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    verify_coordinate_contract()
    scene_data = json.loads(scene_path.read_text())
    stage_name = scene_data["stage_instance"]["template_name"]
    stage_config_path = asset_root / "configs/stages" / f"{stage_name}.stage_config.json"
    stage_config = json.loads(stage_config_path.read_text())
    if stage_config.get("up") != [0, 1, 0]:
        raise RuntimeError(f"unexpected stage up vector: {stage_config.get('up')}")
    stage_asset = (stage_config_path.parent / stage_config["render_asset"]).resolve()
    if not stage_asset.is_file():
        raise FileNotFoundError(stage_asset)

    rigid_specs: list[dict] = []
    for row in scene_data["object_instances"]:
        template = row["template_name"].rsplit("/", 1)[-1]
        config_path = asset_root / "configs/objects" / f"{template}.object_config.json"
        config = json.loads(config_path.read_text())
        asset_path = (config_path.parent / config["render_asset"]).resolve()
        if not asset_path.is_file():
            raise FileNotFoundError(asset_path)
        scale = row.get("uniform_scale", 1.0)
        rigid_specs.append(
            {
                "template_name": row["template_name"],
                "asset_path": asset_path,
                "position_g": habitat_xyz_to_genesis_xyz(row["translation"]),
                "quaternion_g_wxyz": habitat_xyzw_to_genesis_wxyz(row["rotation"]),
                "scale": scale,
            }
        )

    art_urdf_rel = {
        "fridge": "urdf/fridge/fridge.urdf",
        "kitchen_counter": "urdf/kitchen_counter/kitchen_counter.urdf",
        "kitchenCupboard_01": "urdf/kitchen_cupboards/kitchenCupboard_01.urdf",
        "chestOfDrawers_01": "urdf/chest_of_drawers/chestOfDrawers_01.urdf",
        "cabinet": "urdf/cabinet/cabinet.urdf",
        "door2": "urdf/doors/door2.urdf",
    }
    art_specs = []
    for row in scene_data.get("articulated_object_instances", []):
        template = row["template_name"]
        rel = art_urdf_rel.get(template)
        if rel is None:
            raise RuntimeError(f"no checked URDF mapping for articulated asset {template}")
        path = asset_root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        art_specs.append(
            {
                "template_name": template,
                "asset_path": path,
                "position_g": habitat_xyz_to_genesis_xyz(row["translation"]),
                "quaternion_g_wxyz": habitat_xyzw_to_genesis_wxyz(row["rotation"]),
                "scale": row.get("uniform_scale", 1.0),
                "fixed_base": row.get("fixed_base", False),
                "translation_origin": row.get("translation_origin"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    # ReplicaCAD v1.5 uses comma-separated URDF mesh scale vectors, which the
    # bundled Genesis urdfpy parser rejects, and the kitchen-counter URDF has
    # one additional comma in a box size vector. Normalize only these delimiters
    # in run-local copies; preserve every numeric value and the shared source.
    compat_root = args.output_dir / "compat_assets"
    compat_patches: list[dict] = []
    copied_dirs: dict[Path, Path] = {}
    mesh_scale_pattern = re.compile(r'(<mesh\b[^>]*\bscale=")([^"]+)(")')
    bad_counter_size = '<box size="0.617 0.145, 0.894"/>'
    fixed_counter_size = '<box size="0.617 0.145 0.894"/>'
    for spec in art_specs:
        source_urdf = spec["asset_path"]
        source_dir = source_urdf.parent
        overlay_dir = copied_dirs.get(source_dir)
        if overlay_dir is None:
            overlay_dir = compat_root / source_dir.relative_to(asset_root)
            overlay_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, overlay_dir)
            copied_dirs[source_dir] = overlay_dir
        overlay_urdf = overlay_dir / source_urdf.name
        source_text = source_urdf.read_text()
        scale_replacements = 0

        def normalize_scale(match: re.Match[str]) -> str:
            nonlocal scale_replacements
            raw_value = match.group(2)
            parts = [part for part in re.split(r"[\s,]+", raw_value.strip()) if part]
            if len(parts) != 3:
                raise RuntimeError(f"unexpected mesh scale in {source_urdf}: {raw_value!r}")
            values = [float(part) for part in parts]
            if not np.isfinite(values).all():
                raise RuntimeError(f"nonfinite mesh scale in {source_urdf}: {raw_value!r}")
            if "," in raw_value:
                scale_replacements += 1
            return match.group(1) + " ".join(parts) + match.group(3)

        patched_text = mesh_scale_pattern.sub(normalize_scale, source_text)
        size_replacements = 0
        if spec["template_name"] == "kitchen_counter":
            size_replacements = patched_text.count(bad_counter_size)
            if size_replacements != 1:
                raise RuntimeError(
                    "expected exactly one known malformed kitchen-counter box vector; "
                    f"found {size_replacements}"
                )
            patched_text = patched_text.replace(bad_counter_size, fixed_counter_size)
        overlay_urdf.write_text(patched_text)
        spec["asset_path"] = overlay_urdf
        compat_patches.append(
            {
                "template_name": spec["template_name"],
                "source_urdf": str(source_urdf),
                "source_urdf_sha256": sha256_file(source_urdf),
                "patched_urdf": str(overlay_urdf),
                "patched_urdf_sha256": sha256_file(overlay_urdf),
                "mesh_scale_delimiter_replacements": scale_replacements,
                "malformed_counter_size_replacements": size_replacements,
                "numeric_values_changed": False,
            }
        )

    # Validate all patched files with the exact bundled parser before asking
    # Genesis to build the complete layout, so any remaining contract issue is
    # attributed to a specific asset rather than a late scene failure.
    from genesis.ext.urdfpy import URDF

    for spec in art_specs:
        parsed = URDF.load(str(spec["asset_path"]))
        if not parsed.links:
            raise RuntimeError(f"patched URDF contains no links: {spec['asset_path']}")

    gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, -9.81)),
            vis_options=gs.options.VisOptions(ambient_light=(0.55, 0.55, 0.55)),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=False,
        )
        scene.add_entity(
            gs.morphs.Mesh(
                file=str(stage_asset),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
                scale=1.0,
                visualization=True,
                collision=False,
                fixed=True,
                file_meshes_are_zup=False,
            )
        )
        for spec in rigid_specs:
            scene.add_entity(
                gs.morphs.Mesh(
                    file=str(spec["asset_path"]),
                    pos=tuple(spec["position_g"]),
                    quat=tuple(spec["quaternion_g_wxyz"]),
                    scale=spec["scale"],
                    visualization=True,
                    collision=False,
                    fixed=True,
                    file_meshes_are_zup=False,
                )
            )
        for spec in art_specs:
            scene.add_entity(
                gs.morphs.URDF(
                    file=str(spec["asset_path"]),
                    pos=tuple(spec["position_g"]),
                    quat=tuple(spec["quaternion_g_wxyz"]),
                    scale=spec["scale"],
                    fixed=True,
                    visualization=True,
                    collision=False,
                    file_meshes_are_zup=False,
                )
            )

        # Registered Habitat camera: (1.5, 1.55, 7.1), looking toward (1.5, 1.1, 2.0).
        camera = scene.add_camera(
            res=(320, 240),
            pos=tuple(habitat_xyz_to_genesis_xyz([1.5, 1.55, 7.1])),
            lookat=tuple(habitat_xyz_to_genesis_xyz([1.5, 1.1, 2.0])),
            fov=72,
            GUI=False,
            near=0.03,
            far=30.0,
        )
        scene.build()
        # Match the Stage-4 HDR overlay's separate render passes: the scene-linear
        # radiance request and the geometry depth/segmentation pass must not be
        # conflated with the ordinary 8-bit display RGB output.
        radiance, _, _, _ = camera.render(
            rgb=True, depth=False, segmentation=False, radiance=True, force_render=True
        )
        rgb, _, _, _ = camera.render(
            rgb=True, depth=False, segmentation=False, force_render=True
        )
        _, depth, segmentation, _ = camera.render(
            rgb=False, depth=True, segmentation=True, force_render=True
        )
        radiance = np.asarray(radiance)
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
        segmentation = np.asarray(segmentation)
        if radiance.shape != (240, 320, 3) or not np.isfinite(radiance).all():
            raise RuntimeError(f"invalid scene-linear radiance output shape/values: {radiance.shape}")
        if rgb.shape != (240, 320, 3) or not np.isfinite(rgb).all():
            raise RuntimeError(f"invalid RGB output shape/values: {rgb.shape}")
        if depth.shape != (240, 320) or not np.isfinite(depth).all():
            raise RuntimeError(f"invalid depth output shape/values: {depth.shape}")
        if segmentation.shape != (240, 320):
            raise RuntimeError(f"invalid segmentation shape: {segmentation.shape}")
        if rgb.dtype == np.uint8 or float(rgb.max()) > 1.5:
            rgb_u8 = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
        else:
            rgb_u8 = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb_u8, mode="RGB").save(args.output_dir / "layout_rgb.png")
        np.save(args.output_dir / "layout_radiance.npy", radiance.astype(np.float32, copy=False))
        np.save(args.output_dir / "layout_depth.npy", depth)
        np.save(args.output_dir / "layout_segmentation.npy", segmentation)
        motion_result = capture_camera_motion_probe(
            scene, camera, args.output_dir, tuple(args.translation_habitat_xyz)
        )
        result = {
            "schema": "genesis_replicad_layout_camera_motion_pilot_v1",
            "status": "passed_static_scene_and_camera_motion_flow_event_probe",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "genesis_version": gs.__version__,
            "backend": str(gs.backend),
            "device": str(gs.device),
            "cuda_visible_devices": "empty",
            "offline_cache_redirected_to_run_local_tmp": True,
            "renderer_overlay": "/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2",
            "radiance_signal": {
                "request": "camera.render(rgb=True, radiance=True, force_render=True)",
                "payload": "scene-linear float32 from the isolated Genesis HDR rasterizer overlay",
                "physical_radiometric_calibration": "unverified",
                "shape": list(radiance.shape),
                "finite_fraction": float(np.isfinite(radiance).mean()),
                "range": [float(radiance.min()), float(radiance.max())],
            },
            "scene_id": scene_id,
            "scene_config": str(scene_path),
            "scene_config_sha256": sha256_file(scene_path),
            "stage_asset": str(stage_asset),
            "stage_asset_sha256": sha256_file(stage_asset),
            "source_archive_sha256": "d571c2c1315601c7faed35452e54a15c56646393a4c330235f20fb6fc8d8edfd",
            "coordinate_contract": {
                "habitat_world": "right-handed Y-up; rotation coefficients XYZW",
                "genesis_world": "right-handed Z-up; quaternion WXYZ",
                "basis_matrix": COORD_H_TO_G.tolist(),
                "position_mapping": "(x, y, z)_habitat -> (x, -z, y)_genesis",
                "rotation_mapping": "q_g = q_basis * q_h * inverse(q_basis); q_basis is +90deg about X",
                "glb_meshes": "file_meshes_are_zup=False; Genesis applies the documented Y-up to Z-up mesh conversion",
            },
            "counts": {
                "stage_meshes": 1,
                "rigid_scene_instances": len(rigid_specs),
                "unique_rigid_templates": len({s["template_name"] for s in rigid_specs}),
                "articulated_urdf_instances": len(art_specs),
                "entities_added": 1 + len(rigid_specs) + len(art_specs),
            },
            "camera": {
                "resolution_hw": [240, 320],
                "position_habitat": [1.5, 1.55, 7.1],
                "lookat_habitat": [1.5, 1.1, 2.0],
                "fov_degrees": 72,
                "intrinsics": np.asarray(camera.intrinsics).tolist(),
            },
            "outputs": {
                "rgb_shape": list(rgb.shape),
                "rgb_range": [float(rgb.min()), float(rgb.max())],
                "radiance_shape": list(radiance.shape),
                "radiance_dtype": str(radiance.dtype),
                "radiance_range": [float(radiance.min()), float(radiance.max())],
                "depth_shape": list(depth.shape),
                "depth_finite_fraction": float(np.isfinite(depth).mean()),
                "depth_positive_fraction": float((depth > 0).mean()),
                "segmentation_shape": list(segmentation.shape),
                "segmentation_ids": np.unique(segmentation).tolist(),
            },
            "camera_motion_probe": motion_result,
            "articulated_instances": [
                {**spec, "asset_path": str(spec["asset_path"])} for spec in art_specs
            ],
            "run_local_compatibility_patches": compat_patches,
            "source_assets_modified": False,
            "rigid_instance_assets": [
                {k: v for k, v in s.items() if k != "asset_path"} | {"asset_path": str(s["asset_path"])}
                for s in rigid_specs
            ],
            "elapsed_seconds": time.monotonic() - started,
            "limitations": [
                "only a registered 30 ms camera-only trajectory was exercised; no independently moving objects",
                "EventGenesis event path was exercised, but the exact same frames have not yet been processed by V2E",
                "no training or real data",
                "articulated URDF joint state is its importer default and was not animated",
                "this is an import/appearance pilot, not a validated scene conversion or coverage pass",
            ],
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(json.dumps({"status": result["status"], "counts": result["counts"],
                          "rgb_range": result["outputs"]["rgb_range"],
                          "depth_positive_fraction": result["outputs"]["depth_positive_fraction"],
                          "segmentation_ids": result["outputs"]["segmentation_ids"],
                          "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
