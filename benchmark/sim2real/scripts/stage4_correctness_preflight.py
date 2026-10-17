#!/usr/bin/env python3
"""Execute renderer, CUDA-operator, and corrected-bank preflight controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO / "benchmark/sim2real/compat"))

from generate_stage4_flow_bank import (
    GeometryCameraProxy, as_numpy, geometry_intrinsics, renderer_segmentation_index_map,
    segmentation_render_to_link,
    registered_episodes, trajectory_pose,
)
from run_steflow_stage2_pilot import compute_epe, pure_torch_spatial_correlation, state_sha256, verify_correlation
from steflow_align_corners import patch_ste
from stage4_launch_gate import CORE_FILES, build_fingerprint
from prepare_steflow_stage0 import gaussian_count_representation


FOUNDATION = [
    "renderer_segmentation_mapping", "depth_K_pixel_center", "trajectory_initialization",
    "steflow_model_and_loss_warp_cuda", "epe_controls", "correlation_operator_cuda",
    "bank_manifest_and_geometry", "paired_adapters_and_shared_labels", "event_window_recount",
    "event_representation_and_edge_cases", "fingerprint_completeness",
    "frozen_core_unchanged", "synthetic_only_guard",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_renderer(device_name: str, event_config: dict) -> tuple[dict, dict]:
    import genesis as gs
    from genesis_event_plugin.genesis_adapter import camera_to_world_cv, link_to_world_cv
    from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
    from genesis_event_plugin.noise_model import DvsNoiseConfig
    from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
    from genesis_event_plugin.physics_warp import project_endpoint_to_alpha

    gs.init(backend=gs.gpu, logging_level="error")
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=.001, gravity=(0., 0., -9.81)),
            vis_options=gs.options.VisOptions(ambient_light=(.38, .40, .43), segmentation_level="link"),
            show_viewer=False,
        )
        plane = scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(.38, .40, .43, 1.)))
        fixture_episode = registered_episodes()[8]
        base0 = np.asarray([0.02, 0.0, .11], dtype=np.float64)
        registered_initial = trajectory_pose(0., fixture_episode, 0, base0)
        boxes = []
        for index, (xyz, color) in enumerate(((tuple(registered_initial), (.9, .1, .1, 1.)),
                                                ((.22, .05, .12), (.1, .8, .2, 1.)),
                                                ((-.20, -.10, .10), (.1, .2, .9, 1.)))):
            boxes.append(scene.add_entity(gs.morphs.Box(pos=xyz, size=(.12, .10, .13)),
                                          surface=gs.surfaces.Default(color=color)))
        camera = scene.add_camera(res=(346, 260), pos=(1.05, 1.12, .78), lookat=(0., 0., .24),
                                  fov=58., GUI=False, near=.03, far=8.)
        scene.build()
        actual_initial = link_to_world_cv(boxes[0].links[0])[:3, 3]
        initial_pose_error = float(np.max(np.abs(actual_initial - registered_initial)))
        if initial_pose_error > 1e-4:
            raise RuntimeError(f"registered trajectory_pose(0) was not used as initial body pose: {initial_pose_error}")
        K_renderer, K_geometry = geometry_intrinsics(camera)
        proxy = GeometryCameraProxy(camera, K_geometry)
        _, _, _, _ = camera.render(rgb=True, depth=False, segmentation=False, radiance=True, force_render=True)
        _, depth_raw, seg_raw, _ = camera.render(rgb=False, depth=True, segmentation=True, force_render=True)
        raw_seg = as_numpy(seg_raw).astype(np.int32)
        render_to_link = {int(k): int(v) for k, v in renderer_segmentation_index_map(camera).items()}
        seg = segmentation_render_to_link(raw_seg, render_to_link)
        if render_to_link.get(0) != -1:
            raise RuntimeError("actual renderer background did not map to physical background -1")
        plane_id = int(plane.links[0].idx)
        object_ids = [int(body.links[0].idx) for body in boxes]
        if plane_id not in set(map(int, np.unique(seg))) or not any(oid in set(map(int, np.unique(seg))) for oid in object_ids):
            raise RuntimeError("actual renderer did not expose the expected plane/object physical link IDs")

        # Independent synthetic mapping fixture proves mapping is dictionary
        # based and supports non-contiguous physical IDs.
        fixture = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
        mapped_fixture = segmentation_render_to_link(fixture, {0: -1, 1: 17, 2: 5, 3: 42})
        if mapped_fixture.tolist() != [[-1, 17, 5, 42]]:
            raise RuntimeError(f"non-contiguous renderer mapping fixture failed: {mapped_fixture}")

        depth = as_numpy(depth_raw).astype(np.float32)
        Tcw = camera_to_world_cv(camera)
        pose = {int(link.idx): link_to_world_cv(link) for link in scene.rigid_solver.links}
        uv_static, _, valid_static = project_endpoint_to_alpha(
            depth, seg, K_geometry, Tcw, Tcw, 1., object_world0=pose, object_world1=pose,
        )
        xx, yy = np.meshgrid(np.arange(346, dtype=np.float64), np.arange(260, dtype=np.float64), indexing="xy")
        plane_mask = (seg == plane_id)
        eroded = cv2.erode(plane_mask.astype(np.uint8), np.ones((9, 9), np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        camera_points = np.stack(((xx - K_geometry[0, 2]) * depth / K_geometry[0, 0],
                                  (yy - K_geometry[1, 2]) * depth / K_geometry[1, 1], depth), axis=0).reshape(3, -1)
        world = (Tcw @ np.vstack((camera_points, np.ones((1, camera_points.shape[1])))))[:3]
        plane_error = np.abs(world[2].reshape(260, 346)[eroded])
        plane_p99 = float(np.quantile(plane_error, .99))
        static_max = float(np.max(np.abs(uv_static - np.stack((xx, yy), axis=0))[:, valid_static]))
        if plane_p99 > .001:
            raise RuntimeError(f"depth/K plane p99 {plane_p99} m exceeds 0.001 m")
        if static_max > 1e-5:
            raise RuntimeError(f"static renderer scene produced {static_max}px flow")

        target_id = object_ids[0]
        moved = {key: value.copy() for key, value in pose.items()}
        moved[target_id][0, 3] += .01
        uv_moved, _, valid_moved = project_endpoint_to_alpha(
            depth, seg, K_geometry, Tcw, Tcw, 1., object_world0=pose, object_world1=moved,
        )
        object_mask = (seg == target_id) & valid_moved
        other_mask = (seg != target_id) & valid_moved
        object_mean = float(np.mean(np.abs(uv_moved[:, object_mask] - np.stack((xx, yy), axis=0)[:, object_mask]))) if np.any(object_mask) else 0.
        other_max = float(np.max(np.abs(uv_moved[:, other_mask] - np.stack((xx, yy), axis=0)[:, other_mask]))) if np.any(other_mask) else 0.
        if not np.any(object_mask):
            raise RuntimeError("target object has no valid rendered source pixels")
        sample_y, sample_x = np.argwhere(object_mask)[len(np.argwhere(object_mask)) // 2]
        p_cam = np.asarray([(sample_x - K_geometry[0, 2]) * depth[sample_y, sample_x] / K_geometry[0, 0],
                            (sample_y - K_geometry[1, 2]) * depth[sample_y, sample_x] / K_geometry[1, 1],
                            depth[sample_y, sample_x], 1.], dtype=np.float64)
        p_world = Tcw @ p_cam
        delta = moved[target_id] @ np.linalg.inv(pose[target_id])
        moved_world = delta @ p_world
        moved_cam = np.linalg.inv(Tcw) @ moved_world
        expected_u = K_geometry[0, 0] * moved_cam[0] / moved_cam[2] + K_geometry[0, 2]
        expected_v = K_geometry[1, 1] * moved_cam[1] / moved_cam[2] + K_geometry[1, 2]
        independent_flow = [float(expected_u - sample_x), float(expected_v - sample_y)]
        observed_flow = [float(uv_moved[0, sample_y, sample_x] - sample_x),
                         float(uv_moved[1, sample_y, sample_x] - sample_y)]
        independent_error = float(np.max(np.abs(np.asarray(independent_flow) - observed_flow)))
        if object_mean <= 1e-5 or other_max > 1e-5:
            raise RuntimeError(f"object-only physical displacement failed: target mean={object_mean}, other max={other_max}")
        if independent_error > 1e-5 or independent_flow[0] <= 0:
            raise RuntimeError(f"known +world-X object shift has wrong sign/scale: expected={independent_flow}, observed={observed_flow}")

        # Exercise the real frozen plugin with the same renderer-map/depth/K
        # adapter used by the exporter before generating a full rollout bank.
        plugin_frame_cache = {}
        def plugin_frame_provider(active_scene, active_camera):
            radiance, _, _, _ = active_camera.render(rgb=True, depth=False, segmentation=False,
                                                       radiance=True, force_render=True)
            _, depth_value, raw_labels, _ = active_camera.render(rgb=False, depth=True,
                                                                   segmentation=True, force_render=True)
            label_mapping = renderer_segmentation_index_map(active_camera)
            physical_labels = segmentation_render_to_link(as_numpy(raw_labels).astype(np.int32), label_mapping)
            payload = {"rgb": as_numpy(radiance), "radiance": as_numpy(radiance),
                       "depth": as_numpy(depth_value), "seg": physical_labels}
            plugin_frame_cache["last"] = payload
            return payload
        def plugin_state_provider(active_scene, active_camera):
            return {"camera_to_world": camera_to_world_cv(active_camera),
                    "object_to_world": {int(link.idx): link_to_world_cv(link)
                                        for link in active_scene.rigid_solver.links}}
        sensor = DvsNoiseConfig(
            pos_thres=float(event_config["pos_thres"]), neg_thres=float(event_config["neg_thres"]),
            sigma_thres=float(event_config["sigma_thres"]), cutoff_hz=float(event_config["cutoff_hz"]),
            leak_rate_hz=float(event_config["leak_rate_hz"]),
            leak_jitter_fraction=float(event_config["leak_jitter_fraction"]),
            noise_rate_cov_decades=float(event_config["noise_rate_cov_decades"]),
            shot_noise_rate_hz=float(event_config["shot_noise_rate_hz"]),
            refractory_period_s=float(event_config["refractory_period_s"]),
            seed=int(event_config["seed"]), radiance_input=True, response_mode="physical_log",
            radiometric_calibration_status="uncalibrated")
        plugin = GenesisPhysicsEventPlugin(
            input_space="linear", response_mode="physical_log", radiometric_calibration_status="uncalibrated",
            noise_config=sensor, sampler_config=AdaptiveSamplingConfig(),
            frame_provider=plugin_frame_provider, motion_state_provider=plugin_state_provider,
            device=event_config["device"], warp_backend=event_config["warp_backend"])
        plugin.attach(scene, GeometryCameraProxy(camera, K_geometry)).start_episode()
        initial_events = np.asarray(plugin.capture(), dtype=np.float64).reshape(-1, 4)
        scene.step()
        boxes[0].set_pos(registered_initial + np.asarray([.001, 0., 0.]))
        scene.step()
        interval_events = np.asarray(plugin.capture(), dtype=np.float64).reshape(-1, 4)
        if len(initial_events) or (len(interval_events) and (not np.isfinite(interval_events).all()
                                                               or not np.all(np.diff(interval_events[:, 0]) >= 0))):
            raise RuntimeError("EventGenesis small-scene frame-provider event contract failed")
        if plugin.total_steps != 2 or plugin.interpolator.K.shape != K_geometry.shape or not np.allclose(plugin.interpolator.K, K_geometry):
            raise RuntimeError("EventGenesis did not receive the registered pixel-center geometry K")

        # The actual first-frame placement condition is also asserted on the
        # deterministic path by the exporter before each rollout is captured.
        report = {
            "status": "PASS", "genesis_version": getattr(gs, "__version__", "unknown"),
            "renderer_segmentation_idxc_to_link": render_to_link,
            "renderer_mapping_api": "Camera._visualizer.segmentation_idx_dict (pinned Genesis 1.2.2)",
            "physical_link_ids": {"plane": plane_id, "objects": object_ids},
            "noncontiguous_mapping_fixture": mapped_fixture.tolist(),
            "depth_only_pass": True, "renderer_K": K_renderer.tolist(), "geometry_K": K_geometry.tolist(),
            "plane_height_abs_p99_m": plane_p99, "plane_height_tolerance_m": .001,
            "static_scene_flow_max_px": static_max, "static_flow_tolerance_px": 1e-5,
            "target_object_mean_abs_flow_px": object_mean, "all_other_pixels_max_abs_flow_px": other_max,
            "independent_projected_object_flow_px": independent_flow,
            "observed_projected_object_flow_px": observed_flow,
            "independent_projection_max_abs_error_px": independent_error,
            "trajectory_initial_pose_max_abs_error_m": initial_pose_error,
            "eventgenesis_plugin_frame_provider": {"capture_steps": plugin.total_steps,
                "event_count_second_interval": int(len(interval_events)),
                "geometry_K_matches_registered": bool(np.allclose(plugin.interpolator.K, K_geometry)),
                "segmentation_mapping_applied": True, "depth_only": True},
            "trajectory_initialization": "generator enforces first captured link translation equals trajectory_pose(0) <= 1e-4 m",
        }
        return report, {"genesis_runtime": gs.__file__, "raw_seg_unique": np.unique(raw_seg).tolist(),
                        "render_to_link": render_to_link, "plane_id": plane_id, "object_ids": object_ids}
    finally:
        gs.destroy()


def check_epe() -> dict:
    gt = np.zeros((2, 2, 3), dtype=np.float32); valid = np.ones((2, 3), dtype=np.uint8)
    pred = np.zeros_like(gt)
    perfect, _ = compute_epe(pred, gt, valid, prediction_unit="px_per_interval", gt_unit="px_per_interval")
    gt[0] = 3.; gt[1] = 4.
    both, _ = compute_epe(pred, gt, valid, prediction_unit="px_per_interval", gt_unit="px_per_interval")
    vertical = gt.copy(); vertical[0] = 0.
    v_only, _ = compute_epe(pred, vertical, valid, prediction_unit="px_per_interval", gt_unit="px_per_interval")
    masked = valid.copy(); masked[:, 0] = 0
    masked_mean, count = compute_epe(pred, gt, masked, prediction_unit="px_per_interval", gt_unit="px_per_interval")
    empty_rejected = False
    try:
        compute_epe(pred, gt, np.zeros_like(valid), prediction_unit="px_per_interval", gt_unit="px_per_interval")
    except ValueError:
        empty_rejected = True
    if (perfect, both, v_only, masked_mean, count) != (0., 5., 4., 5., 4) or not empty_rejected:
        raise RuntimeError("strict EPE independent hand examples failed")
    return {"status": "PASS", "perfect_2d": perfect, "u3_v4": both,
            "vertical_only_v4": v_only, "valid_region_mean": masked_mean,
            "valid_count": count, "empty_mask_rejected": empty_rejected,
            "shape_unit_finite_broadcast_checks": "compute_epe requires exact [2,H,W], explicit unit and finite values"}


def check_cuda_warps(model_root: Path) -> tuple[dict, dict]:
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    sys.path.insert(0, str(model_root))
    import importlib
    corr_module = importlib.import_module("models.corr")
    corr_module.spatial_correlation_sample = pure_torch_spatial_correlation
    import models
    import multiscaleloss
    patch_record = patch_ste(models, multiscaleloss, model_root)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(19); torch.cuda.manual_seed_all(19)
    h, w = 33, 41
    image = torch.rand((1, 2, h, w), device=device)
    model = models.steflow().to(device).eval()
    rows = []
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        flow = torch.zeros((1, 2, h, w), device=device, requires_grad=True)
        with torch.no_grad():
            flow[:, 0] = du; flow[:, 1] = dv
        loss_out = multiscaleloss.warp(image, flow)
        model_out = model.flow_warp(image, flow, pad="zeros")
        y1, x1 = h - max(dv, 0), w - max(du, 0)
        reference = image[:, :, dv:dv + y1, du:du + x1]
        loss_delta = (loss_out[:, :, :y1, :x1] - reference).abs()
        model_delta = (model_out[:, :, :y1, :x1] - reference).abs()
        loss_mae = float(loss_delta.mean().cpu())
        model_mae = float(model_delta.mean().cpu())
        loss_err = float(loss_delta.max().cpu())
        model_err = float(model_delta.max().cpu())
        (model_out[:, :, :y1, :x1].square().mean()).backward()
        grad_finite = bool(flow.grad is not None and torch.isfinite(flow.grad).all())
        rows.append({"du": du, "dv": dv, "loss_warp_mae": loss_mae,
                     "model_warp_mae": model_mae, "loss_warp_max_abs": loss_err,
                     "model_warp_max_abs": model_err, "flow_gradient_finite": grad_finite})
        if max(loss_mae, model_mae) > 1e-6 or max(loss_err, model_err) > 5e-6 or not grad_finite:
            raise RuntimeError(f"CUDA STE warp contract failed: {rows[-1]}")

    # Exercise the actual photometric loss with a one-pixel translated texture.
    prev = torch.zeros((1, 1, 32, 40), device=device)
    prev[:, :, 6:26, 7:33] = torch.rand((1, 1, 20, 26), device=device)
    nxt = torch.zeros_like(prev); nxt[:, :, :, 1:] = prev[:, :, :, :-1]
    flow_zero = torch.zeros((1, 2, 32, 40), device=device, requires_grad=True)
    flow_known = torch.zeros_like(flow_zero, requires_grad=True)
    with torch.no_grad():
        flow_known[:, 0] = 1.
    pnp = prev[:, 0].detach().cpu().numpy(); nnp = nxt[:, 0].detach().cpu().numpy()
    l0 = multiscaleloss.compute_photometric_loss(pnp, nnp, None, [flow_zero], weights=[1.])
    l1 = multiscaleloss.compute_photometric_loss(pnp, nnp, None, [flow_known], weights=[1.])
    l1.backward()
    photo = {"zero_flow_charbonnier": float(l0.detach().cpu()),
             "known_positive_u_flow_charbonnier": float(l1.detach().cpu()),
             "known_warp_has_finite_gradient": bool(flow_known.grad is not None and torch.isfinite(flow_known.grad).all())}
    if not photo["known_warp_has_finite_gradient"] or not photo["known_positive_u_flow_charbonnier"] < photo["zero_flow_charbonnier"]:
        raise RuntimeError(f"actual CUDA photometric loss translation direction failed: {photo}")

    # Run real model forward at training resolution and inspect both vector components.
    model_input = torch.zeros((1, 2, 256, 256, 5), device=device)
    with torch.no_grad():
        output = model(model_input, 256, use_dt4=False, flow_seq=False)
    if tuple(output.shape) != (1, 2, 256, 256) or not torch.isfinite(output).all():
        raise RuntimeError(f"STE full forward contract failed: {tuple(output.shape)}")
    return ({"status": "PASS", "device": torch.cuda.get_device_name(device),
             "dtype": str(image.dtype), "warp_controls": rows, "photometric_loss_control": photo,
             "warp_error_tolerances": {"interior_mae": 1e-6, "max_abs_float32": 5e-6},
             "full_model_forward_shape": list(output.shape), "full_model_output_finite": True,
             "compatibility": patch_record},
            {"model_state_sha256": state_sha256(model), "pytorch": torch.__version__, "cuda": torch.version.cuda})


def check_bank(bank: Path, v2e: Path, data: Path) -> tuple[dict, dict]:
    bank_manifest = json.loads((bank / "bank_manifest.json").read_text())
    if bank_manifest.get("schema") != "stage4_shared_flow_bank_v2_geometry_corrected":
        raise RuntimeError(f"wrong bank schema: {bank_manifest.get('schema')}")
    if sha256(bank / "shared_trajectory.h5") != bank_manifest.get("bank_sha256"):
        raise RuntimeError("bank h5 hash mismatch")
    split = json.loads((bank / "registered_split.json").read_text())
    if len(split) != 32 or sum(row["split"] == "train" for row in split) != 24:
        raise RuntimeError("registered rollout split mismatch")
    v2e_manifest = json.loads((v2e / "v2e_complete_manifest.json").read_text())
    if v2e_manifest.get("bank_manifest_sha256") != sha256(bank / "bank_manifest.json"):
        raise RuntimeError("V2E raw events are not bound to corrected shared bank")
    adapter = json.loads((data / "training_bank_manifest.json").read_text())
    if adapter.get("shared_bank_sha256") != sha256(bank / "shared_trajectory.h5"):
        raise RuntimeError("adapter is not bound to corrected shared HDF5")
    entries = {(row["generator"], row["rollout_id"]): row for row in adapter["entries"]}
    if len(entries) != 64 or not adapter.get("paired_labels_identical"):
        raise RuntimeError("paired adapter manifest is incomplete or labels are not paired")
    map_counts, ground_cases, max_ground_flow = {}, 0, 0.
    initial_pose_errors = {}
    event_windows = 0
    with h5py.File(bank / "shared_trajectory.h5", "r") as h5:
        if h5.attrs.get("schema") != "stage4_shared_flow_bank_v2_geometry_corrected":
            raise RuntimeError("HDF5 schema does not match corrected geometry bank")
        for rollout in sorted(h5["episodes"].keys()):
            group = h5[f"episodes/{rollout}"]
            raw = np.asarray(group["segmentation_render_idx"], dtype=np.int32)
            physical = np.asarray(group["segmentation_link_idx"], dtype=np.int32)
            mapping = json.loads(bytes(group["segmentation_render_to_link_json"][()]).decode())
            rebuilt = segmentation_render_to_link(raw, mapping)
            if not np.array_equal(rebuilt, physical):
                raise RuntimeError(f"{rollout}: stored segmentation is not the renderer mapping")
            K_raw = np.asarray(group["camera_intrinsics_K_renderer"], dtype=np.float64)
            K_geo = np.asarray(group["camera_intrinsics_K_px"], dtype=np.float64)
            expected_K = K_raw.copy(); expected_K[0, 2] -= .5; expected_K[1, 2] -= .5
            if not np.allclose(K_geo, expected_K, atol=0, rtol=0):
                raise RuntimeError(f"{rollout}: renderer and integer-pixel K contract differs")
            if group.attrs.get("trajectory_initialization_max_abs_error_m", 999.) > 1e-4:
                raise RuntimeError(f"{rollout}: initial registered trajectory state mismatch")
            initial_pose_errors[rollout] = float(group.attrs["trajectory_initialization_max_abs_error_m"])
            if len(group["frame_timestamps_s"]) != 201 or not np.allclose(np.diff(group["frame_timestamps_s"][:]), .01, atol=1e-7, rtol=0):
                raise RuntimeError(f"{rollout}: endpoint timing contract failed")
            physical_ids = set(map(int, group["object_segmentation_link_ids"][:]))
            present = set(map(int, np.unique(physical)))
            if not present.issubset(physical_ids | {-1}):
                raise RuntimeError(f"{rollout}: segmentation labels lack a matching physical pose")
            map_counts[rollout] = {"raw_render_ids": sorted(map(int, np.unique(raw))),
                                   "physical_ids": sorted(present)}
            if rollout.startswith("s1_"):
                ground_id = int(group.attrs["ground_segmentation_link_id"])
                flow = np.asarray(group["flow_forward_30ms_px"], dtype=np.float32)
                valid = np.asarray(group["flow_valid_30ms"], dtype=bool)
                for index in (0, 32, 65):
                    mask = valid[index] & (physical[index * 3] == ground_id)
                    if np.any(mask):
                        ground_cases += 1
                        value = float(np.max(np.linalg.norm(flow[index, :, mask], axis=0)))
                        max_ground_flow = max(max_ground_flow, value)
            if sha256(Path(entries[("eventgenesis", rollout)]["path"])) != entries[("eventgenesis", rollout)]["sha256"]:
                raise RuntimeError(f"eventgenesis adapter hash mismatch for {rollout}")
            if sha256(Path(entries[("v2e", rollout)]["path"])) != entries[("v2e", rollout)]["sha256"]:
                raise RuntimeError(f"v2e adapter hash mismatch for {rollout}")
            for generator in ("eventgenesis", "v2e"):
                row = entries[(generator, rollout)]
                path = Path(row["path"])
                with np.load(path, allow_pickle=False) as npz:
                    counts = np.asarray(npz["event_counts"], dtype=np.int64)
                    starts = np.asarray(npz["event_start_s"], dtype=np.float64)
                    ends = np.asarray(npz["event_end_s"], dtype=np.float64)
                    if counts.shape != (66,) or starts.shape != (66,) or ends.shape != (66,):
                        raise RuntimeError(f"{generator}/{rollout}: malformed sample windows")
                    source_times = np.asarray(group["frame_timestamps_s"], dtype=np.float64)
                    if not np.array_equal(starts, source_times[np.arange(0, 198, 3)]) or not np.array_equal(ends, source_times[np.arange(3, 201, 3)]):
                        raise RuntimeError(f"{generator}/{rollout}: event endpoints and label timestamps differ")
                    if not np.array_equal(npz["flow_gt"], group["flow_forward_30ms_px"][:].astype(np.float32)):
                        raise RuntimeError(f"{generator}/{rollout}: adapter changed shared physical labels")
                    if not np.array_equal(npz["flow_valid"], group["flow_valid_30ms"][:].astype(np.uint8)):
                        raise RuntimeError(f"{generator}/{rollout}: adapter changed common physical validity")
                    event_windows += int(len(counts))
    if ground_cases == 0 or max_ground_flow > 1e-5:
        raise RuntimeError(f"static-camera ground physical flow contract failed: cases={ground_cases}, max={max_ground_flow}")
    if len(initial_pose_errors) != 32 or max(initial_pose_errors.values()) > 1e-4:
        raise RuntimeError(f"all-rollout initial pose contract failed: count={len(initial_pose_errors)}, max={max(initial_pose_errors.values(), default=float('inf'))}")
    if event_windows != 4224:
        raise RuntimeError(f"expected 4224 paired event windows, got {event_windows}")
    summary = {"status": "PASS", "shared_bank_sha256": sha256(bank / "shared_trajectory.h5"),
               "bank_manifest_sha256": sha256(bank / "bank_manifest.json"),
               "v2e_manifest_sha256": sha256(v2e / "v2e_complete_manifest.json"),
               "adapter_manifest_sha256": sha256(data / "training_bank_manifest.json"),
               "rollouts": 32, "paired_adapter_files": 64, "paired_windows": event_windows,
               "static_camera_ground_cases": ground_cases, "max_static_ground_flow_px_per_30ms": max_ground_flow,
               "first_frame_initial_pose_max_abs_error_m": max(initial_pose_errors.values()),
               "event_count_filtering": "none"}
    return summary, {"per_rollout_segmentation_ids": map_counts}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--bank-root", type=Path, required=True)
    p.add_argument("--v2e-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--genesis-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=("predata", "full"), default="full")
    args = p.parse_args()
    args.repo_root = args.repo_root.resolve(); args.config = args.config.resolve()
    args.bank_root = args.bank_root.resolve(); args.v2e_root = args.v2e_root.resolve()
    args.data_root = args.data_root.resolve(); args.model_root = args.model_root.resolve()
    args.genesis_root = args.genesis_root.resolve()
    results = []
    evidence = {}
    exact_command = " ".join(shlex.quote(part) for part in sys.argv)
    def run_check(name, fn):
        start = time.time()
        try:
            value, extra = fn()
            results.append({"check": name, "status": "PASS", "actual": value,
                            "tolerance": "see actual fields", "command": exact_command,
                            "evidence": str(args.output),
                            "elapsed_s": round(time.time() - start, 3)})
            evidence[name] = extra
        except Exception as exc:
            results.append({"check": name, "status": "FAIL", "actual": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(), "command": exact_command,
                            "evidence": str(args.output),
                            "elapsed_s": round(time.time() - start, 3)})

    config_for_preflight = json.loads(args.config.read_text())
    run_check("renderer_segmentation_mapping", lambda: check_renderer(
        "cuda:0", config_for_preflight["generators"]["eventgenesis"]))
    # Geometry and segmentation are measured together in the real renderer
    # fixture; expose a separate machine check as required by the receipt.
    render_row = next((row for row in results if row["check"] == "renderer_segmentation_mapping"), None)
    if render_row and render_row["status"] == "PASS":
        run_check("depth_K_pixel_center", lambda: ({"status": "PASS",
                  "plane_height_abs_p99_m": render_row["actual"]["plane_height_abs_p99_m"],
                  "tolerance_m": .001, "static_scene_flow_max_px": render_row["actual"]["static_scene_flow_max_px"]}, {}))
        run_check("trajectory_initialization", lambda: ({"status": "PASS",
                  "fixture_pose_max_abs_error_m": render_row["actual"]["trajectory_initial_pose_max_abs_error_m"],
                  "tolerance_m": 1e-4,
                  "all_rollouts_checked_after_generation": args.stage == "full"}, {}))
    else:
        results.append({"check": "depth_K_pixel_center", "status": "NOT_RUN", "actual": "renderer fixture did not pass"})

    def warp_fn():
        value, extra = check_cuda_warps(args.model_root)
        return value, extra
    run_check("steflow_model_and_loss_warp_cuda", warp_fn)
    run_check("epe_controls", lambda: (check_epe(), {}))
    def corr_fn():
        import torch
        return verify_correlation(torch.device("cuda:0")), {}
    run_check("correlation_operator_cuda", corr_fn)
    if args.stage == "predata":
        passed = all(row["status"] == "PASS" for row in results)
        report = {"schema": "stage4_predata_controls_v1", "stage": "predata",
                  "overall_status": "PASS" if passed else "BLOCKED", "created_unix": time.time(),
                  "checks": results, "evidence": evidence,
                  "data_generation_authorized": passed,
                  "warning": "This pre-data probe is not a full training receipt."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"overall_status": report["overall_status"], "report": str(args.output),
                          "checks": [{"check": row["check"], "status": row["status"]} for row in results]}, indent=2), flush=True)
        if not passed:
            raise SystemExit(2)
        return
    run_check("bank_manifest_and_geometry", lambda: check_bank(args.bank_root, args.v2e_root, args.data_root))

    # Paired event window recount by timestamps for every raw stream/window.
    def recount():
        adapter_rows = {(row["generator"], row["rollout_id"]): row
                        for row in json.loads((args.data_root / "training_bank_manifest.json").read_text())["entries"]}
        with h5py.File(args.bank_root / "shared_trajectory.h5", "r") as h5:
            v2e_manifest = json.loads((args.v2e_root / "v2e_complete_manifest.json").read_text())
            v2e_rows = {row["rollout_id"]: row for row in v2e_manifest["episodes"]}
            checked = 0
            for rollout in sorted(h5["episodes"].keys()):
                group = h5[f"episodes/{rollout}"]
                times = np.asarray(group["frame_timestamps_s"], dtype=np.float64)
                sources = {
                    "eventgenesis": np.asarray(group["eventgenesis_events_txyp"], dtype=np.float64)[:, 0],
                    "v2e": None,
                }
                v2e_path = Path(v2e_rows[rollout]["raw_events_path"])
                with h5py.File(v2e_path, "r") as raw_h5:
                    v2e_us = np.asarray(raw_h5["events"], dtype=np.uint32)[:, 0]
                sources["v2e"] = v2e_us.astype(np.float64) * 1e-6
                for generator, event_times in sources.items():
                    adapter_file = Path(adapter_rows[(generator, rollout)]["path"])
                    with np.load(adapter_file, allow_pickle=False) as npz:
                        expected = np.asarray(npz["event_counts"], dtype=np.int64)
                    direct = np.asarray([np.count_nonzero((event_times > times[i]) & (event_times <= times[i + 3]))
                                         for i in range(0, 198, 3)], dtype=np.int64)
                    if not np.array_equal(direct, expected):
                        raise RuntimeError(f"{generator}/{rollout}: independent timestamp recount mismatch")
                    checked += len(direct)
        if checked != 4224:
            raise RuntimeError(f"timestamp recount covered {checked}, expected 4224")
        return {"status": "PASS", "windows_recounted": checked,
                "interval_rule": "(t_i,t_i+3] independently counted from raw timestamps"}, {}
    run_check("event_window_recount", recount)

    def pair_check():
        manifest = json.loads((args.data_root / "training_bank_manifest.json").read_text())
        rows = {(e["generator"], e["rollout_id"]): e for e in manifest["entries"]}
        raw_v2e_manifest = json.loads((args.v2e_root / "v2e_complete_manifest.json").read_text())
        raw_v2e_rows = {row["rollout_id"]: row for row in raw_v2e_manifest["episodes"]}
        if len(rows) != 64:
            raise RuntimeError("expected 64 train/validation adapter rollouts")
        train_count = val_count = 0
        checked = 0
        for rid in sorted({key[1] for key in rows}):
            left = rows[("eventgenesis", rid)]; right = rows[("v2e", rid)]
            if left["label_sha256"] != right["label_sha256"]:
                raise RuntimeError(f"paired physical labels differ for {rid}")
            for generator in ("eventgenesis", "v2e"):
                row = rows[(generator, rid)]
                if row["split"] == "train": train_count += 66
                else: val_count += 66
                with np.load(row["path"], allow_pickle=False) as npz:
                    if npz["events"].shape != (66, 2, 260, 346, 5):
                        raise RuntimeError(f"{generator}/{rid}: event-only input shape mismatch")
                    if not np.isfinite(npz["events"]).all():
                        raise RuntimeError(f"{generator}/{rid}: non-finite event input")
                    if any(key in npz for key in ("depth", "segmentation", "pose", "mask", "box")):
                        raise RuntimeError(f"{generator}/{rid}: adapter contains privileged input key")
                    checked += 66
        if (train_count, val_count, checked) != (3168, 1056, 4224):
            raise RuntimeError(f"paired sample census mismatch: {(train_count,val_count,checked)}")
        return {"status": "PASS", "paired_rollouts": 32, "paired_sample_windows": checked,
                "train_samples_each_generator": 1584, "validation_samples_each_generator": 528,
                "input_contract": "five-bin ON/OFF event representation only"}, {}
    run_check("paired_adapters_and_shared_labels", pair_check)

    def representation_check():
        empty = np.empty((0, 4), dtype=np.float64)
        empty_rep = gaussian_count_representation(empty, 8, 9)
        if int(empty_rep.sum()) != 0:
            raise RuntimeError("empty event window did not map to zero representation")
        tied = np.asarray([[.001, 2, 3, 1], [.001, 4, 3, -1], [.001, 2, 3, 1]], dtype=np.float64)
        tied_rep = gaussian_count_representation(tied, 8, 9)
        if tied_rep.shape != (2, 8, 9, 5) or not np.isfinite(tied_rep).all() or tied_rep[0].sum() <= 0 or tied_rep[1].sum() <= 0:
            raise RuntimeError("equal-timestamp / dual-polarity representation control failed")
        manifest = json.loads((args.data_root / "training_bank_manifest.json").read_text())
        entry_rows = {(row["generator"], row["rollout_id"]): row for row in manifest["entries"]}
        v2e_rows = {row["rollout_id"]: row for row in json.loads((args.v2e_root / "v2e_complete_manifest.json").read_text())["episodes"]}
        selected = [f"s{stratum}_{local:02d}" for stratum in range(4) for local in (0, 6)]
        checked = []
        with h5py.File(args.bank_root / "shared_trajectory.h5", "r") as shared:
            for rollout in selected:
                group = shared[f"episodes/{rollout}"]
                times = np.asarray(group["frame_timestamps_s"], dtype=np.float64)
                for generator in ("eventgenesis", "v2e"):
                    if generator == "eventgenesis":
                        raw = np.asarray(group["eventgenesis_events_txyp"], dtype=np.float64).reshape(-1, 4)
                    else:
                        raw_path = Path(v2e_rows[rollout]["raw_events_path"])
                        with h5py.File(raw_path, "r") as source:
                            data = np.asarray(source["events"], dtype=np.uint32)
                        raw = np.stack((data[:, 0].astype(np.float64) * 1e-6, data[:, 1].astype(np.float64),
                                        data[:, 2].astype(np.float64), np.where(data[:, 3] > 0, 1., -1.)), axis=1)
                    # Boundary cases: first interval, immediately adjacent
                    # interval, and final registered interval before the 20ms tail.
                    with np.load(Path(entry_rows[(generator, rollout)]["path"]), allow_pickle=False) as data:
                        for index in (0, 1, 65):
                            left, right = float(times[index * 3]), float(times[index * 3 + 3])
                            window = raw[(raw[:, 0] > left) & (raw[:, 0] <= right)].copy()
                            if len(window): window[:, 0] -= left
                            expected = gaussian_count_representation(window, 260, 346)
                            actual = np.asarray(data["events"][index])
                            if not np.array_equal(actual, expected):
                                raise RuntimeError(f"{generator}/{rollout}/window{index}: stored representation differs from raw-event reconstruction")
                            checked.append({"generator": generator, "rollout": rollout, "window": index,
                                            "event_count": int(len(window)), "start_s": left, "end_s": right})
        return {"status": "PASS", "edge_controls": {"empty_zero": True, "equal_timestamp_dual_polarity": True},
                "raw_windows_recomputed": len(checked), "windows": checked}, {}
    run_check("event_representation_and_edge_cases", representation_check)

    def core_check():
        subprocess.run(["git", "-C", str(args.repo_root), "cat-file", "-e", "e79e656ac00f7847b98db9e12bf43b093f3afff1^{commit}"], check=True)
        diff = subprocess.run(["git", "-C", str(args.repo_root), "diff", "--quiet",
                               "e79e656ac00f7847b98db9e12bf43b093f3afff1", "--", "genesis_event_plugin"], check=False)
        if diff.returncode != 0:
            raise RuntimeError("frozen simulator core differs from e79e656")
        hashes = {path: sha256(args.repo_root / path) for path in CORE_FILES}
        return {"status": "PASS", "frozen_core_commit": "e79e656ac00f7847b98db9e12bf43b093f3afff1",
                "files": hashes}, {}
    run_check("frozen_core_unchanged", core_check)

    def isolation():
        config = json.loads(args.config.read_text())
        if config.get("real_scoring_enabled") is not False or config.get("real_inference_enabled_in_this_stage") is not False:
            raise RuntimeError("configuration enables real scoring/inference")
        return {"status": "PASS", "real_data_opened": False, "real_feedback_used": False,
                "inference_inputs": "event-only", "training_supervision": "synthetic grayscale photometric/smoothness only"}, {}
    run_check("synthetic_only_guard", isolation)

    # Canonical fingerprints are computed after all checks; they do not depend
    # on this report or on logs/checkpoints outside the fixed data manifests.
    try:
        fingerprint = build_fingerprint(args.repo_root, args.config, args.bank_root, args.v2e_root,
                                         args.data_root, args.model_root, args.genesis_root)
        results.append({"check": "fingerprint_completeness", "status": "PASS",
                        "actual": {"fingerprint_sha256": fingerprint["fingerprint_sha256"],
                                   "code_inputs": len(fingerprint["code"]),
                                   "input_components": len(fingerprint["inputs"])},
                        "command": exact_command, "evidence": str(args.output)})
    except Exception as exc:
        fingerprint = None
        results.append({"check": "fingerprint_completeness", "status": "FAIL",
                        "actual": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                        "command": exact_command, "evidence": str(args.output)})
    all_pass = len(results) == len(FOUNDATION) and all(row["status"] == "PASS" for row in results)
    report = {"schema": "stage4_preflight_result_v1", "stage": "foundation",
              "created_unix": time.time(), "host": os.uname().nodename,
              "overall_status": "PASS" if all_pass else "BLOCKED", "checks": results,
              "evidence": evidence, "fingerprint": fingerprint, "command": exact_command,
              "real_data_scoring": "disabled", "required_for_full_release": FOUNDATION + [
                  "paired_formal_epoch_pilot", "resume_equivalence_epoch_boundary", "launcher_negative_controls"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"overall_status": report["overall_status"], "report": str(args.output),
                      "fingerprint_sha256": fingerprint["fingerprint_sha256"],
                      "checks": [{"check": row["check"], "status": row["status"]} for row in results]}, indent=2), flush=True)
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
