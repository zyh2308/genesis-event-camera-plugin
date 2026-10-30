#!/usr/bin/env python3
"""Validate the diagnostic coverage-splat candidate on CPU.

This script leaves the frozen plugin untouched. It compares the diagnostic
NumPy candidate with a tensor-native Torch implementation on the same cached
projected coordinates, exercises a moving tilted surface and layered occlusion
controls, compares the active NumPy/Torch interpolator paths on the archived
episodes, and classifies residual image errors against saved Direct-HDR frames.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator
from genesis_event_plugin.physics_warp import (
    _splat_projected_diagnostics,
    depth_aware_bidirectional_warp_projected_diagnostics,
    project_endpoint_to_alpha,
)
from genesis_event_plugin.torch_physics import project_endpoint_to_alpha_stack_torch


DEFAULT_CACHE = ROOT / "benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
BASE_DIAGNOSTIC = ROOT / "benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_pixel_rule_attribution.py"
DEPTH_TIE_M = 1e-5
LOG_EPS = 1e-6


def load_base_diagnostic():
    spec = importlib.util.spec_from_file_location("r1_5_pixel_rule_diagnostic", BASE_DIAGNOSTIC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def log_error(a, b):
    return np.abs(np.log(np.maximum(np.asarray(a, dtype=np.float32), LOG_EPS)) -
                  np.log(np.maximum(np.asarray(b, dtype=np.float32), LOG_EPS)))


def coverage_splat_torch(image, source_depth, target_uv, target_z, source_valid=None):
    """Torch CPU equivalent of the diagnostic NumPy front-to-back resolver.

    It groups all contributors within the depth tie band of the current
    nearest layer, accumulates their bilinear footprint, and lets farther
    layers fill only the remaining target-pixel coverage.
    """
    image = torch.as_tensor(np.asarray(image, dtype=np.float32), dtype=torch.float32, device="cpu")
    source_depth = torch.as_tensor(np.asarray(source_depth, dtype=np.float32), dtype=torch.float32, device="cpu")
    target_uv = torch.as_tensor(np.asarray(target_uv, dtype=np.float32), dtype=torch.float32, device="cpu")
    target_z = torch.as_tensor(np.asarray(target_z, dtype=np.float32), dtype=torch.float32, device="cpu")
    height, width = source_depth.shape
    scalar = image.ndim == 2
    src = image.unsqueeze(-1) if scalar else image
    valid = torch.isfinite(src).all(dim=-1)
    valid &= torch.isfinite(source_depth) & (source_depth > 1e-6)
    valid &= torch.isfinite(target_z) & (target_z > 1e-6)
    if source_valid is not None:
        valid &= torch.as_tensor(np.asarray(source_valid, dtype=bool), dtype=torch.bool, device="cpu")

    tx, ty = target_uv[0], target_uv[1]
    x0, y0 = torch.floor(tx).long(), torch.floor(ty).long()
    wx, wy = tx - x0, ty - y0
    source_index = torch.arange(height * width, dtype=torch.long).reshape(height, width)
    pieces = []
    for ox, oy, weight in (
        (0, 0, (1 - wx) * (1 - wy)),
        (1, 0, wx * (1 - wy)),
        (0, 1, (1 - wx) * wy),
        (1, 1, wx * wy),
    ):
        x, y = x0 + ox, y0 + oy
        keep = valid & (weight > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if bool(keep.any()):
            pixel = (y[keep] * width + x[keep]).long()
            pieces.append((pixel, weight[keep], target_z[keep], src[keep]))

    pixel_count = height * width
    output = torch.zeros((pixel_count, src.shape[-1]), dtype=torch.float32)
    coverage = torch.zeros(pixel_count, dtype=torch.float32)
    remaining_coverage = torch.ones(pixel_count, dtype=torch.float32)
    if pieces:
        pixel = torch.cat([piece[0] for piece in pieces])
        weight = torch.cat([piece[1] for piece in pieces])
        depth = torch.cat([piece[2] for piece in pieces])
        value = torch.cat([piece[3] for piece in pieces])
        unresolved = torch.ones(pixel.shape[0], dtype=torch.bool)
        max_layers = int(torch.bincount(pixel, minlength=pixel_count).max())
        for _ in range(max_layers):
            min_depth = torch.full((pixel_count,), float("inf"), dtype=torch.float32)
            min_depth.scatter_reduce_(0, pixel[unresolved], depth[unresolved], reduce="amin", include_self=True)
            layer = unresolved & (depth <= min_depth[pixel] + DEPTH_TIE_M)
            if not bool(layer.any()):
                break
            raw_coverage = torch.zeros(pixel_count, dtype=torch.float32)
            raw_coverage.index_add_(0, pixel[layer], weight[layer])
            value_sum = torch.zeros((pixel_count, src.shape[-1]), dtype=torch.float32)
            value_sum.index_add_(0, pixel[layer], value[layer] * weight[layer, None])
            alpha = torch.minimum(raw_coverage, remaining_coverage)
            layer_mean = value_sum / torch.clamp(raw_coverage[:, None], min=1e-30)
            output += alpha[:, None] * layer_mean
            coverage += alpha
            remaining_coverage = torch.clamp(remaining_coverage - alpha, min=0.0)
            unresolved &= ~layer

    visible = coverage > 1e-8
    output[visible] /= coverage[visible, None]
    output = output.reshape(height, width, -1)
    if scalar:
        output = output[..., 0]
    return output, visible.reshape(height, width), coverage.reshape(height, width)


def dilated_label_boundary(segmentation):
    seg = np.asarray(segmentation)
    boundary = np.zeros(seg.shape, dtype=bool)
    boundary[:, 1:] |= seg[:, 1:] != seg[:, :-1]
    boundary[:, :-1] |= seg[:, :-1] != seg[:, 1:]
    boundary[1:, :] |= seg[1:, :] != seg[:-1, :]
    boundary[:-1, :] |= seg[:-1, :] != seg[1:, :]
    expanded = boundary.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y0, y1 = max(0, dy), min(seg.shape[0], seg.shape[0] + dy)
            x0, x1 = max(0, dx), min(seg.shape[1], seg.shape[1] + dx)
            expanded[y0:y1, x0:x1] |= boundary[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return expanded


def synthetic_controls(base):
    # A linearly shaded, tilted continuous surface. Its depth varies smoothly
    # across source samples, and its true image under pure x translation is
    # analytically known for interior pixels.
    height, width = 24, 32
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    image = 0.2 + 0.003 * xx + 0.002 * yy
    depth = 1.0 + 0.0003 * xx + 0.0002 * yy
    shifts = [-0.5, -0.25, -0.01, -1e-6, 0.0, 1e-6, 0.01, 0.25, 0.5]
    ramp_rows = []
    ramp_numpy = []
    ramp_torch = []
    ramp_legacy = []
    for shift in shifts:
        uv = np.stack((xx + shift, yy))
        numpy_image, numpy_valid, numpy_coverage = base.coverage_splat(image, depth, uv, depth)
        torch_image, torch_valid, torch_coverage = coverage_splat_torch(image, depth, uv, depth)
        legacy_image, _ = _splat_projected_diagnostics(image, depth, uv, depth)
        torch_image = torch_image.numpy()
        torch_valid = torch_valid.numpy()
        torch_coverage = torch_coverage.numpy()
        interior = np.s_[2:-2, 2:-2]
        analytic = 0.2 + 0.003 * (xx - shift) + 0.002 * yy
        ramp_rows.append({
            "x_shift_px": shift,
            "numpy_torch_max_abs_luma": float(np.max(np.abs(numpy_image - torch_image))),
            "numpy_torch_visibility_mismatch_px": int(np.count_nonzero(numpy_valid != torch_valid)),
            "coverage_max_abs_delta": float(np.max(np.abs(numpy_coverage - torch_coverage))),
            "analytic_tilted_plane_interior_max_abs_error": float(np.max(np.abs(numpy_image[interior] - analytic[interior]))),
        })
        ramp_numpy.append(numpy_image)
        ramp_torch.append(torch_image)
        ramp_legacy.append(legacy_image)
    near_zero_idx = shifts.index(1e-6)
    center = (height // 2, width // 2)
    ramp_continuity = {
        "legacy_center_luma_by_shift": [
            {"x_shift_px": shift, "legacy_luma": float(frame[center])}
            for shift, frame in zip(shifts, ramp_legacy)
        ],
        "legacy_delta_from_zero_at_plus_1e-6_px": float(abs(ramp_legacy[near_zero_idx][center] - ramp_legacy[shifts.index(0.0)][center])),
        "legacy_delta_from_zero_at_minus_1e-6_px": float(abs(ramp_legacy[shifts.index(-1e-6)][center] - ramp_legacy[shifts.index(0.0)][center])),
        "candidate_numpy_delta_from_zero_at_plus_1e-6_px": float(abs(ramp_numpy[near_zero_idx][center] - ramp_numpy[shifts.index(0.0)][center])),
        "candidate_torch_delta_from_zero_at_plus_1e-6_px": float(abs(ramp_torch[near_zero_idx][center] - ramp_torch[shifts.index(0.0)][center])),
        "candidate_numpy_delta_from_zero_at_minus_1e-6_px": float(abs(ramp_numpy[shifts.index(-1e-6)][center] - ramp_numpy[shifts.index(0.0)][center])),
        "candidate_torch_delta_from_zero_at_minus_1e-6_px": float(abs(ramp_torch[shifts.index(-1e-6)][center] - ramp_torch[shifts.index(0.0)][center])),
        "all_shift_results": ramp_rows,
    }

    # Foreground rectangle in front of a constant background, translated by a
    # quarter pixel. The edge pixel has analytically known area coverage.
    image = np.full((height, width), 0.02, dtype=np.float32)
    depth = np.full((height, width), 2.0, dtype=np.float32)
    image[7:17, 8:18] = 0.8
    depth[7:17, 8:18] = 1.0
    layer_rows = []
    for shift in (-0.5, -0.25, 0.0, 0.25, 0.5):
        uv = np.stack((xx + shift, yy))
        numpy_image, numpy_valid, numpy_coverage = base.coverage_splat(image, depth, uv, depth)
        torch_image, torch_valid, torch_coverage = coverage_splat_torch(image, depth, uv, depth)
        torch_image, torch_valid, torch_coverage = torch_image.numpy(), torch_valid.numpy(), torch_coverage.numpy()
        layer_rows.append({
            "x_shift_px": shift,
            "numpy_torch_max_abs_luma": float(np.max(np.abs(numpy_image - torch_image))),
            "numpy_torch_visibility_mismatch_px": int(np.count_nonzero(numpy_valid != torch_valid)),
            "numpy_torch_coverage_max_abs_delta": float(np.max(np.abs(numpy_coverage - torch_coverage))),
            "foreground_interior_luma": float(numpy_image[10, 12]),
            "background_interior_luma": float(numpy_image[10, 25]),
            "left_edge_luma": float(numpy_image[10, 8]),
            "right_edge_luma": float(numpy_image[10, 18]),
        })
    return {
        "moving_tilted_continuous_surface": ramp_continuity,
        "moving_foreground_over_background": {
            "expected_plus_0p25_left_edge_luma": 0.75 * 0.8 + 0.25 * 0.02,
            "expected_plus_0p25_right_edge_luma": 0.25 * 0.8 + 0.75 * 0.02,
            "shifts": layer_rows,
        },
    }


def load_episode(cache_root: Path, ep: str):
    episode_dir = cache_root / f"episode_{ep}"
    cache_path = episode_dir / "same_state_source_cache.h5"
    evidence_path = episode_dir / "rendered_frame_evidence.h5"
    with h5py.File(cache_path, "r") as f:
        src = {name: np.asarray(f[name][:]) for name in (
            "timestamps_s", "radiance_luma", "depth", "segmentation", "camera_to_world_cv",
            "object_poses", "segmentation_pose_ids")}
        width, height = map(int, f.attrs["camera_resolution_wh"])
        fov = float(f.attrs["vertical_fov_deg"])
    with h5py.File(evidence_path, "r") as f:
        direct_times = np.asarray(f["dense_times_s"][:], dtype=np.float64)
        direct = np.asarray(f["dense_direct_hdr_luminance"][:], dtype=np.float32)
        direct_seg = np.asarray(f["dense_direct_segmentation"][:], dtype=np.int32)
    return src, direct_times, direct, direct_seg, width, height, fov, cache_path, evidence_path


def frame_edge_metrics(segmentation):
    boundary = np.zeros(segmentation.shape, dtype=bool)
    boundary[:, 1:] |= segmentation[:, 1:] != segmentation[:, :-1]
    boundary[:, :-1] |= segmentation[:, :-1] != segmentation[:, 1:]
    boundary[1:, :] |= segmentation[1:, :] != segmentation[:-1, :]
    boundary[:-1, :] |= segmentation[:-1, :] != segmentation[1:, :]
    return dilated_label_boundary(segmentation), boundary


def run_episode(base, cache_root: Path, ep: str):
    src, direct_times, direct, direct_seg, width, height, fov, cache_path, evidence_path = load_episode(cache_root, ep)
    times = src["timestamps_s"]
    left = int(np.flatnonzero(np.isclose(times, 0.5805, atol=1e-9, rtol=0))[0])
    right = int(np.flatnonzero(np.isclose(times, 0.5905, atol=1e-9, rtol=0))[0])
    if right != left + 1:
        raise RuntimeError(f"{ep}: cached interval endpoints are not adjacent")
    focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    K = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    object0 = {int(seg_id): src["object_poses"][left, j] for j, seg_id in enumerate(src["segmentation_pose_ids"])}
    object1 = {int(seg_id): src["object_poses"][right, j] for j, seg_id in enumerate(src["segmentation_pose_ids"])}
    T0, T1 = src["camera_to_world_cv"][left], src["camera_to_world_cv"][right]
    I0, I1 = src["radiance_luma"][left], src["radiance_luma"][right]
    depth0, depth1 = src["depth"][left], src["depth"][right]
    seg0, seg1 = src["segmentation"][left], src["segmentation"][right]
    endpoint0_candidate_max = 0.0
    endpoint1_candidate_max = 0.0
    endpoint0_active_max = 0.0
    endpoint1_active_max = 0.0
    candidate_vis_mismatch = 0
    baseline_vis_mismatch = 0
    boundary_residual = 0
    residual_count = 0
    residual_unknown = 0
    residual_border = 0
    residual_labels = {}
    residual_csv = []
    per_frame = []
    all_candidate_errors = []

    np_interp = PhysicsInterpolator(K, sampler_config=AdaptiveSamplingConfig(min_samples=10, max_samples=10),
                                    disocclusion_policy="hold", composite="next_primary", warp_backend="numpy")
    torch_interp = PhysicsInterpolator(K, sampler_config=AdaptiveSamplingConfig(min_samples=10, max_samples=10),
                                       disocclusion_policy="hold", composite="next_primary", warp_backend="torch", device="cpu")
    numpy_active, numpy_times = np_interp.interpolate(
        I0, depth0, seg0, I1, depth1, seg1, T0, T1, None, float(times[left]), float(times[right]),
        input_space="linear", object_world0=object0, object_world1=object1)
    torch_active, torch_times = torch_interp.interpolate(
        I0, depth0, seg0, I1, depth1, seg1, T0, T1, None, float(times[left]), float(times[right]),
        input_space="linear", object_world0=object0, object_world1=object1)
    numpy_active = np.asarray(numpy_active, dtype=np.float32)
    torch_active = np.asarray(torch_active, dtype=np.float32)
    if not np.allclose(numpy_times, torch_times, atol=1e-8, rtol=0):
        raise RuntimeError(f"{ep}: NumPy/Torch interpolator timestamp mismatch")
    if not np.allclose(numpy_times, direct_times, atol=1e-8, rtol=0):
        raise RuntimeError(f"{ep}: interpolator timestamps do not align with saved direct renders")
    np_valid = np_interp.last_visibility["intermediate_visibility"]
    torch_valid = torch_interp.last_visibility["intermediate_visibility"]
    baseline_vis_mismatch = int(np.count_nonzero(np_valid != torch_valid))
    active_delta = np.abs(numpy_active - torch_active)

    # Retain the actual tensor-native projection outputs as well as the full
    # interpolator comparison, so tiny coordinate differences can be related
    # directly to downstream z-buffer changes.
    device = torch.device("cpu")
    alpha_stack = torch.as_tensor(np.arange(1, len(direct_times) + 1, dtype=np.float32) / len(direct_times), device=device)
    T0_t = torch.as_tensor(T0, dtype=torch.float32, device=device)
    T1_t = torch.as_tensor(T1, dtype=torch.float32, device=device)
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
    object0_t = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in object0.items()}
    object1_t = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in object1.items()}
    depth0_t = torch.as_tensor(depth0, dtype=torch.float32, device=device)
    depth1_t = torch.as_tensor(depth1, dtype=torch.float32, device=device)
    seg0_t = torch.as_tensor(seg0, device=device)
    seg1_t = torch.as_tensor(seg1, device=device)
    uv0_t, z0_t, valid0_t = project_endpoint_to_alpha_stack_torch(
        depth0_t, seg0_t, K_t, T0_t, T0_t, T1_t, alpha_stack, 0, object0_t, object1_t, None)
    uv1_t, z1_t, valid1_t = project_endpoint_to_alpha_stack_torch(
        depth1_t, seg1_t, K_t, T1_t, T0_t, T1_t, alpha_stack, 1, object0_t, object1_t, None)
    projection_rows = []
    candidate_native_backend_delta_max = 0.0
    candidate_native_backend_delta_p99 = 0.0
    candidate_native_backend_delta_over_1e_3 = 0
    candidate_native_backend_delta_over_0p2 = 0
    candidate_native_visibility_mismatch = 0
    candidate_native_backend_delta_values = []
    active_divergence_rows = []
    projection_records = []

    for qi, t in enumerate(direct_times):
        alpha = float((t - times[left]) / (times[right] - times[left]))
        p0 = project_endpoint_to_alpha(depth0, seg0, K, T0, T0, alpha, 0, object0, object1, None)
        p1 = project_endpoint_to_alpha(depth1, seg1, K, T1, T0, alpha, 1, object0, object1, None)
        p0_tensor = (uv0_t[qi].numpy(), z0_t[qi].numpy(), valid0_t[qi].numpy())
        p1_tensor = (uv1_t[qi].numpy(), z1_t[qi].numpy(), valid1_t[qi].numpy())
        projection_records.append((p0, p1, p0_tensor, p1_tensor))
        for endpoint, numpy_projection, torch_projection in ((0, p0, p0_tensor), (1, p1, p1_tensor)):
            common = numpy_projection[2] & torch_projection[2]
            uv_delta = np.abs(numpy_projection[0] - torch_projection[0])
            z_delta = np.abs(numpy_projection[1] - torch_projection[1])
            projection_rows.append({
                "time_s": float(t), "endpoint": endpoint,
                "common_valid_pixels": int(common.sum()),
                "validity_mismatch_pixels": int(np.count_nonzero(numpy_projection[2] != torch_projection[2])),
                "uv_max_abs_delta_px": float(uv_delta[:, common].max()) if common.any() else 0.0,
                "uv_p99_abs_delta_px": float(np.quantile(uv_delta[:, common], 0.99)) if common.any() else 0.0,
                "z_max_abs_delta_m": float(z_delta[common].max()) if common.any() else 0.0,
            })
        # Explicit candidate/backend parity on exact shared NumPy-projected geometry.
        n0, nv0, nc0 = base.coverage_splat(I0, depth0, *p0)
        t0, tv0, tc0 = coverage_splat_torch(I0, depth0, *p0)
        n1, nv1, nc1 = base.coverage_splat(I1, depth1, *p1)
        t1, tv1, tc1 = coverage_splat_torch(I1, depth1, *p1)
        endpoint0_candidate_max = max(endpoint0_candidate_max, float(np.max(np.abs(n0 - t0.numpy()))))
        endpoint1_candidate_max = max(endpoint1_candidate_max, float(np.max(np.abs(n1 - t1.numpy()))))
        candidate_vis_mismatch += int(np.count_nonzero(nv0 != tv0.numpy()) + np.count_nonzero(nv1 != tv1.numpy()))
        endpoint0_active, d0 = _splat_projected_diagnostics(I0, depth0, *p0)
        endpoint1_active, d1 = _splat_projected_diagnostics(I1, depth1, *p1)
        # Exercise the actual Torch backend splat on the same projected endpoint data.
        from genesis_event_plugin.torch_physics import _splat_projected_batch_torch
        torch_device_args = [torch.as_tensor(x, dtype=torch.float32, device="cpu") for x in (I0[None], depth0[None], p0[0][None], p0[1][None])]
        active_t0, active_v0 = _splat_projected_batch_torch(*torch_device_args[:2], torch_device_args[2], torch_device_args[3],
                                                              torch.as_tensor(p0[2][None], dtype=torch.bool))
        torch_device_args = [torch.as_tensor(x, dtype=torch.float32, device="cpu") for x in (I1[None], depth1[None], p1[0][None], p1[1][None])]
        active_t1, active_v1 = _splat_projected_batch_torch(*torch_device_args[:2], torch_device_args[2], torch_device_args[3],
                                                              torch.as_tensor(p1[2][None], dtype=torch.bool))
        endpoint0_active_max = max(endpoint0_active_max, float(np.max(np.abs(endpoint0_active - active_t0[0].numpy()))))
        endpoint1_active_max = max(endpoint1_active_max, float(np.max(np.abs(endpoint1_active - active_t1[0].numpy()))))

        candidate_valid = nv0 | nv1
        candidate = np.full_like(I0, np.nan, dtype=np.float32)
        candidate[nv1] = n1[nv1]
        candidate[~nv1 & nv0] = n0[~nv1 & nv0]
        candidate = base.hold_fill(candidate, candidate_valid, alpha, I0, I1)
        n0_native, nv0_native, _ = coverage_splat_torch(I0, depth0, *p0_tensor)
        n1_native, nv1_native, _ = coverage_splat_torch(I1, depth1, *p1_tensor)
        n0_native, nv0_native = n0_native.numpy(), nv0_native.numpy()
        n1_native, nv1_native = n1_native.numpy(), nv1_native.numpy()
        candidate_native_valid = nv0_native | nv1_native
        candidate_native = np.full_like(I0, np.nan, dtype=np.float32)
        candidate_native[nv1_native] = n1_native[nv1_native]
        candidate_native[~nv1_native & nv0_native] = n0_native[~nv1_native & nv0_native]
        candidate_native = base.hold_fill(candidate_native, candidate_native_valid, alpha, I0, I1)
        native_delta = np.abs(candidate - candidate_native)
        candidate_native_backend_delta_values.append(native_delta.ravel())
        candidate_native_backend_delta_max = max(candidate_native_backend_delta_max, float(native_delta.max()))
        candidate_native_backend_delta_over_1e_3 += int((native_delta > 1e-3).sum())
        candidate_native_backend_delta_over_0p2 += int((native_delta > 0.2).sum())
        candidate_native_visibility_mismatch += int(np.count_nonzero(candidate_valid != candidate_native_valid))
        baseline, baseline_valid, _ = depth_aware_bidirectional_warp_projected_diagnostics(
            I0, depth0, p0[0], p0[1], p0[2], I1, depth1, p1[0], p1[1], p1[2], composite="next_primary")
        baseline = base.hold_fill(baseline, baseline_valid, alpha, I0, I1)
        ref = direct[qi]
        frame_delta = active_delta[qi]
        if np.any(frame_delta):
            for flat in np.argsort(frame_delta.ravel())[-20:][::-1]:
                y, x = divmod(int(flat), width)
                if frame_delta[y, x] <= 0:
                    break
                active_divergence_rows.append({
                    "time_s": float(t), "x": x, "y": y,
                    "numpy_active_luma": float(numpy_active[qi, y, x]),
                    "torch_active_luma": float(torch_active[qi, y, x]),
                    "abs_luma_delta": float(frame_delta[y, x]),
                    "direct_luma": float(ref[y, x]),
                    "direct_segmentation_id_posthoc": int(direct_seg[qi, y, x]),
                })
        err = log_error(candidate, ref)
        base_err = log_error(baseline, ref)
        bad = err > 0.2
        edge_band, _ = frame_edge_metrics(direct_seg[qi])
        residual_count += int(bad.sum())
        boundary_residual += int((bad & edge_band).sum())
        residual_unknown += int((bad & ~candidate_valid).sum())
        yy, xx = np.mgrid[:height, :width]
        residual_border += int((bad & ((xx < 2) | (xx >= width - 2) | (yy < 2) | (yy >= height - 2))).sum())
        labels, counts = np.unique(direct_seg[qi][bad], return_counts=True)
        for label, count in zip(labels, counts):
            residual_labels[str(int(label))] = residual_labels.get(str(int(label)), 0) + int(count)
        all_candidate_errors.append(err)
        per_frame.append({
            "time_s": float(t),
            "candidate_residual_px_over_0p2": int(bad.sum()),
            "baseline_bad_px_over_0p2": int((base_err > 0.2).sum()),
            "candidate_unknown_px": int((~candidate_valid).sum()),
            "residual_on_direct_segmentation_boundary_band": int((bad & edge_band).sum()),
        })
        if bad.any():
            order = np.argsort(err.ravel())[::-1]
            taken = 0
            for flat in order:
                y, x = divmod(int(flat), width)
                if err[y, x] <= 0.2:
                    break
                selected_endpoint = 1 if nv1[y, x] else 0
                selected_cov = float((nc1 if selected_endpoint else nc0)[y, x])
                residual_csv.append({
                    "episode": ep, "time_s": float(t), "x": x, "y": y,
                    "candidate_luma": float(candidate[y, x]), "direct_luma": float(ref[y, x]),
                    "candidate_abs_log_error": float(err[y, x]), "baseline_abs_log_error": float(base_err[y, x]),
                    "direct_segmentation_id_posthoc": int(direct_seg[qi, y, x]),
                    "within_1px_of_any_segmentation_boundary_posthoc": bool(edge_band[y, x]),
                    "candidate_had_projected_support": bool(candidate_valid[y, x]),
                    "selected_endpoint": selected_endpoint, "selected_endpoint_coverage": selected_cov,
                })
                taken += 1
                if taken >= 20:
                    break

    candidate_error = np.concatenate([e[None] for e in all_candidate_errors], axis=0)
    backend_trace_rows = []
    for flat in np.argsort(active_delta.ravel())[-10:][::-1]:
        qi, local = divmod(int(flat), height * width)
        y, x = divmod(local, width)
        numpy_projection0, numpy_projection1, torch_projection0, torch_projection1 = projection_records[qi]
        for endpoint, (image, depth, segmentation, numpy_projection, torch_projection) in enumerate((
            (I0, depth0, seg0, numpy_projection0, torch_projection0),
            (I1, depth1, seg1, numpy_projection1, torch_projection1),
        )):
            for backend, projection in (("numpy_projection", numpy_projection), ("torch_projection", torch_projection)):
                trace = trace_target_contributors(base, image, depth, segmentation, projection, x, y)
                backend_trace_rows.append({
                    "episode": ep,
                    "time_s": float(direct_times[qi]),
                    "x": int(x), "y": int(y), "endpoint": endpoint,
                    "projection_backend": backend,
                    "active_numpy_luma": float(numpy_active[qi, y, x]),
                    "active_torch_luma": float(torch_active[qi, y, x]),
                    "combined_active_abs_luma_delta": float(active_delta[qi, y, x]),
                    "endpoint_splat_trace_json": json.dumps(trace, separators=(",", ":")),
                })
    return {
        "episode": ep,
        "input_hashes": {"source_cache_sha256": sha256(cache_path), "direct_frame_cache_sha256": sha256(evidence_path)},
        "active_numpy_vs_torch_interpolator": {
            "max_abs_luma_delta": float(active_delta.max()),
            "p99_abs_luma_delta": float(np.quantile(active_delta, 0.99)),
            "visibility_mismatch_pixel_times": baseline_vis_mismatch,
            "timestamp_match": True,
        },
        "projected_splat_backend_comparison": {
            "shared_numpy_geometry_candidate_numpy_vs_torch_max_abs_luma": {
                "endpoint0": endpoint0_candidate_max, "endpoint1": endpoint1_candidate_max,
            },
            "shared_numpy_geometry_candidate_visibility_mismatch_pixels": candidate_vis_mismatch,
            "active_numpy_splat_vs_torch_splat_max_abs_luma": {
                "endpoint0": endpoint0_active_max, "endpoint1": endpoint1_active_max,
            },
            "candidate_native_numpy_geometry_vs_torch_geometry": {
                "max_abs_luma_delta": candidate_native_backend_delta_max,
                "p99_abs_luma_delta": float(np.quantile(np.concatenate(candidate_native_backend_delta_values), 0.99)),
                "pixel_times_over_1e_3": candidate_native_backend_delta_over_1e_3,
                "pixel_times_over_0p2": candidate_native_backend_delta_over_0p2,
                "visibility_mismatch_pixels": candidate_native_visibility_mismatch,
            },
            "projection_rows": projection_rows,
        },
        "active_backend_divergence_examples_top20_per_frame": active_divergence_rows,
        "active_backend_divergence_source_traces_top10": backend_trace_rows,
        "candidate_residual_summary": {
            "pixel_times_over_0p2": residual_count,
            "edge_band_pixel_times": boundary_residual,
            "unknown_support_pixel_times": residual_unknown,
            "image_border_pixel_times": residual_border,
            "label_counts_posthoc": residual_labels,
            "mean_abs_log_error": float(candidate_error.mean()),
            "p95_abs_log_error": float(np.quantile(candidate_error, 0.95)),
            "max_abs_log_error": float(candidate_error.max()),
            "by_frame": per_frame,
        },
        "residual_examples_top20_per_frame": residual_csv,
    }


def trace_target_contributors(base, image, depth, segmentation, projection, x, y):
    contributions = base.collect_contributions(image, depth, *projection)
    target = int(y * depth.shape[1] + x)
    selected = np.flatnonzero(contributions["target"] == target)
    selected = selected[np.argsort(contributions["z"][selected])]
    rows = []
    for i in selected[:8]:
        source = int(contributions["source"][i])
        sy, sx = divmod(source, depth.shape[1])
        rows.append({
            "source_xy": [sx, sy],
            "source_seg_id": int(segmentation[sy, sx]),
            "bilinear_weight": float(contributions["weight"][i]),
            "projected_z_m": float(contributions["z"][i]),
            "source_luma": float(contributions["value"][i, 0]),
        })
    splat, diagnostics = _splat_projected_diagnostics(image, depth, *projection)
    return {
        "splat_luma": float(splat[y, x]),
        "visible": bool(diagnostics["visibility"][y, x]),
        "z_buffer_m": float(diagnostics["z_buffer"][y, x]),
        "contributors": rows,
    }


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(root: Path):
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory: {output}")
    output.mkdir(parents=True)
    base = load_base_diagnostic()
    results = [run_episode(base, args.cache_root.resolve(), ep) for ep in ("004", "005")]
    synthetic = synthetic_controls(base)
    summary = {
        "purpose": "CPU NumPy/Torch parity, general moving-surface controls, and residual attribution for the diagnostic coverage-splat candidate",
        "scope": "cached episodes plus deterministic synthetic controls; no Genesis renderer, physics, GPU, training, or frozen-core modification",
        "repository_head_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "base_diagnostic_script_sha256": sha256(BASE_DIAGNOSTIC),
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                    "torch_device": "CPU", "platform": platform.platform()},
        "runtime_module_hashes": {
            str(name): {"imported_from": str(Path(module.__file__).resolve()),
                        "sha256": sha256(Path(module.__file__).resolve())}
            for name, module in (
                ("genesis_event_plugin.physics_interpolator", sys.modules["genesis_event_plugin.physics_interpolator"]),
                ("genesis_event_plugin.physics_warp", sys.modules["genesis_event_plugin.physics_warp"]),
                ("genesis_event_plugin.torch_physics", sys.modules["genesis_event_plugin.torch_physics"]),
            )
        },
        "thresholds": {"log_luma_error": 0.2, "depth_tie_band_m": DEPTH_TIE_M},
        "synthetic_controls": synthetic,
        "episodes": results,
        "candidate_status": "diagnostic-only; no production switch",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_csv(output / "residual_top20_per_frame.csv", [row for result in results for row in result["residual_examples_top20_per_frame"]])
    write_csv(output / "backend_projection_pixel_traces.csv", [row for result in results for row in result["active_backend_divergence_source_traces_top10"]])
    write_manifest(output)
    print(json.dumps({
        "output": str(output),
        "torch_parity": [r["projected_splat_backend_comparison"] for r in results],
        "active_path": [r["active_numpy_vs_torch_interpolator"] for r in results],
        "residual": [r["candidate_residual_summary"] for r in results],
        "manifest_sha256": sha256(output / "SHA256SUMS"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
