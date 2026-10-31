#!/usr/bin/env python3
"""Bounded CPU acceptance replay for the isolated R1.5 core candidate.

This reuses immutable scene-linear/HDR caches and the already captured g1
shadow x MSAA factorial. It does not initialize Genesis, render, advance
physics, train, or alter the frozen branch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator, RenderRequiredError
from genesis_event_plugin.physics_warp import (
    _splat_projected_coverage_diagnostics,
    depth_aware_bidirectional_warp_projected_diagnostics,
)
from genesis_event_plugin.radiometry import linear_luminance
from genesis_event_plugin.torch_physics import _splat_projected_coverage_batch_torch
from genesis_event_plugin.torch_physics import depth_aware_bidirectional_warp_projected_batch_torch

DEFAULT_CACHE = ROOT / "benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
DEFAULT_FACTORIAL = ROOT / "benchmark/robot_closed_loop/results/r1_5_shadow_multisample_cached_states_20261029_g1_gpu0_previewfix_855b4b0"
DEFAULT_CONFIG = ROOT / "benchmark/robot_closed_loop/configs/r1_5_core_candidate_20261030.json"
LOG_EPS = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def log_error(a, b):
    return np.abs(np.log(np.maximum(np.asarray(a, dtype=np.float32), LOG_EPS)) -
                  np.log(np.maximum(np.asarray(b, dtype=np.float32), LOG_EPS)))


def ideal_dvs(frames, times, initial_frame, initial_time):
    """Run the unchanged deterministic physical-log DVS model."""
    h, w = initial_frame.shape
    torch.set_num_threads(1)
    config = DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
        cutoff_hz=0.0, leak_rate_hz=0.0, leak_jitter_fraction=0.0,
        shot_noise_rate_hz=0.0, noise_rate_cov_decades=0.0,
        refractory_period_s=0.0, radiance_input=True,
        response_mode="physical_log", readout_model="ideal_readout", seed=42,
    )
    emulator = DvsEmulator(res=(h, w), config=config, device="cpu")
    emulator.initialize(initial_frame, float(initial_time))
    chunks = [emulator.generate_events(frame, float(t)) for frame, t in zip(frames, times)]
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)


def event_metrics(reference, candidate, width):
    """Compare event support, polarity, per-pixel count, and matched time."""
    def index(events):
        rows = {}
        for row in np.asarray(events):
            t, x, y, p = row
            key = (int(x), int(y), int(np.sign(p)))
            rows.setdefault(key, []).append(float(t))
        for times in rows.values():
            times.sort()
        return rows

    ref, cand = index(reference), index(candidate)
    all_keys = ref.keys() | cand.keys()
    count_l1 = sum(abs(len(ref.get(k, ())) - len(cand.get(k, ()))) for k in all_keys)
    matched_time_errors = []
    for key in ref.keys() & cand.keys():
        n = min(len(ref[key]), len(cand[key]))
        matched_time_errors.extend(abs(a - b) for a, b in zip(ref[key][:n], cand[key][:n]))
    shared_slots = sum(min(len(ref.get(k, ())), len(cand.get(k, ()))) for k in all_keys)
    times = np.asarray(matched_time_errors, dtype=np.float64)
    return {
        "reference_event_count": int(len(reference)),
        "candidate_event_count": int(len(candidate)),
        "reference_on_count": int(np.count_nonzero(np.asarray(reference)[:, 3] > 0)) if len(reference) else 0,
        "candidate_on_count": int(np.count_nonzero(np.asarray(candidate)[:, 3] > 0)) if len(candidate) else 0,
        "reference_off_count": int(np.count_nonzero(np.asarray(reference)[:, 3] < 0)) if len(reference) else 0,
        "candidate_off_count": int(np.count_nonzero(np.asarray(candidate)[:, 3] < 0)) if len(candidate) else 0,
        "pixel_polarity_count_l1": int(count_l1),
        "matched_event_slots": int(shared_slots),
        "matched_event_time_abs_error_median_s": float(np.median(times)) if times.size else None,
        "matched_event_time_abs_error_p95_s": float(np.quantile(times, 0.95)) if times.size else None,
        "matched_event_time_abs_error_max_s": float(times.max()) if times.size else None,
        "coordinate_convention": "raw [t,x,y,p], polarity {-1,+1}",
    }


def dilated_boundary(seg):
    seg = np.asarray(seg)
    edge = np.zeros(seg.shape, dtype=bool)
    edge[:, 1:] |= seg[:, 1:] != seg[:, :-1]
    edge[:, :-1] |= seg[:, 1:] != seg[:, :-1]
    edge[1:, :] |= seg[1:, :] != seg[:-1, :]
    edge[:-1, :] |= seg[1:, :] != seg[:-1, :]
    padded = np.pad(edge, 1, constant_values=False)
    expanded = np.zeros_like(edge)
    h, w = edge.shape
    for dy in range(3):
        for dx in range(3):
            expanded |= padded[dy:dy+h, dx:dx+w]
    return expanded


def run_unit_controls():
    h, w = 24, 32
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    image = 0.2 + 0.003 * xx + 0.002 * yy
    depth = 1.0 + 0.0003 * xx + 0.0002 * yy
    labels = np.ones((h, w), dtype=np.int32)
    center = (h // 2, w // 2)
    shift_rows = []
    shift_outputs = {}
    for shift in (-1e-6, 0.0, 1e-6):
        uv = np.stack((xx + shift, yy))
        out, diag = _splat_projected_coverage_diagnostics(
            image, depth, uv, depth, source_surface_id=labels,
        )
        shift_outputs[shift] = out
        shift_rows.append({
            "shift_px": shift,
            "visible_fraction": float(diag["visibility"].mean()),
            "supported_fraction": float(diag["support"].mean()),
            "center_luma": float(out[center]),
        })
    zero = shift_outputs[0.0][2:-2, 2:-2]
    for row in shift_rows:
        frame = shift_outputs[row["shift_px"]][2:-2, 2:-2]
        row["max_interior_abs_delta_from_zero"] = float(np.max(np.abs(frame - zero)))

    # One source contributes approximately 1e-6 of a neighboring target pixel.
    sparse_image = np.zeros((4, 4), dtype=np.float32)
    sparse_depth = np.ones((4, 4), dtype=np.float32)
    sparse_uv = np.stack(np.meshgrid(np.arange(4), np.arange(4), indexing="xy")).astype(np.float32)
    sparse_uv[0, 1, 1] = 1.000001
    sparse_valid = np.zeros((4, 4), dtype=bool)
    sparse_valid[1, 1] = True
    sparse_image[1, 1] = 1.0
    sparse, sparse_diag = _splat_projected_coverage_diagnostics(
        sparse_image, sparse_depth, sparse_uv, sparse_depth,
        source_valid=sparse_valid, source_surface_id=np.ones((4, 4), dtype=np.int32),
    )
    tiny_coverage = float(sparse_diag["coverage"][1, 2])
    tiny_supported = bool(sparse_diag["support"][1, 2])

    # Two half-pixel contributors straddle the configured depth tie band.
    # Front-to-back coverage should keep the radiance continuous on both sides
    # of the layer classification boundary.
    tie_image = np.zeros((4, 5), dtype=np.float32)
    tie_depth = np.ones((4, 5), dtype=np.float32)
    tie_valid = np.zeros((4, 5), dtype=bool)
    tie_uv = np.stack(np.meshgrid(np.arange(5), np.arange(4), indexing="xy")).astype(np.float32)
    tie_uv[0, 1, 1] = 1.5
    tie_uv[0, 1, 2] = 2.5
    tie_image[1, 1], tie_image[1, 2] = 0.8, 0.02
    tie_valid[1, 1] = tie_valid[1, 2] = True
    tie_seg = np.ones((4, 5), dtype=np.int32)
    tie_seg[1, 1], tie_seg[1, 2] = 2, 3
    depth_tolerance = max(1e-7, 8.0 * np.finfo(np.float32).eps)
    tie_rows = []
    for gap in (0.9 * depth_tolerance, 1.1 * depth_tolerance):
        current_depth = tie_depth.copy()
        current_depth[1, 2] += gap
        tie_out, tie_diag = _splat_projected_coverage_diagnostics(
            tie_image, current_depth, tie_uv, current_depth,
            source_valid=tie_valid, source_surface_id=tie_seg,
        )
        tie_torch, _, tie_torch_diag = _splat_projected_coverage_batch_torch(
            torch.from_numpy(tie_image[None]), torch.from_numpy(current_depth[None]),
            torch.from_numpy(tie_uv[None]), torch.from_numpy(current_depth[None]),
            torch.from_numpy(tie_valid[None]), torch.from_numpy(tie_seg[None]),
        )
        tie_rows.append({
            "depth_gap_m": float(gap),
            "configured_tie_tolerance_m_at_z1": float(depth_tolerance),
            "target_luma": float(tie_out[1, 2]),
            "target_coverage": float(tie_diag["coverage"][1, 2]),
            "target_support": bool(tie_diag["support"][1, 2]),
            "target_surface_consistent": bool(tie_diag["surface_consistent"][1, 2]),
            "torch_target_luma": float(tie_torch[0, 1, 2]),
            "numpy_torch_target_abs_delta": float(abs(tie_out[1, 2] - tie_torch[0, 1, 2].item())),
            "torch_target_coverage": float(tie_torch_diag["coverage"][0, 1, 2]),
        })

    # Foreground patch over a farther background, shifted by one quarter pixel.
    occ = np.full((h, w), 0.02, dtype=np.float32)
    occ_depth = np.full((h, w), 2.0, dtype=np.float32)
    occ_labels = np.ones((h, w), dtype=np.int32)
    occ[7:17, 8:18] = 0.8
    occ_depth[7:17, 8:18] = 1.0
    occ_labels[7:17, 8:18] = 2
    occ_uv = np.stack((xx + 0.25, yy))
    occ_out, occ_diag = _splat_projected_coverage_diagnostics(
        occ, occ_depth, occ_uv, occ_depth, source_surface_id=occ_labels,
    )
    occ_metrics = {
        "foreground_interior_luma": float(occ_out[10, 12]),
        "background_interior_luma": float(occ_out[10, 25]),
        "mixed_edge_luma_x8": float(occ_out[10, 8]),
        "mixed_edge_luma_x18": float(occ_out[10, 18]),
        "expected_edge_high": float(0.75 * 0.8 + 0.25 * 0.02),
        "expected_edge_low": float(0.25 * 0.8 + 0.75 * 0.02),
        "supported_fraction": float(occ_diag["support"].mean()),
    }

    # CPU NumPy/Torch parity for a moving layered surface, with same surfaces.
    uv = np.stack((xx + 0.25, yy))
    np_out, np_diag = _splat_projected_coverage_diagnostics(
        occ, occ_depth, uv, occ_depth, source_surface_id=occ_labels,
    )
    torch_out, torch_vis, torch_diag = _splat_projected_coverage_batch_torch(
        torch.from_numpy(occ[None]), torch.from_numpy(occ_depth[None]),
        torch.from_numpy(uv[None]), torch.from_numpy(occ_depth[None]),
        torch.ones((1, h, w), dtype=torch.bool), torch.from_numpy(occ_labels[None]),
    )
    torch_out_np = torch_out[0].numpy()
    torch_support_np = torch_diag["support"][0].numpy()
    parity = {
        "max_abs_luma_delta": float(np.nanmax(np.abs(np_out - torch_out_np))),
        "max_abs_coverage_delta": float(np.max(np.abs(np_diag["coverage"] - torch_diag["coverage"][0].numpy()))),
        "support_mismatch_pixels": int(np.count_nonzero(np_diag["support"] != torch_support_np)),
        "visibility_mismatch_pixels": int(np.count_nonzero(np_diag["visibility"] != torch_vis[0].numpy())),
    }

    # Exact t=1 must return the observed frame even where endpoint depth is invalid.
    H = W = 8
    yy8, xx8 = np.mgrid[:H, :W].astype(np.float32)
    ident = np.stack((xx8, yy8))
    valid = np.ones((H, W), dtype=bool)
    invalid = valid.copy()
    invalid[2, 3] = False
    f0 = np.full((H, W), 0.1, dtype=np.float32)
    f1 = np.full((H, W), 0.37, dtype=np.float32)
    d = np.ones((H, W), dtype=np.float32)
    lab = np.ones((H, W), dtype=np.int32)
    exact_np, exact_np_valid, _ = depth_aware_bidirectional_warp_projected_diagnostics(
        f0, d, ident, d, valid, f1, d, ident, d, invalid,
        composite="same_surface_log_linear", alpha=1.0,
        surface_id0=lab, surface_id1=lab,
    )
    exact_torch, exact_torch_valid, _ = depth_aware_bidirectional_warp_projected_batch_torch(
        torch.from_numpy(f0[None]), torch.from_numpy(d[None]),
        torch.from_numpy(ident[None, None]), torch.from_numpy(d[None, None]),
        torch.from_numpy(valid[None, None]),
        torch.from_numpy(f1[None]), torch.from_numpy(d[None]),
        torch.from_numpy(ident[None, None]), torch.from_numpy(d[None, None]),
        torch.from_numpy(invalid[None, None]),
        composite="same_surface_log_linear", return_diagnostics=True,
        alpha=torch.tensor([1.0]), surface_id0=torch.from_numpy(lab[None]),
        surface_id1=torch.from_numpy(lab[None]),
    )
    exact_torch_image = exact_torch[0, 0].numpy()
    exact_torch_visibility = exact_torch_valid[0, 0].numpy()
    exact_endpoint = {
        "max_abs_radiance_delta": float(np.max(np.abs(exact_np - f1))),
        "finite_direct_pixel_still_known_with_invalid_depth": bool(exact_np_valid[2, 3]),
        "torch_max_abs_radiance_delta": float(np.max(np.abs(exact_torch_image - f1))),
        "torch_finite_direct_pixel_still_known_with_invalid_depth": bool(exact_torch_visibility[2, 3]),
    }

    return {
        "moving_continuous_surface": shift_rows,
        "tiny_support": {
            "neighbor_luma": float(sparse[1, 2]),
            "neighbor_coverage": tiny_coverage,
            "neighbor_visible": bool(sparse_diag["visibility"][1, 2]),
            "neighbor_supported": tiny_supported,
        },
        "depth_tie_band_continuity": tie_rows,
        "front_to_back_occlusion": occ_metrics,
        "numpy_torch_cpu_parity": parity,
        "exact_endpoint_identity": exact_endpoint,
    }


def ramp_control():
    h = w = 4
    t0, t1 = 0.0, 0.010
    l0, l1 = 0.1, 0.2
    depth = np.ones((h, w), dtype=np.float32)
    seg = np.ones((h, w), dtype=np.int32)
    eye = np.eye(4, dtype=np.float64)
    K = np.asarray([[10.0, 0, 2], [0, 10.0, 2], [0, 0, 1]], dtype=np.float64)
    direct = []
    direct_times = []
    for i in range(1, 11):
        t = i * 0.001
        lum = l0 * np.exp((t / (t1 - t0)) * np.log(l1 / l0))
        direct.append(np.full((h, w), lum, dtype=np.float32))
        direct_times.append(t)
    interpolator = PhysicsInterpolator(
        K, sampler_config=AdaptiveSamplingConfig(min_samples=2, max_samples=2),
        disocclusion_policy="nan", composite="same_surface_log_linear",
        warp_backend="numpy", strict_fidelity=True,
        max_photometric_interval_s=0.005,
    )
    requests_without_provider = None
    try:
        interpolator.interpolate(
            np.full((h, w), l0, np.float32), depth, seg,
            np.full((h, w), l1, np.float32), depth, seg,
            eye, eye, None, t0, t1, input_space="linear",
        )
    except RenderRequiredError as exc:
        requests_without_provider = [req.as_dict() for req in exc.requests]

    provider_calls = []
    def exact_radiance_provider(request):
        provider_calls.append(request.as_dict())
        sample_luma = l0 * np.exp((request.t - t0) / (t1 - t0) * np.log(l1 / l0))
        return {
            "t": request.t,
            "radiance": np.full((h, w), sample_luma, dtype=np.float32),
            "provenance": "analytic_log_linear_ramp_control",
            "available_at_s": request.t,
        }

    start = time.perf_counter()
    frames, times = interpolator.interpolate(
        np.full((h, w), l0, np.float32), depth, seg,
        np.full((h, w), l1, np.float32), depth, seg,
        eye, eye, None, t0, t1, input_space="linear",
        intermediate_state_provider=exact_radiance_provider,
    )
    api_wall_s = time.perf_counter() - start
    actual = ideal_dvs(frames, times, np.full((h, w), l0, np.float32), t0)
    reference = ideal_dvs(direct, direct_times, np.full((h, w), l0, np.float32), t0)
    static = ideal_dvs(
        [np.full((h, w), l0, np.float32) for _ in direct_times],
        direct_times, np.full((h, w), l0, np.float32), t0,
    )
    expected_t = np.asarray([0.01 * (k * 0.2) / np.log(2.0) for k in (1, 2, 3)], dtype=np.float64)
    per_pixel = np.sort(actual[(actual[:, 1] == 1) & (actual[:, 2] == 1), 0])
    return {
        "strict_missing_sample_requests": requests_without_provider,
        "strict_output_times_s": [float(x) for x in times],
        "strict_uncertainty": interpolator.last_uncertainty,
        "provider_request_count": len(provider_calls),
        "provider_requests": provider_calls,
        "candidate_event_count": int(len(actual)),
        "reference_event_count": int(len(reference)),
        "pixel_1_1_candidate_event_times_s": per_pixel.tolist(),
        "analytic_expected_event_times_s": expected_t.tolist(),
        "max_abs_timing_error_s": float(np.max(np.abs(per_pixel - expected_t))) if len(per_pixel) == 3 else None,
        "candidate_vs_reference_raw_events_exact": bool(actual.shape == reference.shape and np.array_equal(actual, reference)),
        "candidate_vs_reference_event_metrics": event_metrics(reference, actual, w),
        "static_scene_event_count": int(len(static)),
        "cached_state_api_wall_s_cpu": float(api_wall_s),
        "actual_renderer_cost_measured": False,
        "claim_boundary": "known analytic log-linear ramp only; changes between 5 ms real anchors remain unobserved",
    }


def load_cached_episode(cache_root: Path, ep: str):
    folder = cache_root / f"episode_{ep}"
    cache_path = folder / "same_state_source_cache.h5"
    evidence_path = folder / "rendered_frame_evidence.h5"
    with h5py.File(cache_path, "r") as handle:
        source = {name: np.asarray(handle[name][:]) for name in (
            "timestamps_s", "radiance_luma", "depth", "segmentation",
            "camera_to_world_cv", "object_poses", "segmentation_pose_ids",
        )}
        width, height = map(int, handle.attrs["camera_resolution_wh"])
        fov = float(handle.attrs["vertical_fov_deg"])
    with h5py.File(evidence_path, "r") as handle:
        times = np.asarray(handle["dense_times_s"][:], dtype=np.float64)
        direct = np.asarray(handle["dense_direct_hdr_luminance"][:], dtype=np.float32)
        direct_rgb = np.asarray(handle["dense_direct_hdr_rgb"][:], dtype=np.float32)
        seg = np.asarray(handle["dense_direct_segmentation"][:], dtype=np.int32)
    return source, times, direct, direct_rgb, seg, width, height, fov, cache_path, evidence_path


def strict_cached_replay(cache_root: Path, ep: str):
    src, times, direct, _rgb, _seg, width, height, fov, cache_path, evidence_path = load_cached_episode(cache_root, ep)
    t0, t1 = 0.5805, 0.5905
    left = int(np.flatnonzero(np.isclose(src["timestamps_s"], t0, atol=1e-9, rtol=0))[0])
    right = int(np.flatnonzero(np.isclose(src["timestamps_s"], t1, atol=1e-9, rtol=0))[0])
    rows = (times >= t0 + 0.001 - 1e-9) & (times <= t1 + 1e-9)
    dense_times = times[rows]
    dense_direct = direct[rows]
    # Endpoints live in the immutable scene-state cache; the nine interior
    # timestamps are archived exact direct-HDR frames from the same trace.
    interior = [
        {"t": float(t), "radiance": frame,
         "provenance": "archived_exact_direct_hdr_frame",
         "available_at_s": float(t)}
        for t, frame in zip(dense_times[:-1], dense_direct[:-1])
    ]
    focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    K = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    ids = src["segmentation_pose_ids"]
    obj0 = {int(sid): src["object_poses"][left, j] for j, sid in enumerate(ids)}
    obj1 = {int(sid): src["object_poses"][right, j] for j, sid in enumerate(ids)}
    interp = PhysicsInterpolator(
        # The archived direct cache already resolves every 1 ms state in the
        # interval; cap only the geometric preflight at that resolution.
        K, sampler_config=AdaptiveSamplingConfig(min_samples=2, max_samples=10),
        disocclusion_policy="nan", composite="same_surface_log_linear",
        warp_backend="numpy", strict_fidelity=True,
        max_photometric_interval_s=0.005,
    )
    started = time.perf_counter()
    output_frames, output_times = interp.interpolate(
        src["radiance_luma"][left], src["depth"][left], src["segmentation"][left],
        src["radiance_luma"][right], src["depth"][right], src["segmentation"][right],
        src["camera_to_world_cv"][left], src["camera_to_world_cv"][right], None,
        t0, t1, input_space="linear", object_world0=obj0, object_world1=obj1,
        intermediate_states=interior,
    )
    wall = time.perf_counter() - started
    out = np.stack(output_frames)
    if not np.allclose(output_times, dense_times, atol=1e-9, rtol=0):
        raise RuntimeError(f"{ep}: strict cache timestamps do not align with archived direct-HDR frames")
    max_abs = float(np.max(np.abs(out - dense_direct)))
    direct_events = ideal_dvs(dense_direct, dense_times, src["radiance_luma"][left], t0)
    candidate_events = ideal_dvs(out, output_times, src["radiance_luma"][left], t0)
    middle_slot = int(np.flatnonzero(np.isclose(dense_times, 0.5855, atol=1e-9, rtol=0))[0])
    midpoint_only = PhysicsInterpolator(
        K, sampler_config=AdaptiveSamplingConfig(min_samples=2, max_samples=2),
        disocclusion_policy="nan", composite="same_surface_log_linear",
        warp_backend="numpy", strict_fidelity=True,
        max_photometric_interval_s=0.005,
    )
    midpoint_frames, midpoint_times = midpoint_only.interpolate(
        src["radiance_luma"][left], src["depth"][left], src["segmentation"][left],
        src["radiance_luma"][right], src["depth"][right], src["segmentation"][right],
        src["camera_to_world_cv"][left], src["camera_to_world_cv"][right], None,
        t0, t1, input_space="linear", object_world0=obj0, object_world1=obj1,
        intermediate_states=[{
            "t": float(dense_times[middle_slot]),
            "radiance": dense_direct[middle_slot],
            "provenance": "archived_exact_direct_hdr_frame_0p5855",
            "available_at_s": float(dense_times[middle_slot]),
        }],
    )
    midpoint_events = ideal_dvs(
        midpoint_frames, midpoint_times, src["radiance_luma"][left], t0,
    )
    # Production-default motion/adaptive sampling may ask for more than the
    # one photometric midpoint. The exact request list is recorded, not hidden.
    default_probe = PhysicsInterpolator(
        K, disocclusion_policy="nan", composite="same_surface_log_linear",
        warp_backend="numpy", strict_fidelity=True,
        max_photometric_interval_s=0.005,
    )
    try:
        default_probe.interpolate(
            src["radiance_luma"][left], src["depth"][left], src["segmentation"][left],
            src["radiance_luma"][right], src["depth"][right], src["segmentation"][right],
            src["camera_to_world_cv"][left], src["camera_to_world_cv"][right], None,
            t0, t1, input_space="linear", object_world0=obj0, object_world1=obj1,
            intermediate_states=[{
                "t": float(dense_times[middle_slot]),
                "radiance": dense_direct[middle_slot],
                "provenance": "archived_exact_direct_hdr_frame_0p5855",
                "available_at_s": float(dense_times[middle_slot]),
            }],
        )
        default_probe_status = {"status": "midpoint_satisfied_default_sampler"}
    except RenderRequiredError as exc:
        default_probe_status = {
            "status": "additional_real_samples_required_by_default_sampler",
            "adaptive_sample_count": int(default_probe.last_sampling.samples),
            "required_additional_sample_count": len(exc.requests),
            "requests": [request.as_dict() for request in exc.requests],
        }
    midpoint_only_wall = float(midpoint_only.last_uncertainty["output_ready_after_right_endpoint_wall_s"])
    midpoint_only_metrics = event_metrics(direct_events, midpoint_events, width)
    return {
        "episode": ep,
        "input_sha256": {"state_cache": sha256(cache_path), "direct_hdr_evidence": sha256(evidence_path)},
        "timestamp_count": int(len(output_times)),
        "timestamps_s": [float(t) for t in output_times],
        "max_abs_luma_delta_from_archived_direct_frames": max_abs,
        "exact_image_sequence_match": bool(np.array_equal(out, dense_direct)),
        "raw_events": event_metrics(direct_events, candidate_events, width),
        "raw_events_exact_match": bool(direct_events.shape == candidate_events.shape and np.array_equal(direct_events, candidate_events)),
        "one_midpoint_5ms_sampling_ablation": {
            "status": "forced_photometric_minimum; adaptive max_samples=2 to isolate one midpoint",
            "direct_reference_events_1ms": int(len(direct_events)),
            "midpoint_sample_events": int(len(midpoint_events)),
            "event_metrics_vs_dense_1ms_direct": midpoint_only_metrics,
            "strict_extra_render_count_with_cached_midpoint": int(midpoint_only.last_uncertainty["extra_render_count"]),
            "max_anchor_gap_s": float(midpoint_only.last_uncertainty["photometric_max_anchor_gap_s"]),
            "output_ready_after_right_endpoint_wall_s_cpu": midpoint_only_wall,
            "claim_boundary": "5ms piecewise log-radiance contract; does not meet tighter motion-derived sampling if the default sampler requests more",
        },
        "default_motion_sampler_midpoint_only_probe": default_probe_status,
        "extra_render_count": int(interp.last_uncertainty["extra_render_count"]),
        "photometric_anchor_count": int(interp.last_uncertainty["photometric_anchor_count"]),
        "max_anchor_gap_s": float(interp.last_uncertainty["photometric_max_anchor_gap_s"]),
        "output_available_at_s": float(interp.last_uncertainty["output_available_at_s"]),
        "anchor_provenance": interp.last_uncertainty["anchor_provenance"],
        "strict_api_wall_s_cpu_using_cached_frames": float(wall),
        "new_renderer_cost_measured": False,
        "event_arrays": {
            "dense_direct_events": direct_events,
            "strict_dense_cache_replay_events": candidate_events,
            "strict_midpoint_only_events": midpoint_events,
        },
    }


def factorial_candidate(cache_root: Path, factorial_root: Path, ep: str):
    source, _times, _direct, _rgb, _seg, width, height, fov, cache_path, evidence_path = load_cached_episode(cache_root, ep)
    npz_path = factorial_root / f"episode_{ep}_shadow_multisample_frames_and_events.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        hdr = np.asarray(z["hdr_rgb"], dtype=np.float32)
        depth = np.asarray(z["depth"], dtype=np.float32)
        seg = np.asarray(z["segmentation"], dtype=np.int32)
        factorial_times = np.asarray(z["times_s"], dtype=np.float64)
        conditions = [str(x) for x in z["condition_keys"].tolist()]
        direct_events_by_condition = {
            key: np.asarray(z[f"{key}_events_full_frame"], dtype=np.float64)
            for key in conditions
        }
    source_t = source["timestamps_s"]
    left = int(np.flatnonzero(np.isclose(source_t, factorial_times[0], atol=1e-9, rtol=0))[0])
    right = int(np.flatnonzero(np.isclose(source_t, factorial_times[2], atol=1e-9, rtol=0))[0])
    object_ids = source["segmentation_pose_ids"]
    obj0 = {int(sid): source["object_poses"][left, j] for j, sid in enumerate(object_ids)}
    obj1 = {int(sid): source["object_poses"][right, j] for j, sid in enumerate(object_ids)}
    focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    K = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    per_condition = {}
    predictions = {}
    event_arrays = {}
    for ci, key in enumerate(conditions):
        luma = np.stack([linear_luminance(frame) for frame in hdr[ci]])
        interp = PhysicsInterpolator(
            K, sampler_config=AdaptiveSamplingConfig(min_samples=2, max_samples=2),
            disocclusion_policy="hold", composite="same_surface_log_linear",
            warp_backend="numpy", strict_fidelity=False,
            max_photometric_interval_s=0.005,
        )
        start = time.perf_counter()
        generated, generated_times = interp.interpolate(
            luma[0], depth[ci, 0], seg[ci, 0],
            luma[2], depth[ci, 2], seg[ci, 2],
            source["camera_to_world_cv"][left], source["camera_to_world_cv"][right], None,
            float(factorial_times[0]), float(factorial_times[2]),
            input_space="linear", object_world0=obj0, object_world1=obj1,
        )
        warp_wall = time.perf_counter() - start
        if len(generated) != 2 or not np.allclose(generated_times, factorial_times[1:], atol=1e-8, rtol=0):
            raise RuntimeError(f"{ep}/{key}: candidate interval timestamps do not align with observed midpoint/end")
        predicted_mid = np.asarray(generated[0], dtype=np.float32)
        observed_mid = luma[1]
        support = np.asarray(interp.last_visibility["intermediate_visibility"][0], dtype=bool)
        boundary = dilated_boundary(seg[ci, 1])
        support_interior = support & ~boundary
        errors = log_error(predicted_mid, observed_mid)
        finite_errors = errors[support]
        interior_errors = errors[support_interior]
        direct_sparse_events = ideal_dvs(luma[1:], factorial_times[1:], luma[0], float(factorial_times[0]))
        candidate_sparse_events = ideal_dvs(np.asarray(generated), generated_times, luma[0], float(factorial_times[0]))
        strict_interp = PhysicsInterpolator(
            K, sampler_config=AdaptiveSamplingConfig(min_samples=2, max_samples=2),
            disocclusion_policy="nan", composite="same_surface_log_linear",
            warp_backend="numpy", strict_fidelity=True,
            max_photometric_interval_s=0.005,
        )
        strict_frames, strict_times = strict_interp.interpolate(
            luma[0], depth[ci, 0], seg[ci, 0],
            luma[2], depth[ci, 2], seg[ci, 2],
            source["camera_to_world_cv"][left], source["camera_to_world_cv"][right], None,
            float(factorial_times[0]), float(factorial_times[2]),
            input_space="linear", object_world0=obj0, object_world1=obj1,
            intermediate_states=[{
                "t": float(factorial_times[1]),
                "radiance": luma[1],
                "provenance": f"g1_cached_direct_hdr_factorial_{ep}_{key}",
                "available_at_s": float(factorial_times[1]),
            }],
        )
        strict_sparse_events = ideal_dvs(
            strict_frames, strict_times, luma[0], float(factorial_times[0]),
        )
        event_arrays[f"{key}_candidate_sparse_events"] = candidate_sparse_events
        event_arrays[f"{key}_direct_sparse_events"] = direct_sparse_events
        event_arrays[f"{key}_strict_direct_sample_events"] = strict_sparse_events
        predictions[f"{key}_mid_luma"] = predicted_mid
        predictions[f"{key}_mid_support"] = support.astype(np.uint8)
        metrics = {
            "episode": ep,
            "condition": key,
            "shadow": "on" in key.split("_multisample")[0],
            "multisample": key.endswith("_on"),
            "actual_captured_direct_sparse_event_count_from_existing_npz": int(len(direct_events_by_condition[key])),
            "direct_dvs_event_count_replayed_from_three_cached_frames": int(len(direct_sparse_events)),
            "candidate_dvs_event_count_from_warped_midpoint": int(len(candidate_sparse_events)),
            "sparse_event_comparison": event_metrics(direct_sparse_events, candidate_sparse_events, width),
            "strict_cached_sample_event_comparison": event_metrics(direct_sparse_events, strict_sparse_events, width),
            "strict_cached_sample_raw_event_exact_match": bool(
                direct_sparse_events.shape == strict_sparse_events.shape
                and np.array_equal(direct_sparse_events, strict_sparse_events)
            ),
            "strict_cached_sample_extra_render_count": int(strict_interp.last_uncertainty["extra_render_count"]),
            "strict_cached_sample_max_anchor_gap_s": float(strict_interp.last_uncertainty["photometric_max_anchor_gap_s"]),
            "strict_cached_sample_output_ready_wall_s": float(strict_interp.last_uncertainty["output_ready_after_right_endpoint_wall_s"]),
            "midpoint_supported_fraction": float(support.mean()),
            "midpoint_unknown_fraction": float((~support).mean()),
            "supported_log_error_p50": float(np.quantile(finite_errors, 0.50)) if finite_errors.size else None,
            "supported_log_error_p95": float(np.quantile(finite_errors, 0.95)) if finite_errors.size else None,
            "supported_log_error_max": float(finite_errors.max()) if finite_errors.size else None,
            "supported_pixels_over_0p2_log_error": int((finite_errors > 0.2).sum()),
            "supported_interior_log_error_p95": float(np.quantile(interior_errors, 0.95)) if interior_errors.size else None,
            "supported_interior_pixels": int(support_interior.sum()),
            "direct_mid_log_luma_range_p95_over_supported": float(np.quantile(
                np.abs(np.log(np.maximum(luma[1], LOG_EPS)) - np.log(np.maximum(luma[0], LOG_EPS)))[support], 0.95
            )) if support.any() else None,
            "unknown_fraction_of_pixels_near_silhouette": float((~support & boundary).sum() / max(int(boundary.sum()), 1)),
            "candidate_warp_cpu_wall_s": float(warp_wall),
            "endpoint_max_abs_delta_from_direct": float(np.max(np.abs(np.asarray(generated[1]) - luma[2]))),
        }
        per_condition[key] = metrics
    return {
        "episode": ep,
        "factorial_npz_sha256": sha256(npz_path),
        "state_cache_sha256": sha256(cache_path),
        "direct_hdr_evidence_sha256": sha256(evidence_path),
        "times_s": [float(t) for t in factorial_times],
        "conditions": per_condition,
        "prediction_arrays": predictions,
        "sparse_event_arrays": event_arrays,
    }


def simple_contour_candidate(cache_root: Path, factorial_root: Path):
    """Replay the existing render-only moving-box/shadow control."""
    npz_path = factorial_root / "simple_contour_control_frames_and_events.npz"
    experiment_path = ROOT / "benchmark/robot_closed_loop/configs/r1_5_shadow_multisample_cached_states_20261025b.json"
    with np.load(npz_path, allow_pickle=False) as z:
        hdr = np.asarray(z["hdr_rgb"], dtype=np.float32)
        depth = np.asarray(z["depth"], dtype=np.float32)
        seg = np.asarray(z["segmentation"], dtype=np.int32)
        times = np.asarray(z["times_s"], dtype=np.float64)
        conditions = [str(x) for x in z["condition_keys"].tolist()]
        stored_events = {key: np.asarray(z[f"{key}_events_full_frame"], dtype=np.float64) for key in conditions}
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    state_positions = [item["box_position_m"] for item in experiment["simple_contour_control"]["states"]]
    cache_path = cache_root / "episode_004/same_state_source_cache.h5"
    with h5py.File(cache_path, "r") as handle:
        width, height = map(int, handle.attrs["camera_resolution_wh"])
        fov = float(handle.attrs["vertical_fov_deg"])
        camera = np.asarray(handle["camera_to_world_cv"][3], dtype=np.float64)
    focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    K = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    pose0, pose1 = np.eye(4), np.eye(4)
    pose0[:3, 3] = np.asarray(state_positions[0], dtype=np.float64)
    pose1[:3, 3] = np.asarray(state_positions[2], dtype=np.float64)
    per_condition, predictions, event_arrays = {}, {}, {}
    for ci, key in enumerate(conditions):
        luma = np.stack([linear_luminance(frame) for frame in hdr[ci]])
        approx = PhysicsInterpolator(
            K, sampler_config=AdaptiveSamplingConfig(min_samples=2, max_samples=2),
            disocclusion_policy="hold", composite="same_surface_log_linear",
            warp_backend="numpy", strict_fidelity=False,
            max_photometric_interval_s=0.005,
        )
        started = time.perf_counter()
        predicted, predicted_times = approx.interpolate(
            luma[0], depth[ci, 0], seg[ci, 0], luma[2], depth[ci, 2], seg[ci, 2],
            camera, camera, None, float(times[0]), float(times[2]), input_space="linear",
            object_world0={2: pose0}, object_world1={2: pose1},
        )
        warp_wall = time.perf_counter() - started
        predicted_mid = np.asarray(predicted[0], dtype=np.float32)
        support = np.asarray(approx.last_visibility["intermediate_visibility"][0], dtype=bool)
        reference_mid = luma[1]
        error = log_error(predicted_mid, reference_mid)
        floor = seg[ci, 1] == 1
        box = seg[ci, 1] == 2
        direct_replay = ideal_dvs(luma[1:], times[1:], luma[0], float(times[0]))
        candidate_events = ideal_dvs(np.asarray(predicted), predicted_times, luma[0], float(times[0]))
        strict = PhysicsInterpolator(
            K, sampler_config=AdaptiveSamplingConfig(min_samples=2, max_samples=2),
            disocclusion_policy="nan", composite="same_surface_log_linear",
            warp_backend="numpy", strict_fidelity=True,
            max_photometric_interval_s=0.005,
        )
        exact_frames, exact_times = strict.interpolate(
            luma[0], depth[ci, 0], seg[ci, 0], luma[2], depth[ci, 2], seg[ci, 2],
            camera, camera, None, float(times[0]), float(times[2]), input_space="linear",
            object_world0={2: pose0}, object_world1={2: pose1},
            intermediate_states=[{
                "t": float(times[1]), "radiance": luma[1],
                "provenance": f"g1_cached_render_only_contour_{key}",
                "available_at_s": float(times[1]),
            }],
        )
        strict_events = ideal_dvs(exact_frames, exact_times, luma[0], float(times[0]))
        event_arrays[f"{key}_direct_events"] = direct_replay
        event_arrays[f"{key}_candidate_events"] = candidate_events
        event_arrays[f"{key}_strict_cached_events"] = strict_events
        predictions[f"{key}_mid_luma"] = predicted_mid
        predictions[f"{key}_mid_support"] = support.astype(np.uint8)
        supported_error = error[support]
        per_condition[key] = {
            "shadow": "on" in key.split("_multisample")[0],
            "multisample": key.endswith("_on"),
            "supported_fraction": float(support.mean()),
            "supported_log_error_p95": float(np.quantile(supported_error, 0.95)) if supported_error.size else None,
            "supported_log_error_max": float(supported_error.max()) if supported_error.size else None,
            "supported_errors_over_0p2": int((supported_error > 0.2).sum()),
            "floor_supported_fraction": float((support & floor).sum() / max(int(floor.sum()), 1)),
            "floor_log_error_p95": float(np.quantile(error[support & floor], 0.95)) if np.any(support & floor) else None,
            "moving_box_supported_fraction": float((support & box).sum() / max(int(box.sum()), 1)),
            "direct_sparse_event_count": int(len(direct_replay)),
            "stored_factorial_event_count": int(len(stored_events[key])),
            "candidate_sparse_event_count": int(len(candidate_events)),
            "candidate_sparse_event_metrics": event_metrics(direct_replay, candidate_events, width),
            "strict_cached_event_exact_match": bool(
                direct_replay.shape == strict_events.shape and np.array_equal(direct_replay, strict_events)
            ),
            "strict_cached_extra_render_count": int(strict.last_uncertainty["extra_render_count"]),
            "approximate_warp_cpu_wall_s": float(warp_wall),
            "interpretation": "render-only translated box; moving shadow/occlusion control, not a physics trajectory",
        }
    return {
        "source_npz_sha256": sha256(npz_path),
        "source_cache_sha256_for_camera_and_intrinsics": sha256(cache_path),
        "experiment_config_sha256": sha256(experiment_path),
        "times_s": [float(t) for t in times],
        "state_positions_m": state_positions,
        "conditions": per_condition,
        "prediction_arrays": predictions,
        "event_arrays": event_arrays,
    }


def summarize_factorial_source(summary_path: Path, factorial_root: Path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_ep = {}
    for ep, info in summary["episodes"].items():
        conditions = {}
        for key, cell in info["conditions"].items():
            conditions[key] = {
                "shadow": bool(cell["shadow"]),
                "requested_msaa": bool(cell["requested_multisample_enabled"]),
                "actual_gl_msaa_per_pass": cell["actual_gl_multisample_state_per_hdr_pass"],
                "sample_buffers_per_pass": cell["sample_buffers_per_hdr_pass"],
                "sample_count_per_pass": cell["samples_per_hdr_pass"],
                "sample_positions_per_pass": cell["sample_positions_per_hdr_pass"],
                "radiance_readback_passes": len(cell["radiance_readback_audit"]),
                "framebuffer_blits": len(cell["framebuffer_blit_audit"]),
                "render_calls": int(cell["render_calls_included_in_wall"]),
                "captured_render_wall_s": float(cell["captured_render_wall_s"]),
                "scene_build_wall_s": float(cell["scene_build_wall_s"]),
                "condition_wall_s": float(cell["condition_wall_s_including_scene_build_warmup_and_destroy"]),
                "full_frame_events": int(cell["full_frame_events"]),
                "fixed_floor_events": int(cell["fixed_floor_events"]),
                "silhouette_contour_events": cell["silhouette_contour_events"],
            }
        per_ep[ep] = conditions
    simple = summary.get("simple_contour_control", {})
    simple_conditions = {}
    for key, cell in simple.get("conditions", {}).items():
        simple_conditions[key] = {
            "shadow": bool(cell.get("shadow", False)),
            "requested_msaa": bool(cell.get("requested_multisample_enabled", False)),
            "actual_gl_msaa_per_pass": cell.get("actual_gl_multisample_state_per_hdr_pass"),
            "sample_count_per_pass": cell.get("samples_per_hdr_pass"),
            "sample_positions_per_pass": cell.get("sample_positions_per_hdr_pass"),
            "full_frame_events": int(cell.get("full_frame_events", 0)),
            "captured_render_wall_s": float(cell.get("captured_render_wall_s", 0.0)),
            "render_calls": int(cell.get("render_calls_included_in_wall", 0)),
        }
    return {
        "existing_execution_commit": summary["execution_commit"],
        "existing_summary_sha256": sha256(summary_path),
        "existing_factorial_npz_sha256": {
            ep: sha256(factorial_root / f"episode_{ep}_shadow_multisample_frames_and_events.npz")
            for ep in summary["episodes"]
        },
        "existing_renderer_provenance": summary["renderer_provenance"],
        "existing_factorial_measurements": per_ep,
        "existing_simple_moving_contour_control": {
            "purpose": simple.get("purpose"),
            "physics_advanced": simple.get("physics_advanced"),
            "state_times_s": simple.get("state_times_s"),
            "state_positions_m": simple.get("state_positions_m"),
            "conditions": simple_conditions,
        },
        "existing_experiment_wall_s": float(summary["experiment_wall_s_through_render_analysis_and_manifest_preparation"]),
        "limitations": summary["limitations"],
    }


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--factorial-root", type=Path, default=DEFAULT_FACTORIAL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory: {output}")
    output.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    started = time.perf_counter()
    controls = run_unit_controls()
    ramp = ramp_control()
    strict_replays = [strict_cached_replay(args.cache_root.resolve(), ep) for ep in ("004", "005")]
    factorial = [factorial_candidate(args.cache_root.resolve(), args.factorial_root.resolve(), ep) for ep in ("004", "005")]
    simple_control = simple_contour_candidate(args.cache_root.resolve(), args.factorial_root.resolve())
    factorial_source = summarize_factorial_source(
        args.factorial_root.resolve() / "shadow_multisample_summary.json",
        args.factorial_root.resolve(),
    )
    candidate_core_files = [
        ROOT / "genesis_event_plugin/physics_warp.py",
        ROOT / "genesis_event_plugin/torch_physics.py",
        ROOT / "genesis_event_plugin/physics_interpolator.py",
        ROOT / "genesis_event_plugin/physics_plugin.py",
    ]
    summary = {
        "purpose": "acceptance candidate validation using synthetic controls, archived exact states, and existing shadow x MSAA direct renders",
        "scope": "CPU only; no renderer, CUDA, physics advancement, training, ranking, or frozen-core modification",
        "candidate_branch": os.environ.get("R1_5_CANDIDATE_BRANCH") or subprocess.check_output(
            ["git", "-C", str(ROOT), "branch", "--show-current"], text=True
        ).strip(),
        "candidate_base_commit": os.environ.get("R1_5_CANDIDATE_COMMIT") or subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_module_sha256": {p.name: sha256(p) for p in candidate_core_files},
        "script_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                    "torch_device": "CPU", "platform": platform.platform()},
        "fixed_acceptance_criteria": config["validation_acceptance"],
        "approximation_contract": config["photometric_sampling"],
        "unit_controls": controls,
        "known_log_linear_ramp": ramp,
        "strict_cached_direct_hdr_replays": [
            {k: v for k, v in result.items() if k != "event_arrays"}
            for result in strict_replays
        ],
        "paired_shadow_msaa_candidate_replays": [
            {k: v for k, v in result.items() if k not in {"prediction_arrays", "sparse_event_arrays"}}
            for result in factorial
        ],
        "simple_moving_contour_candidate_replay": {
            k: v for k, v in simple_control.items()
            if k not in {"prediction_arrays", "event_arrays"}
        },
        "paired_shadow_msaa_existing_renderer_audit": factorial_source,
        "candidate_wall_s_cpu_total": float(time.perf_counter() - started),
        "g1_cuda_status": os.environ.get(
            "R1_5_G1_CUDA_STATUS",
            "not rerun in this CPU-only validation; see the separately audited CUDA parity run F",
        ),
        "g3_cuda_status": os.environ.get(
            "R1_5_G3_CUDA_STATUS",
            "not exercised by this CPU-only candidate validation",
        ),
        "cost_scope": "cached-frame API and warp CPU time measured here; direct Genesis render cost is inherited from existing 2026-10-29 g1 factorial and is not a fresh candidate timing",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    flat_factorial = [cell for episode in summary["paired_shadow_msaa_candidate_replays"] for cell in episode["conditions"].values()]
    write_csv(output / "shadow_msaa_candidate_midpoint_metrics.csv", flat_factorial)
    (output / "ramp_events.json").write_text(json.dumps(ramp, indent=2) + "\n", encoding="utf-8", newline="\n")
    np.savez_compressed(
        output / "shadow_msaa_candidate_midpoints.npz",
        **{f"ep{episode['episode']}_{key}": value
           for result in factorial for episode in [result]
           for key, value in result["prediction_arrays"].items()},
    )
    np.savez_compressed(
        output / "shadow_msaa_sparse_events.npz",
        **{f"ep{episode['episode']}_{key}": value
           for result in factorial for episode in [result]
           for key, value in result["sparse_event_arrays"].items()},
    )
    np.savez_compressed(
        output / "direct_hdr_sampling_events.npz",
        **{f"ep{item['episode']}_{name}": events
           for item in strict_replays for name, events in item["event_arrays"].items()},
    )
    np.savez_compressed(
        output / "simple_contour_candidate_midpoints.npz",
        **{key: value for key, value in simple_control["prediction_arrays"].items()},
    )
    np.savez_compressed(
        output / "simple_contour_events.npz",
        **{key: value for key, value in simple_control["event_arrays"].items()},
    )
    (output / "config.json").write_text(args.config.resolve().read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    files = [p for p in sorted(output.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (output / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(output).as_posix()}\n" for p in files), encoding="utf-8", newline="\n")
    print(json.dumps({
        "output": str(output),
        "candidate_branch": summary["candidate_branch"],
        "candidate_base_commit": summary["candidate_base_commit"],
        "unit_controls": controls,
        "ramp": ramp,
        "strict_replays": [
            {k: v for k, v in item.items() if k != "event_arrays"}
            for item in strict_replays
        ],
        "factorial_candidate": summary["paired_shadow_msaa_candidate_replays"],
        "manifest_sha256": sha256(output / "SHA256SUMS"),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
