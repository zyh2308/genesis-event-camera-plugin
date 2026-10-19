"""Diagnostic-only image-consistency candidates for Robot R1.5.

This module wraps frozen NumPy geometry primitives without changing simulator
core. It separates exact endpoint handling from an optional conservative
log-radiance interpolation. The latter only interpolates where both warped
sources agree on a visible surface; unresolved pixels request a real render.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis_event_plugin.physics_warp import (
    _splat_projected_diagnostics,
    depth_aware_bidirectional_warp_projected_diagnostics,
    interpolate_se3,
    project_endpoint_to_alpha,
)


def project_endpoint_candidate(
    depth_source: np.ndarray,
    seg_source: np.ndarray,
    K: np.ndarray,
    camera_source: np.ndarray,
    camera_alpha: np.ndarray,
    alpha: float,
    source_index: int,
    object_world0: Optional[Mapping[int, np.ndarray]] = None,
    object_world1: Optional[Mapping[int, np.ndarray]] = None,
    object_world_deltas: Optional[Mapping[int, np.ndarray]] = None,
    *,
    exact_endpoints: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project one endpoint, preserving exact source coordinates at its end.

    Only the mathematically exact endpoint is special-cased: source 0 at
    alpha=0 or source 1 at alpha=1. Intermediate motion remains the original
    continuous SE(3) projection; no coordinate rounding is applied there.
    """
    endpoint_is_identity = (
        (source_index == 0 and float(alpha) == 0.0)
        or (source_index == 1 and float(alpha) == 1.0)
    )
    depth = np.asarray(depth_source, dtype=np.float32)
    if exact_endpoints and endpoint_is_identity:
        height, width = depth.shape
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        valid = np.isfinite(depth) & (depth > 1e-6)
        uv = np.stack((xx, yy), axis=0).astype(np.float32)
        uv[:, ~valid] = 0.0
        z = depth.copy()
        z[~valid] = np.inf
        return uv, z, valid
    return project_endpoint_to_alpha(
        depth_source, seg_source, K, camera_source, camera_alpha, alpha,
        source_index, object_world0, object_world1, object_world_deltas,
    )


def _splat_segment_labels(
    labels: np.ndarray,
    source_depth: np.ndarray,
    target_uv: np.ndarray,
    target_z: np.ndarray,
    source_valid: np.ndarray,
    z_buffer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer integer labels using visible projected samples.

    The nearest-depth visibility epsilon matches the frozen NumPy splatter.
    If equal-weight visible samples disagree on the link label, mark the
    target pixel ambiguous so the photometric candidate requests a render.
    """
    source_depth = np.asarray(source_depth, dtype=np.float32)
    target_uv = np.asarray(target_uv, dtype=np.float32)
    target_z = np.asarray(target_z, dtype=np.float32)
    labels = np.asarray(labels)
    height, width = source_depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    tx, ty = target_uv[0], target_uv[1]
    x0, y0 = np.floor(tx).astype(np.int32), np.floor(ty).astype(np.int32)
    wx, wy = tx - x0, ty - y0
    valid = np.asarray(source_valid, dtype=bool)
    valid &= np.isfinite(source_depth) & (source_depth > 1e-6)
    valid &= np.isfinite(target_z) & (target_z > 1e-6)
    candidates = []
    for ox, oy, weight in (
        (0, 0, (1 - wx) * (1 - wy)),
        (1, 0, wx * (1 - wy)),
        (0, 1, (1 - wx) * wy),
        (1, 1, wx * wy),
    ):
        x, y = x0 + ox, y0 + oy
        inside = valid & (weight > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if np.any(inside):
            index = (y[inside] * width + x[inside]).astype(np.int64)
            candidates.append((index, weight[inside], target_z[inside], labels[inside]))

    zbuf = np.asarray(z_buffer, dtype=np.float32).reshape(-1)
    best_weight = np.full(height * width, -1.0, dtype=np.float32)
    for index, weight, z, _ in candidates:
        front = z <= zbuf[index] + 1e-5
        np.maximum.at(best_weight, index[front], weight[front])
    label_min = np.full(height * width, np.iinfo(np.int64).max, dtype=np.int64)
    label_max = np.full(height * width, -1, dtype=np.int64)
    for index, weight, z, value in candidates:
        front = (z <= zbuf[index] + 1e-5) & (weight >= best_weight[index] - 1e-8)
        if np.any(front):
            ids = value[front].astype(np.int64, copy=False)
            np.minimum.at(label_min, index[front], ids)
            np.maximum.at(label_max, index[front], ids)
    present = label_max >= 0
    conflict = present & (label_min != label_max)
    result = np.full(height * width, -1, dtype=np.int64)
    result[present & ~conflict] = label_min[present & ~conflict]
    result[conflict] = -2
    return result.reshape(height, width), conflict.reshape(height, width)


def compose_log_radiance_candidate(
    warped0: np.ndarray,
    visibility0: np.ndarray,
    z_buffer0: np.ndarray,
    labels0: np.ndarray,
    warped1: np.ndarray,
    visibility1: np.ndarray,
    z_buffer1: np.ndarray,
    labels1: np.ndarray,
    alpha: float,
    *,
    floor: float = 1e-6,
    z_rtol: float = 1e-3,
    z_atol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Interpolate endpoint log radiance only on confident same-surface pixels.

    ``request_render`` is true for disocclusion, label ambiguity/disagreement,
    invalid samples, or depth disagreement. No endpoint is silently used as a
    substitute for unknown temporal radiance.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    a0, a1 = np.asarray(warped0, dtype=np.float32), np.asarray(warped1, dtype=np.float32)
    v0, v1 = np.asarray(visibility0, dtype=bool), np.asarray(visibility1, dtype=bool)
    z0, z1 = np.asarray(z_buffer0, dtype=np.float32), np.asarray(z_buffer1, dtype=np.float32)
    l0, l1 = np.asarray(labels0), np.asarray(labels1)
    if not (a0.shape == a1.shape == v0.shape == v1.shape == z0.shape == z1.shape == l0.shape == l1.shape):
        raise ValueError("warped images, masks, depths, and labels must share target dimensions")
    both = v0 & v1
    label_agreement = (l0 == l1) & (l0 >= 0)
    depth_agreement = np.isclose(z0, z1, rtol=float(z_rtol), atol=float(z_atol))
    finite_positive = np.isfinite(a0) & np.isfinite(a1) & (a0 > 0) & (a1 > 0)
    confident = both & label_agreement & depth_agreement & finite_positive
    out = np.full_like(a0, np.nan, dtype=np.float32)
    log0 = np.log(np.maximum(a0, float(floor)))
    log1 = np.log(np.maximum(a1, float(floor)))
    out[confident] = np.exp((1.0 - float(alpha)) * log0[confident] + float(alpha) * log1[confident])
    request_render = ~confident
    diagnostics = {
        "both_visible_fraction": float(both.mean()),
        "label_agreement_fraction_of_both": float(label_agreement[both].mean()) if np.any(both) else 0.0,
        "depth_agreement_fraction_of_both": float(depth_agreement[both].mean()) if np.any(both) else 0.0,
        "confident_fraction": float(confident.mean()),
        "render_request_fraction": float(request_render.mean()),
        "single_source_or_disocclusion_fraction": float(np.count_nonzero(v0 ^ v1) / v0.size),
        "no_source_fraction": float(np.count_nonzero(~(v0 | v1)) / v0.size),
    }
    return out, confident, request_render, diagnostics


def reconstruct_pair_candidate(
    frame0: np.ndarray,
    depth0: np.ndarray,
    seg0: np.ndarray,
    camera0: np.ndarray,
    frame1: np.ndarray,
    depth1: np.ndarray,
    seg1: np.ndarray,
    camera1: np.ndarray,
    K: np.ndarray,
    object_world0: Mapping[int, np.ndarray],
    object_world1: Mapping[int, np.ndarray],
    alpha: float,
    *,
    mode: str,
    floor: float = 1e-6,
) -> dict:
    """Reconstruct one target time using frozen geometry and a named candidate."""
    if mode not in {
        "frozen", "endpoint_exact", "endpoint_projection_identity",
        "log_radiance_with_render_request",
    }:
        raise ValueError("unknown diagnostic reconstruction mode")
    # Exact endpoints are observed frames. This mode guarantees the complete
    # source image is returned; it does not quantize or alter intermediate
    # motion. ``endpoint_projection_identity`` intentionally bypasses this
    # shortcut at alpha=1 so the lower-level floating-point intervention can
    # be measured on source-1-visible pixels.
    if mode in {"endpoint_exact", "log_radiance_with_render_request"} and float(alpha) in (0.0, 1.0):
        source = np.asarray(frame0 if float(alpha) == 0.0 else frame1, dtype=np.float32).copy()
        valid = np.isfinite(source)
        return {
            "frame": source,
            "valid": valid,
            "request_render": ~valid,
            "diagnostics": {
                "endpoint_identity": "source0" if float(alpha) == 0.0 else "source1",
                "confident_fraction": float(valid.mean()),
            },
            "projection0": None,
            "projection1": None,
        }
    exact = mode != "frozen"
    camera_alpha = interpolate_se3(camera0, camera1, alpha)
    p0 = project_endpoint_candidate(
        depth0, seg0, K, camera0, camera_alpha, alpha, 0,
        object_world0, object_world1, exact_endpoints=exact,
    )
    p1 = project_endpoint_candidate(
        depth1, seg1, K, camera1, camera_alpha, alpha, 1,
        object_world0, object_world1, exact_endpoints=exact,
    )
    if mode != "log_radiance_with_render_request":
        frame, valid, diagnostics = depth_aware_bidirectional_warp_projected_diagnostics(
            frame0, depth0, *p0, frame1, depth1, *p1,
            composite="next_primary",
        )
        return {
            "frame": frame,
            "valid": valid,
            "request_render": ~valid,
            "diagnostics": diagnostics,
            "projection0": p0,
            "projection1": p1,
        }

    warped0, diag0 = _splat_projected_diagnostics(frame0, depth0, p0[0], p0[1], p0[2])
    warped1, diag1 = _splat_projected_diagnostics(frame1, depth1, p1[0], p1[1], p1[2])
    target_labels0, ambiguous0 = _splat_segment_labels(seg0, depth0, p0[0], p0[1], p0[2], diag0["z_buffer"])
    target_labels1, ambiguous1 = _splat_segment_labels(seg1, depth1, p1[0], p1[1], p1[2], diag1["z_buffer"])
    target_labels0[ambiguous0] = -2
    target_labels1[ambiguous1] = -2
    frame, valid, request, photometric = compose_log_radiance_candidate(
        warped0, diag0["visibility"], diag0["z_buffer"], target_labels0,
        warped1, diag1["visibility"], diag1["z_buffer"], target_labels1,
        alpha, floor=floor,
    )
    return {
        "frame": frame,
        "valid": valid,
        "request_render": request,
        "diagnostics": {"source0": diag0, "source1": diag1, **photometric},
        "projection0": p0,
        "projection1": p1,
    }


def _analytic_control() -> dict:
    """Reproduce static and smooth-log-ramp timing with the candidate rule."""
    from genesis_event_plugin.dvs_emulator import DvsEmulator
    from genesis_event_plugin.noise_model import DvsNoiseConfig

    shape = (16, 16)
    floor = 1e-6
    start = np.ones(shape, dtype=np.float32)
    final = np.full(shape, np.exp(0.5), dtype=np.float32)
    depth = np.ones(shape, dtype=np.float32)
    labels = np.ones(shape, dtype=np.int32)
    visible = np.ones(shape, dtype=bool)
    z = np.ones(shape, dtype=np.float32)
    cfg = DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
        cutoff_hz=0.0, leak_rate_hz=0.0, leak_jitter_fraction=0.0,
        shot_noise_rate_hz=0.0, noise_rate_cov_decades=0.0,
        refractory_period_s=0.0, log_input=False, radiance_input=True,
        response_mode="physical_log", radiometric_floor=floor,
        readout_model="ideal_readout", seed=42,
    )

    def run_sequence(frames):
        emulator = DvsEmulator(res=shape, config=cfg, device="cpu")
        emulator.initialize(frames[0], 0.0)
        chunks = []
        for idx, frame in enumerate(frames[1:], 1):
            event = emulator.generate_events(frame, idx * 0.001)
            if len(event):
                chunks.append(np.asarray(event))
        events = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4))
        return {
            "count": int(len(events)),
            "unique_times_s": np.unique(np.round(events[:, 0], 9)).tolist() if len(events) else [],
        }

    static_frames = [start.copy() for _ in range(11)]
    static_events = run_sequence(static_frames)
    candidate_frames = []
    request_masks = []
    for idx in range(1, 11):
        alpha = idx / 10.0
        result = compose_log_radiance_candidate(
            start, visible, z, labels,
            final, visible, z, labels,
            alpha, floor=floor,
        )
        frame, confident, request, _ = result
        if not np.all(confident) or np.any(request):
            raise RuntimeError("analytic ramp should be fully supported by both endpoint samples")
        candidate_frames.append(frame)
        request_masks.append(request)
    ramp_frames = [start, *candidate_frames]
    ramp_events = run_sequence(ramp_frames)
    direct_frames = [start] + [np.full(shape, np.exp(0.5 * idx / 10.0), dtype=np.float32) for idx in range(1, 11)]
    direct_events = run_sequence(direct_frames)
    height, width = shape
    focal = 40.0
    intrinsics = np.asarray([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
    camera = np.eye(4, dtype=np.float64)
    camera[:3, 3] = [0.37, -0.11, 0.62]
    angle = 0.17
    object0 = np.eye(4, dtype=np.float64)
    object1 = np.eye(4, dtype=np.float64)
    object1[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.0],
                       [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    object1[:3, 3] = [0.041, 0.013, 0.0]
    raw_endpoint = project_endpoint_to_alpha(
        depth, labels, intrinsics, camera, camera, 1.0, 1, {1: object0}, {1: object1},
    )
    corrected_endpoint = project_endpoint_candidate(
        depth, labels, intrinsics, camera, camera, 1.0, 1,
        {1: object0}, {1: object1}, exact_endpoints=True,
    )
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixel_grid = np.stack((xx, yy), axis=0).astype(np.float32)
    raw_endpoint_error = float(np.max(np.abs(raw_endpoint[0][:, raw_endpoint[2]] - pixel_grid[:, raw_endpoint[2]])))
    exact_endpoint_uv_error = float(np.max(np.abs(corrected_endpoint[0][:, corrected_endpoint[2]] - pixel_grid[:, corrected_endpoint[2]])))
    raw_mid = project_endpoint_to_alpha(
        depth, labels, intrinsics, camera, camera, 0.5, 0, {1: object0}, {1: object1},
    )
    candidate_mid = project_endpoint_candidate(
        depth, labels, intrinsics, camera, camera, 0.5, 0,
        {1: object0}, {1: object1}, exact_endpoints=True,
    )
    mid_motion_px = float(np.max(np.abs(raw_mid[0][:, raw_mid[2]] - pixel_grid[:, raw_mid[2]])))
    intermediate_projection_delta_px = float(np.max(np.abs(raw_mid[0] - candidate_mid[0])))
    endpoint_frame0 = (0.5 + np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape) / 512.0)
    endpoint_frame1 = endpoint_frame0[::-1].copy() + 0.1
    endpoint_projection = reconstruct_pair_candidate(
        endpoint_frame0, depth, labels, camera, endpoint_frame1, depth, labels, camera,
        intrinsics, {1: object0}, {1: object1}, 1.0, mode="endpoint_projection_identity",
    )
    endpoint_return = reconstruct_pair_candidate(
        endpoint_frame0, depth, labels, camera, endpoint_frame1, depth, labels, camera,
        intrinsics, {1: object0}, {1: object1}, 1.0, mode="endpoint_exact",
    )
    endpoint_projection_abs_error = float(np.max(np.abs(endpoint_projection["frame"] - endpoint_frame1)))
    endpoint_return_abs_error = float(np.max(np.abs(endpoint_return["frame"] - endpoint_frame1)))
    if endpoint_projection_abs_error > 1e-6 or endpoint_return_abs_error != 0.0:
        raise RuntimeError("exact endpoint candidate failed its synthetic identity control")
    return {
        "candidate": "diagnostic log-radiance blend on same visible label and depth; otherwise render request",
        "static_max_abs_error": float(max(np.max(np.abs(frame - start)) for frame in static_frames)),
        "static_event_count": static_events["count"],
        "raw_endpoint_max_uv_error_px": raw_endpoint_error,
        "endpoint_identity_max_uv_error_px": exact_endpoint_uv_error,
        "intermediate_motion_max_uv_displacement_px": mid_motion_px,
        "intermediate_candidate_vs_frozen_projection_max_difference_px": intermediate_projection_delta_px,
        "projected_endpoint_identity_max_abs_image_error": endpoint_projection_abs_error,
        "exact_endpoint_return_max_abs_image_error": endpoint_return_abs_error,
        "ramp_expected_first_log_mean": float(np.log(candidate_frames[0]).mean()),
        "ramp_expected_last_log_mean": float(np.log(candidate_frames[-1]).mean()),
        "candidate_ramp_events": ramp_events,
        "direct_smooth_ramp_events": direct_events,
        "all_ramp_pixels_confident": not any(bool(mask.any()) for mask in request_masks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-control", action="store_true")
    args = parser.parse_args()
    if not args.analytic_control:
        parser.error("the cached-trajectory runner is separate; use --analytic-control for pure controls")
    print(json.dumps(_analytic_control(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
