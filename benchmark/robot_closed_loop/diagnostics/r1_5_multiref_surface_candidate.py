"""Selective multi-reference surface candidate for Robot R1.5 diagnostics.

Static scene samples are reused at their fixed camera pixels. Only moving
segmentation labels are reprojected and triangle-rasterized into the target
pose. Several already-rendered keyframes can contribute complementary visible
surfaces; a per-pixel nearest-depth decision composes them with the static
background. No spatial inpainting is performed: pixels with no observed
surface, or ambiguous equal-depth labels, remain render requests.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from genesis_event_plugin.physics_warp import interpolate_se3


def interpolate_body_poses(poses0: dict[int, np.ndarray], poses1: dict[int, np.ndarray],
                           alpha: float, moving_ids: Iterable[int]) -> dict[int, np.ndarray]:
    """Interpolate the benchmark's rigid-link transforms for one target time."""
    out = {}
    for object_id in sorted({int(x) for x in moving_ids}):
        if object_id not in poses0 or object_id not in poses1:
            continue
        out[object_id] = interpolate_se3(poses0[object_id], poses1[object_id], alpha)
    return out


def project_moving_source_to_target(
    depth: np.ndarray,
    labels: np.ndarray,
    K: np.ndarray,
    camera_source: np.ndarray,
    camera_target: np.ndarray,
    source_poses: dict[int, np.ndarray],
    target_poses: dict[int, np.ndarray],
    moving_ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project only explicitly moving source samples into a target camera pose.

    The fixed camera used by the robot benchmark is a common source/target
    transform, but both transforms are accepted here. Static background pixels
    are deliberately excluded because their image coordinates do not change.
    """
    depth = np.asarray(depth, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    K = np.asarray(K, dtype=np.float64)
    if depth.ndim != 2 or labels.shape != depth.shape or K.shape != (3, 3):
        raise ValueError("depth/labels must be matching HxW arrays and K must be 3x3")
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float64),
                         np.arange(width, dtype=np.float64), indexing="ij")
    valid_depth = np.isfinite(depth) & (depth > 1e-6)
    requested = np.isin(labels, np.asarray(sorted({int(x) for x in moving_ids}), dtype=np.int64))
    valid = valid_depth & requested
    uv = np.zeros((2, height, width), dtype=np.float32)
    z = np.full((height, width), np.inf, dtype=np.float32)
    if not valid.any():
        return uv, z, valid

    source_indices = np.flatnonzero(valid.reshape(-1))
    source_z = depth.reshape(-1)[source_indices].astype(np.float64)
    source_x = xx.reshape(-1)[source_indices]
    source_y = yy.reshape(-1)[source_indices]
    camera_points = np.stack(((source_x - K[0, 2]) * source_z / K[0, 0],
                              (source_y - K[1, 2]) * source_z / K[1, 1],
                              source_z, np.ones_like(source_z)), axis=0)
    world_points = np.asarray(camera_source, dtype=np.float64) @ camera_points
    camera_inverse = np.linalg.inv(np.asarray(camera_target, dtype=np.float64))
    flat_labels = labels.reshape(-1)
    flat_uv = uv.reshape(2, -1)
    flat_z = z.reshape(-1)
    for object_id in sorted({int(x) for x in moving_ids}):
        if object_id not in source_poses or object_id not in target_poses:
            continue
        local = flat_labels[source_indices] == object_id
        if not local.any():
            continue
        delta = np.asarray(target_poses[object_id], dtype=np.float64) @ np.linalg.inv(
            np.asarray(source_poses[object_id], dtype=np.float64))
        target_world = delta @ world_points[:, local]
        target_camera = camera_inverse @ target_world
        target_z = target_camera[2]
        good = np.isfinite(target_camera).all(axis=0) & (target_z > 1e-6)
        indices = source_indices[local][good]
        target_z = target_z[good]
        flat_uv[0, indices] = K[0, 0] * target_camera[0, good] / target_z + K[0, 2]
        flat_uv[1, indices] = K[1, 1] * target_camera[1, good] / target_z + K[1, 2]
        flat_z[indices] = target_z.astype(np.float32)
    valid &= np.isfinite(uv).all(axis=0) & np.isfinite(z) & (z > 1e-6)
    uv[:, ~valid] = 0.0
    z[~valid] = np.inf
    return uv, z, valid


def _frontmost_composite(
    radiance: np.ndarray,
    depth: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    times: np.ndarray,
    query_time: float,
    *,
    floor: float,
    depth_rtol: float,
    depth_atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Z-select surface evidence and interpolate only observations of that surface."""
    radiance = np.asarray(radiance, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(radiance) & np.isfinite(depth) & (labels >= 0)
    if radiance.ndim != 3 or depth.shape != radiance.shape or labels.shape != radiance.shape or valid.shape != radiance.shape:
        raise ValueError("multi-reference arrays must have shape (sources, H, W)")
    times = np.asarray(times, dtype=np.float64)
    if times.shape != (radiance.shape[0],) or not np.isfinite(times).all():
        raise ValueError("times must provide one finite timestamp per source")

    masked_depth = np.where(valid, depth, np.inf)
    min_depth = np.min(masked_depth, axis=0)
    has_surface = np.isfinite(min_depth)
    nearest_idx = np.argmin(masked_depth, axis=0)
    front_label = np.take_along_axis(labels, nearest_idx[None, ...], axis=0)[0]
    depth_tol = np.maximum(float(depth_atol), float(depth_rtol) * np.maximum(min_depth, 0.0))
    contributors = valid & (labels == front_label[None, ...])
    depth_delta = np.zeros_like(depth, dtype=np.float32)
    np.subtract(depth, min_depth[None, ...], out=depth_delta,
                where=np.isfinite(depth) & np.isfinite(min_depth[None, ...]))
    depth_delta = np.abs(depth_delta)
    contributors &= depth_delta <= depth_tol[None, ...]
    # Distinct surfaces within the numerical coincidence band are not ordered
    # confidently. They are left unknown instead of hiding one arbitrarily.
    conflicting = np.zeros_like(has_surface)
    # Detect another segmentation surface at the same depth within the local
    # numerical band; such a contact/interpenetration is not tie-broken.
    near = valid & (depth_delta <= depth_tol[None, ...])
    for i in range(radiance.shape[0]):
        conflicting |= near[i] & (labels[i] != front_label)

    before_scores = np.where(contributors & (times[:, None, None] <= query_time),
                             times[:, None, None], -np.inf)
    after_scores = np.where(contributors & (times[:, None, None] >= query_time),
                            times[:, None, None], np.inf)
    before_idx = np.argmax(before_scores, axis=0)
    after_idx = np.argmin(after_scores, axis=0)
    has_before = np.isfinite(np.max(before_scores, axis=0))
    has_after = np.isfinite(np.min(after_scores, axis=0))
    two_sided = has_before & has_after
    one_sided = has_before ^ has_after
    before_value = np.take_along_axis(radiance, before_idx[None, ...], axis=0)[0]
    after_value = np.take_along_axis(radiance, after_idx[None, ...], axis=0)[0]
    before_time = times[before_idx]
    after_time = times[after_idx]
    span = np.maximum(after_time - before_time, 1e-12)
    frac = np.clip((float(query_time) - before_time) / span, 0.0, 1.0)
    before_log = np.log(np.maximum(before_value, float(floor)))
    after_log = np.log(np.maximum(after_value, float(floor)))
    out = np.full(min_depth.shape, np.nan, dtype=np.float32)
    out[two_sided] = np.exp(((1.0 - frac) * before_log + frac * after_log)[two_sided])
    only_before = has_before & ~has_after
    only_after = has_after & ~has_before
    out[only_before] = before_value[only_before]
    out[only_after] = after_value[only_after]
    out_labels = np.where(has_surface, front_label, -1).astype(np.int64)
    support = has_surface & (two_sided | one_sided) & ~conflicting
    out[~support] = np.nan
    out_labels[~support] = -1
    out_depth = np.where(support, min_depth, np.inf).astype(np.float32)
    stats = {
        "supported_pixels": int(support.sum()),
        "two_sided_photometric_pixels": int((support & two_sided).sum()),
        "one_sided_photometric_pixels": int((support & one_sided).sum()),
        "unobserved_pixels": int((~has_surface).sum()),
        "ambiguous_pixels": int((has_surface & conflicting).sum()),
    }
    return out, out_depth, out_labels, {"valid": support, **stats}


def compose_multireference_scene(
    moving_references: Sequence[dict],
    static_references: Sequence[dict],
    moving_ids: Iterable[int],
    query_time: float,
    *,
    floor: float = 1e-6,
    depth_rtol: float = 1e-3,
    depth_atol: float = 2e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Merge dynamic surface warps and fixed-coordinate static observations.

    Each reference has ``time_s``, ``radiance``, ``depth``, ``labels`` and
    ``visible``. The visibility decision is a nearest-depth z test across all
    complementary views. No-source and depth-conflict pixels request a real
    render.
    """
    if not moving_references or not static_references:
        raise ValueError("at least one moving and one static reference are required")
    moving_times = np.asarray([x["time_s"] for x in moving_references], dtype=np.float64)
    moving = _frontmost_composite(
        np.stack([x["radiance"] for x in moving_references]),
        np.stack([x["depth"] for x in moving_references]),
        np.stack([x["labels"] for x in moving_references]),
        np.stack([x["visible"] for x in moving_references]), moving_times,
        query_time, floor=floor, depth_rtol=depth_rtol, depth_atol=depth_atol,
    )
    moving_luma, moving_depth, moving_label, moving_stats = moving
    motion_ids = np.asarray(sorted({int(x) for x in moving_ids}), dtype=np.int64)
    static_radiance, static_depths, static_labels = [], [], []
    for frame in static_references:
        labels = np.asarray(frame["labels"], dtype=np.int64)
        dynamic_mask = np.isin(labels, motion_ids)
        static_radiance.append(np.asarray(frame["radiance"], dtype=np.float32))
        static_depths.append(np.where(dynamic_mask, np.inf, frame["depth"]).astype(np.float32))
        static_labels.append(np.where(dynamic_mask, -1, labels).astype(np.int64))
    static_valid = np.stack([np.isfinite(z) & (z > 1e-6) & (l >= 0)
                             for z, l in zip(static_depths, static_labels)])
    static = _frontmost_composite(
        np.stack(static_radiance), np.stack(static_depths), np.stack(static_labels),
        static_valid, np.asarray([x["time_s"] for x in static_references], dtype=np.float64),
        query_time, floor=floor, depth_rtol=depth_rtol, depth_atol=depth_atol,
    )
    static_luma, static_depth, static_label, static_stats = static
    moving_evidence = {key: value for key, value in moving_stats.items() if key != "valid"}
    static_evidence = {key: value for key, value in static_stats.items() if key != "valid"}
    shape = static_luma.shape
    moving_valid = np.isfinite(moving_luma) & np.isfinite(moving_depth) & (moving_label >= 0)
    static_valid_out = np.isfinite(static_luma) & np.isfinite(static_depth) & (static_label >= 0)
    out = np.full(shape, np.nan, dtype=np.float32)
    out_depth = np.full(shape, np.inf, dtype=np.float32)
    out_label = np.full(shape, -1, dtype=np.int64)
    has_move = moving_valid
    has_static = static_valid_out
    depth_gap = np.zeros_like(moving_depth, dtype=np.float32)
    overlap = has_move & has_static
    np.subtract(moving_depth, static_depth, out=depth_gap, where=overlap)
    tie = overlap & (np.abs(depth_gap) <= np.maximum(
        float(depth_atol), float(depth_rtol) * np.minimum(moving_depth, static_depth)))
    moving_front = has_move & (~has_static | (moving_depth < static_depth)) & ~tie
    static_front = has_static & (~has_move | (static_depth < moving_depth)) & ~tie
    out[moving_front] = moving_luma[moving_front]
    out_depth[moving_front] = moving_depth[moving_front]
    out_label[moving_front] = moving_label[moving_front]
    out[static_front] = static_luma[static_front]
    out_depth[static_front] = static_depth[static_front]
    out_label[static_front] = static_label[static_front]
    supported = moving_front | static_front
    request = ~supported
    diagnostics = {
        "moving_supported_fraction": float(moving_valid.mean()),
        "static_supported_fraction": float(static_valid_out.mean()),
        "composited_supported_fraction": float(supported.mean()),
        "render_request_fraction": float(request.mean()),
        "dynamic_front_fraction": float(moving_front.mean()),
        "static_front_fraction": float(static_front.mean()),
        "depth_tie_request_fraction": float(tie.mean()),
        "moving_surface_evidence": moving_evidence,
        "static_surface_evidence": static_evidence,
        "target_labels": out_label,
        "target_depth": out_depth,
        "supported": supported,
        "request_render": request,
        "moving_ids": motion_ids.tolist(),
        "visibility": "per-pixel nearest interpolated depth across complementary source surfaces",
    }
    return out, supported, request, diagnostics
