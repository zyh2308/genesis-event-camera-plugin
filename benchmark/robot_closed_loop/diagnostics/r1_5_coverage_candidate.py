"""Isolated R1.5 front-to-back coverage-splat candidate helpers.

This module is diagnostic-only. It does not modify or monkey-patch the frozen
plugin. The temporal sample schedule is computed with the frozen
``PhysicsInterpolator`` sampler; only its per-sample image reconstruction is
replaced by the candidate coverage resolve.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np

from genesis_event_plugin.physics_warp import interpolate_se3, project_endpoint_to_alpha


DEPTH_TIE_M = 1e-5


def collect_contributions(
    image: np.ndarray,
    source_depth: np.ndarray,
    target_uv: np.ndarray,
    target_z: np.ndarray,
    source_valid: Optional[np.ndarray] = None,
):
    """Flatten the four bilinear footprint taps for every valid source sample."""
    image = np.asarray(image, dtype=np.float32)
    depth = np.asarray(source_depth, dtype=np.float32)
    uv = np.asarray(target_uv, dtype=np.float32)
    z = np.asarray(target_z, dtype=np.float32)
    height, width = depth.shape
    src = image[..., None] if image.ndim == 2 else image
    valid = np.isfinite(src).all(axis=-1) & np.isfinite(depth) & (depth > 1e-6)
    valid &= np.isfinite(z) & (z > 1e-6) & np.isfinite(uv).all(axis=0)
    if source_valid is not None:
        valid &= np.asarray(source_valid, dtype=bool)

    tx, ty = uv[0], uv[1]
    x0, y0 = np.floor(tx).astype(np.int32), np.floor(ty).astype(np.int32)
    wx, wy = tx - x0, ty - y0
    source_index = np.arange(height * width, dtype=np.int32).reshape(height, width)
    pieces = []
    for ox, oy, weight in (
        (0, 0, (1 - wx) * (1 - wy)),
        (1, 0, wx * (1 - wy)),
        (0, 1, (1 - wx) * wy),
        (1, 1, wx * wy),
    ):
        x, y = x0 + ox, y0 + oy
        keep = valid & (weight > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if np.any(keep):
            pieces.append((
                (y[keep] * width + x[keep]).astype(np.int32),
                source_index[keep], weight[keep].astype(np.float32),
                z[keep].astype(np.float32), src[keep].astype(np.float32),
            ))
    if not pieces:
        channels = src.shape[-1]
        return (
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32),
            np.empty((0, channels), dtype=np.float32),
        )
    return tuple(np.concatenate([piece[index] for piece in pieces], axis=0) for index in range(5))


def coverage_splat(
    image: np.ndarray,
    source_depth: np.ndarray,
    target_uv: np.ndarray,
    target_z: np.ndarray,
    source_valid: Optional[np.ndarray] = None,
    depth_tie_m: float = DEPTH_TIE_M,
):
    """Resolve bilinear footprints front-to-back while retaining area support.

    The implementation is vectorized over target pixels and uses the same
    greedy depth tie-band rule as the slower diagnostic attribution helper.
    Incomplete support is normalized over observed contributors, preserving
    that helper's no-background assumption.
    """
    image = np.asarray(image, dtype=np.float32)
    depth = np.asarray(source_depth, dtype=np.float32)
    scalar = image.ndim == 2
    src = image[..., None] if scalar else image
    height, width = depth.shape
    pixel_count = height * width
    pixel, source, weight, z, value = collect_contributions(
        image, depth, target_uv, target_z, source_valid,
    )
    output = np.zeros((pixel_count, src.shape[-1]), dtype=np.float64)
    coverage = np.zeros(pixel_count, dtype=np.float64)
    if pixel.size:
        unresolved = np.ones(pixel.size, dtype=bool)
        remaining = np.ones(pixel_count, dtype=np.float64)
        weight64 = weight.astype(np.float64)
        z32 = z.astype(np.float32, copy=False)
        value64 = value.astype(np.float64)
        # Every iteration resolves at least one layer for an occupied target
        # pixel. The loop stops early when all observed support is resolved.
        while np.any(unresolved):
            active_mask = unresolved & (remaining[pixel] > 1e-8)
            active = np.flatnonzero(active_mask)
            min_depth = np.full(pixel_count, np.inf, dtype=np.float32)
            np.minimum.at(min_depth, pixel[active], z32[active])
            layer_mask = active_mask & (z32 <= min_depth[pixel] + depth_tie_m)
            layer = np.flatnonzero(layer_mask)
            if layer.size == 0:
                break
            raw_coverage = np.bincount(
                pixel[layer], weights=weight64[layer], minlength=pixel_count,
            )
            alpha = np.minimum(raw_coverage, np.maximum(1.0 - coverage, 0.0))
            for channel in range(src.shape[-1]):
                value_sum = np.bincount(
                    pixel[layer], weights=weight64[layer] * value64[layer, channel],
                    minlength=pixel_count,
                )
                layer_mean = np.divide(
                    value_sum, raw_coverage, out=np.zeros_like(value_sum),
                    where=raw_coverage > 0,
                )
                output[:, channel] += alpha * layer_mean
            coverage += alpha
            unresolved[layer] = False
            unresolved &= remaining[pixel] > 1e-8
            if not np.any(unresolved):
                break

    visible = coverage > 1e-8
    output[visible] /= coverage[visible, None]
    shaped = output.astype(np.float32).reshape(height, width, -1)
    if scalar:
        shaped = shaped[..., 0]
    return shaped, visible.reshape(height, width), coverage.astype(np.float32).reshape(height, width)


def candidate_frame(
    image0: np.ndarray,
    depth0: np.ndarray,
    seg0: np.ndarray,
    image1: np.ndarray,
    depth1: np.ndarray,
    seg1: np.ndarray,
    camera0: np.ndarray,
    camera1: np.ndarray,
    intrinsics: np.ndarray,
    alpha: float,
    object_world0: Optional[Mapping[int, np.ndarray]] = None,
    object_world1: Optional[Mapping[int, np.ndarray]] = None,
) -> np.ndarray:
    """Build one candidate intermediate frame using cached endpoint states."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    image0 = np.asarray(image0, dtype=np.float32)
    image1 = np.asarray(image1, dtype=np.float32)
    if alpha <= 1e-8:
        return image0.copy()
    if alpha >= 1.0 - 1e-8:
        return image1.copy()

    camera_alpha = interpolate_se3(camera0, camera1, alpha)
    p0 = project_endpoint_to_alpha(
        depth0, seg0, intrinsics, camera0, camera_alpha, alpha, 0,
        object_world0, object_world1, None,
    )
    p1 = project_endpoint_to_alpha(
        depth1, seg1, intrinsics, camera1, camera_alpha, alpha, 1,
        object_world0, object_world1, None,
    )
    warped0, valid0, _ = coverage_splat(image0, depth0, *p0)
    warped1, valid1, _ = coverage_splat(image1, depth1, *p1)
    valid = valid0 | valid1
    output = np.full_like(image0, np.nan, dtype=np.float32)
    output[valid1] = warped1[valid1]
    output[~valid1 & valid0] = warped0[~valid1 & valid0]
    # This intentionally matches the current demo candidate's legacy hold
    # convention. It does not claim to reconstruct unknown mid-interval light.
    invalid = ~valid
    if np.any(invalid):
        source = image0 if alpha < 0.5 else image1
        output[invalid] = source[invalid]
    return output
