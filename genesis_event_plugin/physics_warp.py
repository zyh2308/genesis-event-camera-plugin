"""Depth-aware image warping driven by Genesis rigid-body poses."""

from typing import Dict, Mapping, Optional, Tuple

import numpy as np


def _check_pose(T: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    return T


def _pixel_rays(depth: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    K = np.asarray(K, dtype=np.float64)
    if depth.ndim != 2 or K.shape != (3, 3):
        raise ValueError("depth must be HxW and K must be 3x3")
    H, W = depth.shape
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij"
    )
    X = (xx - K[0, 2]) * depth / K[0, 0]
    Y = (yy - K[1, 2]) * depth / K[1, 1]
    return np.stack((X, Y, depth), axis=-1), np.stack((xx, yy), axis=0)


def compute_se3_flow(
    depth0: np.ndarray,
    seg0: np.ndarray,
    K: np.ndarray,
    camera_to_world0: np.ndarray,
    camera_to_world1: np.ndarray,
    object_world_deltas: Optional[Mapping[int, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a full camera/object SE(3) interval into image displacement.

    ``object_world_deltas[id]`` maps a point on object ``id`` from its world
    pose at ``t0`` to its world pose at ``t1``.  Pixels not listed in the map
    are treated as static world geometry, while the camera pose is always
    applied.  This handles camera translation/rotation, axial motion and
    articulated-link transforms without estimating optical flow from RGB.
    """
    Tcw0 = _check_pose(camera_to_world0, "camera_to_world0")
    Tcw1 = _check_pose(camera_to_world1, "camera_to_world1")
    depth = np.asarray(depth0, dtype=np.float32)
    seg = np.asarray(seg0)
    p0_cam, uv0 = _pixel_rays(depth, K)
    H, W = depth.shape
    p0_flat = p0_cam.reshape(-1, 3)
    p0_world = (Tcw0 @ np.concatenate(
        (p0_flat, np.ones((p0_flat.shape[0], 1), dtype=np.float64)), axis=1
    ).T).T[:, :3]
    p1_world = p0_world.copy()

    if object_world_deltas:
        seg_flat = seg.reshape(-1)
        for object_id, delta in object_world_deltas.items():
            mask = seg_flat == object_id
            if not np.any(mask):
                continue
            T = _check_pose(delta, f"object_world_deltas[{object_id}]")
            p = np.concatenate((p0_world[mask], np.ones((mask.sum(), 1))), axis=1)
            p1_world[mask] = (T @ p.T).T[:, :3]

    Twc1 = np.linalg.inv(Tcw1)
    p1_cam = (Twc1 @ np.concatenate(
        (p1_world, np.ones((p1_world.shape[0], 1), dtype=np.float64)), axis=1
    ).T).T[:, :3]
    z1 = p1_cam[:, 2]
    valid = (
        np.isfinite(depth.reshape(-1))
        & np.isfinite(p1_cam).all(axis=1)
        & (depth.reshape(-1) > 1e-6)
        & (z1 > 1e-6)
    )
    u1 = np.zeros_like(z1, dtype=np.float64)
    v1 = np.zeros_like(z1, dtype=np.float64)
    u1[valid] = K[0, 0] * p1_cam[valid, 0] / z1[valid] + K[0, 2]
    v1[valid] = K[1, 1] * p1_cam[valid, 1] / z1[valid] + K[1, 2]
    flow = np.stack(
        (u1 - uv0[0].reshape(-1), v1 - uv0[1].reshape(-1)), axis=0
    ).reshape(2, H, W).astype(np.float32)
    valid = valid.reshape(H, W)
    flow[:, ~valid] = 0.0
    return flow, valid


def _splat(
    image: np.ndarray,
    depth: np.ndarray,
    flow: np.ndarray,
    amount: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward splat an image with a nearest-depth z-buffer."""
    image = np.asarray(image, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    flow = np.asarray(flow, dtype=np.float32)
    H, W = depth.shape
    channels = (1,) if image.ndim == 2 else (image.shape[-1],)
    src = image[..., None] if image.ndim == 2 else image
    out = np.zeros((H, W) + channels, dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    valid = np.isfinite(src).all(axis=-1) & np.isfinite(depth) & (depth > 1e-6)
    tx = xx.astype(np.float32) + flow[0] * amount
    ty = yy.astype(np.float32) + flow[1] * amount
    x0 = np.floor(tx).astype(np.int32)
    y0 = np.floor(ty).astype(np.int32)
    wx = tx - x0
    wy = ty - y0

    candidates = []
    for ox, oy, w in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)),
                       (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        x = x0 + ox
        y = y0 + oy
        inside = valid & (w > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if np.any(inside):
            candidates.append((x[inside], y[inside], w[inside], depth[inside], src[inside]))

    for x, y, w, z, values in candidates:
        np.minimum.at(zbuf, (y, x), z)
    for x, y, w, z, values in candidates:
        keep = z <= zbuf[y, x] + 1e-5
        if not np.any(keep):
            continue
        x, y, w, values = x[keep], y[keep], w[keep], values[keep]
        np.add.at(weight, (y, x), w)
        for c in range(values.shape[1]):
            np.add.at(out[..., c], (y, x), w * values[:, c])
    nonzero = weight > 1e-8
    out[nonzero] /= weight[nonzero, None]
    return (out[..., 0] if image.ndim == 2 else out), nonzero


def depth_aware_bidirectional_warp(
    frame0: np.ndarray,
    depth0: np.ndarray,
    flow01: np.ndarray,
    frame1: np.ndarray,
    depth1: np.ndarray,
    flow10: np.ndarray,
    alpha: float,
    composite: str = "weighted",
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp both endpoints to ``alpha`` and fuse only visible samples.

    ``weighted`` preserves the original v2 blend.  ``next_primary`` follows
    the EVIS/Genesis motion-vector convention: the backward-warped next
    keyframe is authoritative wherever it is visible, while the previous
    keyframe only fills disocclusion holes.  The latter avoids averaging two
    differently exposed silhouette fringes and is the recommended event path.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if composite not in {"weighted", "next_primary"}:
        raise ValueError("composite must be 'weighted' or 'next_primary'")
    a0, v0 = _splat(frame0, depth0, flow01, alpha)
    a1, v1 = _splat(frame1, depth1, flow10, 1.0 - alpha)
    if composite == "next_primary":
        valid = v0 | v1
        if frame0.ndim == 2:
            out = np.full_like(a0, np.nan, dtype=np.float32)
            out[v1] = a1[v1]
            out[~v1 & v0] = a0[~v1 & v0]
        else:
            out = np.full_like(a0, np.nan, dtype=np.float32)
            out[v1] = a1[v1]
            out[~v1 & v0] = a0[~v1 & v0]
        return out, valid
    w0 = (1.0 - alpha) * v0.astype(np.float32)
    w1 = alpha * v1.astype(np.float32)
    denom = w0 + w1
    if frame0.ndim == 2:
        out = np.zeros_like(a0, dtype=np.float32)
        out[:] = np.nan
        good = denom > 1e-8
        out[good] = (a0[good] * w0[good] + a1[good] * w1[good]) / denom[good]
    else:
        out = np.full_like(a0, np.nan, dtype=np.float32)
        good = denom > 1e-8
        out[good] = (
            a0[good] * w0[good, None] + a1[good] * w1[good, None]
        ) / denom[good, None]
    return out, denom > 1e-8
