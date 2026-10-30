"""Depth-aware image warping driven by Genesis rigid-body poses."""

from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np


def _check_pose(T: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    return T


def _so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """Rodrigues exponential for a 3-vector rotation logarithm."""
    w = np.asarray(rotation_vector, dtype=np.float64)
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
        return np.eye(3) + K
    a = w / theta
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    cosine = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cosine))
    if theta < 1e-10:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    sine = np.sin(theta)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * sine)
    return axis * theta


def interpolate_se3(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate an SE(3) pose with linear translation and constant twist.

    This is an endpoint interpolation assumption, not a substitute for the
    actual Genesis physics trajectory.  It uses the SO(3) logarithm/exponential
    for rotation and linear translation between the supplied endpoint poses.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    A = _check_pose(T0, "T0")
    B = _check_pose(T1, "T1")
    out = np.eye(4, dtype=np.float64)
    rel_R = A[:3, :3].T @ B[:3, :3]
    out[:3, :3] = A[:3, :3] @ _so3_exp(float(alpha) * _so3_log(rel_R))
    out[:3, 3] = (1.0 - float(alpha)) * A[:3, 3] + float(alpha) * B[:3, 3]
    return out


def _object_motion_delta(
    object_id: int,
    alpha: float,
    source_index: int,
    object_world0: Optional[Mapping[int, np.ndarray]],
    object_world1: Optional[Mapping[int, np.ndarray]],
    object_world_deltas: Optional[Mapping[int, np.ndarray]],
) -> Optional[np.ndarray]:
    if object_world0 is not None and object_world1 is not None and object_id in object_world0 and object_id in object_world1:
        T0 = _check_pose(object_world0[object_id], f"object_world0[{object_id}]")
        T1 = _check_pose(object_world1[object_id], f"object_world1[{object_id}]")
        T_alpha = interpolate_se3(T0, T1, alpha)
        source = T0 if source_index == 0 else T1
        return T_alpha @ np.linalg.inv(source)
    if object_world_deltas is not None and object_id in object_world_deltas:
        delta = _check_pose(object_world_deltas[object_id], f"object_world_deltas[{object_id}]")
        T_alpha = interpolate_se3(np.eye(4), delta, alpha)
        return T_alpha if source_index == 0 else T_alpha @ np.linalg.inv(delta)
    return None


def project_endpoint_to_alpha(
    depth_source: np.ndarray,
    seg_source: np.ndarray,
    K: np.ndarray,
    camera_source: np.ndarray,
    camera_alpha: np.ndarray,
    alpha: float,
    source_index: int = 0,
    object_world0: Optional[Mapping[int, np.ndarray]] = None,
    object_world1: Optional[Mapping[int, np.ndarray]] = None,
    object_world_deltas: Optional[Mapping[int, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project an endpoint's 3-D samples into an explicitly interpolated pose.

    Returns ``(uv, z_alpha, valid)``.  The z-buffer depth is the projected
    intermediate depth, not either endpoint depth.
    """
    T_source = _check_pose(camera_source, "camera_source")
    T_alpha = _check_pose(camera_alpha, "camera_alpha")
    p_source, uv_source = _pixel_rays(depth_source, K)
    H, W = np.asarray(depth_source).shape
    p_flat = p_source.reshape(-1, 3)
    p_world = (T_source @ np.c_[p_flat, np.ones(len(p_flat))].T).T[:, :3]
    p_alpha_world = p_world.copy()
    flat_seg = np.asarray(seg_source).reshape(-1)
    object_ids = set()
    if object_world0 is not None:
        object_ids.update(object_world0.keys())
    if object_world1 is not None:
        object_ids.update(object_world1.keys())
    if object_world_deltas is not None:
        object_ids.update(object_world_deltas.keys())
    for object_id in object_ids:
        mask = flat_seg == int(object_id)
        if not np.any(mask):
            continue
        delta = _object_motion_delta(
            int(object_id), alpha, source_index,
            object_world0, object_world1, object_world_deltas,
        )
        if delta is not None:
            p_alpha_world[mask] = (delta @ np.c_[p_world[mask], np.ones(mask.sum())].T).T[:, :3]
    p_alpha_cam = (np.linalg.inv(T_alpha) @ np.c_[p_alpha_world, np.ones(len(p_alpha_world))].T).T[:, :3]
    z = p_alpha_cam[:, 2]
    valid = np.isfinite(p_alpha_cam).all(axis=1) & np.isfinite(np.asarray(depth_source).reshape(-1))
    valid &= (np.asarray(depth_source).reshape(-1) > 1e-6) & (z > 1e-6)
    uv = np.zeros((2, len(z)), dtype=np.float64)
    uv[0, valid] = K[0, 0] * p_alpha_cam[valid, 0] / z[valid] + K[0, 2]
    uv[1, valid] = K[1, 1] * p_alpha_cam[valid, 1] / z[valid] + K[1, 2]
    uv = uv.reshape(2, H, W).astype(np.float32)
    z = z.reshape(H, W).astype(np.float32)
    valid = valid.reshape(H, W)
    uv[:, ~valid] = 0.0
    z[~valid] = np.inf
    return uv, z, valid


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


def _splat_projected(
    image: np.ndarray,
    source_depth: np.ndarray,
    target_uv: np.ndarray,
    target_z: np.ndarray,
    source_valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward splat using explicit intermediate projected coordinates/depth."""
    output, diagnostics = _splat_projected_diagnostics(
        image, source_depth, target_uv, target_z, source_valid
    )
    return output, diagnostics["visibility"]


def _splat_projected_diagnostics(
    image: np.ndarray,
    source_depth: np.ndarray,
    target_uv: np.ndarray,
    target_z: np.ndarray,
    source_valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Forward splat and return masks in the *projected target image domain*.

    ``source_valid`` is evaluated on the source grid. ``occupancy`` and
    ``visibility`` are target-grid masks: occupancy means at least one source
    sample landed in the target pixel, while visibility means the sample
    survived the target ``z``-buffer and contributed to the output. Keeping
    these domains explicit prevents same-shape arrays from being mistaken for
    the same physical correspondence.
    """
    image = np.asarray(image, dtype=np.float32)
    source_depth = np.asarray(source_depth, dtype=np.float32)
    target_uv = np.asarray(target_uv, dtype=np.float32)
    target_z = np.asarray(target_z, dtype=np.float32)
    H, W = source_depth.shape
    scalar = image.ndim == 2
    src = image[..., None] if scalar else image
    C = src.shape[-1]
    out = np.zeros((H, W, C), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    occupancy = np.zeros((H, W), dtype=bool)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    valid = np.isfinite(src).all(axis=-1) & np.isfinite(source_depth) & (source_depth > 1e-6)
    valid &= np.isfinite(target_z) & (target_z > 1e-6)
    if source_valid is not None:
        valid &= np.asarray(source_valid, dtype=bool)
    tx, ty = target_uv[0], target_uv[1]
    x0, y0 = np.floor(tx).astype(np.int32), np.floor(ty).astype(np.int32)
    wx, wy = tx - x0, ty - y0
    candidates = []
    for ox, oy, w in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)),
                      (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        x, y = x0 + ox, y0 + oy
        inside = valid & (w > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if np.any(inside):
            candidates.append((x[inside], y[inside], w[inside], target_z[inside], src[inside]))
    for x, y, w, z, values in candidates:
        np.logical_or.at(occupancy, (y, x), True)
    for x, y, w, z, values in candidates:
        np.minimum.at(zbuf, (y, x), z)
    for x, y, w, z, values in candidates:
        keep = z <= zbuf[y, x] + 1e-5
        if not np.any(keep):
            continue
        x, y, w, values = x[keep], y[keep], w[keep], values[keep]
        np.add.at(weight, (y, x), w)
        for c in range(C):
            np.add.at(out[..., c], (y, x), w * values[:, c])
    nonzero = weight > 1e-8
    out[nonzero] /= weight[nonzero, None]
    return (out[..., 0] if scalar else out), {
        "occupancy": occupancy,
        "visibility": nonzero,
        "z_buffer": zbuf,
    }


def _splat_projected_coverage_diagnostics(
    image: np.ndarray,
    source_depth: np.ndarray,
    target_uv: np.ndarray,
    target_z: np.ndarray,
    source_valid: Optional[np.ndarray] = None,
    source_surface_id: Optional[np.ndarray] = None,
    min_coverage_fraction: float = 0.995,
    depth_tie_abs_m: float = 1e-7,
    depth_tie_rtol: float = 8.0 * np.finfo(np.float32).eps,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Candidate front-to-back splat with approximate footprint coverage.

    Bilinear weights are only an area approximation, not exact renderer
    coverage. A target pixel whose known samples cover less than the declared
    fraction remains unsupported even though a conditional radiance estimate
    is returned for diagnostics.
    """
    if not 0.0 < min_coverage_fraction <= 1.0:
        raise ValueError("min_coverage_fraction must be in (0,1]")
    if depth_tie_abs_m < 0 or depth_tie_rtol < 0:
        raise ValueError("depth tie tolerances must be non-negative")
    image = np.asarray(image, dtype=np.float32)
    depth = np.asarray(source_depth, dtype=np.float32)
    uv = np.asarray(target_uv, dtype=np.float32)
    z = np.asarray(target_z, dtype=np.float32)
    H, W = depth.shape
    scalar = image.ndim == 2
    src = image[..., None] if scalar else image
    if src.shape[:2] != (H, W) or uv.shape != (2, H, W) or z.shape != (H, W):
        raise ValueError("image/depth/target projection shapes do not agree")
    labels = None if source_surface_id is None else np.asarray(source_surface_id, dtype=np.int64)
    if labels is not None and labels.shape != (H, W):
        raise ValueError("source_surface_id must have shape (H,W)")
    valid = np.isfinite(src).all(axis=-1) & np.isfinite(depth) & (depth > 1e-6)
    valid &= np.isfinite(z) & (z > 1e-6) & np.isfinite(uv).all(axis=0)
    if source_valid is not None:
        valid &= np.asarray(source_valid, dtype=bool)

    tx, ty = uv[0], uv[1]
    x0, y0 = np.floor(tx).astype(np.int32), np.floor(ty).astype(np.int32)
    wx, wy = tx - x0, ty - y0
    candidates = []
    occupancy = np.zeros((H, W), dtype=bool)
    z_buffer = np.full((H, W), np.inf, dtype=np.float32)
    for ox, oy, weight in (
        (0, 0, (1.0 - wx) * (1.0 - wy)),
        (1, 0, wx * (1.0 - wy)),
        (0, 1, (1.0 - wx) * wy),
        (1, 1, wx * wy),
    ):
        x, y = x0 + ox, y0 + oy
        keep = valid & (weight > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if np.any(keep):
            entry = (x[keep], y[keep], weight[keep], z[keep], src[keep])
            if labels is not None:
                entry += (labels[keep],)
            else:
                entry += (None,)
            candidates.append(entry)
            np.logical_or.at(occupancy, (y[keep], x[keep]), True)
            np.minimum.at(z_buffer, (y[keep], x[keep]), z[keep])

    out = np.zeros((H, W, src.shape[-1]), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=np.float32)
    surface_id = np.full((H, W), -1, dtype=np.int64)
    surface_consistent = np.zeros((H, W), dtype=bool)
    if candidates:
        count = np.zeros((H, W), dtype=np.int32)
        for x, y, *_ in candidates:
            np.add.at(count, (y, x), 1)
        unresolved = [np.ones(len(item[0]), dtype=bool) for item in candidates]
        for _ in range(int(count.max())):
            nearest = np.full((H, W), np.inf, dtype=np.float32)
            for (x, y, _weight, zc, _value, _label), active in zip(candidates, unresolved):
                if np.any(active):
                    np.minimum.at(nearest, (y[active], x[active]), zc[active])
            if not np.isfinite(nearest).any():
                break
            tie = np.maximum(depth_tie_abs_m, depth_tie_rtol * np.abs(nearest))
            layer_weight = np.zeros((H, W), dtype=np.float32)
            layer_value = np.zeros_like(out)
            label_min = np.full((H, W), np.iinfo(np.int64).max, dtype=np.int64)
            label_max = np.full((H, W), np.iinfo(np.int64).min, dtype=np.int64)
            next_unresolved = []
            for (x, y, weight, zc, value, label), active in zip(candidates, unresolved):
                take = active & (zc <= nearest[y, x] + tie[y, x])
                next_unresolved.append(active & ~take)
                if not np.any(take):
                    continue
                xt, yt, wt, vt = x[take], y[take], weight[take], value[take]
                np.add.at(layer_weight, (yt, xt), wt)
                for channel in range(src.shape[-1]):
                    np.add.at(layer_value[..., channel], (yt, xt), wt * vt[:, channel])
                if label is not None:
                    np.minimum.at(label_min, (yt, xt), label[take])
                    np.maximum.at(label_max, (yt, xt), label[take])
            first_layer = coverage <= 1e-8
            is_first = first_layer & (layer_weight > 0)
            if labels is not None:
                surface_id[is_first] = label_min[is_first]
                surface_consistent[is_first] = label_min[is_first] == label_max[is_first]
            accepted = np.minimum(layer_weight, np.maximum(1.0 - coverage, 0.0))
            scale = np.divide(accepted, layer_weight, out=np.zeros_like(accepted), where=layer_weight > 0)
            out += layer_value * scale[..., None]
            coverage = np.minimum(coverage + accepted, 1.0)
            unresolved = next_unresolved

    visible = coverage > 1e-8
    out[visible] /= coverage[visible, None]
    if labels is None:
        surface_consistent[visible] = True
        surface_id[visible] = 0
    return (out[..., 0] if scalar else out), {
        "occupancy": occupancy,
        "visibility": visible,
        "coverage": coverage,
        "support": coverage >= min_coverage_fraction,
        "z_buffer": z_buffer,
        "surface_id": surface_id,
        "surface_consistent": surface_consistent,
    }


def projected_target_visibility(
    source_depth: np.ndarray,
    target_uv: np.ndarray,
    target_z: np.ndarray,
    source_valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target-domain occupancy, visibility, and projected z-buffer.

    The returned masks all use the target image coordinate system. This is the
    canonical diagnostic API for endpoint correspondence and disocclusion.
    """
    depth = np.asarray(source_depth, dtype=np.float32)
    dummy = np.ones_like(depth, dtype=np.float32)
    _, diagnostics = _splat_projected_diagnostics(
        dummy, depth, target_uv, target_z, source_valid
    )
    return (
        diagnostics["occupancy"],
        diagnostics["visibility"],
        diagnostics["z_buffer"],
    )


def depth_aware_bidirectional_warp_projected(
    frame0: np.ndarray,
    depth0: np.ndarray,
    uv0: np.ndarray,
    z0: np.ndarray,
    valid0: np.ndarray,
    frame1: np.ndarray,
    depth1: np.ndarray,
    uv1: np.ndarray,
    z1: np.ndarray,
    valid1: np.ndarray,
    composite: str = "next_primary",
    alpha: float = 0.5,
    surface_id0: Optional[np.ndarray] = None,
    surface_id1: Optional[np.ndarray] = None,
    min_coverage_fraction: float = 0.995,
    depth_tie_abs_m: float = 1e-7,
    depth_tie_rtol: float = 8.0 * np.finfo(np.float32).eps,
    surface_depth_abs_m: float = 1e-3,
    surface_depth_rtol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp endpoints using true projections in one target image domain.

    The second return value is the intermediate target-domain visibility mask,
    not either endpoint's source-valid mask.
    """
    if composite == "same_surface_log_linear":
        out, valid, _ = depth_aware_bidirectional_warp_projected_diagnostics(
            frame0, depth0, uv0, z0, valid0,
            frame1, depth1, uv1, z1, valid1,
            composite=composite, alpha=alpha,
            surface_id0=surface_id0, surface_id1=surface_id1,
            min_coverage_fraction=min_coverage_fraction,
            depth_tie_abs_m=depth_tie_abs_m, depth_tie_rtol=depth_tie_rtol,
            surface_depth_abs_m=surface_depth_abs_m,
            surface_depth_rtol=surface_depth_rtol,
        )
        return out, valid
    a0, d0 = _splat_projected_diagnostics(frame0, depth0, uv0, z0, valid0)
    a1, d1 = _splat_projected_diagnostics(frame1, depth1, uv1, z1, valid1)
    v0, v1 = d0["visibility"], d1["visibility"]
    if composite == "next_primary":
        valid = v0 | v1
        out = np.full_like(a0, np.nan, dtype=np.float32)
        out[v1] = a1[v1]
        out[~v1 & v0] = a0[~v1 & v0]
        return out, valid
    if composite != "weighted":
        raise ValueError("composite must be 'weighted' or 'next_primary'")
    w0 = v0.astype(np.float32)
    w1 = v1.astype(np.float32)
    denom = w0 + w1
    out = np.full_like(a0, np.nan, dtype=np.float32)
    good = denom > 1e-8
    if frame0.ndim == 2:
        out[good] = (a0[good] * w0[good] + a1[good] * w1[good]) / denom[good]
    else:
        out[good] = (a0[good] * w0[good, None] + a1[good] * w1[good, None]) / denom[good, None]
    return out, good


def depth_aware_bidirectional_warp_projected_diagnostics(
    frame0: np.ndarray,
    depth0: np.ndarray,
    uv0: np.ndarray,
    z0: np.ndarray,
    valid0: np.ndarray,
    frame1: np.ndarray,
    depth1: np.ndarray,
    uv1: np.ndarray,
    z1: np.ndarray,
    valid1: np.ndarray,
    composite: str = "next_primary",
    alpha: float = 0.5,
    surface_id0: Optional[np.ndarray] = None,
    surface_id1: Optional[np.ndarray] = None,
    min_coverage_fraction: float = 0.995,
    depth_tie_abs_m: float = 1e-7,
    depth_tie_rtol: float = 8.0 * np.finfo(np.float32).eps,
    surface_depth_abs_m: float = 1e-3,
    surface_depth_rtol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Warp and expose source/target visibility diagnostics.

    ``disocclusion`` is defined in the projected target image domain as the
    complement of the merged endpoint visibility. There is no claim that this
    mask identifies hidden content beyond the support of both endpoint
    projections; such content requires a real Genesis intermediate render.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if composite == "same_surface_log_linear":
        exact_frame, exact_valid, exact_endpoint = None, None, None
        if alpha == 0.0:
            exact_frame, exact_valid, exact_endpoint = frame0, valid0, 0
        elif alpha == 1.0:
            exact_frame, exact_valid, exact_endpoint = frame1, valid1, 1
        if exact_frame is not None:
            # Direct endpoint radiance remains observed even when depth is
            # missing; depth support only limits geometric interpolation.
            exact_valid = (
                np.isfinite(exact_frame).all(axis=-1)
                if exact_frame.ndim == 3 else np.isfinite(exact_frame)
            )
            shape = exact_valid.shape
            false = np.zeros(shape, dtype=bool)
            zero = np.zeros(shape, dtype=np.float32)
            sid0 = np.asarray(surface_id0) if surface_id0 is not None else np.zeros(shape, dtype=np.int64)
            sid1 = np.asarray(surface_id1) if surface_id1 is not None else np.zeros(shape, dtype=np.int64)
            diag = {
                "occupancy0": exact_valid.copy() if exact_endpoint == 0 else false,
                "occupancy1": exact_valid.copy() if exact_endpoint == 1 else false,
                "visibility0": exact_valid.copy() if exact_endpoint == 0 else false,
                "visibility1": exact_valid.copy() if exact_endpoint == 1 else false,
                "support0": exact_valid.copy() if exact_endpoint == 0 else false,
                "support1": exact_valid.copy() if exact_endpoint == 1 else false,
                "coverage0": exact_valid.astype(np.float32) if exact_endpoint == 0 else zero,
                "coverage1": exact_valid.astype(np.float32) if exact_endpoint == 1 else zero,
                "z_buffer0": np.asarray(z0, dtype=np.float32) if exact_endpoint == 0 else np.full(shape, np.inf, dtype=np.float32),
                "z_buffer1": np.asarray(z1, dtype=np.float32) if exact_endpoint == 1 else np.full(shape, np.inf, dtype=np.float32),
                "surface_id0": sid0 if exact_endpoint == 0 else np.full(shape, -1, dtype=np.int64),
                "surface_id1": sid1 if exact_endpoint == 1 else np.full(shape, -1, dtype=np.int64),
                "surface_consistent0": exact_valid.copy() if exact_endpoint == 0 else false,
                "surface_consistent1": exact_valid.copy() if exact_endpoint == 1 else false,
                "surface_match": exact_valid.copy(),
                "intermediate_visibility": exact_valid.copy(),
                "intermediate_unknown": ~exact_valid,
                "intermediate_coverage": exact_valid.astype(np.float32),
                "disocclusion": ~exact_valid,
            }
            return np.asarray(exact_frame, dtype=np.float32).copy(), exact_valid, diag
        a0, d0 = _splat_projected_coverage_diagnostics(
            frame0, depth0, uv0, z0, valid0, surface_id0,
            min_coverage_fraction, depth_tie_abs_m, depth_tie_rtol,
        )
        a1, d1 = _splat_projected_coverage_diagnostics(
            frame1, depth1, uv1, z1, valid1, surface_id1,
            min_coverage_fraction, depth_tie_abs_m, depth_tie_rtol,
        )
        zmatch = np.abs(d0["z_buffer"] - d1["z_buffer"]) <= np.maximum(
            float(surface_depth_abs_m),
            float(surface_depth_rtol) * np.maximum(np.abs(d0["z_buffer"]), np.abs(d1["z_buffer"])),
        )
        surface_match = (
            d0["support"] & d1["support"]
            & d0["surface_consistent"] & d1["surface_consistent"]
            & (d0["surface_id"] == d1["surface_id"])
            & zmatch
        )
        out = np.full_like(a0, np.nan, dtype=np.float32)
        safe0 = np.maximum(a0, 1e-6)
        safe1 = np.maximum(a1, 1e-6)
        mixed_log = (1.0 - float(alpha)) * np.log(safe0) + float(alpha) * np.log(safe1)
        out[surface_match] = np.exp(mixed_log[surface_match]).astype(np.float32)
        visible = surface_match
        diag = {
            "occupancy0": d0["occupancy"], "occupancy1": d1["occupancy"],
            "visibility0": d0["visibility"], "visibility1": d1["visibility"],
            "support0": d0["support"], "support1": d1["support"],
            "coverage0": d0["coverage"], "coverage1": d1["coverage"],
            "z_buffer0": d0["z_buffer"], "z_buffer1": d1["z_buffer"],
            "surface_id0": d0["surface_id"], "surface_id1": d1["surface_id"],
            "surface_consistent0": d0["surface_consistent"],
            "surface_consistent1": d1["surface_consistent"],
            "surface_match": surface_match,
            "intermediate_visibility": visible,
            "intermediate_unknown": ~visible,
            "intermediate_coverage": np.minimum(d0["coverage"], d1["coverage"]),
            "disocclusion": ~visible,
        }
        return out, visible, diag

    a0, d0 = _splat_projected_diagnostics(frame0, depth0, uv0, z0, valid0)
    a1, d1 = _splat_projected_diagnostics(frame1, depth1, uv1, z1, valid1)
    v0, v1 = d0["visibility"], d1["visibility"]
    if composite == "next_primary":
        visible = v0 | v1
        out = np.full_like(a0, np.nan, dtype=np.float32)
        out[v1] = a1[v1]
        out[~v1 & v0] = a0[~v1 & v0]
    elif composite == "weighted":
        w0, w1 = v0.astype(np.float32), v1.astype(np.float32)
        denom = w0 + w1
        visible = denom > 1e-8
        out = np.full_like(a0, np.nan, dtype=np.float32)
        if frame0.ndim == 2:
            out[visible] = (a0[visible] * w0[visible] + a1[visible] * w1[visible]) / denom[visible]
        else:
            out[visible] = (a0[visible] * w0[visible, None] + a1[visible] * w1[visible, None]) / denom[visible, None]
    else:
        raise ValueError("composite must be 'weighted' or 'next_primary'")
    diagnostics = {
        "occupancy0": d0["occupancy"],
        "occupancy1": d1["occupancy"],
        "visibility0": v0,
        "visibility1": v1,
        "z_buffer0": d0["z_buffer"],
        "z_buffer1": d1["z_buffer"],
        "intermediate_visibility": visible,
        "disocclusion": ~visible,
    }
    return out, visible, diagnostics


def endpoint_radiance_residual(
    frame0: np.ndarray,
    frame1: np.ndarray,
    depth0: np.ndarray,
    target_uv0: np.ndarray,
    target_z0: np.ndarray,
    valid0: np.ndarray,
    eps: float = 1e-6,
    return_support: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Measure target-domain log-radiance residual.

    The warped source is in the target image domain. Support is therefore the
    projected target visibility from the splat and the target frame's own
    finite-pixel mask; ``valid0`` is never compared elementwise with a target
    mask. Set ``return_support=True`` to inspect the support used for the
    residual.
    """
    log0 = np.log(np.maximum(np.asarray(frame0, dtype=np.float32), eps))
    log1 = np.log(np.maximum(np.asarray(frame1, dtype=np.float32), eps))
    warped, diagnostics = _splat_projected_diagnostics(
        log0, depth0, target_uv0, target_z0, valid0
    )
    target_frame = np.asarray(frame1, dtype=np.float32)
    target_support = (
        diagnostics["visibility"]
        & np.isfinite(log1)
        & np.isfinite(target_frame)
        & (target_frame >= 0.0)
    )
    residual = np.full_like(log1, np.nan, dtype=np.float32)
    residual[target_support] = np.abs(log1[target_support] - warped[target_support])
    if return_support:
        return residual, target_support
    return residual


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
