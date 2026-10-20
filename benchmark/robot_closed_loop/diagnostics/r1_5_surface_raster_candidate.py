"""Diagnostic-only continuous-surface rasterizer for Robot R1.5.

This is deliberately outside ``genesis_event_plugin``: the frozen NumPy and
Torch simulator paths remain untouched until this candidate passes the
general controls and is reviewed. Unlike point splatting, this module turns
locally continuous depth/segmentation samples into projected triangles,
rasterizes covered target pixels, and performs visibility on interpolated
surface depth. Unsupported pixels remain explicit render requests.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


try:  # Optional acceleration only; the implementation has a NumPy fallback.
    from numba import njit

    _NUMBA = True
except Exception:  # pragma: no cover - depends on the execution environment.
    _NUMBA = False

    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


@njit(cache=False)
def _raster_triangles_numba(image, labels, uv, z, vertex_valid, edge_valid,
                            out, zbuf, out_label, tie_tolerance, height, width):
    triangle_vertices = np.array(((0, 1, 3), (0, 3, 2)), dtype=np.int64)
    for y in range(height - 1):
        for x in range(width - 1):
            ids = np.array((y * width + x, y * width + x + 1,
                            (y + 1) * width + x, (y + 1) * width + x + 1), dtype=np.int64)
            for tri_index in range(2):
                a = ids[triangle_vertices[tri_index, 0]]
                b = ids[triangle_vertices[tri_index, 1]]
                c = ids[triangle_vertices[tri_index, 2]]
                if not (vertex_valid[a] and vertex_valid[b] and vertex_valid[c]):
                    continue
                if labels[a] != labels[b] or labels[a] != labels[c]:
                    continue
                # edge_valid stores horizontal, vertical and down-right
                # continuity.  These are topology checks, not z-buffer epsilons.
                if tri_index == 0:
                    if not (edge_valid[0, a] and edge_valid[1, b] and edge_valid[2, a]):
                        continue
                else:
                    if not (edge_valid[1, a] and edge_valid[0, c] and edge_valid[2, a]):
                        continue

                x0, y0 = uv[0, a], uv[1, a]
                x1, y1 = uv[0, b], uv[1, b]
                x2, y2 = uv[0, c], uv[1, c]
                area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                if not np.isfinite(area) or abs(area) < 1e-10:
                    continue
                min_x = max(0, int(np.ceil(min(x0, x1, x2) - 1e-7)))
                max_x = min(width - 1, int(np.floor(max(x0, x1, x2) + 1e-7)))
                min_y = max(0, int(np.ceil(min(y0, y1, y2) - 1e-7)))
                max_y = min(height - 1, int(np.floor(max(y0, y1, y2) + 1e-7)))
                inv_area = 1.0 / area
                for py in range(min_y, max_y + 1):
                    for px in range(min_x, max_x + 1):
                        # Pixel centers are integer coordinates throughout the
                        # Genesis camera/ray convention used by this project.
                        l0 = ((x1 - px) * (y2 - py) - (y1 - py) * (x2 - px)) * inv_area
                        l1 = ((x2 - px) * (y0 - py) - (y2 - py) * (x0 - px)) * inv_area
                        l2 = 1.0 - l0 - l1
                        if l0 < -1e-6 or l1 < -1e-6 or l2 < -1e-6:
                            continue
                        # Perspective-correct interpolation of radiance and
                        # depth over the projected 3D triangle.
                        q0, q1, q2 = l0 / z[a], l1 / z[b], l2 / z[c]
                        qsum = q0 + q1 + q2
                        if qsum <= 1e-12:
                            continue
                        fragment_z = 1.0 / qsum
                        if fragment_z < zbuf[py, px] - tie_tolerance:
                            zbuf[py, px] = fragment_z
                            out[py, px] = (q0 * image[a] + q1 * image[b] + q2 * image[c]) / qsum
                            out_label[py, px] = labels[a]
                        elif abs(fragment_z - zbuf[py, px]) <= tie_tolerance:
                            # Shared triangle edges represent the same sheet;
                            # average only genuinely coincident fragments.
                            out[py, px] = 0.5 * (out[py, px] +
                                                  (q0 * image[a] + q1 * image[b] + q2 * image[c]) / qsum)
                            if labels[a] != out_label[py, px]:
                                # Exact coplanar overlap with distinct labels
                                # is ambiguous; mark it so callers can render.
                                out_label[py, px] = -2


def _continuous_edges(depth: np.ndarray, valid: np.ndarray, K: np.ndarray,
                      labels: np.ndarray, max_pixel_scale: float) -> np.ndarray:
    """Test local 3D sample connectivity using a focal-length-scaled bound."""
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    points = np.stack(((xx - K[0, 2]) * depth / K[0, 0],
                       (yy - K[1, 2]) * depth / K[1, 1], depth), axis=-1)
    edge_ok = np.zeros((3, height, width), dtype=bool)
    pairs = (
        (0, 1.0 / K[0, 0]),  # horizontal
        (1, 1.0 / K[1, 1]),  # vertical
        (2, np.hypot(1.0 / K[0, 0], 1.0 / K[1, 1])),  # down-right
    )
    for slot, footprint_per_m in pairs:
        # These local edges make a top/right/down-right triangulation without
        # ever joining across an invalid, different-label, or depth-disjoint
        # surface boundary.
        if slot == 0:
            a, b = (slice(None), slice(0, width - 1)), (slice(None), slice(1, width))
        elif slot == 1:
            a, b = (slice(0, height - 1), slice(None)), (slice(1, height), slice(None))
        else:
            a, b = (slice(0, height - 1), slice(0, width - 1)), (slice(1, height), slice(1, width))
        pa, pb = points[a], points[b]
        za, zb = depth[a], depth[b]
        expected = 0.5 * (za + zb) * footprint_per_m
        distance = np.linalg.norm(pb - pa, axis=-1)
        connected = valid[a] & valid[b] & (labels[a] == labels[b])
        connected &= np.isfinite(distance) & (distance <= float(max_pixel_scale) * expected)
        edge_ok[slot][a] = connected
    return edge_ok


def surface_raster_warp(
    image: np.ndarray,
    source_depth: np.ndarray,
    source_labels: np.ndarray,
    target_uv: np.ndarray,
    target_z: np.ndarray,
    source_valid: np.ndarray,
    K: np.ndarray,
    *,
    max_pixel_scale: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Rasterize local continuous surfaces into the intermediate camera.

    Returns radiance, interpolated target depth, target labels, visibility and
    diagnostics.  Pixels outside all supported triangles are invalid (and must
    be directly rendered or treated as unknown by the caller).
    """
    image = np.asarray(image, dtype=np.float32)
    depth = np.asarray(source_depth, dtype=np.float32)
    labels = np.asarray(source_labels, dtype=np.int64)
    uv = np.asarray(target_uv, dtype=np.float32)
    z = np.asarray(target_z, dtype=np.float32)
    valid = np.asarray(source_valid, dtype=bool).copy()
    K = np.asarray(K, dtype=np.float64)
    if image.ndim != 2 or image.shape != depth.shape or labels.shape != depth.shape:
        raise ValueError("candidate expects same-size HxW radiance, depth, and labels")
    if uv.shape != (2,) + depth.shape or z.shape != depth.shape or valid.shape != depth.shape:
        raise ValueError("projected coordinates, depth, and validity have incompatible shapes")
    if K.shape != (3, 3) or max_pixel_scale <= 0:
        raise ValueError("K must be 3x3 and max_pixel_scale must be positive")
    valid &= np.isfinite(image) & np.isfinite(depth) & (depth > 1e-6)
    valid &= np.isfinite(z) & (z > 1e-6) & np.isfinite(uv).all(axis=0)
    valid &= labels >= 0
    edges = _continuous_edges(depth, valid, K, labels, max_pixel_scale)
    height, width = depth.shape
    out = np.full((height, width), np.nan, dtype=np.float32)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    out_labels = np.full((height, width), -1, dtype=np.int64)
    finite_z = z[valid]
    tie_tolerance = max(1e-7, 8.0 * np.finfo(np.float32).eps * float(np.median(finite_z))) if finite_z.size else 1e-7
    _raster_triangles_numba(
        image.reshape(-1), labels.reshape(-1), uv.reshape(2, -1), z.reshape(-1),
        valid.reshape(-1), edges.reshape(3, -1), out, zbuf, out_labels,
        tie_tolerance, height, width,
    )
    visibility = np.isfinite(zbuf) & np.isfinite(out)
    diag = {
        "backend": "numba" if _NUMBA else "numpy_python_fallback",
        "visible_fraction": float(visibility.mean()),
        "unknown_fraction": float((~visibility).mean()),
        "input_valid_fraction": float(valid.mean()),
        "continuous_edge_fraction": float(edges.mean()),
        "projected_triangle_coverage": int(visibility.sum()),
        "max_pixel_scale": float(max_pixel_scale),
        "surface_topology": "same_label_and_focal_scaled_3d_edge",
        "visibility_method": "perspective_correct_triangle_zbuffer",
    }
    return out, zbuf, out_labels, visibility, diag


def compose_surface_warps(
    radiance0: np.ndarray,
    depth0: np.ndarray,
    labels0: np.ndarray,
    visible0: np.ndarray,
    radiance1: np.ndarray,
    depth1: np.ndarray,
    labels1: np.ndarray,
    visible1: np.ndarray,
    alpha: float,
    *,
    floor: float = 1e-6,
    correspondence_rtol: float = 2e-3,
    correspondence_atol: float = 2e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Log-interpolate matching surfaces and depth-select true occlusions.

    One-sided support is retained as a provisional value but marked for a real
    render. No-source pixels stay NaN and are also render requests.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    arrays = [np.asarray(x) for x in (radiance0, depth0, labels0, visible0,
                                      radiance1, depth1, labels1, visible1)]
    shape = arrays[0].shape
    if any(x.shape != shape for x in arrays):
        raise ValueError("both projected surfaces must share target dimensions")
    a0, z0, l0, v0, a1, z1, l1, v1 = arrays
    v0 = v0.astype(bool) & np.isfinite(a0) & np.isfinite(z0)
    v1 = v1.astype(bool) & np.isfinite(a1) & np.isfinite(z1)
    out = np.full(shape, np.nan, dtype=np.float32)
    out_labels = np.full(shape, -1, dtype=np.int64)
    has = v0 | v1
    one_sided = v0 ^ v1
    both = v0 & v1
    same_surface = both & (l0 == l1) & (l0 >= 0)
    same_surface &= np.isclose(z0, z1, rtol=float(correspondence_rtol),
                               atol=float(correspondence_atol))
    coplanar_label_conflict = both & (l0 != l1) & np.isclose(
        z0, z1, rtol=float(correspondence_rtol), atol=float(correspondence_atol)
    )
    coplanar_label_conflict |= both & ((l0 == -2) | (l1 == -2))
    x0 = np.log(np.maximum(np.asarray(a0, dtype=np.float32), float(floor)))
    x1 = np.log(np.maximum(np.asarray(a1, dtype=np.float32), float(floor)))
    out[same_surface] = np.exp((1.0 - float(alpha)) * x0[same_surface]
                                + float(alpha) * x1[same_surface])
    out_labels[same_surface] = l0[same_surface].astype(np.int64)
    choose0 = both & ~same_surface & (z0 < z1)
    choose1 = both & ~same_surface & ~choose0
    out[choose0] = a0[choose0]
    out[choose1] = a1[choose1]
    out_labels[choose0] = l0[choose0].astype(np.int64)
    out_labels[choose1] = l1[choose1].astype(np.int64)
    out[v0 & ~v1] = a0[v0 & ~v1]
    out[v1 & ~v0] = a1[v1 & ~v0]
    out_labels[v0 & ~v1] = l0[v0 & ~v1].astype(np.int64)
    out_labels[v1 & ~v0] = l1[v1 & ~v0].astype(np.int64)
    render_request = ~has | one_sided | coplanar_label_conflict
    diagnostics = {
        "same_surface_fraction": float(same_surface.mean()),
        "depth_occlusion_selection_fraction": float((choose0 | choose1).mean()),
        "one_sided_support_fraction": float(one_sided.mean()),
        "no_source_fraction": float((~has).mean()),
        "coplanar_label_conflict_fraction": float(coplanar_label_conflict.mean()),
        "render_request_fraction": float(render_request.mean()),
        "target_labels": out_labels,
    }
    return out, has, render_request, diagnostics
