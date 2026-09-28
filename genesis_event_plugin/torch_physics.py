"""Torch/CUDA reference kernels for the v2 physics path.

The NumPy implementation remains the numerical reference.  These kernels
avoid the per-pixel Python loops and are intended for the later integrated
training path; they return tensors so a future tensor-native DVS step can
avoid GPU↔CPU copies.
"""

from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


def prepare_linear_radiance_torch(
    image: torch.Tensor,
    input_space: str = "linear",
    weights=(0.2126, 0.7152, 0.0722),
) -> torch.Tensor:
    if not torch.is_tensor(image):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[-1] != 3):
        raise ValueError(f"expected HxW or HxWx3, got {tuple(image.shape)}")
    x = image.to(dtype=torch.float32)
    if image.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        x = x / 255.0
    if input_space == "srgb":
        x = torch.clamp(x, 0.0, 1.0)
        x = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    elif input_space != "linear":
        raise ValueError("input_space must be 'linear' or 'srgb'")
    if not torch.isfinite(x).all() or bool((x < 0).any()):
        raise ValueError("radiance must be finite and non-negative")
    if x.ndim == 2:
        return x
    w = torch.as_tensor(weights, dtype=x.dtype, device=x.device)
    return torch.tensordot(x, w, dims=([-1], [0]))


def compute_se3_flow_torch(
    depth0: torch.Tensor,
    seg0: torch.Tensor,
    K: torch.Tensor,
    camera_to_world0: torch.Tensor,
    camera_to_world1: torch.Tensor,
    object_world_deltas: Optional[Mapping[int, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA-friendly counterpart of :func:`compute_se3_flow`."""
    if depth0.ndim != 2 or seg0.shape != depth0.shape:
        raise ValueError("depth0 and seg0 must be HxW")
    device = depth0.device
    dtype = torch.float32
    depth = depth0.to(dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    T0 = camera_to_world0.to(device=device, dtype=dtype)
    T1 = camera_to_world1.to(device=device, dtype=dtype)
    H, W = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij"
    )
    p0 = torch.stack(
        ((xx - K[0, 2]) * depth / K[0, 0],
         (yy - K[1, 2]) * depth / K[1, 1], depth), dim=-1
    ).reshape(-1, 3)
    ones = torch.ones((p0.shape[0], 1), device=device, dtype=dtype)
    p0h = torch.cat((p0, ones), dim=1)
    p0w = (T0 @ p0h.T).T[:, :3]
    p1w = p0w.clone()
    if object_world_deltas:
        flat_seg = seg0.reshape(-1)
        for object_id, delta in object_world_deltas.items():
            mask = flat_seg == int(object_id)
            if bool(mask.any()):
                Td = delta.to(device=device, dtype=dtype)
                p1w[mask] = (Td @ torch.cat((p0w[mask], torch.ones((int(mask.sum()), 1), device=device, dtype=dtype)), dim=1).T).T[:, :3]
    p1c = (torch.linalg.inv(T1) @ torch.cat((p1w, ones), dim=1).T).T[:, :3]
    z1 = p1c[:, 2]
    valid = torch.isfinite(p1c).all(dim=1) & torch.isfinite(depth.reshape(-1))
    valid &= (depth.reshape(-1) > 1e-6) & (z1 > 1e-6)
    u1 = torch.zeros_like(z1)
    v1 = torch.zeros_like(z1)
    u1[valid] = K[0, 0] * p1c[valid, 0] / z1[valid] + K[0, 2]
    v1[valid] = K[1, 1] * p1c[valid, 1] / z1[valid] + K[1, 2]
    flow = torch.stack((u1 - xx.reshape(-1), v1 - yy.reshape(-1)), dim=0).reshape(2, H, W)
    flow = torch.where(valid.reshape(1, H, W), flow, torch.zeros_like(flow))
    return flow, valid.reshape(H, W)


def _splat_torch(image, depth, flow, amount):
    H, W = depth.shape
    scalar = image.ndim == 2
    src = image[..., None] if scalar else image
    C = src.shape[-1]
    device = src.device
    dtype = src.dtype
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij"
    )
    tx = xx + flow[0].to(dtype) * amount
    ty = yy + flow[1].to(dtype) * amount
    x0, y0 = torch.floor(tx).long(), torch.floor(ty).long()
    wx, wy = tx - x0, ty - y0
    valid_src = torch.isfinite(src).all(dim=-1) & torch.isfinite(depth) & (depth > 1e-6)
    zbuf = torch.full((H * W,), float("inf"), device=device, dtype=dtype)
    candidates = []
    for ox, oy, weight in ((0, 0, (1-wx)*(1-wy)), (1, 0, wx*(1-wy)),
                            (0, 1, (1-wx)*wy), (1, 1, wx*wy)):
        x, y = x0 + ox, y0 + oy
        valid = valid_src & (weight > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if bool(valid.any()):
            idx = (y[valid] * W + x[valid]).long()
            candidates.append((idx, weight[valid], depth[valid], src[valid]))
            zbuf.scatter_reduce_(0, idx, depth[valid], reduce="amin", include_self=True)
    out = torch.zeros((H * W, C), device=device, dtype=dtype)
    norm = torch.zeros((H * W, 1), device=device, dtype=dtype)
    for idx, weight, z, values in candidates:
        keep = z <= zbuf[idx] + 1e-5
        if bool(keep.any()):
            idx, weight, values = idx[keep], weight[keep], values[keep]
            out.scatter_add_(0, idx[:, None].expand(-1, C), values * weight[:, None])
            norm.scatter_add_(0, idx[:, None], weight[:, None])
    out = out / torch.clamp(norm, min=1e-8)
    valid = norm[:, 0] > 1e-8
    out = out.reshape(H, W, C)
    return (out[..., 0] if scalar else out), valid.reshape(H, W)


def depth_aware_bidirectional_warp_torch(
    frame0, depth0, flow01, frame1, depth1, flow10, alpha: float,
    composite: str = "weighted",
):
    if composite not in {"weighted", "next_primary"}:
        raise ValueError("composite must be 'weighted' or 'next_primary'")
    a0, v0 = _splat_torch(frame0, depth0, flow01, alpha)
    a1, v1 = _splat_torch(frame1, depth1, flow10, 1.0 - alpha)
    if composite == "next_primary":
        valid = v0 | v1
        if frame0.ndim == 2:
            out = torch.full_like(a0, float("nan"))
            out = torch.where(v1, a1, torch.where(v0, a0, out))
        else:
            out = torch.full_like(a0, float("nan"))
            mask = v1[..., None]
            out = torch.where(mask, a1, torch.where(v0[..., None], a0, out))
        return out, valid
    w0 = (1.0 - alpha) * v0.to(a0.dtype)
    w1 = alpha * v1.to(a1.dtype)
    denom = w0 + w1
    if frame0.ndim == 2:
        out = (a0 * w0 + a1 * w1) / torch.clamp(denom, min=1e-8)
    else:
        out = (a0 * w0[..., None] + a1 * w1[..., None]) / torch.clamp(denom[..., None], min=1e-8)
    return out, denom > 1e-8
