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


def _so3_log_torch(rotation: torch.Tensor) -> torch.Tensor:
    """Stable logarithm of one 3x3 rotation matrix."""
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    vee = 0.5 * torch.stack((
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ))
    scale = theta / torch.clamp(torch.sin(theta), min=1e-6)
    return torch.where(theta < 1e-5, vee, vee * scale)


def _so3_exp_batch_torch(rotation_vectors: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential for a ``[T,3]`` stack of rotation vectors."""
    theta = torch.linalg.norm(rotation_vectors, dim=-1)
    x, y, z = rotation_vectors.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    K = torch.stack((
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ), dim=-1).reshape(-1, 3, 3)
    theta2 = theta * theta
    a = torch.where(theta < 1e-5, 1.0 - theta2 / 6.0, torch.sin(theta) / torch.clamp(theta, min=1e-6))
    b = torch.where(theta < 1e-5, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / torch.clamp(theta2, min=1e-6))
    eye = torch.eye(3, dtype=rotation_vectors.dtype, device=rotation_vectors.device).expand(rotation_vectors.shape[0], -1, -1)
    return eye + a[:, None, None] * K + b[:, None, None] * torch.bmm(K, K)


def interpolate_se3_stack_torch(
    camera_to_world0: torch.Tensor,
    camera_to_world1: torch.Tensor,
    alphas: torch.Tensor,
) -> torch.Tensor:
    """GPU endpoint SE(3) interpolation for a stack of alphas."""
    T0 = camera_to_world0.to(dtype=torch.float32)
    T1 = camera_to_world1.to(device=T0.device, dtype=torch.float32)
    a = alphas.to(device=T0.device, dtype=T0.dtype).reshape(-1)
    rel = T0[:3, :3].transpose(0, 1) @ T1[:3, :3]
    w = _so3_log_torch(rel)
    Rinc = _so3_exp_batch_torch(a[:, None] * w[None, :])
    R = torch.einsum("ij,tjk->tik", T0[:3, :3], Rinc)
    trans = (1.0 - a[:, None]) * T0[:3, 3] + a[:, None] * T1[:3, 3]
    output = torch.eye(4, dtype=T0.dtype, device=T0.device).expand(a.shape[0], -1, -1).clone()
    output[:, :3, :3] = R
    output[:, :3, 3] = trans
    return output


def project_endpoint_to_alpha_stack_torch(
    depth_source: torch.Tensor,
    seg_source: torch.Tensor,
    K: torch.Tensor,
    camera_source: torch.Tensor,
    camera0: torch.Tensor,
    camera1: torch.Tensor,
    alphas: torch.Tensor,
    source_index: int = 0,
    object_world0: Optional[Mapping[int, torch.Tensor]] = None,
    object_world1: Optional[Mapping[int, torch.Tensor]] = None,
    object_world_deltas: Optional[Mapping[int, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one endpoint at all alphas on-device.

    This is the Torch counterpart of the NumPy ``project_endpoint_to_alpha``
    reference.  It keeps the alpha stack on the selected device; object IDs
    are the only Python-level loop because Genesis segmentation labels are a
    sparse mapping, not a dense tensor dimension.
    """
    if depth_source.ndim != 2 or seg_source.shape != depth_source.shape:
        raise ValueError("depth_source and seg_source must be HxW")
    device = depth_source.device
    dtype = torch.float32
    depth = depth_source.to(dtype=dtype)
    seg = seg_source.to(device=device)
    K = K.to(device=device, dtype=dtype)
    source = camera_source.to(device=device, dtype=dtype)
    camera_stack = interpolate_se3_stack_torch(camera0.to(device=device), camera1.to(device=device), alphas)
    H, W = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij"
    )
    p_cam = torch.stack(((xx - K[0, 2]) * depth / K[0, 0],
                         (yy - K[1, 2]) * depth / K[1, 1], depth), dim=-1).reshape(-1, 3)
    ones = torch.ones((p_cam.shape[0], 1), dtype=dtype, device=device)
    p_world = (source @ torch.cat((p_cam, ones), dim=1).T).T[:, :3]
    p_world_stack = p_world[None].expand(camera_stack.shape[0], -1, -1).clone()
    object_ids = set()
    for mapping in (object_world0, object_world1, object_world_deltas):
        if mapping is not None:
            object_ids.update(mapping.keys())
    flat_seg = seg.reshape(-1)
    for object_id in object_ids:
        mask = flat_seg == int(object_id)
        if not bool(mask.any()):
            continue
        if object_world0 is not None and object_world1 is not None and object_id in object_world0 and object_id in object_world1:
            O0 = object_world0[object_id].to(device=device, dtype=dtype)
            O1 = object_world1[object_id].to(device=device, dtype=dtype)
            O_stack = interpolate_se3_stack_torch(O0, O1, alphas)
            source_object = O0 if source_index == 0 else O1
            motion = torch.bmm(O_stack, torch.linalg.inv(source_object).expand(O_stack.shape[0], -1, -1))
        elif object_world_deltas is not None and object_id in object_world_deltas:
            delta = object_world_deltas[object_id].to(device=device, dtype=dtype)
            identity = torch.eye(4, dtype=dtype, device=device)
            O_stack = interpolate_se3_stack_torch(identity, delta, alphas)
            motion = O_stack if source_index == 0 else torch.bmm(O_stack, torch.linalg.inv(delta).expand(O_stack.shape[0], -1, -1))
        else:
            continue
        points = torch.cat((p_world[mask], torch.ones((int(mask.sum()), 1), dtype=dtype, device=device)), dim=1)
        p_world_stack[:, mask] = torch.bmm(motion, points.T[None].expand(motion.shape[0], -1, -1)).transpose(1, 2)[:, :, :3]
    camera_inv = torch.linalg.inv(camera_stack)
    homogeneous = torch.cat((p_world_stack, torch.ones((*p_world_stack.shape[:2], 1), dtype=dtype, device=device)), dim=-1)
    p_alpha_cam = torch.einsum("tij,tnj->tni", camera_inv, homogeneous)[..., :3]
    z = p_alpha_cam[..., 2]
    base_valid = torch.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 1e-6)
    valid = torch.isfinite(p_alpha_cam).all(dim=-1) & base_valid[None, :] & (z > 1e-6)
    u = torch.where(valid, K[0, 0] * p_alpha_cam[..., 0] / torch.clamp(z, min=1e-6) + K[0, 2], torch.zeros_like(z))
    v = torch.where(valid, K[1, 1] * p_alpha_cam[..., 1] / torch.clamp(z, min=1e-6) + K[1, 2], torch.zeros_like(z))
    uv = torch.stack((u, v), dim=1).reshape(-1, 2, H, W)
    return uv, z.reshape(-1, H, W), valid.reshape(-1, H, W)


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


def _splat_projected_batch_torch(
    image: torch.Tensor,
    source_depth: torch.Tensor,
    target_uv: torch.Tensor,
    target_z: torch.Tensor,
    source_valid: Optional[torch.Tensor] = None,
    return_diagnostics: bool = False,
):
    """Forward-splat ``B*T`` projected frames without CPU round-trips.

    ``image`` is ``[M,H,W]`` or ``[M,H,W,C]`` and the remaining arguments are
    ``[M,H,W]``/``[M,2,H,W]``.  The implementation uses one flattened z-buffer
    per batch/sample and four bilinear candidates.  The small four-way Python
    loop is over stencil offsets, not over pixels, environments, or temporal
    samples, so it remains a tensor-native kernel on CUDA.
    """
    if image.ndim not in (3, 4):
        raise ValueError("image must have shape [M,H,W] or [M,H,W,C]")
    if source_depth.ndim != 3:
        raise ValueError("source_depth must have shape [M,H,W]")
    if target_uv.ndim != 4 or target_uv.shape[1] != 2:
        raise ValueError("target_uv must have shape [M,2,H,W]")
    if target_z.ndim != 3 or target_z.shape != source_depth.shape:
        raise ValueError("target_z must have shape [M,H,W]")
    M, H, W = source_depth.shape
    if image.shape[:3] != (M, H, W) or target_uv.shape[0] != M:
        raise ValueError("projected batch dimensions do not agree")
    scalar = image.ndim == 3
    src = image.unsqueeze(-1) if scalar else image
    C = src.shape[-1]
    dtype = src.dtype
    device = src.device
    source_depth = source_depth.to(device=device, dtype=dtype)
    target_uv = target_uv.to(device=device, dtype=dtype)
    target_z = target_z.to(device=device, dtype=dtype)
    if source_valid is None:
        source_valid = torch.ones((M, H, W), dtype=torch.bool, device=device)
    else:
        source_valid = source_valid.to(device=device, dtype=torch.bool)
    if source_valid.shape != (M, H, W):
        raise ValueError("source_valid must have shape [M,H,W]")

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij"
    )
    tx, ty = target_uv[:, 0], target_uv[:, 1]
    x0, y0 = torch.floor(tx).long(), torch.floor(ty).long()
    wx, wy = tx - x0, ty - y0
    finite_values = torch.isfinite(src).all(dim=-1)
    valid_src = source_valid & finite_values
    valid_src &= torch.isfinite(source_depth) & (source_depth > 1e-6)
    valid_src &= torch.isfinite(target_z) & (target_z > 1e-6)

    offsets = (
        (0, 0, (1.0 - wx) * (1.0 - wy)),
        (1, 0, wx * (1.0 - wy)),
        (0, 1, (1.0 - wx) * wy),
        (1, 1, wx * wy),
    )
    batch_offset = (torch.arange(M, device=device, dtype=torch.long) * (H * W))[:, None, None]
    total = M * H * W
    zbuf = torch.full((total,), float("inf"), device=device, dtype=dtype)
    occupancy = torch.zeros((total,), dtype=torch.bool, device=device)
    candidates = []
    for ox, oy, weight in offsets:
        x, y = x0 + ox, y0 + oy
        valid = valid_src & (weight > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if bool(valid.any()):
            idx = (batch_offset + y * W + x)[valid]
            w = weight[valid]
            z = target_z[valid]
            values = src[valid]
            candidates.append((idx, w, z, values))
            occupancy.scatter_(0, idx, True)
            zbuf.scatter_reduce_(0, idx, z, reduce="amin", include_self=True)

    out = torch.zeros((total, C), device=device, dtype=dtype)
    norm = torch.zeros((total, 1), device=device, dtype=dtype)
    for idx, weight, z, values in candidates:
        keep = z <= zbuf[idx] + 1e-5
        if bool(keep.any()):
            idx_k = idx[keep]
            w_k = weight[keep]
            v_k = values[keep]
            out.scatter_add_(0, idx_k[:, None].expand(-1, C), v_k * w_k[:, None])
            norm.scatter_add_(0, idx_k[:, None], w_k[:, None])
    out = out / torch.clamp(norm, min=1e-8)
    valid = norm[:, 0] > 1e-8
    out = out.reshape(M, H, W, C)
    valid = valid.reshape(M, H, W)
    output = (out[..., 0] if scalar else out)
    if return_diagnostics:
        return output, valid, {
            "occupancy": occupancy.reshape(M, H, W),
            "visibility": valid,
            "z_buffer": zbuf.reshape(M, H, W),
        }
    return output, valid


def depth_aware_bidirectional_warp_projected_batch_torch(
    frame0: torch.Tensor,
    depth0: torch.Tensor,
    uv0: torch.Tensor,
    z0: torch.Tensor,
    valid0: torch.Tensor,
    frame1: torch.Tensor,
    depth1: torch.Tensor,
    uv1: torch.Tensor,
    z1: torch.Tensor,
    valid1: torch.Tensor,
    composite: str = "next_primary",
    return_diagnostics: bool = False,
):
    """Warp ``B`` endpoint frames at ``T`` projected intermediate poses.

    Endpoint images are ``[B,H,W]`` or ``[B,H,W,C]``.  Projection tensors are
    ``[B,T,2,H,W]`` and ``[B,T,H,W]``.  Returned scalar frames are
    ``[B,T,H,W]``; RGB frames are ``[B,T,H,W,C]``.  All splatting, z-buffering,
    compositing and validity masks stay on the input device. With
    ``return_diagnostics=True`` a third dictionary contains target-domain
    occupancy, visibility, z-buffer and intermediate-unknown masks.
    """
    if composite not in {"weighted", "next_primary"}:
        raise ValueError("composite must be 'weighted' or 'next_primary'")
    if frame0.shape != frame1.shape or depth0.shape != depth1.shape:
        raise ValueError("endpoint frame/depth shapes must agree")
    if depth0.ndim != 3:
        raise ValueError("depth tensors must have shape [B,H,W]")
    B, H, W = depth0.shape
    if (
        uv0.ndim != 5 or uv0.shape[0] != B or uv0.shape[2] != 2
        or uv0.shape[3:] != (H, W)
    ):
        raise ValueError("uv tensors must have shape [B,T,2,H,W]")
    T = uv0.shape[1]
    for value, name in ((uv1, "uv1"), (z0, "z0"), (z1, "z1"),
                        (valid0, "valid0"), (valid1, "valid1")):
        expected = (B, T, 2, H, W) if name == "uv1" else (B, T, H, W)
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if frame0.ndim == 4 and frame0.shape[:3] != (B, H, W):
        raise ValueError("RGB frame tensors must have shape [B,H,W,C]")

    # Flatten the temporal dimension into the same batched z-buffer.  The
    # endpoint is repeated as a view-like expand, then made contiguous once.
    def flatten_endpoint(frame, depth, uv, z, valid):
        M = B * T
        frame_m = frame[:, None].expand((B, T) + tuple(frame.shape[1:])).reshape((M,) + tuple(frame.shape[1:]))
        depth_m = depth[:, None].expand(B, T, H, W).reshape(M, H, W)
        uv_m = uv.reshape(M, 2, H, W)
        z_m = z.reshape(M, H, W)
        valid_m = valid.reshape(M, H, W)
        return _splat_projected_batch_torch(
            frame_m, depth_m, uv_m, z_m, valid_m,
            return_diagnostics=return_diagnostics,
        )

    result0 = flatten_endpoint(frame0, depth0, uv0, z0, valid0)
    result1 = flatten_endpoint(frame1, depth1, uv1, z1, valid1)
    if return_diagnostics:
        warped0, visible0, diagnostics0 = result0
        warped1, visible1, diagnostics1 = result1
    else:
        warped0, visible0 = result0
        warped1, visible1 = result1
    if composite == "next_primary":
        visible = visible0 | visible1
        if warped0.ndim == 3:
            output = torch.full_like(warped0, float("nan"))
            output = torch.where(visible1, warped1, torch.where(visible0, warped0, output))
        else:
            output = torch.full_like(warped0, float("nan"))
            mask1, mask0 = visible1[..., None], visible0[..., None]
            output = torch.where(mask1, warped1, torch.where(mask0, warped0, output))
    else:
        w0 = visible0.to(warped0.dtype)
        w1 = visible1.to(warped1.dtype)
        denom = w0 + w1
        if warped0.ndim == 3:
            output = (warped0 * w0 + warped1 * w1) / torch.clamp(denom, min=1e-8)
        else:
            output = (warped0 * w0[..., None] + warped1 * w1[..., None]) / torch.clamp(denom[..., None], min=1e-8)
        visible = denom > 1e-8
    if output.ndim == 3:
        output = output.reshape(B, T, H, W)
    else:
        output = output.reshape(B, T, H, W, output.shape[-1])
    visible = visible.reshape(B, T, H, W)
    if return_diagnostics:
        diagnostics = {
            "occupancy0": diagnostics0["occupancy"].reshape(B, T, H, W),
            "occupancy1": diagnostics1["occupancy"].reshape(B, T, H, W),
            "visibility0": diagnostics0["visibility"].reshape(B, T, H, W),
            "visibility1": diagnostics1["visibility"].reshape(B, T, H, W),
            "z_buffer0": diagnostics0["z_buffer"].reshape(B, T, H, W),
            "z_buffer1": diagnostics1["z_buffer"].reshape(B, T, H, W),
            "intermediate_visibility": visible,
            "disocclusion": ~visible,
        }
        return output, visible, diagnostics
    return output, visible
