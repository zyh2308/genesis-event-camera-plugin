"""Narrow Stage-5 compatibility adapter for the pinned E2VID TC loss.

Upstream normalizes pixel coordinates with W-1/H-1, which is the
``align_corners=True`` convention. Its two ``grid_sample`` calls omit that
argument and therefore inherit PyTorch's False default. Keep upstream source
unchanged and replace only the instance's sampling function at the benchmark
boundary.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


COMPAT_VERSION = "e2vid_temporal_consistency_align_corners_true_v1"
UPSTREAM_RELATIVE_PATH = "utils/loss.py"
PINNED_UPSTREAM_SHA256 = "3d94483e3bc05582560efd9fa2d50f751ecff52321ec812ae04429b36c88aeb4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_upstream(model_root: Path) -> dict:
    path = model_root / UPSTREAM_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"pinned E2VID loss source is missing: {path}")
    actual = sha256_file(path)
    if actual != PINNED_UPSTREAM_SHA256:
        raise RuntimeError(
            f"pinned E2VID loss source hash mismatch: {actual} != {PINNED_UPSTREAM_SHA256}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "compat_version": COMPAT_VERSION,
        "upstream_delta": "explicit align_corners=True for both image0 and processed0 grid_sample calls",
    }


def temporal_consistency_loss_align_corners_true(
    image0: torch.Tensor,
    image1: torch.Tensor,
    processed0: torch.Tensor,
    processed1: torch.Tensor,
    flow01: torch.Tensor,
    alpha: float = 50.0,
    output_images: bool = False,
):
    """Upstream TC math with explicit pixel-center sampling convention.

    ``flow01`` is the displacement added to each image1/output pixel to sample
    image0/processed0. The E2VID wrapper reverses its forward flow before this
    function is called. Boundary padding remains the upstream zero padding.
    """
    tensors = (image0, image1, processed0, processed1)
    if any(t.ndim != 4 for t in tensors):
        raise ValueError("TC images must all have shape [B,C,H,W]")
    if any(t.shape != image0.shape for t in tensors[1:]):
        raise ValueError("TC image and processed-image shapes must match exactly")
    if image0.shape[-2] <= 1 or image0.shape[-1] <= 1:
        raise ValueError("TC spatial dimensions must both exceed one pixel")
    expected_flow = (image0.shape[0], 2, image0.shape[2], image0.shape[3])
    if tuple(flow01.shape) != expected_flow:
        raise ValueError(f"TC flow must have shape {expected_flow}, got {tuple(flow01.shape)}")
    if flow01.device != image0.device or any(t.device != image0.device for t in tensors[1:]):
        raise ValueError("TC images and flow must be on the same device")

    _, _, t_height, t_width = image0.shape
    xx, yy = torch.meshgrid(
        torch.arange(t_width, device=image0.device),
        torch.arange(t_height, device=image0.device),
        indexing="ij",
    )
    xx = xx.transpose(0, 1).to(dtype=flow01.dtype)
    yy = yy.transpose(0, 1).to(dtype=flow01.dtype)
    warping_grid_x = xx + flow01[:, 0]
    warping_grid_y = yy + flow01[:, 1]
    warping_grid_x = 2 * warping_grid_x / (t_width - 1) - 1
    warping_grid_y = 2 * warping_grid_y / (t_height - 1) - 1
    warping_grid = torch.stack((warping_grid_x, warping_grid_y), dim=3)

    image0_warped_to1 = F.grid_sample(
        image0, warping_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    visibility_mask = torch.exp(-alpha * (image1 - image0_warped_to1) ** 2)
    processed0_warped_to1 = F.grid_sample(
        processed0, warping_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    tc_map = visibility_mask * torch.abs(processed1 - processed0_warped_to1) / (
        torch.abs(processed1) + torch.abs(processed0_warped_to1) + 1e-5
    )
    tc_loss = tc_map.mean()
    if not output_images:
        return tc_loss
    return tc_loss, {
        "image0": image0,
        "image1": image1,
        "image0_warped_to1": image0_warped_to1,
        "processed0_warped_to1": processed0_warped_to1,
        "visibility_mask": visibility_mask,
        "error_map": tc_map,
    }


def patch_temporal_loss_instance(loss_instance, model_root: Path) -> dict:
    """Install the adapter on one upstream temporal-loss instance only."""
    upstream = validate_upstream(model_root)
    loss_instance.loss = temporal_consistency_loss_align_corners_true
    if loss_instance.loss is not temporal_consistency_loss_align_corners_true:
        raise RuntimeError("failed to install benchmark E2VID TC compatibility adapter")
    return upstream
