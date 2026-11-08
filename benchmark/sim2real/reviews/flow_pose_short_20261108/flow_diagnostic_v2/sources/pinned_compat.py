"""Narrow STE-FlowNet grid-sampling compatibility patch.

The pinned upstream functions normalize integer pixel coordinates with
2*x/(W-1)-1, which is the align_corners=True convention. PyTorch defaults to
False, so the model's internal warp and the photometric-loss warp must state
True explicitly. The training augmentation uses pixel-center coordinates and
align_corners=False; it is deliberately untouched.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import torch
import torch.nn.functional as F


PINNED_SHA256 = {
    "multiscaleloss.py": "ff401317e49ce37ae0f8e9bfd6b82f133549b4a0271be35095f74bdc50a439c0",
    "models/STEFlow.py": "eaf1f02ba5addc319c9736307960172cb2e3693a714d536d2bad478c0964c785",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_upstream(root: Path) -> dict[str, str]:
    actual = {}
    for relative, expected in PINNED_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned STE-FlowNet source is missing: {path}")
        actual[relative] = sha256(path)
        if actual[relative] != expected:
            raise RuntimeError(f"pinned STE-FlowNet hash mismatch for {relative}: {actual[relative]}")
    return actual


def loss_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or flow.shape != (x.shape[0], 2, x.shape[2], x.shape[3]):
        raise ValueError(f"STE loss warp expects x [B,C,H,W] and flow [B,2,H,W], got {x.shape}, {flow.shape}")
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=flow.dtype),
        torch.arange(w, device=x.device, dtype=flow.dtype), indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(b, -1, -1, -1) + flow
    grid[:, 0] = 2.0 * grid[:, 0] / max(w - 1, 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / max(h - 1, 1) - 1.0
    grid = grid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    mask = F.grid_sample(torch.ones_like(x[:, :1]), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    mask = (mask >= 0.9999).to(output.dtype)
    return output * mask


def model_flow_warp(self, x: torch.Tensor, flow12: torch.Tensor, pad="border", mode="bilinear") -> torch.Tensor:
    if x.ndim != 4 or flow12.shape != (x.shape[0], 2, x.shape[2], x.shape[3]):
        raise ValueError(f"STE model warp expects x [B,C,H,W] and flow [B,2,H,W], got {x.shape}, {flow12.shape}")
    b, _, h, w = x.shape
    base = self.mesh_grid(b, h, w).to(device=x.device, dtype=x.dtype)
    grid = self.norm_grid(base + flow12)
    return F.grid_sample(x, grid, mode=mode, padding_mode=pad, align_corners=True)


def patch_ste(models_module, loss_module, upstream_root: Path) -> dict:
    hashes = validate_upstream(upstream_root)
    loss_module.warp = loss_warp
    model_impl = importlib.import_module("models.STEFlow")
    model_impl.FlowNetS_spike.flow_warp = model_flow_warp
    return {
        "upstream_hashes": hashes,
        "loss_warp": "2*x/(W-1)-1 with align_corners=True",
        "model_flow_warp": "STE norm_grid 2*x/(W-1)-1 with align_corners=True",
        "augmentation": "unchanged: pixel-center normalized grid with align_corners=False",
    }
