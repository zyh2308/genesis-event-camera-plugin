"""Small spatial event-to-state model and shared causal voxel encoder."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch import nn


INPUT_CONTRACT = "causal five 5ms bins, two polarities, 64x64, log1p counts; output state at cutoff"


def five_bin_tensor(events: np.ndarray, cutoff_s: float, width: int, height: int) -> np.ndarray:
    """Build (time, polarity, y, x) counts on [cutoff-25ms, cutoff)."""
    arr = np.asarray(events, dtype=np.float64).reshape(-1, 4)
    start = float(cutoff_s) - 0.025
    keep = (arr[:, 0] >= start) & (arr[:, 0] < float(cutoff_s))
    arr = arr[keep]
    output = np.zeros((5, 2, 64, 64), dtype=np.float32)
    if not len(arr):
        return output
    time_idx = np.floor((arr[:, 0] - start) / 0.005).astype(np.int64)
    x_idx = np.clip((arr[:, 1] * 64 / width).astype(np.int64), 0, 63)
    y_idx = np.clip((arr[:, 2] * 64 / height).astype(np.int64), 0, 63)
    polarity_idx = np.where(arr[:, 3] > 0, 0, 1)
    flat = (((time_idx * 2 + polarity_idx) * 64 + y_idx) * 64 + x_idx)
    return np.bincount(flat, minlength=5 * 2 * 64 * 64).reshape(output.shape).astype(np.float32)


class SpatialEventStateNet(nn.Module):
    """Compact CNN that preserves coarse image coordinates through its head."""

    architecture = "conv10-16-pool-conv16-24-pool-conv24-32-pool8x8-linear64-linear6-v1"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.GroupNorm(6, 24), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 6),
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(events))


class HeatmapEventStateNet(nn.Module):
    """Pixel-space target heatmap with a separate height/velocity head."""

    architecture = "event-heatmap64-aux-z-velocity-v2"

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16), nn.SiLU(),
        )
        self.stage32 = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.GroupNorm(6, 24), nn.SiLU(),
        )
        self.stage16 = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(72, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.motion_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 4),
        )

    def forward(self, events: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature64 = self.stem(events)
        feature32 = self.stage32(feature64)
        feature16 = self.stage16(feature32)
        fused = torch.cat((
            feature64,
            torch.nn.functional.interpolate(feature32, size=(64, 64), mode="bilinear", align_corners=False),
            torch.nn.functional.interpolate(feature16, size=(64, 64), mode="bilinear", align_corners=False),
        ), dim=1)
        return self.heatmap_head(fused)[:, 0], self.motion_head(feature16)


def _softargmax_uv(
    heatmap_logits: torch.Tensor,
    resolution_wh: tuple[int, int],
    temperature: float = 0.5,
) -> torch.Tensor:
    """Return continuous image coordinates at heatmap-cell centers."""
    batch, height, width = heatmap_logits.shape
    probabilities = torch.softmax(heatmap_logits.reshape(batch, -1) / temperature, dim=1)
    x_grid = (torch.arange(width, device=heatmap_logits.device, dtype=heatmap_logits.dtype) + 0.5)
    y_grid = (torch.arange(height, device=heatmap_logits.device, dtype=heatmap_logits.dtype) + 0.5)
    x_grid = x_grid * (float(resolution_wh[0]) / width)
    y_grid = y_grid * (float(resolution_wh[1]) / height)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
    return torch.stack((probabilities @ xx.reshape(-1), probabilities @ yy.reshape(-1)), dim=1)


def decode_heatmap_state(
    output: tuple[torch.Tensor, torch.Tensor],
    label_mean: torch.Tensor,
    label_scale: torch.Tensor,
    camera_to_world_cv: torch.Tensor,
    intrinsics: torch.Tensor,
    resolution_wh: tuple[int, int],
) -> torch.Tensor:
    """Decode UV plus predicted height into world xyz and velocity at cutoff."""
    heatmap_logits, auxiliary_normalized = output
    uv = _softargmax_uv(heatmap_logits, resolution_wh)
    auxiliary = auxiliary_normalized * label_scale[2:] + label_mean[2:]
    z = auxiliary[:, 0]
    fx, fy, cx, cy = intrinsics
    ray_camera = torch.stack(((uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy,
                              torch.ones_like(z)), dim=1)
    rotation = camera_to_world_cv[:3, :3]
    origin = camera_to_world_cv[:3, 3]
    ray_world = ray_camera @ rotation.T
    scale = (z - origin[2]) / ray_world[:, 2]
    position = origin[None, :] + scale[:, None] * ray_world
    velocity = auxiliary[:, 1:]
    return torch.cat((position, velocity), dim=1)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[SpatialEventStateNet, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture")
    model_types = {
        SpatialEventStateNet.architecture: SpatialEventStateNet,
        HeatmapEventStateNet.architecture: HeatmapEventStateNet,
    }
    if architecture not in model_types:
        raise ValueError(f"unsupported event-state architecture: {architecture}")
    model = model_types[architecture]().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = torch.as_tensor(checkpoint["label_mean"], dtype=torch.float32, device=device)
    scale = torch.as_tensor(checkpoint["label_scale"], dtype=torch.float32, device=device)
    return model, mean, scale, checkpoint
