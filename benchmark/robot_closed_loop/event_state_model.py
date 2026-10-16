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


def load_checkpoint(path: str, device: str = "cpu") -> tuple[SpatialEventStateNet, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != SpatialEventStateNet.architecture:
        raise ValueError(f"unsupported event-state architecture: {checkpoint.get('architecture')}")
    model = SpatialEventStateNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = torch.as_tensor(checkpoint["label_mean"], dtype=torch.float32, device=device)
    scale = torch.as_tensor(checkpoint["label_scale"], dtype=torch.float32, device=device)
    return model, mean, scale, checkpoint
