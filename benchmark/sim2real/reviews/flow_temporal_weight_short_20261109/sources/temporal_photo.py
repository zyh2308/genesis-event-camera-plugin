"""Training-only spatial weights for the pinned STE photometric residual."""
from __future__ import annotations

import cv2
import numpy as np
import torch

CONTRAST_CAP = 0.05


def endpoint_weights(previous, following):
    """Normalize each window independently, preserving its total pixel mass."""
    if previous.shape != following.shape or previous.ndim != 4 or previous.shape[1] != 1:
        raise ValueError('Aligned B×1×H×W endpoint tensors required')
    for image in (previous, following):
        tolerance = 8 * torch.finfo(image.dtype).eps
        if not bool(torch.isfinite(image).all() & (image.min() >= -tolerance) & (image.max() <= 1 + tolerance)):
            raise ValueError('Interpolated endpoints must stay in finite [0,1] within rounding error')
    with torch.no_grad():
        raw = 1 + ((following - previous).abs() / CONTRAST_CAP).clamp_max(1)
        return raw / raw.mean(dim=(-2, -1), keepdim=True)


def prepare(previous_cpu, following_cpu, outputs):
    """Use the original OpenCV linear image pyramid at the four square sizes."""
    previous = np.asarray(previous_cpu)
    following = np.asarray(following_cpu)
    if previous.shape != following.shape or previous.ndim != 3 or len(outputs) != 4:
        raise ValueError('Four native flow scales and aligned B×H×W gray required')
    levels = []
    for flow in outputs:
        batch, channels, height, width = flow.shape
        if batch != len(previous) or channels != 2 or height != width or flow.dtype != torch.float32:
            raise ValueError('Unexpected pinned float32 square flow pyramid')
        def resized(images):
            result = np.stack([cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR) for image in images])
            return torch.from_numpy(result.astype(np.float32, copy=False)).unsqueeze(1).to(flow.device)
        first, last = resized(previous), resized(following)
        levels.append((first, last, endpoint_weights(first, last)))
    return levels


def photometric_terms(outputs, levels, warp, level_weights):
    """Keep the original warp, Charbonnier and four-scale sum normalization."""
    if len(outputs) != 4 or len(levels) != 4 or tuple(level_weights) != (1., 1., 1., 1.):
        raise ValueError('Only the registered four-scale native recipe is supported')
    native = 0.
    weighted = 0.
    for index, (flow, (previous, following, weight)) in enumerate(zip(outputs, levels)):
        residual = warp(following, flow) - previous
        values = torch.pow(torch.mul(residual, residual) + torch.mul(1e-3, 1e-3), .45)
        coefficient = level_weights[len(level_weights) - index - 1]
        native = native + coefficient * values.sum()
        weighted = weighted + coefficient * (weight * values).sum()
    return native / len(outputs), weighted / len(outputs)
