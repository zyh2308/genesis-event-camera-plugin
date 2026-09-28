"""Conservative over-render margins for fast moving Genesis cameras.

Forward warping can expose pixels that were outside the nominal camera view at
the previous keyframe.  Padding the render target gives those pixels a chance
to be rendered before the event path is cropped back to the sensor window.
This module only describes the camera contract; the caller must create the
Genesis camera with the expanded resolution and then crop the returned frame.
"""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CameraMargin:
    """Pixel padding around the nominal sensor image.

    ``resolution_wh`` follows Genesis' ``(width, height)`` convention.  The
    principal point must be shifted by ``(left, top)`` in the expanded image.
    """

    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0

    def __post_init__(self):
        values = (self.left, self.right, self.top, self.bottom)
        if any(int(v) != v or int(v) < 0 for v in values):
            raise ValueError("camera margins must be non-negative integers")

    @property
    def horizontal(self) -> int:
        return int(self.left + self.right)

    @property
    def vertical(self) -> int:
        return int(self.top + self.bottom)

    def expanded_resolution(self, resolution_wh: Tuple[int, int]) -> Tuple[int, int]:
        width, height = (int(resolution_wh[0]), int(resolution_wh[1]))
        if width <= 0 or height <= 0:
            raise ValueError("resolution must be positive")
        return width + self.horizontal, height + self.vertical

    def validate_for(self, resolution_wh: Tuple[int, int]) -> None:
        width, height = (int(resolution_wh[0]), int(resolution_wh[1]))
        if width <= 0 or height <= 0:
            raise ValueError("resolution must be positive")
        if self.left + self.right >= width or self.top + self.bottom >= height:
            raise ValueError("camera margin consumes the nominal image")


def estimate_required_margin(
    flow: np.ndarray,
    safety_px: float = 1.0,
    percentile: float = 100.0,
) -> CameraMargin:
    """Estimate a conservative symmetric over-render margin from flow.

    The flow is the whole-interval displacement in pixels with shape
    ``(2,H,W)``.  We use the maximum absolute displacement (or a chosen
    percentile) independently in x/y, then add ``safety_px``.  Symmetric
    padding is intentionally conservative and robust to the unknown direction
    of newly exposed regions; later Genesis-native variants may use directional
    margins once the renderer's clip/crop semantics are exposed.
    """

    value = np.asarray(flow, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 2:
        raise ValueError("flow must have shape (2,H,W)")
    if not 0 < float(percentile) <= 100:
        raise ValueError("percentile must be in (0,100]")
    if float(safety_px) < 0:
        raise ValueError("safety_px must be non-negative")
    finite = np.isfinite(value)
    if not finite.any():
        return CameraMargin()
    x = np.abs(value[0][finite[0]])
    y = np.abs(value[1][finite[1]])
    max_x = float(np.percentile(x, percentile)) if x.size else 0.0
    max_y = float(np.percentile(y, percentile)) if y.size else 0.0
    return CameraMargin(
        left=int(math.ceil(max_x + safety_px)),
        right=int(math.ceil(max_x + safety_px)),
        top=int(math.ceil(max_y + safety_px)),
        bottom=int(math.ceil(max_y + safety_px)),
    )


def pad_intrinsics(K: np.ndarray, margin: CameraMargin) -> np.ndarray:
    """Return intrinsics for an expanded render target."""

    matrix = np.asarray(K, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("K must be a finite 3x3 matrix")
    shifted = matrix.copy()
    shifted[0, 2] += margin.left
    shifted[1, 2] += margin.top
    return shifted


def crop_margin(image: np.ndarray, margin: CameraMargin) -> np.ndarray:
    """Crop an expanded image back to the nominal sensor window."""

    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    height, width = array.shape[:2]
    y0, y1 = margin.top, height - margin.bottom
    x0, x1 = margin.left, width - margin.right
    if y0 >= y1 or x0 >= x1:
        raise ValueError("camera margin consumes the image")
    return array[y0:y1, x0:x1, ...]

