"""Physics-constrained temporal sampling controller.

The controller operates on a displacement field over one endpoint interval.
It separates the contrast bound from visibility/motion guards and returns
diagnostics so experiments can report *why* an interval was oversampled.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .radiometry import log_radiance


@dataclass(frozen=True)
class AdaptiveSamplingConfig:
    contrast_threshold: float = 0.2
    safety_factor: float = 0.5
    min_samples: int = 2
    max_samples: int = 300
    flow_percentile: float = 100.0
    max_boundary_motion_px: float = 1.0
    max_relative_depth_change: float = 0.1
    max_camera_motion_px: float = 1.0
    max_articulation_motion_px: float = 1.0
    use_visibility_guard: bool = True
    strict: bool = True

    def validate(self) -> None:
        if self.contrast_threshold <= 0:
            raise ValueError("contrast_threshold must be positive")
        if not 0 < self.safety_factor <= 1:
            raise ValueError("safety_factor must be in (0, 1]")
        if self.min_samples < 1 or self.max_samples < self.min_samples:
            raise ValueError("invalid sample bounds")
        if not 0 < self.flow_percentile <= 100:
            raise ValueError("flow_percentile must be in (0, 100]")
        if self.max_boundary_motion_px <= 0:
            raise ValueError("max_boundary_motion_px must be positive")
        if self.max_relative_depth_change <= 0:
            raise ValueError("max_relative_depth_change must be positive")
        if self.max_camera_motion_px <= 0:
            raise ValueError("max_camera_motion_px must be positive")
        if self.max_articulation_motion_px <= 0:
            raise ValueError("max_articulation_motion_px must be positive")


@dataclass(frozen=True)
class AdaptiveSamplingResult:
    samples: int
    contrast_samples: int
    visibility_samples: int
    depth_samples: int
    max_log_change: float
    max_boundary_flow_px: float
    max_relative_depth_change: float
    camera_samples: int
    articulation_samples: int
    max_camera_flow_px: float
    max_articulation_flow_px: float
    uncapped_samples: int
    capped: bool


def _seg_boundary(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg)
    boundary = np.zeros(seg.shape, dtype=bool)
    boundary[1:] |= seg[1:] != seg[:-1]
    boundary[:-1] |= seg[:-1] != seg[1:]
    boundary[:, 1:] |= seg[:, 1:] != seg[:, :-1]
    boundary[:, :-1] |= seg[:, :-1] != seg[:, 1:]
    return boundary


def _motion_samples(
    flow: Optional[np.ndarray],
    max_motion_px: float,
    min_samples: int,
    percentile: float,
) -> tuple[int, float]:
    """Convert a pixel displacement field into a conservative sample bound.

    The bound is deliberately based on image displacement rather than RGB
    optical flow.  Camera and articulation motion therefore remain available
    even in textureless regions, which is the useful property of Genesis's
    physical state for event simulation.
    """
    if flow is None:
        return min_samples, 0.0
    value = np.asarray(flow, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 2:
        raise ValueError("motion flow must have shape (2,H,W)")
    magnitude = np.linalg.norm(value, axis=0)
    magnitude = magnitude[np.isfinite(magnitude)]
    if magnitude.size == 0:
        return min_samples, 0.0
    max_motion = float(np.percentile(magnitude, percentile))
    samples = max(min_samples, int(math.ceil(max_motion / max_motion_px)))
    return samples, max_motion


class PhysicsAdaptiveSampler:
    """Compute a conservative sample count for one rendered interval."""

    def __init__(self, config: AdaptiveSamplingConfig = None):
        self.config = config or AdaptiveSamplingConfig()
        self.config.validate()
        self.last_result: Optional[AdaptiveSamplingResult] = None

    def compute(
        self,
        luminance0: np.ndarray,
        flow_displacement: np.ndarray,
        dt: float,
        depth0: np.ndarray = None,
        depth1: np.ndarray = None,
        seg0: np.ndarray = None,
        seg1: np.ndarray = None,
        camera_flow: np.ndarray = None,
        articulation_flow: np.ndarray = None,
    ) -> AdaptiveSamplingResult:
        """Compute ``U`` from a per-interval pixel displacement field.

        ``flow_displacement`` has units of pixels over the whole interval.
        If a caller has velocity in px/s it must multiply by ``dt`` first;
        this explicit contract avoids the legacy implementation's ambiguity.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        lum = np.asarray(luminance0, dtype=np.float32)
        flow = np.asarray(flow_displacement, dtype=np.float32)
        if lum.ndim != 2 or flow.shape != (2,) + lum.shape:
            raise ValueError(
                f"expected luminance HxW and flow (2,H,W), got {lum.shape}, {flow.shape}"
            )
        log_lum = log_radiance(lum)
        gy, gx = np.gradient(log_lum)
        signal = np.abs(flow[0] * gx + flow[1] * gy)
        valid = np.isfinite(signal)
        signal = signal[valid]
        signal = signal[signal > 1e-8]
        if signal.size:
            max_log_change = float(np.percentile(signal, self.config.flow_percentile))
        else:
            max_log_change = 0.0
        contrast_samples = max(
            self.config.min_samples,
            int(math.ceil(max_log_change / (
                self.config.safety_factor * self.config.contrast_threshold
            ))) if max_log_change > 0 else self.config.min_samples,
        )

        visibility_samples = self.config.min_samples
        max_boundary_flow = 0.0
        if self.config.use_visibility_guard and seg0 is not None:
            boundary = _seg_boundary(np.asarray(seg0))
            flow_mag = np.linalg.norm(flow, axis=0)
            if np.any(boundary):
                max_boundary_flow = float(np.nanmax(flow_mag[boundary]))
                visibility_samples = max(
                    self.config.min_samples,
                    int(math.ceil(max_boundary_flow / self.config.max_boundary_motion_px)),
                )

        depth_samples = self.config.min_samples
        max_rel_depth = 0.0
        if (
            self.config.use_visibility_guard
            and depth0 is not None
            and depth1 is not None
        ):
            d0 = np.asarray(depth0, dtype=np.float32)
            d1 = np.asarray(depth1, dtype=np.float32)
            mask = np.isfinite(d0) & np.isfinite(d1) & (d0 > 1e-6) & (d1 > 1e-6)
            if np.any(mask):
                rel = np.abs(d1[mask] - d0[mask]) / np.maximum(d0[mask], 1e-6)
                max_rel_depth = float(np.nanmax(rel))
                depth_samples = max(
                    self.config.min_samples,
                    int(math.ceil(max_rel_depth / self.config.max_relative_depth_change)),
                )

        # Keep the four physical causes separate in the diagnostics.  The
        # final U is their maximum, so a fast camera or articulated link can
        # never be hidden by a low-contrast background.
        camera_samples, max_camera_flow = _motion_samples(
            camera_flow,
            self.config.max_camera_motion_px,
            self.config.min_samples,
            self.config.flow_percentile,
        )
        articulation_samples, max_articulation_flow = _motion_samples(
            articulation_flow,
            self.config.max_articulation_motion_px,
            self.config.min_samples,
            self.config.flow_percentile,
        )
        # Visibility is the stricter of boundary crossing and depth change.
        visibility_samples = max(visibility_samples, depth_samples)

        uncapped_samples = max(
            self.config.min_samples,
            contrast_samples,
            visibility_samples,
            camera_samples,
            articulation_samples,
        )
        samples = min(self.config.max_samples, uncapped_samples)
        result = AdaptiveSamplingResult(
            samples=samples,
            contrast_samples=contrast_samples,
            visibility_samples=visibility_samples,
            depth_samples=depth_samples,
            max_log_change=max_log_change,
            max_boundary_flow_px=max_boundary_flow,
            max_relative_depth_change=max_rel_depth,
            camera_samples=camera_samples,
            articulation_samples=articulation_samples,
            max_camera_flow_px=max_camera_flow,
            max_articulation_flow_px=max_articulation_flow,
            uncapped_samples=uncapped_samples,
            capped=uncapped_samples > self.config.max_samples,
        )
        self.last_result = result
        return result
