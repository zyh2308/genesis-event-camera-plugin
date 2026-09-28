"""Physics-constrained temporal sampling controller.

The controller operates on a displacement field over one endpoint interval.
It separates the contrast bound from visibility/motion guards and returns
diagnostics so experiments can report *why* an interval was oversampled.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

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
    radiance_residual_threshold: float = 0.1
    sensor_eps_target: float = 0.05
    sensor_time_constant_s: Optional[float] = None
    max_disocclusion_fraction: float = 0.0
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
        if self.radiance_residual_threshold <= 0:
            raise ValueError("radiance_residual_threshold must be positive")
        if self.sensor_eps_target <= 0:
            raise ValueError("sensor_eps_target must be positive")
        if self.sensor_time_constant_s is not None and self.sensor_time_constant_s <= 0:
            raise ValueError("sensor_time_constant_s must be positive when provided")
        if not 0 <= self.max_disocclusion_fraction <= 1:
            raise ValueError("max_disocclusion_fraction must be in [0, 1]")


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
    radiance_samples: int = 1
    sensor_samples: int = 1
    max_radiance_residual: float = 0.0
    p99_radiance_residual: float = 0.0
    max_sensor_eps: float = 0.0
    disocclusion_fraction: float = 0.0
    used_correspondence_depth: bool = False
    recommend_render: bool = False


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
) -> Tuple[int, float]:
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
        endpoint_radiance_residual: np.ndarray = None,
        disocclusion_mask: np.ndarray = None,
        correspondence_depth0: np.ndarray = None,
        correspondence_depth1: np.ndarray = None,
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
        if self.config.use_visibility_guard and (seg0 is not None or seg1 is not None):
            boundary = np.zeros(lum.shape, dtype=bool)
            if seg0 is not None:
                boundary |= _seg_boundary(np.asarray(seg0))
            if seg1 is not None:
                boundary |= _seg_boundary(np.asarray(seg1))
            flow_mag = np.linalg.norm(flow, axis=0)
            if np.any(boundary):
                max_boundary_flow = float(np.nanmax(flow_mag[boundary]))
                visibility_samples = max(
                    self.config.min_samples,
                    int(math.ceil(max_boundary_flow / self.config.max_boundary_motion_px)),
                )

        depth_samples = self.config.min_samples
        max_rel_depth = 0.0
        d0 = correspondence_depth0 if correspondence_depth0 is not None else depth0
        d1 = correspondence_depth1 if correspondence_depth1 is not None else depth1
        used_correspondence_depth = correspondence_depth0 is not None and correspondence_depth1 is not None
        if self.config.use_visibility_guard and d0 is not None and d1 is not None:
            d0 = np.asarray(d0, dtype=np.float32)
            d1 = np.asarray(d1, dtype=np.float32)
            mask = np.isfinite(d0) & np.isfinite(d1) & (d0 > 1e-6) & (d1 > 1e-6)
            if np.any(mask):
                rel = np.abs(d1[mask] - d0[mask]) / np.maximum(d0[mask], 1e-6)
                max_rel_depth = float(np.nanmax(rel))
                depth_samples = max(
                    self.config.min_samples,
                    int(math.ceil(max_rel_depth / self.config.max_relative_depth_change)),
                )

        radiance_samples = self.config.min_samples
        max_residual = 0.0
        p99_residual = 0.0
        if endpoint_radiance_residual is not None:
            residual = np.asarray(endpoint_radiance_residual, dtype=np.float32)
            values = residual[np.isfinite(residual)]
            if values.size:
                max_residual = float(np.max(values))
                p99_residual = float(np.percentile(values, 99.0))
                radiance_samples = max(
                    self.config.min_samples,
                    int(math.ceil(p99_residual / self.config.radiance_residual_threshold)),
                )

        sensor_samples = self.config.min_samples
        max_sensor_eps = 0.0
        if self.config.sensor_time_constant_s is not None:
            max_sensor_eps = float(dt / self.config.sensor_time_constant_s)
            sensor_samples = max(
                self.config.min_samples,
                int(math.ceil(max_sensor_eps / self.config.sensor_eps_target)),
            )

        disocclusion_fraction = 0.0
        if disocclusion_mask is not None:
            invalid = np.asarray(disocclusion_mask, dtype=bool)
            disocclusion_fraction = float(invalid.mean())
            if disocclusion_fraction > self.config.max_disocclusion_fraction:
                visibility_samples = max(visibility_samples, self.config.min_samples + 1)

        # Keep the independent physical causes separate. The final U is their
        # maximum, but disocclusion and unexplained radiance also produce an
        # explicit render recommendation rather than pretending more samples
        # alone can create missing content.
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
        uncapped_samples = max(
            self.config.min_samples,
            contrast_samples,
            visibility_samples,
            depth_samples,
            camera_samples,
            articulation_samples,
            radiance_samples,
            sensor_samples,
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
            radiance_samples=radiance_samples,
            sensor_samples=sensor_samples,
            max_radiance_residual=max_residual,
            p99_radiance_residual=p99_residual,
            max_sensor_eps=max_sensor_eps,
            disocclusion_fraction=disocclusion_fraction,
            used_correspondence_depth=used_correspondence_depth,
            recommend_render=(
                disocclusion_fraction > self.config.max_disocclusion_fraction
                or max_residual > self.config.radiance_residual_threshold
                or uncapped_samples > self.config.max_samples
            ),
        )
        self.last_result = result
        return result
