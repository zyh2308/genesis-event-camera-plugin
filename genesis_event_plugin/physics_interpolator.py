"""Physics-first interpolation with explicit uncertainty and render requests.

The v3 path interpolates endpoint poses with an explicitly documented
constant-twist approximation and projects each source endpoint into the
intermediate camera pose. It never claims that this endpoint interpolation is
the hidden Genesis trajectory. NumPy remains the auditable reference; Torch is
used for the projected warp when available.
"""

from dataclasses import dataclass
import time
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .adaptive_sampler import AdaptiveSamplingConfig, PhysicsAdaptiveSampler
from .physics_warp import (
    compute_se3_flow,
    depth_aware_bidirectional_warp_projected_diagnostics,
    endpoint_radiance_residual,
    interpolate_se3,
    project_endpoint_to_alpha,
    projected_target_visibility,
)
from .radiometry import prepare_linear_radiance, log_radiance


class RenderRequiredError(RuntimeError):
    """Raised when fidelity mode cannot safely fill an uncertain interval."""

    def __init__(self, message: str, requests: List["RenderRequest"]):
        super().__init__(message)
        self.requests = requests


@dataclass
class RenderRequest:
    """A request for a real Genesis subframe at an exact historical time.

    ``available_at`` is the earliest simulation timestamp at which the
    requested state is causally observable (normally the bracket's right
    endpoint). It does not include wall-clock rendering or processing time.
    """

    t: float
    alpha: float
    reason: str
    mask: Optional[np.ndarray]
    diagnostics: dict
    available_at: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "t": float(self.t),
            "alpha": float(self.alpha),
            "reason": self.reason,
            "invalid_fraction": float(self.diagnostics.get("invalid_fraction", 0.0)),
            "radiance_residual_p99": float(self.diagnostics.get("radiance_residual_p99", 0.0)),
            "recommend_render": bool(self.diagnostics.get("recommend_render", True)),
            "available_at": None if self.available_at is None else float(self.available_at),
            "available_at_semantics": "simulation_state_observable_time; excludes wall-clock render/processing latency",
        }


class PhysicsInterpolator:
    """Generate intermediate linear-radiance frames from physical endpoint data.

    ``interpolate_se3`` is a constant-twist endpoint approximation: translation
    is linear and rotation follows the SO(3) log/exp path. It is not the
    actual physics trajectory and is marked as such in ``last_uncertainty``.
    Endpoint occupancy, visibility, and unknown/disoccluded regions are kept
    in ``last_visibility``; disocclusion is defined in the projected target
    image domain, never by intersecting same-shape source masks.
    """

    def __init__(
        self,
        K: np.ndarray,
        sampler_config: AdaptiveSamplingConfig = None,
        disocclusion_policy: str = "hold",
        composite: str = "same_surface_log_linear",
        device: str = None,
        warp_backend: str = "auto",
        strict_fidelity: bool = False,
        min_coverage_fraction: float = 0.995,
        depth_tie_abs_m: float = 1e-7,
        depth_tie_rtol: float = 8.0 * np.finfo(np.float32).eps,
        surface_depth_abs_m: float = 1e-3,
        surface_depth_rtol: float = 1e-3,
        max_photometric_interval_s: float = 5e-3,
        max_render_anchors: int = 128,
    ):
        self.K = np.asarray(K, dtype=np.float64)
        if self.K.shape != (3, 3):
            raise ValueError("K must have shape (3,3)")
        if disocclusion_policy not in {"hold", "nan", "raise", "request_render"}:
            raise ValueError("disocclusion_policy must be hold, nan, raise, or request_render")
        if composite not in {"weighted", "next_primary", "same_surface_log_linear"}:
            raise ValueError("unknown warp composite")
        if warp_backend not in {"auto", "numpy", "torch"}:
            raise ValueError("warp_backend must be 'auto', 'numpy', or 'torch'")
        self.sampler = PhysicsAdaptiveSampler(sampler_config)
        self.disocclusion_policy = disocclusion_policy
        self.composite = composite
        self.device = device
        if warp_backend == "auto":
            self.warp_backend = "torch" if device and str(device).startswith("cuda") else "numpy"
        else:
            self.warp_backend = warp_backend
        self.strict_fidelity = bool(strict_fidelity)
        if not 0.0 < min_coverage_fraction <= 1.0:
            raise ValueError("min_coverage_fraction must be in (0,1]")
        if max_photometric_interval_s <= 0:
            raise ValueError("max_photometric_interval_s must be positive")
        if max_render_anchors < 1:
            raise ValueError("max_render_anchors must be positive")
        self.min_coverage_fraction = float(min_coverage_fraction)
        self.depth_tie_abs_m = float(depth_tie_abs_m)
        self.depth_tie_rtol = float(depth_tie_rtol)
        self.surface_depth_abs_m = float(surface_depth_abs_m)
        self.surface_depth_rtol = float(surface_depth_rtol)
        self.max_photometric_interval_s = float(max_photometric_interval_s)
        self.max_render_anchors = int(max_render_anchors)
        self.last_flow01 = None
        self.last_flow10 = None
        self.last_camera_flow01 = None
        self.last_articulation_flow01 = None
        self.last_invalid_fraction = 0.0
        self.last_invalid_mask = None
        self._endpoint_invalid_mask = None
        self.last_high_gradient_invalid_fraction = 0.0
        self.last_sampling = None
        self.last_uncertainty = {}
        self.last_render_requests: List[RenderRequest] = []
        self.last_intermediate_z = []
        self.last_visibility = {}
        self.last_frames_torch = None

    @staticmethod
    def _luminance(frame: np.ndarray, input_space: str) -> np.ndarray:
        return prepare_linear_radiance(frame, input_space=input_space)

    @staticmethod
    def _pixel_grid(H: int, W: int) -> np.ndarray:
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        return np.stack((xx, yy), axis=0).astype(np.float32)

    def _projection_stack(
        self,
        depth0,
        seg0,
        depth1,
        seg1,
        camera0,
        camera1,
        object_world0,
        object_world1,
        object_delta,
        samples: int,
    ):
        uv0, z0, valid0, uv1, z1, valid1 = [], [], [], [], [], []
        for k in range(1, samples + 1):
            alpha = k / float(samples)
            camera_alpha = interpolate_se3(camera0, camera1, alpha)
            p0 = project_endpoint_to_alpha(
                depth0, seg0, self.K, camera0, camera_alpha, alpha, 0,
                object_world0, object_world1, object_delta,
            )
            p1 = project_endpoint_to_alpha(
                depth1, seg1, self.K, camera1, camera_alpha, alpha, 1,
                object_world0, object_world1, object_delta,
            )
            uv0.append(p0[0]); z0.append(p0[1]); valid0.append(p0[2])
            uv1.append(p1[0]); z1.append(p1[1]); valid1.append(p1[2])
        return (
            np.stack(uv0), np.stack(z0), np.stack(valid0),
            np.stack(uv1), np.stack(z1), np.stack(valid1),
        )

    def _record_uncertainty(self, sampling, invalid, residual, gradient):
        invalid = np.asarray(invalid, dtype=bool)
        self.last_invalid_fraction = float(invalid.mean()) if invalid.size else 0.0
        self.last_invalid_mask = invalid.copy()
        if gradient is None or not invalid.size:
            self.last_high_gradient_invalid_fraction = 0.0
        else:
            high = np.asarray(gradient) >= np.nanpercentile(gradient, 90.0)
            self.last_high_gradient_invalid_fraction = float(invalid[high].mean()) if np.any(high) else 0.0
        residual_values = np.asarray(residual)[np.isfinite(residual)] if residual is not None else np.empty(0)
        self.last_uncertainty = {
            "invalid_fraction": self.last_invalid_fraction,
            "high_gradient_invalid_fraction": self.last_high_gradient_invalid_fraction,
            "endpoint_radiance_residual_max": float(np.max(residual_values)) if residual_values.size else 0.0,
            "endpoint_radiance_residual_p99": float(np.percentile(residual_values, 99)) if residual_values.size else 0.0,
            "constant_twist_endpoint_assumption": True,
            "recommend_render": bool(sampling.recommend_render or np.any(invalid)),
            "radiometric_calibration_status": "provided_by_caller",
            "disocclusion_domain": "projected_target_image",
            "depth_correspondence_method": sampling.depth_correspondence_method,
        }

    def _handle_invalid(self, warped, valid, alpha, t0, t1, sampling, gradient, strict_override=None):
        invalid = ~np.asarray(valid, dtype=bool)
        residual_diag = {
            "invalid_fraction": float(invalid.mean()),
            "radiance_residual_p99": float(sampling.p99_radiance_residual),
            "recommend_render": True,
        }
        if np.any(invalid):
            request = RenderRequest(
                t=float(t0 + alpha * (t1 - t0)),
                alpha=float(alpha),
                reason="disocclusion_or_invalid_projection",
                mask=invalid.copy(),
                diagnostics=residual_diag,
                available_at=float(t1),
            )
            self.last_render_requests.append(request)
            strict = self.strict_fidelity if strict_override is None else bool(strict_override)
            should_raise = (
                self.disocclusion_policy in {"raise", "request_render"} or strict
            )
            if strict_override is False:
                should_raise = False
            if should_raise:
                raise RenderRequiredError(
                    f"physics warp has {invalid.mean():.3%} invalid pixels; a real Genesis subframe render is required",
                    self.last_render_requests,
                )
            if self.disocclusion_policy == "hold":
                fallback = self._current_I0 if alpha < 0.5 else self._current_I1
                warped[invalid] = fallback[invalid]
        return warped.astype(np.float32, copy=False), invalid

    def _interpolate_pair(
        self,
        frame0: np.ndarray,
        depth0: np.ndarray,
        seg0: np.ndarray,
        frame1: np.ndarray,
        depth1: np.ndarray,
        seg1: np.ndarray,
        camera_to_world0: np.ndarray,
        camera_to_world1: np.ndarray,
        object_world_delta: Optional[Mapping[int, np.ndarray]],
        t0: float,
        t1: float,
        input_space: str = "linear",
        object_world0: Optional[Mapping[int, np.ndarray]] = None,
        object_world1: Optional[Mapping[int, np.ndarray]] = None,
        return_torch: bool = False,
        _allow_approximate: bool = False,
    ) -> Tuple[List[np.ndarray], List[float]]:
        if t1 <= t0:
            raise ValueError("t1 must be greater than t0")
        self.last_invalid_fraction = 0.0
        self.last_invalid_mask = None
        self._endpoint_invalid_mask = None
        self.last_render_requests = []
        self.last_visibility = {}
        self.last_frames_torch = None
        I0 = self._luminance(frame0, input_space)
        I1 = self._luminance(frame1, input_space)
        dt = float(t1 - t0)
        if object_world_delta is None and object_world0 is not None and object_world1 is not None:
            object_world_delta = {
                int(key): np.asarray(object_world1[key], dtype=np.float64)
                @ np.linalg.inv(np.asarray(object_world0[key], dtype=np.float64))
                for key in object_world0.keys() & object_world1.keys()
            }
        camera_flow01, _ = compute_se3_flow(
            depth0, seg0, self.K, camera_to_world0, camera_to_world1,
            object_world_deltas=None,
        )
        flow01, _ = compute_se3_flow(
            depth0, seg0, self.K, camera_to_world0, camera_to_world1,
            object_world_deltas=object_world_delta,
        )
        inverse_deltas = {
            int(key): np.linalg.inv(np.asarray(value, dtype=np.float64))
            for key, value in (object_world_delta or {}).items()
        }
        flow10, _ = compute_se3_flow(
            depth1, seg1, self.K, camera_to_world1, camera_to_world0,
            object_world_deltas=inverse_deltas,
        )
        articulation_flow01 = flow01 - camera_flow01
        endpoint0 = project_endpoint_to_alpha(
            depth0, seg0, self.K, camera_to_world0, camera_to_world1, 1.0, 0,
            object_world0, object_world1, object_world_delta,
        )
        endpoint1 = project_endpoint_to_alpha(
            depth1, seg1, self.K, camera_to_world1, camera_to_world0, 0.0, 1,
            object_world0, object_world1, object_world_delta,
        )
        occupancy01, visibility01, zbuf01 = projected_target_visibility(
            depth0, endpoint0[0], endpoint0[1], endpoint0[2]
        )
        occupancy10, visibility10, zbuf10 = projected_target_visibility(
            depth1, endpoint1[0], endpoint1[1], endpoint1[2]
        )
        target_valid0 = np.isfinite(depth0) & (np.asarray(depth0) > 1e-6)
        target_valid1 = np.isfinite(depth1) & (np.asarray(depth1) > 1e-6)
        # Each mask is compared only in its own projected target domain:
        # t0->t1 is evaluated against endpoint-1 validity, and t1->t0 against
        # endpoint-0 validity. They are not intersected pixelwise.
        disocclusion01 = target_valid1 & ~visibility01
        disocclusion10 = target_valid0 & ~visibility10
        unknown01 = (~visibility01) | (~target_valid1)
        unknown10 = (~visibility10) | (~target_valid0)
        residual, residual_support = endpoint_radiance_residual(
            I0, I1, depth0, endpoint0[0], endpoint0[1], endpoint0[2],
            eps=1e-6, return_support=True,
        )
        source_depth = np.asarray(depth0, dtype=np.float32)
        correspondence_relative_depth = np.full_like(source_depth, np.nan, dtype=np.float32)
        depth_support = endpoint0[2] & np.isfinite(source_depth) & (source_depth > 1e-6)
        correspondence_relative_depth[depth_support] = (
            np.abs(endpoint0[1][depth_support] - source_depth[depth_support])
            / np.maximum(source_depth[depth_support], 1e-6)
        )
        self.last_visibility = {
            "source_valid0": np.asarray(endpoint0[2], dtype=bool),
            "source_valid1": np.asarray(endpoint1[2], dtype=bool),
            "occupancy_0_to_1": occupancy01,
            "visibility_0_to_1": visibility01,
            "target_valid1": target_valid1,
            "disocclusion_0_to_1": disocclusion01,
            "unknown_region_0_to_1": unknown01,
            "occupancy_1_to_0": occupancy10,
            "visibility_1_to_0": visibility10,
            "target_valid0": target_valid0,
            "disocclusion_1_to_0": disocclusion10,
            "unknown_region_1_to_0": unknown10,
            "z_buffer_0_to_1": zbuf01,
            "z_buffer_1_to_0": zbuf10,
            "residual_support_target1": residual_support,
        }
        sampler_config = self.sampler.config
        sampling = self.sampler.compute(
            I0, flow01, dt,
            depth0=depth0,
            depth1=depth1,
            seg0=seg0,
            seg1=seg1,
            camera_flow=camera_flow01,
            articulation_flow=articulation_flow01,
            endpoint_radiance_residual=residual,
            # This is a target-domain unsupported mask. It includes both
            # newly exposed valid pixels and target pixels with no reliable
            # depth, so strict fidelity cannot mistake missing geometry for
            # a valid black/background correspondence.
            disocclusion_mask=unknown01,
            correspondence_relative_depth=correspondence_relative_depth,
        )
        self.last_flow01 = flow01
        self.last_flow10 = flow10
        self.last_camera_flow01 = camera_flow01
        self.last_articulation_flow01 = articulation_flow01
        self.last_sampling = sampling
        self._current_I0, self._current_I1 = I0, I1

        H, W = I0.shape
        gradient = np.hypot(*np.gradient(log_radiance(I0)))
        self._endpoint_invalid_mask = unknown01.copy()
        self._record_uncertainty(sampling, self._endpoint_invalid_mask, residual, gradient)
        if (
            sampling.recommend_render
            and not _allow_approximate
            and (self.strict_fidelity or self.disocclusion_policy in {"raise", "request_render"})
        ):
            request = RenderRequest(
                t=float(t0 + 0.5 * dt),
                alpha=0.5,
                reason="radiance_residual_or_sampling_bound",
                mask=None,
                diagnostics={
                    "invalid_fraction": float(sampling.disocclusion_fraction),
                    "radiance_residual_p99": float(sampling.p99_radiance_residual),
                    "recommend_render": True,
                },
                available_at=float(t1),
            )
            self.last_render_requests.append(request)
            raise RenderRequiredError(
                "adaptive physics uncertainty exceeds the configured fidelity "
                "bound; a real Genesis subframe render is required",
                self.last_render_requests,
            )
        alphas = np.arange(1, sampling.samples + 1, dtype=np.float32) / float(sampling.samples)
        timestamps = [float(t0 + float(a) * dt) for a in alphas]

        # The projected NumPy path is the numerical reference. The Torch path
        # computes the same endpoint geometry with the tensor-native projection
        # stack and keeps all warp arithmetic on device.
        if self.warp_backend == "torch":
            import torch
            from .torch_physics import (
                depth_aware_bidirectional_warp_projected_batch_torch,
                project_endpoint_to_alpha_stack_torch,
            )

            torch_device = torch.device(self.device or "cuda")
            alphas_t = torch.as_tensor(alphas, dtype=torch.float32, device=torch_device)
            T0_t = torch.as_tensor(camera_to_world0, dtype=torch.float32, device=torch_device)
            T1_t = torch.as_tensor(camera_to_world1, dtype=torch.float32, device=torch_device)
            K_t = torch.as_tensor(self.K, dtype=torch.float32, device=torch_device)
            d0_t = torch.as_tensor(depth0, dtype=torch.float32, device=torch_device)
            d1_t = torch.as_tensor(depth1, dtype=torch.float32, device=torch_device)
            seg0_t = torch.as_tensor(seg0, device=torch_device)
            seg1_t = torch.as_tensor(seg1, device=torch_device)
            object0_t = {
                int(key): torch.as_tensor(value, dtype=torch.float32, device=torch_device)
                for key, value in (object_world0 or {}).items()
            }
            object1_t = {
                int(key): torch.as_tensor(value, dtype=torch.float32, device=torch_device)
                for key, value in (object_world1 or {}).items()
            }
            delta_t = {
                int(key): torch.as_tensor(value, dtype=torch.float32, device=torch_device)
                for key, value in (object_world_delta or {}).items()
            }
            uv0, z0, vv0 = project_endpoint_to_alpha_stack_torch(
                d0_t, seg0_t, K_t, T0_t, T0_t, T1_t, alphas_t, 0,
                object0_t, object1_t, delta_t,
            )
            uv1, z1, vv1 = project_endpoint_to_alpha_stack_torch(
                d1_t, seg1_t, K_t, T1_t, T0_t, T1_t, alphas_t, 1,
                object0_t, object1_t, delta_t,
            )
            self.last_intermediate_z = [z0.detach().cpu().numpy(), z1.detach().cpu().numpy()]
            frame0_t = torch.as_tensor(I0, dtype=torch.float32, device=torch_device)
            frame1_t = torch.as_tensor(I1, dtype=torch.float32, device=torch_device)
            batch = depth_aware_bidirectional_warp_projected_batch_torch(
                frame0_t.unsqueeze(0), d0_t.unsqueeze(0),
                uv0.unsqueeze(0), z0.unsqueeze(0), vv0.unsqueeze(0),
                frame1_t.unsqueeze(0), d1_t.unsqueeze(0),
                uv1.unsqueeze(0), z1.unsqueeze(0), vv1.unsqueeze(0),
                composite=self.composite,
                return_diagnostics=True,
                alpha=alphas_t,
                surface_id0=torch.as_tensor(seg0, dtype=torch.long, device=torch_device).unsqueeze(0),
                surface_id1=torch.as_tensor(seg1, dtype=torch.long, device=torch_device).unsqueeze(0),
                min_coverage_fraction=self.min_coverage_fraction,
                depth_tie_abs_m=self.depth_tie_abs_m,
                depth_tie_rtol=self.depth_tie_rtol,
                surface_depth_abs_m=self.surface_depth_abs_m,
                surface_depth_rtol=self.surface_depth_rtol,
            )
            warped_t, valid_t, torch_visibility = batch
            # ``warped_t`` is [B,T,H,W] for luminance and [B,T,H,W,C] for
            # colour.  Keep the batch/temporal axes explicit; callers that
            # need NumPy frames can opt into the documented conversion below.
            self.last_frames_torch = warped_t.contiguous()
            for key in torch_visibility:
                self.last_visibility[key] = torch_visibility[key][0].detach().cpu().numpy()
            self.last_visibility["intermediate_visibility"] = valid_t[0].detach().cpu().numpy().astype(bool)
            self.last_visibility["intermediate_unknown"] = ~self.last_visibility["intermediate_visibility"]
            if return_torch and self.disocclusion_policy == "raise" and bool((~valid_t).any()):
                raise RenderRequiredError(
                    "Torch projected warp has invalid pixels; a real Genesis subframe is required",
                    self.last_render_requests,
                )
            frames = []
            for index, alpha in enumerate(alphas):
                warped = self.last_frames_torch[0, index]
                valid = valid_t[0, index]
                warped_np = warped.detach().cpu().numpy()
                valid_np = valid.detach().cpu().numpy().astype(bool)
                warped_np, invalid = self._handle_invalid(
                    warped_np, valid_np, float(alpha), t0, t1, sampling, gradient,
                    strict_override=self.strict_fidelity and not _allow_approximate,
                )
                if self.disocclusion_policy == "nan":
                    warped_np[invalid] = np.nan
                frames.append(warped_np)
            if return_torch:
                return self.last_frames_torch, timestamps
            return frames, timestamps

        projections = self._projection_stack(
            depth0, seg0, depth1, seg1, camera_to_world0, camera_to_world1,
            object_world0, object_world1, object_world_delta, sampling.samples,
        )
        self.last_intermediate_z = [projections[1], projections[4]]
        frames: List[np.ndarray] = []
        intermediate_visibility = []
        for index, alpha in enumerate(alphas):
            warped, valid, diagnostics = depth_aware_bidirectional_warp_projected_diagnostics(
                I0, depth0, projections[0][index], projections[1][index], projections[2][index],
                I1, depth1, projections[3][index], projections[4][index], projections[5][index],
                composite=self.composite,
                alpha=float(alpha),
                surface_id0=seg0,
                surface_id1=seg1,
                min_coverage_fraction=self.min_coverage_fraction,
                depth_tie_abs_m=self.depth_tie_abs_m,
                depth_tie_rtol=self.depth_tie_rtol,
                surface_depth_abs_m=self.surface_depth_abs_m,
                surface_depth_rtol=self.surface_depth_rtol,
            )
            intermediate_visibility.append(diagnostics["intermediate_visibility"])
            warped, invalid = self._handle_invalid(
                warped, valid, float(alpha), t0, t1, sampling, gradient,
                strict_override=self.strict_fidelity and not _allow_approximate,
            )
            if self.disocclusion_policy == "nan":
                warped[invalid] = np.nan
            frames.append(warped)
        self.last_visibility["intermediate_visibility"] = np.stack(intermediate_visibility)
        self.last_visibility["intermediate_unknown"] = ~self.last_visibility["intermediate_visibility"]
        return frames, timestamps

    @staticmethod
    def _validated_anchor(anchor: Mapping, expected_shape, expected_time: Optional[float] = None):
        required = {"t", "radiance", "depth", "seg", "camera_to_world", "object_to_world"}
        if not isinstance(anchor, Mapping) or not required.issubset(anchor):
            raise ValueError(f"rendered state must contain {sorted(required)}")
        t = float(anchor["t"])
        if not np.isfinite(t) or (expected_time is not None and abs(t - expected_time) > 1e-8):
            raise ValueError("rendered state timestamp does not match its request")
        radiance = np.asarray(anchor["radiance"], dtype=np.float32)
        depth = np.asarray(anchor["depth"], dtype=np.float32)
        seg = np.asarray(anchor["seg"])
        if radiance.shape[:2] != expected_shape or depth.shape != expected_shape or seg.shape != expected_shape:
            raise ValueError("rendered state image/depth/segmentation shapes do not match endpoints")
        camera = np.asarray(anchor["camera_to_world"], dtype=np.float64)
        objects = {int(k): np.asarray(v, dtype=np.float64) for k, v in anchor["object_to_world"].items()}
        if camera.shape != (4, 4) or not np.isfinite(camera).all():
            raise ValueError("rendered camera pose must be finite 4x4")
        if any(v.shape != (4, 4) or not np.isfinite(v).all() for v in objects.values()):
            raise ValueError("rendered object poses must be finite 4x4")
        out = dict(anchor)
        out.update({"t": t, "radiance": radiance, "depth": depth, "seg": seg,
                    "camera_to_world": camera, "object_to_world": objects})
        return out

    @staticmethod
    def _validated_radiance_sample(
        sample: Mapping,
        expected_shape,
        expected_time: Optional[float] = None,
        *,
        require_provenance: bool = False,
        available_at_limit: Optional[float] = None,
    ):
        """Validate a real time-stamped radiance frame without GT geometry."""
        if not isinstance(sample, Mapping) or not {"t", "radiance"}.issubset(sample):
            raise ValueError("radiance sample must contain t and radiance")
        t = float(sample["t"])
        radiance = np.asarray(sample["radiance"], dtype=np.float32)
        if not np.isfinite(t) or (expected_time is not None and abs(t - expected_time) > 1e-8):
            raise ValueError("radiance sample timestamp does not match its request")
        if radiance.shape[:2] != expected_shape or not np.isfinite(radiance).all():
            raise ValueError("radiance sample shape/data do not match interval endpoints")
        provenance = str(sample.get("provenance", "")).strip()
        if require_provenance and not provenance:
            raise ValueError("strict radiance samples must identify their render/cache provenance")
        available_at = float(sample.get("available_at_s", t))
        if not np.isfinite(available_at) or available_at < t - 1e-9:
            raise ValueError("radiance sample cannot be available before its observed timestamp")
        if available_at_limit is not None and available_at > available_at_limit + 1e-9:
            raise ValueError("radiance sample availability exceeds the interval right endpoint")
        return {"t": t, "radiance": radiance,
                "provenance": provenance or "unspecified_radiance_sample",
                "available_at_s": available_at}

    def interpolate(
        self,
        frame0: np.ndarray,
        depth0: np.ndarray,
        seg0: np.ndarray,
        frame1: np.ndarray,
        depth1: np.ndarray,
        seg1: np.ndarray,
        camera_to_world0: np.ndarray,
        camera_to_world1: np.ndarray,
        object_world_delta: Optional[Mapping[int, np.ndarray]],
        t0: float,
        t1: float,
        input_space: str = "linear",
        object_world0: Optional[Mapping[int, np.ndarray]] = None,
        object_world1: Optional[Mapping[int, np.ndarray]] = None,
        return_torch: bool = False,
        intermediate_states: Optional[List[Mapping]] = None,
        intermediate_state_provider: Optional[Callable[[RenderRequest], Mapping]] = None,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Interpolate with explicit radiometric sampling bounds.

        The endpoint-only log-radiance model is an approximation. In strict
        mode, intervals wider than ``max_photometric_interval_s`` request
        real scene-state samples. A supplied provider must return an actual
        rendered/state-cache observation at the requested timestamp; it may
        not extrapolate from future frames. Event output is available no
        earlier than the right endpoint of the bracket.
        """
        call_started = time.perf_counter()
        if t1 <= t0:
            raise ValueError("t1 must be greater than t0")
        h, w = np.asarray(depth0).shape
        start = self._validated_anchor({
            "t": t0, "radiance": self._luminance(frame0, input_space), "depth": depth0, "seg": seg0,
            "camera_to_world": camera_to_world0,
            "object_to_world": object_world0 or {},
            "provenance": "caller_endpoint",
        }, (h, w), t0)
        end = self._validated_anchor({
            "t": t1, "radiance": self._luminance(frame1, input_space), "depth": depth1, "seg": seg1,
            "camera_to_world": camera_to_world1,
            "object_to_world": object_world1 or {},
            "provenance": "caller_endpoint",
        }, (h, w), t1)
        if self.strict_fidelity:
            # First ask the physics sampler how densely this interval needs
            # observations. These warped frames are used only to size the
            # schedule; strict output is made from actual cached/direct frames.
            self._interpolate_pair(
                start["radiance"], start["depth"], start["seg"],
                end["radiance"], end["depth"], end["seg"],
                start["camera_to_world"], end["camera_to_world"],
                object_world_delta, t0, t1, input_space="linear",
                object_world0=start["object_to_world"], object_world1=end["object_to_world"],
                return_torch=False, _allow_approximate=True,
            )
            sampling_count = int(self.last_sampling.samples)
            spacing_count = max(
                1,
                int(np.ceil((t1 - t0) / self.max_photometric_interval_s - 1e-9)),
            )
            sample_count = max(1, sampling_count, spacing_count)
            sampled = {
                start["t"]: {"t": start["t"], "radiance": start["radiance"], "provenance": start["provenance"]},
                end["t"]: {"t": end["t"], "radiance": end["radiance"], "provenance": end["provenance"]},
            }
            for state in intermediate_states or []:
                checked = self._validated_radiance_sample(
                    state, (h, w), require_provenance=True,
                    available_at_limit=float(t1),
                )
                if not t0 < checked["t"] < t1:
                    raise ValueError("intermediate state timestamp must be strictly inside the interval")
                sampled[checked["t"]] = checked
            # Existing exact observations can be denser and non-uniform. Use
            # them as schedule anchors; request additional radiance only inside
            # gaps that exceed the adaptive/photometric target spacing.
            target_gap = float(t1 - t0) / float(sample_count)
            ordered_times = sorted(sampled)
            requests = []
            for left_t, right_t in zip(ordered_times[:-1], ordered_times[1:]):
                gap = right_t - left_t
                segments = max(1, int(np.ceil(gap / target_gap - 1e-9)))
                for index in range(1, segments):
                    sample_t = float(left_t + gap * index / segments)
                    requests.append(RenderRequest(
                        t=sample_t,
                        alpha=float((sample_t - t0) / (t1 - t0)),
                        reason="strict_real_radiance_sample",
                        mask=None,
                        diagnostics={
                            "recommend_render": True,
                            "maximum_spacing_s": min(self.max_photometric_interval_s, target_gap),
                            "adaptive_sample_count": sampling_count,
                            "required_sample_count": sample_count,
                            "sampling_role": "real_intermediate_observation",
                        },
                        available_at=float(t1),
                    ))
            if len(requests) > self.max_render_anchors:
                request = RenderRequest(
                    t=float(requests[0].t if requests else t0 + 0.5 * (t1 - t0)),
                    alpha=float(requests[0].alpha if requests else 0.5),
                    reason="real_sample_budget_exceeded", mask=None,
                    diagnostics={
                        "recommend_render": True,
                        "required_interior_samples": int(len(requests)),
                        "configured_max_render_anchors": self.max_render_anchors,
                        "target_sample_gap_s": target_gap,
                    },
                    available_at=float(t1),
                )
                self.last_render_requests = [request]
                self.last_uncertainty.update({
                    "recommend_render": True,
                    "strict_sampling_status": "budget_exceeded",
                    "required_real_sample_count": int(len(requests)),
                    "output_available_at_s": float(t1),
                })
                raise RenderRequiredError("strict real-sampling budget exceeded", [request])
            if requests and intermediate_state_provider is None:
                self.last_render_requests = requests
                self.last_uncertainty.update({
                    "recommend_render": True,
                    "strict_sampling_status": "awaiting_real_state_provider",
                    "required_real_sample_count": len(requests),
                    "max_photometric_interval_s": self.max_photometric_interval_s,
                    "output_available_at_s": float(t1),
                    "future_state_used": False,
                })
                raise RenderRequiredError(
                    "strict fidelity requires real intermediate radiance/state observations",
                    requests,
                )
            provider_seconds = 0.0
            for request in requests:
                begin = time.perf_counter()
                state = intermediate_state_provider(request)
                provider_seconds += time.perf_counter() - begin
                checked = self._validated_radiance_sample(
                    state, (h, w), request.t,
                    require_provenance=True,
                    available_at_limit=float(t1),
                )
                sampled[checked["t"]] = checked
            ordered = sorted(sampled.values(), key=lambda item: item["t"])
            exact_times = [float(item["t"]) for item in ordered[1:]]
            exact_frames = [self._luminance(item["radiance"], "linear") for item in ordered[1:]]
            gaps = [b["t"] - a["t"] for a, b in zip(ordered[:-1], ordered[1:])]
            base = dict(self.last_uncertainty)
            base.update({
                "strict_sampling_status": "actual_state_samples",
                "recommend_render": False,
                "photometric_request_resolved": True,
                "strict_sample_output": "direct_radiance_frames_at_observed_states",
                "photometric_temporal_model": "piecewise_log_radiance_between_actual_samples",
                "photometric_anchor_count": len(ordered),
                "photometric_max_anchor_gap_s": float(max(gaps) if gaps else 0.0),
                "max_photometric_interval_s": self.max_photometric_interval_s,
                "adaptive_sample_count": sampling_count,
                "extra_render_count": len(requests),
                "direct_sample_frame_count": len(ordered) - 2,
                "render_provider_wall_s": float(provider_seconds),
                "anchor_provenance": [item.get("provenance", "unspecified") for item in ordered],
                "anchor_available_at_s": [
                    float(item.get("available_at_s", t1)) if item["t"] not in {t0, t1} else float(item["t"])
                    for item in ordered
                ],
                "output_available_at_s": float(t1),
                "output_ready_after_right_endpoint_wall_s": float(time.perf_counter() - call_started),
                "future_state_used": False,
                "geometry_warp_used_for_strict_output": False,
                "causal_input_contract": "timestamp_match_plus_explicit_sample_provenance; output_available_no_earlier_than_right_endpoint",
            })
            self.last_uncertainty = base
            self.last_render_requests = []
            visibility = np.stack([np.isfinite(frame) for frame in exact_frames]) if exact_frames else np.empty((0, h, w), dtype=bool)
            self.last_visibility = {
                "intermediate_visibility": visibility,
                "intermediate_unknown": ~visibility,
                "strict_direct_state_samples": visibility.copy(),
            }
            if return_torch:
                import torch
                self.last_frames_torch = torch.as_tensor(
                    np.stack(exact_frames)[None], dtype=torch.float32,
                    device=torch.device(self.device or "cpu"),
                )
                return self.last_frames_torch, exact_times
            return exact_frames, exact_times

        anchors = [start, end]
        for state in intermediate_states or []:
            checked = self._validated_anchor(state, (h, w))
            if not t0 < checked["t"] < t1:
                raise ValueError("intermediate state timestamp must be strictly inside the interval")
            anchors.append(checked)
        anchors.sort(key=lambda item: item["t"])
        if any(b["t"] - a["t"] <= 1e-10 for a, b in zip(anchors[:-1], anchors[1:])):
            raise ValueError("duplicate or non-increasing intermediate state timestamps")

        required_segments = max(
            1,
            int(np.ceil((t1 - t0) / self.max_photometric_interval_s - 1e-9)),
        )
        required_times = [t0 + (t1 - t0) * k / required_segments for k in range(1, required_segments)]
        anchor_times = [item["t"] for item in anchors]
        missing = [
            t for t in required_times
            if min(abs(t - observed) for observed in anchor_times) > 1e-8
        ]
        pending = [RenderRequest(
            t=float(t), alpha=float((t - t0) / (t1 - t0)),
            reason="maximum_photometric_anchor_interval",
            mask=None,
            diagnostics={
                "recommend_render": True,
                "required_max_gap_s": self.max_photometric_interval_s,
                "actual_bracket_gap_s": float(t1 - t0),
                "sampling_role": "real_radiance_anchor",
            },
            available_at=float(t1),
        ) for t in missing]
        self.last_render_requests = []
        self.last_visibility = {}
        render_count = 0
        provider_seconds = 0.0
        resolved_requests = []
        if pending and intermediate_state_provider is None and self.strict_fidelity:
            self.last_render_requests = pending
            self.last_uncertainty = {
                "recommend_render": True,
                "photometric_temporal_model": "unobserved_endpoint_only_approximation",
                "photometric_max_anchor_gap_s": float(t1 - t0),
                "required_real_anchor_count": len(pending),
                "extra_render_count": 0,
                "output_available_at_s": float(t1),
            }
            raise RenderRequiredError(
                "photometric interval exceeds the configured real-anchor spacing",
                pending,
            )
        for request in pending:
            if intermediate_state_provider is None:
                continue
            begin = time.perf_counter()
            state = intermediate_state_provider(request)
            provider_seconds += time.perf_counter() - begin
            checked = self._validated_anchor(state, (h, w), request.t)
            checked["provenance"] = state.get("provenance", "unspecified_state_provider")
            anchors.append(checked)
            anchor_times.append(checked["t"])
            resolved_requests.append(request)
            render_count += 1
        anchors.sort(key=lambda item: item["t"])

        def pair_args(a, b):
            a_objects, b_objects = a["object_to_world"], b["object_to_world"]
            delta = {
                int(key): b_objects[key] @ np.linalg.inv(a_objects[key])
                for key in a_objects.keys() & b_objects.keys()
            }
            return (
                a["radiance"], a["depth"], a["seg"],
                b["radiance"], b["depth"], b["seg"],
                a["camera_to_world"], b["camera_to_world"], delta,
                float(a["t"]), float(b["t"]), "linear",
                a_objects, b_objects, False,
            )

        def solve(a, b, depth=0):
            nonlocal render_count, provider_seconds
            try:
                return self._interpolate_pair(*pair_args(a, b))
            except RenderRequiredError as exc:
                self.last_render_requests.extend(exc.requests)
                if intermediate_state_provider is None:
                    raise
                if depth >= 10 or render_count >= self.max_render_anchors:
                    raise RenderRequiredError(
                        "real-render refinement budget exhausted",
                        exc.requests,
                    ) from exc
                request = next((r for r in exc.requests if abs(r.t - a["t"]) > 1e-8 and abs(r.t - b["t"]) > 1e-8), None)
                if request is None:
                    raise
                begin = time.perf_counter()
                state = intermediate_state_provider(request)
                provider_seconds += time.perf_counter() - begin
                mid = self._validated_anchor(state, (h, w), request.t)
                mid["provenance"] = state.get("provenance", "unspecified_state_provider")
                resolved_requests.append(request)
                render_count += 1
                left_frames, left_times = solve(a, mid, depth + 1)
                right_frames, right_times = solve(mid, b, depth + 1)
                return left_frames + right_frames, left_times + right_times

        frames, timestamps = [], []
        for a, b in zip(anchors[:-1], anchors[1:]):
            pair_frames, pair_times = solve(a, b)
            frames.extend(pair_frames)
            timestamps.extend(pair_times)

        actual_gaps = [b["t"] - a["t"] for a, b in zip(anchors[:-1], anchors[1:])]
        max_gap = max(actual_gaps) if actual_gaps else 0.0
        base_uncertainty = dict(self.last_uncertainty)
        base_uncertainty.update({
            "photometric_temporal_model": "piecewise_log_radiance_between_observed_anchors",
            "photometric_anchor_count": len(anchors),
            "photometric_max_anchor_gap_s": float(max_gap),
            "photometric_anchor_provenance": [a.get("provenance", "unspecified") for a in anchors],
            "extra_render_count": int(render_count),
            "render_provider_wall_s": float(provider_seconds),
            "resolved_render_request_count": len(resolved_requests),
            "output_available_at_s": float(t1),
            "output_ready_after_right_endpoint_wall_s": float(time.perf_counter() - call_started),
            "future_state_used": False,
            "photometric_spacing_satisfied": bool(max_gap <= self.max_photometric_interval_s + 1e-9),
        })
        if max_gap > self.max_photometric_interval_s + 1e-9:
            base_uncertainty["photometric_temporal_model"] = "unobserved_endpoint_log_radiance_approximation"
            base_uncertainty["recommend_render"] = True
        self.last_uncertainty = base_uncertainty
        self.last_render_requests = resolved_requests
        return frames, timestamps
