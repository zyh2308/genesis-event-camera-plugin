"""Physics-first interpolation with explicit uncertainty and render requests.

The v3 path interpolates endpoint poses with an explicitly documented
constant-twist approximation and projects each source endpoint into the
intermediate camera pose. It never claims that this endpoint interpolation is
the hidden Genesis trajectory. NumPy remains the auditable reference; Torch is
used for the projected warp when available.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from .adaptive_sampler import AdaptiveSamplingConfig, PhysicsAdaptiveSampler
from .physics_warp import (
    compute_se3_flow,
    depth_aware_bidirectional_warp,
    depth_aware_bidirectional_warp_projected,
    endpoint_radiance_residual,
    interpolate_se3,
    project_endpoint_to_alpha,
)
from .radiometry import prepare_linear_radiance, log_radiance


class RenderRequiredError(RuntimeError):
    """Raised when fidelity mode cannot safely fill an uncertain interval."""

    def __init__(self, message: str, requests: List["RenderRequest"]):
        super().__init__(message)
        self.requests = requests


@dataclass
class RenderRequest:
    """A request for an upper layer to render a real Genesis subframe."""

    t: float
    alpha: float
    reason: str
    mask: Optional[np.ndarray]
    diagnostics: dict

    def as_dict(self) -> dict:
        return {
            "t": float(self.t),
            "alpha": float(self.alpha),
            "reason": self.reason,
            "invalid_fraction": float(self.diagnostics.get("invalid_fraction", 0.0)),
            "radiance_residual_p99": float(self.diagnostics.get("radiance_residual_p99", 0.0)),
            "recommend_render": bool(self.diagnostics.get("recommend_render", True)),
        }


class PhysicsInterpolator:
    """Generate intermediate linear-radiance frames from physical endpoint data.

    ``interpolate_se3`` is a constant-twist endpoint approximation: translation
    is linear and rotation follows the SO(3) log/exp path. It is not the
    actual physics trajectory and is marked as such in ``last_uncertainty``.
    """

    def __init__(
        self,
        K: np.ndarray,
        sampler_config: AdaptiveSamplingConfig = None,
        disocclusion_policy: str = "hold",
        composite: str = "next_primary",
        device: str = None,
        warp_backend: str = "auto",
        strict_fidelity: bool = False,
    ):
        self.K = np.asarray(K, dtype=np.float64)
        if self.K.shape != (3, 3):
            raise ValueError("K must have shape (3,3)")
        if disocclusion_policy not in {"hold", "nan", "raise", "request_render"}:
            raise ValueError("disocclusion_policy must be hold, nan, raise, or request_render")
        if composite not in {"weighted", "next_primary"}:
            raise ValueError("composite must be 'weighted' or 'next_primary'")
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
        }

    def _handle_invalid(self, warped, valid, alpha, t0, t1, sampling, gradient):
        invalid = ~np.asarray(valid, dtype=bool)
        if self._endpoint_invalid_mask is not None:
            invalid = invalid | self._endpoint_invalid_mask
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
            )
            self.last_render_requests.append(request)
            if self.disocclusion_policy == "raise" or self.disocclusion_policy == "request_render" or self.strict_fidelity:
                raise RenderRequiredError(
                    f"physics warp has {invalid.mean():.3%} invalid pixels; a real Genesis subframe render is required",
                    self.last_render_requests,
                )
            if self.disocclusion_policy == "hold":
                fallback = self._current_I0 if alpha < 0.5 else self._current_I1
                warped[invalid] = fallback[invalid]
        return warped.astype(np.float32, copy=False), invalid

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
    ) -> Tuple[List[np.ndarray], List[float]]:
        if t1 <= t0:
            raise ValueError("t1 must be greater than t0")
        self.last_invalid_fraction = 0.0
        self.last_invalid_mask = None
        self._endpoint_invalid_mask = None
        self.last_render_requests = []
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
        camera_flow01, valid_camera = compute_se3_flow(
            depth0, seg0, self.K, camera_to_world0, camera_to_world1,
            object_world_deltas=None,
        )
        flow01, valid01 = compute_se3_flow(
            depth0, seg0, self.K, camera_to_world0, camera_to_world1,
            object_world_deltas=object_world_delta,
        )
        inverse_deltas = {
            int(key): np.linalg.inv(np.asarray(value, dtype=np.float64))
            for key, value in (object_world_delta or {}).items()
        }
        flow10, valid10 = compute_se3_flow(
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
        residual = endpoint_radiance_residual(
            I0, I1, depth0, endpoint0[0], endpoint0[1], endpoint0[2], eps=1e-6
        )
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
            disocclusion_mask=~(valid01 & valid10),
            correspondence_depth0=endpoint0[1],
            correspondence_depth1=endpoint1[1],
        )
        self.last_flow01 = flow01
        self.last_flow10 = flow10
        self.last_camera_flow01 = camera_flow01
        self.last_articulation_flow01 = articulation_flow01
        self.last_sampling = sampling
        self._current_I0, self._current_I1 = I0, I1

        H, W = I0.shape
        gradient = np.hypot(*np.gradient(log_radiance(I0)))
        self._endpoint_invalid_mask = ~(valid01 & valid10)
        self._record_uncertainty(sampling, self._endpoint_invalid_mask, residual, gradient)
        if (
            sampling.recommend_render
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
            )
            warped_t, valid_t = batch
            # ``warped_t`` is [B,T,H,W] for luminance and [B,T,H,W,C] for
            # colour.  Keep the batch/temporal axes explicit; callers that
            # need NumPy frames can opt into the documented conversion below.
            self.last_frames_torch = warped_t.contiguous()
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
                    warped_np, valid_np, float(alpha), t0, t1, sampling, gradient
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
        for index, alpha in enumerate(alphas):
            warped, valid = depth_aware_bidirectional_warp_projected(
                I0, depth0, projections[0][index], projections[1][index], projections[2][index],
                I1, depth1, projections[3][index], projections[4][index], projections[5][index],
                composite=self.composite,
            )
            warped, invalid = self._handle_invalid(
                warped, valid, float(alpha), t0, t1, sampling, gradient
            )
            if self.disocclusion_policy == "nan":
                warped[invalid] = np.nan
            frames.append(warped)
        return frames, timestamps
