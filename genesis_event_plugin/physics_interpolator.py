"""Radiance-domain, Genesis-pose-driven temporal interpolation."""

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from .adaptive_sampler import AdaptiveSamplingConfig, PhysicsAdaptiveSampler
from .physics_warp import compute_se3_flow, depth_aware_bidirectional_warp
from .torch_physics import depth_aware_bidirectional_warp_torch
from .radiometry import prepare_linear_radiance


class PhysicsInterpolator:
    """Generate intermediate linear-luminance frames from physical motion.

    The class deliberately does not advance Genesis.  The caller owns physics
    integration and supplies endpoint poses; any unsupported/disoccluded
    pixels are reported through ``last_invalid_fraction`` so a production
    plugin can request an actual renderer fallback instead of hallucinating.
    """

    def __init__(
        self,
        K: np.ndarray,
        sampler_config: AdaptiveSamplingConfig = None,
        disocclusion_policy: str = "hold",
        composite: str = "next_primary",
        device: str = None,
        warp_backend: str = "auto",
    ):
        self.K = np.asarray(K, dtype=np.float64)
        if self.K.shape != (3, 3):
            raise ValueError("K must have shape (3,3)")
        if disocclusion_policy not in {"hold", "nan", "raise"}:
            raise ValueError("disocclusion_policy must be hold, nan, or raise")
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
        self.last_flow01 = None
        self.last_flow10 = None
        self.last_camera_flow01 = None
        self.last_articulation_flow01 = None
        self.last_invalid_fraction = 0.0
        self.last_invalid_mask = None
        self.last_sampling = None

    @staticmethod
    def _luminance(frame: np.ndarray, input_space: str) -> np.ndarray:
        return prepare_linear_radiance(frame, input_space=input_space)

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
    ) -> Tuple[List[np.ndarray], List[float]]:
        if t1 <= t0:
            raise ValueError("t1 must be greater than t0")
        self.last_invalid_fraction = 0.0
        self.last_invalid_mask = None
        I0 = self._luminance(frame0, input_space)
        I1 = self._luminance(frame1, input_space)
        dt = float(t1 - t0)
        # Keep camera ego-motion separate from link/object motion.  The full
        # flow is still used for the warp; the split is only for the adaptive
        # controller so its diagnostic U_camera/U_articulation terms are
        # physically interpretable.
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
        sampling = self.sampler.compute(
            I0,
            flow01,
            dt,
            depth0=depth0,
            depth1=depth1,
            seg0=seg0,
            seg1=seg1,
            camera_flow=camera_flow01,
            articulation_flow=articulation_flow01,
        )
        self.last_flow01 = flow01
        self.last_flow10 = flow10
        self.last_camera_flow01 = camera_flow01
        self.last_articulation_flow01 = articulation_flow01
        self.last_sampling = sampling

        frames: List[np.ndarray] = []
        timestamps: List[float] = []
        torch_state = None
        if self.warp_backend == "torch":
            import torch

            torch_device = torch.device(self.device or "cuda")
            torch_state = (
                torch.as_tensor(I0, dtype=torch.float32, device=torch_device),
                torch.as_tensor(depth0, dtype=torch.float32, device=torch_device),
                torch.as_tensor(flow01, dtype=torch.float32, device=torch_device),
                torch.as_tensor(I1, dtype=torch.float32, device=torch_device),
                torch.as_tensor(depth1, dtype=torch.float32, device=torch_device),
                torch.as_tensor(flow10, dtype=torch.float32, device=torch_device),
            )
        for k in range(1, sampling.samples + 1):
            alpha = k / float(sampling.samples)
            if torch_state is None:
                warped, valid = depth_aware_bidirectional_warp(
                    I0, depth0, flow01, I1, depth1, flow10, alpha,
                    composite=self.composite,
                )
            else:
                warped_t, valid_t = depth_aware_bidirectional_warp_torch(
                    *torch_state, alpha, composite=self.composite,
                )
                warped = warped_t.detach().cpu().numpy()
                valid = valid_t.detach().cpu().numpy().astype(bool, copy=False)
            invalid = ~valid
            invalid_fraction = float(invalid.mean())
            self.last_invalid_fraction = max(self.last_invalid_fraction, invalid_fraction)
            if self.last_invalid_mask is None:
                self.last_invalid_mask = invalid.copy()
            else:
                self.last_invalid_mask |= invalid
            if np.any(invalid):
                if self.disocclusion_policy == "raise":
                    raise RuntimeError(
                        f"physics warp has {invalid_fraction:.3%} disoccluded pixels; "
                        "request a real Genesis render for this subinterval"
                    )
                if self.disocclusion_policy == "hold":
                    fallback = I0 if alpha < 0.5 else I1
                    warped[invalid] = fallback[invalid]
            frames.append(warped.astype(np.float32, copy=False))
            timestamps.append(float(t0 + alpha * dt))
        return frames, timestamps
