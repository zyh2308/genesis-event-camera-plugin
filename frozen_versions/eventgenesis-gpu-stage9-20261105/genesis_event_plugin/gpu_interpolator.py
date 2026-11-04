"""Instance-local GPU interpolation; all reference numerical rules retained."""
import torch
from collections.abc import MutableMapping,Sequence
from .physics_interpolator import PhysicsInterpolator,RenderRequest,RenderRequiredError
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

class DeviceRenderRequest(RenderRequest):
    @property
    def mask(self):
        value=self._mask
        if torch.is_tensor(value):value=value.detach().cpu().numpy();self._mask=value
        return value
    @mask.setter
    def mask(self,value):self._mask=value


class DeferredDiagnostics(MutableMapping):
    """Preserve readable NumPy diagnostics, exporting device arrays on access."""
    def __init__(self,initial=()):self.data=dict(initial)
    def __getitem__(self,key):
        value=self.data[key]
        if torch.is_tensor(value):value=value.detach().cpu().numpy();self.data[key]=value
        return value
    def __setitem__(self,key,value):self.data[key]=value
    def __delitem__(self,key):del self.data[key]
    def __iter__(self):return iter(self.data)
    def __len__(self):return len(self.data)
    def materialize(self):return {key:self[key] for key in self}


class DeferredList(Sequence):
    def __init__(self,items):self.items=list(items)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        value=self.items[index]
        if torch.is_tensor(value):value=value.detach().cpu().numpy();self.items[index]=value
        return value
    def __len__(self):return len(self.items)


def finish_geometry(self,warped,valid,alphas,t0,t1,sampling,gradient,allow_approximate,return_torch,mode):
    """Keep frames on GPU; exact mode sends only hole fallback values back."""
    from genesis_event_plugin.physics_interpolator import RenderRequest,RenderRequiredError
    raw=warped[0];invalid=~valid[0];holes=invalid&~torch.isfinite(raw)
    if self.disocclusion_policy!='hold' or self.composite!='coverage_time_log_linear':
        raise NotImplementedError('resident candidate requires the complete current hold/coverage profile')
    # One compact mask transfer replaces all intermediate image transfers.
    statistics=torch.stack((invalid.sum(dim=(1,2)),holes.sum(dim=(1,2))),dim=1).cpu().numpy()
    hole_pixels=int(statistics[:,1].sum());hole_values=[]
    encoded=holes.cpu().numpy() if mode=='exact' and hole_pixels else None
    if mode=='gpu' and hole_pixels:
        log0=torch.log(torch.clamp(torch.as_tensor(self._current_I0,device=raw.device),min=1e-6))
        log1=torch.log(torch.clamp(torch.as_tensor(self._current_I1,device=raw.device),min=1e-6))
        a=torch.as_tensor(alphas,dtype=torch.float32,device=raw.device)[:,None,None]
        complement=torch.as_tensor(1.0-alphas,dtype=torch.float32,device=raw.device)[:,None,None]
        fallback=torch.exp(complement*log0+a*log1)
        resident=torch.where(holes,fallback,raw)
    else:resident=raw
    logs=None
    for index,alpha in enumerate(alphas):
        bad_count=int(statistics[index,0])
        if not bad_count:continue
        diagnostics=dict(invalid_fraction=float(bad_count/(raw.shape[1]*raw.shape[2])),radiance_residual_p99=float(sampling.p99_radiance_residual),recommend_render=True)
        self.last_render_requests.append(DeviceRenderRequest(t=float(t0+float(alpha)*(t1-t0)),alpha=float(alpha),reason='disocclusion_or_invalid_projection',mask=invalid[index],diagnostics=diagnostics,available_at=float(t1)))
        if self.strict_fidelity and not allow_approximate:
            raise RenderRequiredError('resident warp requires a real Genesis subframe',self.last_render_requests)
        count=int(statistics[index,1]);missing=encoded[index] if encoded is not None else None
        if count and mode=='exact':
            if logs is None:logs=(np.log(np.maximum(self._current_I0,1e-6)),np.log(np.maximum(self._current_I1,1e-6)))
            fallback=np.exp((1-float(alpha))*logs[0]+float(alpha)*logs[1])
            hole_values.append(fallback[missing])
    if mode=='exact' and hole_values:
        values=torch.as_tensor(np.concatenate(hole_values),device=raw.device,dtype=raw.dtype)
        resident=raw.clone();resident.masked_scatter_(holes,values)
    self._resident_frames=resident
    self._resident_last_hole_pixels=hole_pixels
    self.last_visibility['intermediate_visibility']=valid[0]
    self.last_visibility['intermediate_unknown']=invalid
    if return_torch:
        self.last_frames_torch=resident[None]
        return self.last_frames_torch,[float(t0+float(a)*(t1-t0)) for a in alphas]
    return list(resident.unbind(0)),[float(t0+float(a)*(t1-t0)) for a in alphas]


class ResidentPhysicsInterpolator(PhysicsInterpolator):
    @classmethod
    def from_reference(cls,reference):
        result=cls.__new__(cls)
        result.__dict__.update(reference.__dict__)
        return result

    def interpolate_device(self,*args,**kwargs):
        kwargs.pop('return_torch',None)
        return super().interpolate(*args,return_torch=False,**kwargs)

    def interpolate(self,*args,**kwargs):
        requested=kwargs.pop('return_torch',False)
        frames,times=self.interpolate_device(*args,**kwargs)
        if requested:
            self.last_frames_torch=torch.stack(frames)[None]
            return self.last_frames_torch,times
        return list(torch.stack(frames).cpu().numpy()),times

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
        # Keep the pose interpolation parameters in double precision. The
        # NumPy geometry path evaluates k / samples as Python float; matching
        # that here prevents the Torch caller from rounding alpha before its
        # float64 projection kernel receives it.
        alphas = np.arange(1, sampling.samples + 1, dtype=np.float64) / float(sampling.samples)
        timestamps = [float(t0 + float(a) * dt) for a in alphas]

        # The projected NumPy path is the numerical reference. The Torch path
        # computes the same endpoint geometry with the tensor-native projection
        # stack and keeps all warp arithmetic on device.
        if self.warp_backend == "torch":
            import torch
            from .gpu_geometry import (
                depth_aware_bidirectional_warp_projected_batch_torch,
                project_endpoint_to_alpha_stack_torch,
            )

            torch_device = torch.device(self.device or "cuda")
            # Keep geometry inputs double precision until projection. The
            # projected uv/z arrays are cast to float32 by the Torch kernel,
            # matching the NumPy public contract without losing precision at
            # the call boundary.
            alphas_t = torch.as_tensor(alphas, dtype=torch.float64, device=torch_device)
            T0_t = torch.as_tensor(camera_to_world0, dtype=torch.float64, device=torch_device)
            T1_t = torch.as_tensor(camera_to_world1, dtype=torch.float64, device=torch_device)
            K_t = torch.as_tensor(self.K, dtype=torch.float64, device=torch_device)
            d0_t = torch.as_tensor(depth0, dtype=torch.float32, device=torch_device)
            d1_t = torch.as_tensor(depth1, dtype=torch.float32, device=torch_device)
            seg0_t = torch.as_tensor(seg0, device=torch_device)
            seg1_t = torch.as_tensor(seg1, device=torch_device)
            object0_t = {
                int(key): torch.as_tensor(value, dtype=torch.float64, device=torch_device)
                for key, value in (object_world0 or {}).items()
            }
            object1_t = {
                int(key): torch.as_tensor(value, dtype=torch.float64, device=torch_device)
                for key, value in (object_world1 or {}).items()
            }
            delta_t = {
                int(key): torch.as_tensor(value, dtype=torch.float64, device=torch_device)
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
            self.last_intermediate_z = DeferredList([z0,z1])
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
            self.last_visibility = DeferredDiagnostics(self.last_visibility)
            for key in torch_visibility:
                self.last_visibility[key] = torch_visibility[key][0]
            if return_torch and self.disocclusion_policy == "raise" and bool((~valid_t).any()):
                raise RenderRequiredError(
                    "Torch projected warp has invalid pixels; a real Genesis subframe is required",
                    self.last_render_requests,
                )
            return finish_geometry(self,warped_t,valid_t,alphas,t0,t1,sampling,gradient,_allow_approximate,return_torch,'exact')

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
