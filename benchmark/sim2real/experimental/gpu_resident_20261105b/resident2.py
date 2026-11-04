"""Isolated GPU frame pipeline; the frozen simulator files are never changed."""
import inspect,sys,textwrap
from collections.abc import MutableMapping,Sequence
import numpy as np
import torch
import prior_candidates as prior

from genesis_event_plugin.physics_interpolator import RenderRequest

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

def sensor_device(emu,frames,times):
    """Reuse the validated FP64 sensor kernel and RNG call sequence exactly."""
    import prior_candidates as module
    if not hasattr(module,'fused_device'):
        source=textwrap.dedent(inspect.getsource(module.fused)).replace('def fused(','def fused_device(',1)
        old='prepared=np.stack([emu._prepare_input(f) for f in frames])\n    stack=torch.as_tensor(prepared,device=emu.device,dtype=torch.float64)'
        assert source.count(old)==1
        new="assert torch.is_tensor(frames) and frames.is_cuda\n    if tuple(frames.shape[1:])!=(emu.height,emu.width):raise ValueError('sensor frame shape')\n    if not bool(torch.isfinite(frames).all()) or bool((frames<0).any()):raise ValueError('invalid radiance')\n    stack=frames.to(dtype=torch.float64)*float(emu.cfg.radiance_scale)"
        source=source.replace(old,new)
        source=source.replace('return [emu.generate_events(f,t) for f,t in zip(frames,ts)]','host=frames.detach().cpu().numpy()\n        return [emu.generate_events(f,t) for f,t in zip(host,ts)]')
        source=source.replace('emu.new_frame=stack[-1];emu.log_new_frame=log_frames[-1]','emu.new_frame=stack[-1].clone();emu.log_new_frame=log_frames[-1].clone()')
        exec(source,module.__dict__)
    return module.fused_device(emu,frames,times)

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

def install_geometry(interpolator,mode):
    import genesis_event_plugin.physics_interpolator as module
    source=textwrap.dedent(inspect.getsource(type(interpolator)._interpolate_pair))
    old='self.last_intermediate_z = [z0.detach().cpu().numpy(), z1.detach().cpu().numpy()]'
    assert source.count(old)==1
    source=source.replace(old,'self.last_intermediate_z = DeferredList([z0,z1])')
    old='for key in torch_visibility:\n            self.last_visibility[key] = torch_visibility[key][0].detach().cpu().numpy()\n        self.last_visibility["intermediate_visibility"] = valid_t[0].detach().cpu().numpy().astype(bool)\n        self.last_visibility["intermediate_unknown"] = ~self.last_visibility["intermediate_visibility"]'
    assert source.count(old)==1
    source=source.replace(old,'self.last_visibility = DeferredDiagnostics(self.last_visibility)\n        for key in torch_visibility:\n            self.last_visibility[key] = torch_visibility[key][0]')
    start=source.index('        frames = []');stop=source.index('\n\n    projections =',start)
    source=source[:start]+"        return finish_geometry(self,warped_t,valid_t,alphas,t0,t1,sampling,gradient,_allow_approximate,return_torch,self._resident_mode)"+source[stop:]
    scope=dict(module.__dict__);scope.update(DeferredList=DeferredList,DeferredDiagnostics=DeferredDiagnostics,finish_geometry=finish_geometry)
    exec(source,scope);pair=scope['_interpolate_pair'].__get__(interpolator)
    import genesis_event_plugin.torch_physics as geometry
    original_splat=geometry._splat_projected_coverage_batch_torch
    splat_source=textwrap.dedent(inspect.getsource(original_splat)).replace('unresolved = next_unresolved','unresolved = next_unresolved\n            if not bool(torch.stack([item.any() for item in unresolved]).any()):\n                break')
    splat_scope=dict(geometry.__dict__);exec(splat_source,splat_scope);efficient_splat=splat_scope['_splat_projected_coverage_batch_torch']
    def scoped_pair(*args,**kwargs):
        geometry._splat_projected_coverage_batch_torch=efficient_splat
        try:return pair(*args,**kwargs)
        finally:geometry._splat_projected_coverage_batch_torch=original_splat
    interpolator._interpolate_pair=scoped_pair
    interpolator._resident_mode=mode
    original=interpolator.interpolate
    def interpolate(*args,**kwargs):
        requested=kwargs.pop('return_torch',False)
        frames,times=original(*args,return_torch=False,**kwargs)
        if requested:
            interpolator.last_frames_torch=torch.stack(frames)[None]
            return interpolator.last_frames_torch,times
        return frames,times
    interpolator.interpolate=interpolate

def make_plugin_class(base,mode='exact'):
    import genesis_event_plugin.physics_plugin as module
    source=textwrap.dedent(inspect.getsource(base.capture))
    old='chunks = [\n        self.dvs_emulator.generate_events(frame, ts)\n        for frame, ts in zip(sensor_frames, timestamps)\n    ]'
    assert source.count(old)==1
    source=source.replace(old,'chunks = self._generate_device_sequence(sensor_frames,timestamps)')
    scope=dict(module.__dict__);exec(source,scope)
    class Resident(base):
        def attach(self,*args,**kwargs):
            if self.camera_margin is not None or self.intermediate_state_provider is not None:
                raise NotImplementedError('margin/provider adapters are outside the current cached profile')
            result=super().attach(*args,**kwargs);install_geometry(self.interpolator,mode);return result
        def _generate_device_sequence(self,frames,times):
            value=sensor_device(self.dvs_emulator,torch.stack(frames),times)
            self.fallbacks=getattr(self.dvs_emulator,'_candidate_fallbacks',0);return value
        def materialize_diagnostics(self):
            if isinstance(self.interpolator.last_visibility,DeferredDiagnostics):self.interpolator.last_visibility.materialize()
            list(self.interpolator.last_intermediate_z)
    Resident.capture=scope['capture'];return Resident
