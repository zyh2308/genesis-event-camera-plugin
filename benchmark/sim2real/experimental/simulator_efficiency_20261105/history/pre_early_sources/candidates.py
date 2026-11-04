"""Addons only: frozen geometry, schedule, RNG draws and full current sensor profile."""
import inspect,math,sys,textwrap
from pathlib import Path
import numpy as np,torch
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'build'))
def prepared_method(emu):
    import genesis_event_plugin.dvs_emulator as dm
    if not hasattr(dm,'generate_events_prepared'):
        text=textwrap.dedent(inspect.getsource(dm.DvsEmulator.generate_events))
        text=text.replace('def generate_events(','def generate_events_prepared(',1)
        old='pixel_vals = self._prepare_input(frame)\n    self.new_frame = torch.tensor(pixel_vals, dtype=torch.float64, device=self.device)'
        assert text.count(old)==1
        text=text.replace(old,'self.new_frame = frame\n    assert frame.is_cuda and frame.dtype == torch.float64 and tuple(frame.shape) == (self.height,self.width)')
        exec(text,dm.__dict__)
    return dm.generate_events_prepared.__get__(emu)
def bulk(emu,frames,times):
    prepared=np.stack([emu._prepare_input(f) for f in frames])
    stack=torch.as_tensor(prepared,device=emu.device,dtype=torch.float64)
    method=prepared_method(emu)
    return [method(f,t) for f,t in zip(stack,times)]
def supported(emu):
    c=emu.cfg
    return (c.response_mode=='physical_log' and c.radiance_input and not c.log_input
            and not c.photoreceptor_noise and emu.calibrated_noise_rate_map is None
            and c.readout_model=='ideal_readout' and c.refractory_period_s>0)
def fused(emu,frames,times):
    import frozen_sensor_fused as cuda
    if not supported(emu):raise NotImplementedError('Fused candidate supports the complete current profile; unsupported features are never disabled')
    rng_before=emu.rng.get_state().clone()
    prepared=np.stack([emu._prepare_input(f) for f in frames])
    stack=torch.as_tensor(prepared,device=emu.device,dtype=torch.float64)
    ts=[float(t) for t in times];prev=[float(emu.t_previous)]+ts[:-1]
    dts=[t-p for t,p in zip(ts,prev)]
    if not all(dt>0 for dt in dts):raise ValueError('strictly increasing times required')
    shape=(emu.height,emu.width);c=emu.cfg
    # Keep exactly the original Torch RNG call shapes, dtype and interleaving.
    leaks=[];shots=[];nts=[]
    for _ in ts:
        leaks.append(torch.randn(shape,dtype=torch.float32,device=emu.device,generator=emu.rng) if c.leak_rate_hz>0 else torch.zeros(shape,dtype=torch.float32,device=emu.device))
        shots.append(torch.rand(shape,dtype=torch.float32,device=emu.device,generator=emu.rng) if c.shot_noise_rate_hz>0 else torch.zeros(shape,dtype=torch.float32,device=emu.device))
        nts.append(torch.rand(shape,dtype=torch.float64,device=emu.device,generator=emu.rng) if c.shot_noise_rate_hz>0 else torch.zeros(shape,dtype=torch.float64,device=emu.device))
    rleak=torch.stack(leaks);rshot=torch.stack(shots);rtime=torch.stack(nts)
    log_frames=emu._to_log_input(stack).contiguous();inten=emu._intensity_fraction(stack)
    dt=torch.tensor(dts,dtype=torch.float64,device=emu.device);tp=torch.tensor(prev,dtype=torch.float64,device=emu.device);tt=torch.tensor(ts,dtype=torch.float64,device=emu.device)
    if c.cutoff_hz>0:
        factors=torch.tensor([d/(1/(math.pi*2*c.cutoff_hz)) for d in dts],dtype=torch.float64,device=emu.device)
        eps=torch.clamp(inten*factors[:,None,None],max=1).contiguous()
    else:eps=torch.ones_like(log_frames)
    curr_leak=c.leak_rate_hz*emu.noise_rate_array[None]*(1-c.leak_jitter_fraction*rleak)
    leak=(dt[:,None,None]*curr_leak*emu.pos_thres[None]).contiguous()
    factors=torch.tensor([(c.shot_noise_rate_hz/2)*d for d in dts],dtype=torch.float64,device=emu.device)
    shot_factor=factors[:,None,None]*((c.shot_noise_inten_factor-1)*inten+1)
    noise_on=(rshot>(1-shot_factor*(emu.pos_thres_nominal/emu.pos_thres)[None])).contiguous()
    noise_off=(rshot<(shot_factor*(emu.neg_thres_nominal/emu.neg_thres)[None])).contiguous()
    noise_times=(tp[:,None,None]+dt[:,None,None]*rtime).contiguous()
    earliest=float(emu.timestamp_mem.min().item())
    cap=int(math.ceil((ts[-1]-earliest)/c.refractory_period_s))+2
    result=cuda.run(log_frames,eps,leak,noise_on,noise_off,noise_times,tt,dt,
                    emu.lp_log_frame.contiguous(),emu.base_log_frame.contiguous(),emu.timestamp_mem.contiguous(),
                    emu.pos_thres.contiguous(),emu.neg_thres.contiguous(),emu.photoreceptor_noise_arr.contiguous(),c.refractory_period_s,cap)
    lp,base,mem,diff,packed,keys,counts,flags=result
    flag=int(flags.item())
    if flag:
        # No state was committed. Restore the generator before exact reference fallback.
        emu.rng.set_state(rng_before)
        emu._candidate_fallbacks=getattr(emu,'_candidate_fallbacks',0)+1
        emu._last_fallback_flags=flag
        return [emu.generate_events(f,t) for f,t in zip(frames,ts)]
    mask=torch.arange(cap,device=emu.device)[None,:]<counts[:,None]
    raw=packed[mask].cpu().numpy();order=keys[mask].cpu().numpy()
    ordering=np.argsort(order,kind='stable');raw=raw[ordering];order=order[ordering]
    steps=order>>48;chunks=[]
    for t in range(len(ts)):
        events=raw[steps==t]
        if len(events):events=events[events[:,0].argsort()]
        chunks.append(events)
    emu.lp_log_frame=lp;emu.base_log_frame=base;emu.timestamp_mem=mem;emu.diff_frame=diff
    emu.new_frame=stack[-1];emu.log_new_frame=log_frames[-1]
    emu.t_previous=ts[-1];emu.frame_counter+=len(ts)
    on_count=int((raw[:,3]>0).sum());emu.num_events_on+=on_count;emu.num_events_off+=len(raw)-on_count;emu.num_events_total+=len(raw)
    return chunks
def make_plugin_class(base,variant):
    original=inspect.getsource(base.capture);text=textwrap.dedent(original)
    old='chunks = [\n        self.dvs_emulator.generate_events(frame, ts)\n        for frame, ts in zip(sensor_frames, timestamps)\n    ]'
    assert text.count(old)==1
    text=text.replace(old,'chunks = self._generate_sequence(sensor_frames, timestamps)')
    import genesis_event_plugin.physics_plugin as pm
    namespace={};exec(text,pm.__dict__,namespace)
    class Candidate(base):
        def attach(self,*args,**kwargs):
            result=super().attach(*args,**kwargs)
            if 'cached' in variant or 'chunk' in variant:
                optimize_geometry(self.interpolator,32 if 'chunk' in variant else None)
            return result
        def _generate_sequence(self,frames,times):
            if variant in {'cached','chunk'}:
                result=[self.dvs_emulator.generate_events(f,t) for f,t in zip(frames,times)]
            else:result=(bulk if variant=='bulk' else fused)(self.dvs_emulator,frames,times)
            self.fallbacks=getattr(self.dvs_emulator,'_candidate_fallbacks',0);return result
    Candidate.capture=namespace['capture'];return Candidate

def optimize_geometry(interpolator,chunk_size=None):
    """Scoped candidate: exact endpoint log caching and bulk output transfers.

    No source files are modified. Chunking uses the same frozen Torch function,
    with each timestamp independent; numerical equivalence is tested separately.
    """
    import genesis_event_plugin.physics_interpolator as module
    import genesis_event_plugin.torch_physics as geometry
    source=textwrap.dedent(inspect.getsource(type(interpolator)._interpolate_pair))
    needle='frames = []\n        for index, alpha in enumerate(alphas):'
    assert source.count(needle)==1
    source=source.replace(needle,'warped_bulk = self.last_frames_torch[0].detach().cpu().numpy()\n        valid_bulk = valid_t[0].detach().cpu().numpy().astype(bool)\n        '+needle)
    source=source.replace('warped_np = warped.detach().cpu().numpy()','warped_np = warped_bulk[index]')
    source=source.replace('valid_np = valid.detach().cpu().numpy().astype(bool)','valid_np = valid_bulk[index]')
    scope=dict(module.__dict__)
    exec(source,scope)
    pair=scope['_interpolate_pair'].__get__(interpolator)
    original_warp=geometry.depth_aware_bidirectional_warp_projected_batch_torch
    def warp_chunks(*args,**kwargs):
        n=args[2].shape[1]
        outputs=[]
        for start in range(0,n,chunk_size):
            stop=min(start+chunk_size,n);part=list(args)
            for k in [2,3,4,7,8,9]:part[k]=args[k][:,start:stop]
            options=dict(kwargs);options['alpha']=kwargs['alpha'][start:stop]
            outputs.append(original_warp(*part,**options))
        return (torch.cat([v[0] for v in outputs],dim=1),torch.cat([v[1] for v in outputs],dim=1),
                {k:torch.cat([v[2][k] for v in outputs],dim=1) for k in outputs[0][2]})
    def scoped_pair(*args,**kwargs):
        interpolator._cached_endpoint_logs=None
        if chunk_size:geometry.depth_aware_bidirectional_warp_projected_batch_torch=warp_chunks
        try:return pair(*args,**kwargs)
        finally:
            geometry.depth_aware_bidirectional_warp_projected_batch_torch=original_warp
            interpolator._cached_endpoint_logs=None
    interpolator._interpolate_pair=scoped_pair
    text=textwrap.dedent(inspect.getsource(type(interpolator)._handle_invalid))
    old='fallback = np.exp(\n                    (1.0 - float(alpha)) * np.log(np.maximum(self._current_I0, 1e-6))\n                    + float(alpha) * np.log(np.maximum(self._current_I1, 1e-6))\n                )'
    assert text.count(old)==1
    new='if self._cached_endpoint_logs is None:\n                    self._cached_endpoint_logs = (np.log(np.maximum(self._current_I0, 1e-6)),np.log(np.maximum(self._current_I1, 1e-6)))\n                logs0,logs1=self._cached_endpoint_logs\n                fallback = np.exp((1.0 - float(alpha))*logs0 + float(alpha)*logs1)'
    text=text.replace(old,new);exec(text,scope)
    interpolator._handle_invalid=scope['_handle_invalid'].__get__(interpolator)
