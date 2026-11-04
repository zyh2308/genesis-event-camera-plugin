"""FP64 sensor state machine, original RNG order and exact fallback."""
import math
import numpy as np
import torch

def supported(emu):
    c=emu.cfg
    return (c.response_mode=='physical_log' and c.radiance_input and not c.log_input
            and not c.photoreceptor_noise and emu.calibrated_noise_rate_map is None
            and c.readout_model=='ideal_readout' and c.refractory_period_s>0)


def generate_device_sequence(emu,frames,times):
    from .gpu_kernel import load_kernel
    cuda=load_kernel()
    if not supported(emu):raise NotImplementedError('Fused candidate supports the complete current profile; unsupported features are never disabled')
    rng_before=emu.rng.get_state().clone()
    assert torch.is_tensor(frames) and frames.is_cuda
    if tuple(frames.shape[1:])!=(emu.height,emu.width):raise ValueError('sensor frame shape')
    if not bool(torch.isfinite(frames).all()) or bool((frames<0).any()):raise ValueError('invalid radiance')
    stack=frames.to(dtype=torch.float64)*float(emu.cfg.radiance_scale)
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
    # Every emitted time belongs to this input interval, irrespective of how
    # old a quiet pixel's timestamp memory is. Bound accepted events by the
    # current duration and refractory spacing, not total episode age.
    cap=int(math.ceil((ts[-1]-prev[0])/c.refractory_period_s))+2
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
        host=frames.detach().cpu().numpy()
        return [emu.generate_events(f,t) for f,t in zip(host,ts)]
    mask=torch.arange(cap,device=emu.device)[None,:]<counts[:,None]
    raw=packed[mask].cpu().numpy();order=keys[mask].cpu().numpy()
    ordering=np.argsort(order,kind='stable');raw=raw[ordering];order=order[ordering]
    steps=order>>48;chunks=[]
    for t in range(len(ts)):
        events=raw[steps==t]
        if len(events):events=events[events[:,0].argsort()]
        chunks.append(events)
    emu.lp_log_frame=lp;emu.base_log_frame=base;emu.timestamp_mem=mem;emu.diff_frame=diff
    emu.new_frame=stack[-1].clone();emu.log_new_frame=log_frames[-1].clone()
    emu.t_previous=ts[-1];emu.frame_counter+=len(ts)
    on_count=int((raw[:,3]>0).sum());emu.num_events_on+=on_count;emu.num_events_off+=len(raw)-on_count;emu.num_events_total+=len(raw)
    return chunks
