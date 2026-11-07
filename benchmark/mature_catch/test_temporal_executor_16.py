"""Compare unchanged full vs temporally grouped CUDA kernels on real endpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    uuid=subprocess.check_output(['nvidia-smi','-i',os.environ['EGL_DEVICE_ID'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    os.environ['CUDA_VISIBLE_DEVICES']=uuid
    import h5py
    import numpy as np
    import torch
    from bounded_torch_executor import make_bounded
    from genesis_event_plugin.torch_physics import project_endpoint_to_alpha_stack_torch,depth_aware_bidirectional_warp_projected_batch_torch
    from genesis_event_plugin.radiometry import prepare_linear_radiance
    from genesis_event_plugin.dvs_emulator import DvsEmulator
    from genesis_event_plugin.noise_model import DvsNoiseConfig
    torch.set_num_threads(2);device='cuda:0';start=time.monotonic();rows=[]
    profile=json.loads(Path('/home/zhaoyuhan/Eventbased_WAM/frozen_baselines/robot-sparse-cuda-r2-20261102/event_sparse_reconstruction_20261102.json').read_text())
    with h5py.File(args.cache,'r') as h:
        ids=h['segmentation_pose_ids'][:];K=torch.as_tensor(h['intrinsics'][:],dtype=torch.float64,device=device)
        times=h['time_s'][:]
        for wanted in (.52,.70,.96):
            k=int(np.argmin(abs(times-wanted)));j=k+1
            if j>=len(times):continue
            I0=prepare_linear_radiance(h['radiance'][k],input_space='linear');I1=prepare_linear_radiance(h['radiance'][j],input_space='linear')
            d0=torch.as_tensor(h['depth'][k],dtype=torch.float32,device=device);d1=torch.as_tensor(h['depth'][j],dtype=torch.float32,device=device)
            s0=torch.as_tensor(h['segmentation'][k],device=device);s1=torch.as_tensor(h['segmentation'][j],device=device)
            T0=torch.as_tensor(h['camera_to_world_cv'][k],dtype=torch.float64,device=device);T1=torch.as_tensor(h['camera_to_world_cv'][j],dtype=torch.float64,device=device)
            object0={int(i):torch.as_tensor(v,dtype=torch.float64,device=device) for i,v in zip(ids,h['object_poses'][k])}
            object1={int(i):torch.as_tensor(v,dtype=torch.float64,device=device) for i,v in zip(ids,h['object_poses'][j])}
            alpha=torch.arange(1,33,dtype=torch.float64,device=device)/32
            uv0,z0,v0=project_endpoint_to_alpha_stack_torch(d0,s0,K,T0,T0,T1,alpha,0,object0,object1,{})
            uv1,z1,v1=project_endpoint_to_alpha_stack_torch(d1,s1,K,T1,T0,T1,alpha,1,object0,object1,{})
            a=(torch.as_tensor(I0,dtype=torch.float32,device=device)[None],d0[None],uv0[None],z0[None],v0[None],
               torch.as_tensor(I1,dtype=torch.float32,device=device)[None],d1[None],uv1[None],z1[None],v1[None])
            kw=dict(composite=profile['plugin_kwargs']['warp_composite'],return_diagnostics=True,alpha=alpha,
                    surface_id0=s0.long()[None],surface_id1=s1.long()[None],min_coverage_fraction=.995)
            torch.cuda.reset_peak_memory_stats();full=depth_aware_bidirectional_warp_projected_batch_torch(*a,**kw);full_peak=torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats();bounded=make_bounded(depth_aware_bidirectional_warp_projected_batch_torch,16)(*a,**kw);bounded_peak=torch.cuda.max_memory_allocated()
            assert torch.equal(full[1],bounded[1]),'visibility differs'
            for name in full[2]:
                if full[2][name].dtype in (torch.bool,torch.int64,torch.int32):assert torch.equal(full[2][name],bounded[2][name]),name
            valid=full[1]&torch.isfinite(full[0])&torch.isfinite(bounded[0])
            delta=float((full[0][valid]-bounded[0][valid]).abs().max()) if bool(valid.any()) else 0.
            assert delta<=1e-6,delta
            outputs=[]
            for result in (full,bounded):
                emulator=DvsEmulator(res=I0.shape,config=DvsNoiseConfig(**profile['noise_config_kwargs']),device='cuda:0');emulator.initialize(I0,t=float(times[k]));events=[]
                for n in range(32):
                    frame=result[0][0,n].detach().cpu().numpy().copy();mask=result[1][0,n].detach().cpu().numpy();frame[~mask]=I0[~mask]
                    e=emulator.generate_events(frame,t=float(times[k]+(n+1)/32*(times[j]-times[k])))
                    e=e.detach().cpu().numpy() if hasattr(e,'detach') else np.asarray(e)
                    if len(e):events.append(e)
                e=np.concatenate(events) if events else np.empty((0,4));e=e[np.lexsort((e[:,0],e[:,3],e[:,2],e[:,1]))] if len(e) else e;outputs.append(e)
            assert outputs[0].shape==outputs[1].shape,(outputs[0].shape,outputs[1].shape)
            assert np.array_equal(outputs[0][:,1:],outputs[1][:,1:]),'event coordinates/polarity differ'
            event_dt=float(np.max(np.abs(outputs[0][:,0]-outputs[1][:,0]))) if len(outputs[0]) else 0.
            assert event_dt<=1e-6,event_dt
            rows.append(dict(time_s=float(times[k]),temporal_samples=32,visibility_equal=True,integer_diagnostics_equal=True,
                supported_radiance_max_delta=delta,event_count=len(outputs[0]),event_max_time_delta_s=event_dt,
                full_peak_allocated_bytes=full_peak,grouped_peak_allocated_bytes=bounded_peak))
            del full,bounded,a;torch.cuda.empty_cache()
    result=dict(status='PASS',scope='external grouping of unchanged public kernel; same samples and thresholds',gpu_uuid=uuid,cases=rows,wall_s=time.monotonic()-start)
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
