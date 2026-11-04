"""A saved CUDA simulator state is resumed by a genuinely new OS process."""
import json,math,subprocess,sys
from pathlib import Path
import numpy as np,torch
root=Path(__file__).resolve().parent
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261103-stage9-direct-v1'
profile=json.loads((R/'sources/stage9_sensor_profile.json').read_text())
sys.path.insert(0,json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())['core_path'])
from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from candidates import fused
torch.set_num_threads(2)
out=root/'process_resume_tests'
def inputs():
    H,W=32,48;first=np.full((H,W),.2,np.float32)
    frames=[np.full((H,W),.2*math.exp(2.5*math.sin((i+1)/377*3)),np.float32) for i in range(377)]
    times=[1+(i+1)*.01/377 for i in range(377)]
    return first,frames,times
def init(first):
    e=DvsEmulator(res=first.shape,config=DvsNoiseConfig(**profile['noise_config_kwargs']),device='cuda:0');e.initialize(first,1.0)
    # Regression: a quiet pixel's very old memory must not cause allocation
    # proportional to episode age. All current event times are still >=1s.
    e.timestamp_mem-=100
    return e
def save(e,path):
    state={k:v.detach().cpu().clone() for k,v in vars(e).items() if torch.is_tensor(v)}
    state['rng']=e.rng.get_state().cpu();state['scalars']={k:getattr(e,k) for k in ['t_previous','frame_counter','num_events_on','num_events_off','num_events_total']}
    torch.save(state,path)
if len(sys.argv)==1:
    out.mkdir(exist_ok=False)
    for stage in ['reference','left','right']:
        subprocess.run([sys.executable,'-u',__file__,stage],check=True)
    ref=np.load(out/'reference.npy');actual=np.concatenate([np.load(out/'left.npy'),np.load(out/'right.npy')])
    assert np.array_equal(ref,actual),'events differ'
    a=torch.load(out/'reference.pt',weights_only=False);b=torch.load(out/'right.pt',weights_only=False)
    assert a.keys()==b.keys()
    exact={k:bool(torch.equal(a[k],b[k])) for k in a if k!='scalars'}
    assert all(exact.values()),exact
    assert a['scalars']==b['scalars']
    report=dict(passed=True,independent_process=True,event_bitwise_equal=True,event_count=len(ref),state_bitwise_equal=exact,old_timestamp_memory_regression=True,real_data_used=False)
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
else:
    stage=sys.argv[1];first,frames,times=inputs();e=init(first);split=163
    if stage=='reference':chunks=[e.generate_events(f,t) for f,t in zip(frames,times)]
    elif stage=='left':chunks=fused(e,frames[:split],times[:split])
    else:
        state=torch.load(out/'left.pt',map_location='cpu',weights_only=False)
        for k,v in state.items():
            if k=='rng':e.rng.set_state(v)
            elif k=='scalars':
                for name,val in v.items():setattr(e,name,val)
            else:setattr(e,k,v.to('cuda:0'))
        chunks=fused(e,frames[split:],times[split:])
    np.save(out/(stage+'.npy'),np.concatenate(chunks));save(e,out/(stage+'.pt'))
