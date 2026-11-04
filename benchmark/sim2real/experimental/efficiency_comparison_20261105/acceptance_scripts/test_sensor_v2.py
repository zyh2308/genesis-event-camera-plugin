"""Independent frozen-vs-candidate semantic checks including fallback and resume."""
import copy,hashlib,json,math,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
root=Path(__file__).resolve().parent
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261103-stage9-direct-v1'
cfg=json.loads((R/'sources/stage9_sensor_profile.json').read_text())['noise_config_kwargs']
core=json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())['core_path']
sys.path.insert(0,str(root.parent/'release_candidate_static'))
from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.gpu_sensor import generate_device_sequence as sensor_device
def fused(emu,frames,times):return sensor_device(emu,torch.as_tensor(np.stack(frames),device=emu.device),times)
from genesis_event_plugin.gpu_kernel import load_kernel
cuda=load_kernel()
torch.set_num_threads(2)
report={'thresholds':{'event_time_atol_s':1e-12,'state_atol':1e-12},'tests':[],'real_data_used':False}
out=root/'sensor_tests_v2';out.mkdir(exist_ok=False)
def emu(config,first):
    e=DvsEmulator(res=first.shape,config=DvsNoiseConfig(**config),device='cuda:0');e.initialize(first,1.0);return e
def snapshot(e):
    return {**{k:v.detach().cpu().clone() for k,v in vars(e).items() if torch.is_tensor(v)},'rng':e.rng.get_state().cpu(),
            **{k:getattr(e,k) for k in ['t_previous','frame_counter','num_events_on','num_events_off','num_events_total']}}
def compare_events(a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    assert np.array_equal(a[:,1:],b[:,1:]),'event addresses/order/polarities differ'
    err=float(np.max(np.abs(a[:,0]-b[:,0]),initial=0.0));assert err<=1e-12,err
    return dict(count=len(a),bitwise_equal=bool(np.array_equal(a,b)),max_time_error_s=err)
def compare_states(a,b):
    errors={}
    assert a.keys()==b.keys(),(a.keys(),b.keys())
    for k in a:
        if torch.is_tensor(a[k]):
            assert a[k].shape==b[k].shape,k
            if a[k].is_floating_point():
                err=float((a[k]-b[k]).abs().max());assert err<=1e-12,(k,err);errors[k]=err
            else:assert torch.equal(a[k],b[k]),k
        else:assert a[k]==b[k],k
    return errors
def flatten(chunks):return np.concatenate(chunks) if any(len(x) for x in chunks) else np.empty((0,4),np.float64)
for case in ['static','ramp_up','ramp_down','burst','noise_stress','zero_slope_fallback','overflow_fallback','resume_chunks']:
    H,W=12,16;T=41 if case in ['burst','noise_stress'] else 37
    first=np.full((H,W),.2,dtype=np.float32);times=[1.0+.01*(i+1)/T for i in range(T)];config=dict(cfg)
    if case=='static':frames=[first.copy() for _ in times]
    elif case in ['ramp_up','ramp_down','resume_chunks']:
        direction=-1 if case=='ramp_down' else 1
        frames=[np.full((H,W),.2*math.exp(direction*3*(i+1)/T),np.float32) for i in range(T)]
    else:frames=[np.full((H,W),.01 if i%2 else 1.0,np.float32) for i in range(T)]
    if case=='noise_stress':config.update(shot_noise_rate_hz=500.0,leak_rate_hz=20.0)
    if case=='zero_slope_fallback':
        config.update(cutoff_hz=0,leak_rate_hz=0,shot_noise_rate_hz=0)
        frames=[first.copy() for _ in times]
    ref=emu(config,first);fast=emu(config,first)
    if case=='zero_slope_fallback':ref.base_log_frame-=.8;fast.base_log_frame-=.8
    expected=flatten([ref.generate_events(f,t) for f,t in zip(frames,times)])
    if case=='overflow_fallback':
        actual_run=cuda.run
        with patch.object(cuda,'run',lambda *args:actual_run(*args[:-1],1)):
            actual=flatten(fused(fast,frames,times))
        assert fast._last_fallback_flags&2
    elif case=='resume_chunks':
        split=17;left=fused(fast,frames[:split],times[:split]);state=snapshot(fast);checkpoint=out/'resume.pt';torch.save(state,checkpoint)
        restored=emu(config,first);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for k,v in saved.items():
            if k=='rng':restored.rng.set_state(v)
            elif torch.is_tensor(v):setattr(restored,k,v.to('cuda'))
            else:setattr(restored,k,v)
        right=fused(restored,frames[split:],times[split:]);actual=flatten(left+right);fast=restored
    else:actual=flatten(fused(fast,frames,times))
    ev=compare_events(expected,actual);states=compare_states(snapshot(ref),snapshot(fast))
    if case=='zero_slope_fallback':assert fast._last_fallback_flags&1
    row=dict(case=case,passed=True,events=ev,state_max_abs=max(states.values(),default=0),state_errors=states,fallbacks=getattr(fast,'_candidate_fallbacks',0))
    report['tests'].append(row);(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    del ref,fast
# Public guard checks: no feature silently removed, invalid input does not consume RNG.
first=np.ones((H,W),np.float32);e=emu(dict(cfg,photoreceptor_noise=True),first)
try:fused(e,[first],[1.001]);raise AssertionError('unsupported sensor feature accepted')
except NotImplementedError:pass
report['tests'].append(dict(case='unsupported_feature_rejected',passed=True))
e=emu(cfg,first);before=e.rng.get_state().clone()
try:fused(e,[np.full((H,W),np.nan,np.float32)],[1.001]);raise AssertionError('NaN accepted')
except ValueError:pass
assert torch.equal(before,e.rng.get_state())
report['tests'].append(dict(case='invalid_input_rejected_rng_unchanged',passed=True))
report['passed']=True;(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print('SENSOR_TESTS_PASSED',flush=True)
