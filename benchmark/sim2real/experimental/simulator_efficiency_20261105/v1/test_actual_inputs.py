"""Compare sensor implementations on identical cached physical input frames."""
import argparse,copy,dataclasses,gc,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import h5py,numpy as np,torch
root=Path(__file__).resolve().parent
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261103-stage9-direct-v1'
profile=json.loads((R/'sources/stage9_sensor_profile.json').read_text())
pointer=json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text());core=Path(pointer['core_path'])
for name,h in profile['core_canonical_LF_sha256'].items():assert hashlib.sha256((core/'genesis_event_plugin'/name).read_text().encode()).hexdigest()==h
sys.path.insert(0,str(core))
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from candidates import fused
torch.set_num_threads(2);torch.set_num_interop_threads(1)
out=root/'actual_input_tests';out.mkdir(exist_ok=False);report=dict(cases=[],passed=False,real_data_used=False)
def states(e):
    return {**{k:v.detach().cpu().numpy().copy() for k,v in vars(e).items() if torch.is_tensor(v)},'rng':e.rng.get_state().cpu().numpy()}
for case in ['s0_00','s1_00','s2_00','s3_00','train_000']:
    kind='pose' if case.startswith('train') else 'rich';index=10 if kind=='pose' else 100
    with h5py.File(R/(kind+'_bank/physical')/(case+'.h5'),'r') as h:
        g=h['episodes'][case];ts=np.asarray(g['frame_timestamps_s'][index:index+3],dtype=np.float64)
        K=np.asarray(g['camera_intrinsics_K_px']);ids=np.asarray(g['object_segmentation_link_ids'])
        data={key:np.asarray(g[key][index:index+3]) for key in ['radiance_linear_rgb','depth_m','segmentation_link_idx','object_to_world','camera_to_world_cv']}
    H,W=data['depth_m'].shape[1:];cursor={'i':0};scene=SimpleNamespace(cur_t=float(ts[0]));cam=SimpleNamespace(res=(W,H),intrinsics=K)
    def frame(*args):
        j=cursor['i'];return dict(radiance=data['radiance_linear_rgb'][j],depth=data['depth_m'][j],seg=data['segmentation_link_idx'][j])
    def motion(*args):
        j=cursor['i'];return dict(camera_to_world=data['camera_to_world_cv'][j],object_to_world={int(sid):data['object_to_world'][j,k] for k,sid in enumerate(ids)})
    kwargs=dict(profile['plugin_kwargs']);kwargs.update(device='cuda:0',warp_backend='torch')
    plugin=GenesisPhysicsEventPlugin(noise_config=DvsNoiseConfig(**profile['noise_config_kwargs']),sampler_config=AdaptiveSamplingConfig(**profile['sampler_kwargs']),frame_provider=frame,motion_state_provider=motion,**kwargs)
    plugin.attach(scene,cam).start_episode();plugin.capture();fast=copy.deepcopy(plugin.dvs_emulator)
    frames=[];times=[];expected_chunks=[];gen=plugin.dvs_emulator.generate_events
    def capture_sensor(f,t):
        frames.append(f.copy());times.append(float(t));v=gen(f,t);expected_chunks.append(v);return v
    plugin.dvs_emulator.generate_events=capture_sensor
    for j in [1,2]:cursor['i']=j;scene.cur_t=float(ts[j]);plugin.capture()
    # Replaying the very same intermediate frames removes frozen splat atomics
    # from this sensor arithmetic check.
    actual_chunks=fused(fast,frames,times)
    expected=np.concatenate(expected_chunks);actual=np.concatenate(actual_chunks)
    assert expected.shape==actual.shape,(case,expected.shape,actual.shape)
    assert np.array_equal(expected[:,1:],actual[:,1:]),case
    time_error=float(np.abs(expected[:,0]-actual[:,0]).max(initial=0))
    assert time_error<=1e-12,(case,time_error)
    left=states(plugin.dvs_emulator);right=states(fast);errors={};exact={}
    assert left.keys()==right.keys()
    for key in left:
        assert left[key].shape==right[key].shape,key
        exact[key]=bool(np.array_equal(left[key],right[key]))
        error=float(np.abs(left[key].astype(float)-right[key].astype(float)).max(initial=0))
        errors[key]=error;assert error<=1e-12,(case,key,error)
    row=dict(case=case,intervals=2,sensor_frames=len(times),events=len(expected),event_bitwise_equal=bool(np.array_equal(expected,actual)),time_error_s=time_error,state_errors=errors,state_bitwise_equal=exact,fallbacks=getattr(fast,'_candidate_fallbacks',0),passed=True)
    report['cases'].append(row);(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    plugin.close();del plugin,fast,frames,expected,actual;gc.collect();torch.cuda.empty_cache()
report['passed']=True;(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print('ACTUAL_INPUT_TESTS_PASSED',flush=True)
