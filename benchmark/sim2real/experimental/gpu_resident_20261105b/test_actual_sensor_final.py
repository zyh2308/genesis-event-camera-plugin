"""Actual resident frames: compare every sensor interval to the frozen oracle."""
import copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import h5py,numpy as np,torch
root=Path(__file__).resolve().parent;P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261103-stage9-direct-v1'
profile=json.loads((R/'sources/stage9_sensor_profile.json').read_text());pointer=json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text());core=Path(pointer['core_path'])
for name,digest in profile['core_canonical_LF_sha256'].items():assert hashlib.sha256((core/'genesis_event_plugin'/name).read_text().encode()).hexdigest()==digest
sys.path.insert(0,str(core))
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from resident_final import make_plugin_class
torch.set_num_threads(2);torch.set_num_interop_threads(1)
Resident=make_plugin_class(GenesisPhysicsEventPlugin,'exact');report=dict(cases=[],passed=False,real_data_used=False)
out=root/'actual_sensor_tests_final';out.mkdir(exist_ok=False)
def snapshot(e):
    return {**{k:v.detach().cpu().numpy().copy() for k,v in vars(e).items() if torch.is_tensor(v)},'rng':e.rng.get_state().cpu().numpy()}
cases=['s0_00','s1_00','s2_00','s3_00','train_000']
for case in cases:
    kind='pose' if case.startswith('train') else 'rich';index=10 if kind=='pose' else 100
    with h5py.File(R/(kind+'_bank/physical')/(case+'.h5'),'r') as h:
        g=h['episodes'][case];ts=np.asarray(g['frame_timestamps_s'][index:index+3],dtype=np.float64);K=np.asarray(g['camera_intrinsics_K_px']);ids=np.asarray(g['object_segmentation_link_ids'])
        data={key:np.asarray(g[key][index:index+3]) for key in ['radiance_linear_rgb','depth_m','segmentation_link_idx','object_to_world','camera_to_world_cv']}
    H,W=data['depth_m'].shape[1:];cursor={'i':0};scene=SimpleNamespace(cur_t=float(ts[0]));cam=SimpleNamespace(res=(W,H),intrinsics=K)
    def frame(*args):
        j=cursor['i'];return dict(radiance=data['radiance_linear_rgb'][j],depth=data['depth_m'][j],seg=data['segmentation_link_idx'][j])
    def motion(*args):
        j=cursor['i'];return dict(camera_to_world=data['camera_to_world_cv'][j],object_to_world={int(sid):data['object_to_world'][j,k] for k,sid in enumerate(ids)})
    kwargs=dict(profile['plugin_kwargs']);kwargs.update(device='cuda:0',warp_backend='torch')
    plugin=Resident(noise_config=DvsNoiseConfig(**profile['noise_config_kwargs']),sampler_config=AdaptiveSamplingConfig(**profile['sampler_kwargs']),frame_provider=frame,motion_state_provider=motion,**kwargs)
    plugin.attach(scene,cam).start_episode();plugin.capture();original=plugin._generate_device_sequence;rows=[]
    def checked(frames,times):
        oracle=copy.deepcopy(plugin.dvs_emulator);host=torch.stack(frames).cpu().numpy()
        expected=np.concatenate([oracle.generate_events(f,t) for f,t in zip(host,times)])
        actual=np.concatenate(original(frames,times))
        assert np.array_equal(expected,actual),(case,'sensor events not bitwise equal')
        a=snapshot(oracle);b=snapshot(plugin.dvs_emulator);assert a.keys()==b.keys()
        errors={k:float(np.abs(a[k].astype(float)-b[k].astype(float)).max(initial=0)) for k in a}
        assert max(errors.values())==0,(case,errors)
        for key in ['t_previous','frame_counter','num_events_on','num_events_off','num_events_total']:assert getattr(oracle,key)==getattr(plugin.dvs_emulator,key),(case,key)
        rows.append(dict(frames=len(frames),events=len(expected),events_bitwise_equal=True,state_errors=errors,scalar_counters_equal=True))
        return [actual] # Parent performs the same final public timestamp sort.
    plugin._generate_device_sequence=checked
    for j in [1,2]:cursor['i']=j;scene.cur_t=float(ts[j]);plugin.capture()
    plugin.materialize_diagnostics();report['cases'].append(dict(case=case,intervals=rows,passed=True));(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print('ACTUAL_SENSOR_PASS',case,flush=True)
    plugin.close();del plugin;torch.cuda.empty_cache()
report['passed']=True;(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
