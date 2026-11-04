"""Static package vs the prior candidate, recorder and conservative API fallback."""
import copy,gc,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'release_candidate_static'),str(ROOT.parent/'simulator_gpu_resident_20261105b')]
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.optimized import OptimizedGenesisPhysicsEventPlugin
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.gpu_interpolator import ResidentPhysicsInterpolator
from resident_final import make_plugin_class
torch.set_num_threads(2)
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261103-stage9-direct-v1'
profile=json.loads((R/'sources/stage9_sensor_profile.json').read_text())
Old=make_plugin_class(GenesisPhysicsEventPlugin,'exact')
report=dict(passed=False,cases=[],fallback=[],global_functions_unchanged=True)
out=ROOT/'release_integration_retry2';out.mkdir(exist_ok=False)
def payload(path):
    with h5py.File(path,'r') as h:
        result={}
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset) and any(k in name for k in ['events','timestamps']):result[name]=np.asarray(obj)
        h.visititems(visit)
        return result
for case in ['s0_00','s1_00','s2_00','s3_00','train_000']:
    kind='pose' if case.startswith('train') else 'rich';index=10 if kind=='pose' else 100
    with h5py.File(R/(kind+'_bank/physical')/(case+'.h5'),'r') as h:
        g=h['episodes'][case];ts=np.asarray(g['frame_timestamps_s'][index:index+3],dtype=np.float64)
        K=np.asarray(g['camera_intrinsics_K_px']);ids=np.asarray(g['object_segmentation_link_ids'])
        data={key:np.asarray(g[key][index:index+3]) for key in ['radiance_linear_rgb','depth_m','segmentation_link_idx','object_to_world','camera_to_world_cv']}
    H,W=data['depth_m'].shape[1:];arrays=[];walls=[];paths=[]
    for tag,cls in [('prior_candidate',Old),('static',OptimizedGenesisPhysicsEventPlugin)]:
        cursor={'i':0};scene=SimpleNamespace(cur_t=float(ts[0]));cam=SimpleNamespace(res=(W,H),intrinsics=K)
        def frame(*args):
            j=cursor['i'];return dict(radiance=data['radiance_linear_rgb'][j],depth=data['depth_m'][j],seg=data['segmentation_link_idx'][j])
        def motion(*args):
            j=cursor['i'];return dict(camera_to_world=data['camera_to_world_cv'][j],object_to_world={int(sid):data['object_to_world'][j,k] for k,sid in enumerate(ids)})
        plugin=cls(noise_config=DvsNoiseConfig(**profile['noise_config_kwargs']),sampler_config=AdaptiveSamplingConfig(**profile['sampler_kwargs']),frame_provider=frame,motion_state_provider=motion,output_dir=str(out/(case+'_'+tag)),**profile['plugin_kwargs'])
        plugin.attach(scene,cam).start_episode();plugin.capture();begin=time.perf_counter();chunks=[]
        for j in [1,2]:cursor['i']=j;scene.cur_t=float(ts[j]);chunks.append(plugin.capture())
        torch.cuda.synchronize();walls.append(time.perf_counter()-begin)
        arrays.append(np.concatenate(chunks));plugin.end_episode();plugin.close()
        files=list((out/(case+'_'+tag)).glob('*.h5'));assert len(files)==1;paths.append(files[0])
        if tag=='static':assert plugin.optimization_active and isinstance(plugin.interpolator,ResidentPhysicsInterpolator)
        del plugin;gc.collect();torch.cuda.empty_cache()
    a,b=arrays;assert a.shape==b.shape and np.array_equal(a[:,1:],b[:,1:]),case
    error=float(np.abs(a[:,0]-b[:,0]).max(initial=0));assert error<=3e-9,(case,error)
    pa,pb=payload(paths[0]),payload(paths[1]);assert pa.keys()==pb.keys() and len(pa)>0,(case,list(pa))
    assert all(np.array_equal(pa[k],pb[k]) for k in pa),(case,'recorder payload differs')
    row=dict(case=case,events=len(a),raw_bitwise_equal=bool(np.array_equal(a,b)),max_timestamp_error_s=error,recorder_bitwise_equal=True,prior_wall_s=walls[0],static_wall_s=walls[1],recorder_datasets=list(pa))
    report['cases'].append(row);print('INTEGRATION',json.dumps(row),flush=True)
# Exact CPU fallback for an unsupported device/profile, without changing settings.
H,W=12,16;K=np.array([[30.,0,7.5],[0,30.,5.5],[0,0,1.]])
for why,changes in [('cpu',{}),('sensor_feature',{'refractory_period_s':0.})]:
    cfg=dict(profile['noise_config_kwargs']);cfg.update(changes);scene=SimpleNamespace(cur_t=0.);cam=SimpleNamespace(res=(W,H),intrinsics=K)
    def frame(*args):return dict(radiance=np.full((H,W,3),.2+scene.cur_t,np.float32),depth=np.ones((H,W),np.float32),seg=np.zeros((H,W),np.int32))
    motion=lambda *args:dict(camera_to_world=np.eye(4),object_to_world={})
    kwargs=dict(profile['plugin_kwargs']);kwargs.update(device='cpu',warp_backend='numpy')
    values=[]
    for cls in [GenesisPhysicsEventPlugin,OptimizedGenesisPhysicsEventPlugin]:
        scene.cur_t=0.;p=cls(noise_config=DvsNoiseConfig(**cfg),sampler_config=AdaptiveSamplingConfig(**profile['sampler_kwargs']),frame_provider=frame,motion_state_provider=motion,**kwargs)
        p.attach(scene,cam).start_episode();p.capture();scene.cur_t=.002;values.append(p.capture())
        if cls==OptimizedGenesisPhysicsEventPlugin:assert not p.optimization_active and p.optimization_status['reference_fallback_reasons']
        p.close()
    assert np.array_equal(*values);report['fallback'].append(dict(case=why,exact_reference=True))
# Runtime source contains no dynamic generation or process-global geometry patch.
for name in ['gpu_geometry.py','gpu_interpolator.py','gpu_sensor.py','optimized.py']:
    text=(ROOT/'release_candidate_static/genesis_event_plugin'/name).read_text()
    assert 'exec(' not in text and 'inspect.' not in text
report['passed']=True;(ROOT/'release_integration.json').write_text(json.dumps(report,indent=2)+'\n')
