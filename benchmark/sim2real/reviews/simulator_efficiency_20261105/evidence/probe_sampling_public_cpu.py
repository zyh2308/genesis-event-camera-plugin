"""Read-only CPU probe of the frozen sampler on fixed synthetic training intervals.

Stops immediately after sample-count selection; no event generation, training,
checkpoint access, GPU reservation or changes to frozen source/configuration.
"""
import dataclasses,hashlib,json,os,sys,time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
import h5py
import numpy as np
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261103-stage9-direct-v1'
root=Path(__file__).resolve().parent
profile=json.loads((R/'sources/stage9_sensor_profile.json').read_text())
pointer=json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())
core=Path(pointer['core_path'])
for name,wanted in profile['core_canonical_LF_sha256'].items():
    assert hashlib.sha256((core/'genesis_event_plugin'/name).read_text().encode()).hexdigest()==wanted,name
sys.path.insert(0,str(core))
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
class SelectionCaptured(Exception):pass
rows=[]
for kind,rid,index in [('rich','s0_00',100),('rich','s1_00',100),('rich','s2_00',100),('rich','s3_00',100),('pose','train_000',10)]:
    path=R/(kind+'_bank/physical')/(rid+'.h5')
    with h5py.File(path,'r') as handle:
        g=handle['episodes'][rid];ts=np.asarray(g['frame_timestamps_s']);ids=np.asarray(g['object_segmentation_link_ids'])
        K=np.asarray(g['camera_intrinsics_K_px'])
        kwargs=profile['plugin_kwargs']
        from types import SimpleNamespace
        from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
        from genesis_event_plugin.noise_model import DvsNoiseConfig
        plugin_kwargs=dict(kwargs);plugin_kwargs.update(device='cpu',warp_backend='numpy')
        plugin=GenesisPhysicsEventPlugin(noise_config=DvsNoiseConfig(**profile['noise_config_kwargs']),sampler_config=AdaptiveSamplingConfig(**profile['sampler_kwargs']),frame_provider=lambda *a: None,motion_state_provider=lambda *a: None,**plugin_kwargs)
        plugin.attach(SimpleNamespace(cur_t=float(ts[index])),SimpleNamespace(res=(int(g['depth_m'].shape[2]),int(g['depth_m'].shape[1])),intrinsics=K))
        interpolator=plugin.interpolator
        assert interpolator.sampler.config.sensor_time_constant_s > 0
        original=interpolator.sampler.compute;captured={}
        def capture(*args,**kw):
            selected=original(*args,**kw)
            captured.update(dataclasses.asdict(selected));captured['sampler_dt_s']=float(args[2])
            raise SelectionCaptured()
        interpolator.sampler.compute=capture
        objects=[{int(sid):np.asarray(g['object_to_world'][j])[i] for i,sid in enumerate(ids)} for j in [index,index+1]]
        delta={sid:objects[1][sid]@np.linalg.inv(objects[0][sid]) for sid in objects[0]}
        started=time.perf_counter()
        try:
            interpolator.interpolate(np.asarray(g['radiance_linear_rgb'][index]),np.asarray(g['depth_m'][index]),np.asarray(g['segmentation_link_idx'][index]),
                np.asarray(g['radiance_linear_rgb'][index+1]),np.asarray(g['depth_m'][index+1]),np.asarray(g['segmentation_link_idx'][index+1]),
                np.asarray(g['camera_to_world_cv'][index]),np.asarray(g['camera_to_world_cv'][index+1]),delta,float(ts[index]),float(ts[index+1]),input_space='linear',object_world0=objects[0],object_world1=objects[1])
        except SelectionCaptured:pass
        assert captured,'Probe must stop at the actual frozen sampler'
        rows.append(dict(kind=kind,rollout_id=rid,endpoint_index=index,source=str(path),duration_s=float(ts[-1]-ts[0]),
                         resolution_hw=list(g['depth_m'].shape[1:]),cpu_probe_wall_s=time.perf_counter()-started,sampling=captured))
        print(json.dumps(rows[-1],default=lambda v:v.item()),flush=True)
result=dict(scope='five fixed synthetic training endpoint pairs; public plugin attach-derived sensor bound; first sampler call only',gpu_used=False,
            events_generated=False,training_started=False,core_files_verified=17,profile_sha256=hashlib.sha256((R/'sources/stage9_sensor_profile.json').read_bytes()).hexdigest(),rows=rows)
with (root/'sampling_probe_public_results.json').open('x') as f:f.write(json.dumps(result,indent=2,default=lambda v:v.item())+'\n')
