"""Fresh-state, full-profile frozen/candidate paired cache replay benchmark."""
import argparse,dataclasses,gc,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import h5py,numpy as np,torch
root=Path(__file__).resolve().parent
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261103-stage9-direct-v1'
parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['baseline','bulk','fused','cached','chunk','fused_cached','fused_chunk'],default='baseline');parser.add_argument('--case',default='s2_00');parser.add_argument('--trials',type=int,default=3);parser.add_argument('--intervals',type=int,default=1);parser.add_argument('--out',required=True)
a=parser.parse_args();out=root/a.out;out.mkdir(exist_ok=False)
profile_path=R/'sources/stage9_sensor_profile.json';profile=json.loads(profile_path.read_text())
pointer=json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text());core=Path(pointer['core_path'])
for name,h in profile['core_canonical_LF_sha256'].items():assert hashlib.sha256((core/'genesis_event_plugin'/name).read_text().encode()).hexdigest()==h
sys.path.insert(0,str(core))
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
kind='pose' if a.case.startswith('train') else 'rich';index=10 if kind=='pose' else 100
path=R/(kind+'_bank/physical')/(a.case+'.h5');start=time.perf_counter()
with h5py.File(path,'r') as h:
    g=h['episodes'][a.case];ts=np.asarray(g['frame_timestamps_s'][index:index+a.intervals+1],dtype=np.float64)
    K=np.asarray(g['camera_intrinsics_K_px']);ids=np.asarray(g['object_segmentation_link_ids'])
    data={key:np.asarray(g[key][index:index+a.intervals+1]) for key in ['radiance_linear_rgb','depth_m','segmentation_link_idx','object_to_world','camera_to_world_cv']}
io_s=time.perf_counter()-start
H,W=data['depth_m'].shape[1:];rows=[]
if a.variant!='baseline':
    from candidates import make_plugin_class
    Plugin=make_plugin_class(GenesisPhysicsEventPlugin,a.variant)
else:Plugin=GenesisPhysicsEventPlugin
for trial in range(-1,a.trials):
    cursor={'i':0};scene=SimpleNamespace(cur_t=float(ts[0]));cam=SimpleNamespace(res=(W,H),intrinsics=K)
    def frame(*args):
        j=cursor['i'];return dict(radiance=data['radiance_linear_rgb'][j],depth=data['depth_m'][j],seg=data['segmentation_link_idx'][j])
    def motion(*args):
        j=cursor['i'];return dict(camera_to_world=data['camera_to_world_cv'][j],object_to_world={int(sid):data['object_to_world'][j,k] for k,sid in enumerate(ids)})
    kwargs=dict(profile['plugin_kwargs']);kwargs.update(device='cuda:0',warp_backend='torch')
    plugin=Plugin(noise_config=DvsNoiseConfig(**profile['noise_config_kwargs']),sampler_config=AdaptiveSamplingConfig(**profile['sampler_kwargs']),frame_provider=frame,motion_state_provider=motion,**kwargs)
    setup=time.perf_counter();plugin.attach(scene,cam).start_episode();assert len(plugin.capture())==0;torch.cuda.synchronize();setup=time.perf_counter()-setup
    phase={};frames_hash=hashlib.sha256();samples=[]
    original=plugin.interpolator.interpolate
    def interpolate(*args,**kw):
        t=time.perf_counter();value=original(*args,**kw);phase['geometry_wall_s']=phase.get('geometry_wall_s',0)+time.perf_counter()-t
        samples.append(dataclasses.asdict(plugin.interpolator.last_sampling));return value
    plugin.interpolator.interpolate=interpolate
    if hasattr(plugin,'_generate_sequence'):
        seq=plugin._generate_sequence
        def wrapped(frames,times):
            for f in frames:frames_hash.update(np.ascontiguousarray(f).tobytes())
            t=time.perf_counter();res=seq(frames,times);phase['dvs_wall_s']=phase.get('dvs_wall_s',0)+time.perf_counter()-t;return res
        plugin._generate_sequence=wrapped
    else:
        gen=plugin.dvs_emulator.generate_events
        def generate(f,t):
            frames_hash.update(np.ascontiguousarray(f).tobytes());begin=time.perf_counter();res=gen(f,t);phase['dvs_wall_s']=phase.get('dvs_wall_s',0)+time.perf_counter()-begin;return res
        plugin.dvs_emulator.generate_events=generate
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter();chunks=[]
    for j in range(1,len(ts)):
        cursor['i']=j;scene.cur_t=float(ts[j]);chunks.append(plugin.capture())
    torch.cuda.synchronize();wall=time.perf_counter()-begin
    events=np.concatenate(chunks) if any(len(x) for x in chunks) else np.empty((0,4),dtype=np.float64)
    states={k:v.detach().cpu().numpy().copy() for k,v in vars(plugin.dvs_emulator).items() if torch.is_tensor(v)}
    states['rng_state']=plugin.dvs_emulator.rng.get_state().cpu().numpy()
    masks={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in plugin.interpolator.last_visibility.items() if isinstance(v,np.ndarray) and v.dtype==np.bool_}
    if trial>=0:
        np.savez_compressed(out/('trial'+str(trial)+'.npz'),events=events,**{'state_'+k:v for k,v in states.items()})
        row=dict(trial=trial,wall_s=wall,setup_wall_s=setup,event_count=len(events),frame_sha256=frames_hash.hexdigest(),mask_sha256=masks,sampling=samples,phases=phase,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),fallbacks=getattr(plugin,'fallbacks',0))
        rows.append(row);print(json.dumps(row,default=lambda v:v.item()),flush=True)
    plugin.close();del plugin;gc.collect()
summary=dict(variant=a.variant,case=a.case,intervals=a.intervals,simulated_s=float(ts[-1]-ts[0]),source=str(path),endpoint_index=index,
             device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
             core_release=pointer['release_commit'],core_files_verified=17,io_load_s=io_s,warmup_trials=1,measured_trials=a.trials,rows=rows,
             median_wall_s=float(np.median([r['wall_s'] for r in rows])),real_data_used=False)
(out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda v:v.item())+'\n')
print('BENCH_COMPLETE',a.variant,a.case,summary['median_wall_s'],flush=True)
