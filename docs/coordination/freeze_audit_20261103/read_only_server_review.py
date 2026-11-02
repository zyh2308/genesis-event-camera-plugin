"""Final audit: no original source, baseline or experiment file is written."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,csv,sys,os
import numpy as np
base=Path('/home/zhaoyuhan/Eventbased_WAM')
seal=base/'frozen_baselines/robot-sparse-cuda-r2-20261102'
pointer=json.loads((seal/'BASELINE.json').read_text());d=pointer['descriptor']
assert pointer['release_commit']=='62da4f35ce3c5c96f04268bd1687a27657af9238'
core=Path(pointer['core_path']);benchmark=Path(pointer['benchmark_path'])/'benchmark/robot_closed_loop'
out={'baseline_commit':pointer['release_commit'],'core_files':[],'benchmark_files':[],'raw_manifests':{}}
for name,expected in d['core_canonical_LF_sha256'].items():
 actual=hashlib.sha256((core/'genesis_event_plugin'/name).read_text().encode()).hexdigest();assert actual==expected,name
 out['core_files'].append({'file':name,'match':True})
for name,expected in d['benchmark_source_sha256'].items():
 assert hashlib.sha256((benchmark/name).read_bytes()).hexdigest()==expected,name
 out['benchmark_files'].append({'file':name,'match':True})
assert hashlib.sha256(Path(pointer['checkpoint_path']).read_bytes()).hexdigest()==d['checkpoint_sha256']
for label,record in d['raw_manifests'].items():
 root=Path(record['root']);p=root/'SHA256SUMS.server';assert hashlib.sha256(p.read_bytes()).hexdigest()==record['manifest_sha256'],label
 count=0
 for line in p.read_text().splitlines():
  expected,name=line.split(None,1);target=(root/name).resolve()
  assert os.path.commonpath([str(root.resolve()),str(target)])==str(root.resolve()),name
  assert hashlib.sha256(target.read_bytes()).hexdigest()==expected,(label,name)
  count+=1
 out['raw_manifests'][label]={'files_verified':count}
profile_path=seal/'event_sparse_reconstruction_20261102.json';raw=profile_path.read_bytes();text=raw.decode();normalized=text.replace('\r\n','\n');profile=json.loads(normalized)
raw_hash=hashlib.sha256(raw).hexdigest();lf_hash=hashlib.sha256(normalized.encode()).hexdigest();crlf_hash=hashlib.sha256(normalized.replace('\n','\r\n').encode()).hexdigest()
assert crlf_hash==d['profile_sha256']
out['configuration_hash_review']={'profile_metadata_hash':d['profile_sha256'],'profile_g1_raw_sha256':raw_hash,'canonical_LF_sha256':lf_hash,'Windows_CRLF_sha256':crlf_hash,'raw_byte_hash_matches_original_metadata':raw_hash==d['profile_sha256'],'semantic_JSON_sha256':hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),'finding':'The metadata hash names the Windows CRLF export. Git/G1 carry LF. Same JSON parameters; append audit seal instead of changing frozen commit.'}
assert hashlib.sha256((seal/'shared_control_r2_verified_20261102.json').read_bytes()).hexdigest()==d['control_profile_sha256']
sys.path.insert(0,str(core))
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.noise_model import DvsNoiseConfig
import torch
torch.set_num_threads(1)
h,w=8,10;I0=np.ones((h,w),np.float32);I1=np.full_like(I0,np.exp(.9));depth=np.ones((h,w));seg=np.zeros((h,w),int)
scene=SimpleNamespace(cur_t=0.);cam=SimpleNamespace(res=(w,h),intrinsics=np.array([[20.,0,w/2],[0,20.,h/2],[0,0,1.]]));calls=[]
def frame(s,c):
 calls.append(float(s.cur_t));return dict(radiance=I0 if s.cur_t==0 else I1,depth=depth,seg=seg)
kwargs=dict(profile['plugin_kwargs']);kwargs.update(profile['CPU_equivalent'])
plugin=GenesisPhysicsEventPlugin(noise_config=DvsNoiseConfig(**profile['noise_config_kwargs']),frame_provider=frame,motion_state_provider=lambda s,c:dict(camera_to_world=np.eye(4),object_to_world={}),**kwargs)
plugin.attach(scene,cam);assert len(plugin.capture())==0;scene.cur_t=.01;events=plugin.capture()
expected=np.array([.2,.4,.6,.8])/.9*.01;measured=np.unique(np.round(events[:,0],10))
assert calls==[0.,.01] and len(events)==h*w*4 and np.allclose(measured,expected,atol=2e-8)
assert plugin.interpolator.composite=='coverage_time_log_linear' and plugin.intermediate_state_provider is None
assert np.max(events[:,0])<=.01
plugin.start_episode();scene.cur_t=.01;plugin.capture();scene.cur_t=.02;assert len(plugin.capture())==0
out['frozen_profile_public_capture']={'status':'PASS','backend':'NumPy/CPU equivalent from frozen JSON','ramp_events':len(events),'max_analytic_threshold_time_error_s':float(np.max(np.abs(measured-expected))),'provider_endpoint_calls_for_ramp':[0.,.01],'extra_HDR_frames':0,'static_events':0,'torch_version':torch.__version__}
generation=base/'runs/robot_r1_5_sparse_time_repair_20261101a/sparse_events/episodes'
endpoint_reads=0;episodes=0
for p in sorted(generation.glob('episode_*/reconstruction_audit.json')):
 x=json.loads(p.read_text());assert x['endpoint_only'] and x['renderer_calls']==0 and x['dense_radiance_reads']==0 and x['intermediate_state_provider'] is None
 endpoint_reads+=sum(a['kind']=='frame' for a in x['access']);episodes+=1
assert episodes==9
out['training_input_audit']={'episodes':episodes,'endpoint_reads':endpoint_reads,'renderer_calls':0,'dense_radiance_reads':0}
winner=base/'runs/robot_R2_loaded_guard_20261102o';ep=winner/'R2_kinematic_fit/episodes/episode_000'
with (ep/'episode.csv').open() as f:row=next(csv.DictReader(f))
score=json.loads(row['strict_score']);assert row['success_strict_r0_5']=='True' and float(row['hold_s'])==.5
anchor=np.asarray(score['grasp_relative_anchor_m']);flags=0;future=0;clearance_error=0.;slip_error=0.;qualified=0;source_labels=set()
for line in (ep/'state_control_trace.jsonl').open():
 f=json.loads(line);p=np.asarray(f['object']['position_m']);w,x,y,z=f['object']['orientation_wxyz'];rz=np.array([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)])
 clearance=float(p[2]-.02*np.abs(rz).sum());slip=float(np.linalg.norm(p-np.asarray(f['tcp_pad_center_m'])-anchor));s=f['score'];clearance_error=max(clearance_error,abs(clearance-s['bottom_clearance_m']))
 if s['relative_slip_m'] is not None:slip_error=max(slip_error,abs(slip-s['relative_slip_m']))
 good=bool(f['contacts']['bilateral'] and not f['contacts']['support'] and clearance>=.08 and slip<=.01);flags+=good!=s['qualified'];qualified+=good
 obs=f.get('controller_observation_state')
 if obs:future+=float(obs['time_s'])>float(f['time_s'])-.01+1e-8
 source_labels.add(f.get('state_source'))
assert flags==0 and future==0 and clearance_error<1e-10 and slip_error<1e-10 and source_labels=={'direct_hdr_events_controller'}
out['strict_grasp_recomputation']={'status':'PASS','hold_s':.5,'qualified_physics_steps':score['qualified_hold_samples'],'qualified_trace_rows':qualified,'minimum_clearance_m':score['minimum_true_bottom_clearance_during_longest_qualified_hold_m'],'maximum_slip_m':score['maximum_relative_slip_since_bilateral_grasp_m'],'future_time_violations':future,'qualified_flag_mismatch':flags,'maximum_geometry_residual_m':clearance_error,'maximum_slip_residual_m':slip_error,'state_sources':sorted(source_labels)}
manifest=json.loads((winner/'R2_kinematic_fit/run_manifest.json').read_text())
runtime=manifest['runtime'];environment={}
for path_key,hash_key in [('panda_asset','panda_asset_sha256'),('contact_source','contact_source_sha256')]:
 p=Path(runtime[path_key]);ok=hashlib.sha256(p.read_bytes()).hexdigest()==runtime[hash_key];assert ok,path_key;environment[path_key]=ok
out['recorded_physics_environment_files']=environment
out['status']='review_complete_core_frozen_no_new_algorithm_blocker_within_contract'
print(json.dumps(out))
