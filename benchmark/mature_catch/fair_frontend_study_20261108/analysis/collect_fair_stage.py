"""Close actual fair-frontend stage data; preserve the established raw/window audits.

Derived from the prior analyze_cadence/postprocess contracts. The scientific sample
unit is an explicit physical_case_id+seed block, never a query or candidate repeat.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,subprocess,sys,time

HERE=Path(__file__).resolve().parent
PRIMARY=('ours','v2e')
REFERENCES=('init_only','state_reference')
FRONTEND_KEYS={'CATCHIT_EVENT_OBSERVATION_WINDOW_MS':('0','10'),
 'CATCHIT_EVENT_FLIGHT_FIT':('0','1'),'CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS':('50','100')}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p,canonical=False):
 b=Path(p).read_bytes();return hashlib.sha256(b.replace(b'\r\n',b'\n') if canonical else b).hexdigest()


def rate_key(hz):return format(float(hz),'g')


def frontend(spec,case,method):
 selected=case.get('frontend_overrides',spec.get('frontend_by_method',{}).get(method))
 if selected is None:
  if method in PRIMARY:raise ValueError('Primary method has no explicitly registered frontend: '+str((case['case_id'],method)))
  return None
 if set(selected)!=set(FRONTEND_KEYS):raise ValueError('Frontend must declare exactly the three registered settings')
 selected={k:str(v) for k,v in selected.items()}
 if any(selected[k] not in allowed for k,allowed in FRONTEND_KEYS.items()):raise ValueError('Unregistered frontend value')
 return selected


def frontend_id(env):
 if env is None:return 'not_applicable_reference'
 return 'w%s_fit%s_gap%s'%(env['CATCHIT_EVENT_OBSERVATION_WINDOW_MS'],env['CATCHIT_EVENT_FLIGHT_FIT'],env['CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS'])


def expected_rows(spec):
 result={}
 for case in spec['cases']:
  assert case.get('physical_case_id') is not None,'Explicit physical_case_id is required; do not inherit old sample_unit claims'
  assert case.get('method_order') and len(set(case['method_order']))==len(case['method_order'])
  for method in case['method_order']:
   assert method in PRIMARY+REFERENCES
   key=(case['case_id'],method);assert key not in result,'Duplicate planned actor: '+str(key)
   result[key]=case
 assert result,'Empty study'
 return result


def verify_inputs(spec):
 # Same frozen-input check as sealed run_cadence_comparison.verify and old auditor.
 for name,digest in spec['runtime_files'].items():assert sha(Path(spec['source'])/name)==digest,name
 assert len(spec['core_files'])==17
 for name,digest in spec['core_files'].items():assert sha(Path(spec['core_root'])/'genesis_event_plugin'/name,True)==digest,name
 for name,key in [('checkpoint','checkpoint_sha256'),('profile_file','profile_sha256'),('v2e_checkpoint','v2e_checkpoint_sha256')]:assert sha(spec[name])==spec[key],name
 for name,digest in spec.get('v2e_source_files',{}).items():assert sha(Path(spec['v2e_source'])/name)==digest,name


def retained_path(root,label):
 path=(root/label).resolve()
 assert root.resolve() in path.parents,'Recovery artifact must remain under the actual study root'
 return path


def recovery_receipts(root,spec,shards,plan_path):
 if plan_path is None:return {},set(),None
 path=retained_path(root,plan_path);plan=read(path)
 assert plan['schema']=='fair_workqueue_infra_recovery_v1' and plan['reason']=='cuda_out_of_memory'
 assert plan['protocol_sha256']==sha(root/'EVALUATION_PROTOCOL.json'),'Recovery changed the registered actor protocol'
 import datetime
 registered=datetime.datetime.fromisoformat(plan['registered_utc']);assert registered.utcoffset()==datetime.timedelta(0)
 historical={}
 for entry in plan['historical_failed_workers']:
  worker=entry['worker_id'];assert worker not in historical
  paths={key:retained_path(root,entry[key+'_file']) for key in ('job','result','progress')}
  for key,p in paths.items():assert sha(p)==entry[key+'_sha256'],'Historical failure receipt changed: '+str(p)
  job=read(paths['job']);command=job['command']
  assert command[command.index('--worker-id')+1]==worker
  assert Path(command[command.index('--protocol')+1]).resolve()==(root/'EVALUATION_PROTOCOL.json').resolve()
  failure=read(paths['result']);assert failure['status']=='failed'
  assert failure.get('exit_code',1)!=0 or failure.get('error'),'A successful job cannot be exempted as an OOM failure'
  archived={}
  for attempt in entry['archived_failed_claims']:
   shard=int(attempt['shard']);assert shard in shards and shard not in archived
   claim_path=retained_path(root,attempt['claim_file']);assert sha(claim_path)==attempt['claim_sha256']
   archive=retained_path(root,attempt['archived_attempt_root'])
   assert (root/'infrastructure_failures').resolve() in archive.parents and archive.is_dir()
   assert (root/'infrastructure_failures').resolve() in claim_path.parents
   claim=read(claim_path)
   assert claim['shard']==shard and claim['worker_id']==worker and claim['protocol_sha256']==plan['protocol_sha256']
   assert claim['status']=='infrastructure_failure' and claim['exit_code']!=0
   progress_path=archive/f'shard{shard}_progress.json';attempt_rows=read(progress_path)
   scheduled=[(c,m) for c in spec['cases'] if int(c['shard'])==shard for m in c['method_order']]
   assert 0<len(attempt_rows)<=len(scheduled)
   for row,(case,method) in zip(attempt_rows,scheduled):
    assert row['case_id']==case['case_id'] and row['method']==method and int(row['seed'])==int(case['seed'])
    assert float(row['endpoint_hz'])==float(case['endpoint_hz']) and row.get('frontend_overrides',{})==(frontend(spec,case,method) or {})
    assert (archive/'cases'/case['case_id']/method).is_dir(),'Retain every actual partial actor output'
   assert attempt_rows[-1]['exit_code']!=0 and all(r['exit_code']==0 for r in attempt_rows[:-1])
   assert any(archive.rglob('*.log')),'Retain the original failed attempt logs'
   archived[shard]=claim
  historical[worker]=dict(entry=entry,paths=paths,job=job,archived=archived)
 replacements=plan['replacement_worker_ids']
 assert historical and replacements and len(replacements)==len(set(replacements)) and not set(replacements)&set(historical)
 return historical,set(replacements),dict(path=str(path),sha256=sha(path),registered_utc=plan['registered_utc'],
  reason=plan['reason'],historical_failed_workers=plan['historical_failed_workers'],replacement_worker_ids=replacements)


def queue_receipts(root,spec,shards,selected_results,recovery_plan=None):
 """Finite queue: registered workers are jobs, while claims identify actual shards."""
 queue=root/'workqueue';plan=read(queue/'PLAN.json');protocol_sha=sha(root/'EVALUATION_PROTOCOL.json')
 assert plan['protocol_sha256']==protocol_sha
 assert len(plan['ordered_shards'])==len(shards) and set(plan['ordered_shards'])==set(shards),'Queue plan does not cover registered shards exactly'
 historical,replacements,recovery=recovery_receipts(root,spec,shards,recovery_plan)
 jobs={}
 for path in sorted(root.glob('*.job.json')):
  job=read(path);command=job['command']
  if not any(Path(str(x)).name=='fair_workqueue_worker.py' for x in command):continue
  assert command.count('--worker-id')==command.count('--protocol')==1
  worker=command[command.index('--worker-id')+1]
  assert Path(command[command.index('--protocol')+1]).resolve()==(root/'EVALUATION_PROTOCOL.json').resolve()
  assert path.name==job['name']+'.job.json' and worker not in jobs,'Duplicated registered queue worker'
  if worker in historical:assert sha(path)==historical[worker]['entry']['job_sha256']
  if worker in replacements:assert job['infrastructure_recovery_sha256']==recovery['sha256'],'Replacement was not bound to the preregistered recovery plan'
  jobs[worker]=(path,root/(job['name']+'.result.json'))
 for worker,entry in historical.items():
  jobs[worker]=(entry['paths']['job'],entry['paths']['result'])
 if not replacements<=set(jobs):return None
 assert jobs,'Queue has no registered worker jobs'
 paths=[x[1] for worker,x in jobs.items() if worker not in historical]
 if selected_results:
  assert {x.resolve() for x in paths}=={(root/x).resolve() for x in selected_results},'Queue requires all registered worker results; cannot silently drop a failed worker'
 for path in paths:
  if not path.exists():return None
  receipt=read(path)
  if receipt.get('status')=='failed':raise RuntimeError('Retained queue infrastructure failure: '+str(path)+' '+str(receipt))
  if receipt.get('status')!='completed':return None
  assert receipt.get('exit_code')==0,path
 claims={}
 for shard in shards:
  path=queue/'claims'/f'shard{shard:03d}.json'
  if not path.exists():return None
  claim=read(path)
  if claim.get('status')=='infrastructure_failure':raise RuntimeError('Retained failed shard claim: '+str(path))
  if claim.get('status')!='completed':return None
  assert claim['exit_code']==0 and claim['shard']==shard and claim['protocol_sha256']==protocol_sha
  assert claim['worker_id'] in jobs,'Shard claimed by an unregistered worker'
  claims[shard]=(path,claim)
 assert set((queue/'claims').glob('shard*.json'))=={x[0] for x in claims.values()},'Extra unregistered shard claims'
 for worker in jobs:
  if worker in historical:
   history=historical[worker];reported=read(history['paths']['progress']);own={s:c for s,(p,c) in claims.items() if c['worker_id']==worker}
   assert len(reported)==len({c['shard'] for c in reported})
   completed={c['shard']:c for c in reported if c['status']=='completed'}
   failed={c['shard']:c for c in reported if c['status']=='infrastructure_failure'}
   assert len(completed)+len(failed)==len(reported) and completed==own and failed==history['archived']
   continue
  finished=queue/(worker+'.finished.json');progress=queue/(worker+'.progress.json')
  if not finished.exists():return None
  done=read(finished);own={s:c for s,(p,c) in claims.items() if c['worker_id']==worker}
  assert done['status']=='PASS' and done['shards']==len(own)
  reported=read(progress) if progress.exists() else []
  assert len(reported)==len(own) and {c['shard'] for c in reported}==set(own)
  for claim in reported:assert claim==own[claim['shard']],'Worker receipt differs from final atomic claim'
 return paths,dict(mode='finite_registered_workqueue',registered_workers=list(jobs),infrastructure_recovery=recovery,
  worker_job_receipts=[dict(worker_id=w,job=str(p),job_sha256=sha(p),result=str(r),result_sha256=sha(r)) for w,(p,r) in jobs.items()],
  completed_claim_receipts=[dict(shard=s,path=str(p),sha256=sha(p),worker_id=c['worker_id']) for s,(p,c) in claims.items()])


def finished_progress(root,spec,selected_results=None,recovery_plan=None):
 planned=expected_rows(spec);shards=sorted({int(c['shard']) for c in spec['cases']})
 if (root/'workqueue/PLAN.json').exists():
  queue=queue_receipts(root,spec,shards,selected_results,recovery_plan)
  if queue is None:return None
  paths,scheduling=queue
 else:
  assert recovery_plan is None,'Explicit recovery plan is implemented only for the finite registered workqueue'
  paths=[root/name for name in selected_results] if selected_results else sorted(root.glob('*.result.json'))
  if len(paths)!=len(shards):return None
  scheduling=dict(mode='one_registered_job_per_shard')
 for path in paths:
  assert path.resolve().parent==root.resolve() and path.name.endswith('.result.json')
  if not path.exists():return None
  receipt=read(path)
  if receipt.get('status')=='failed':raise RuntimeError('Retained infrastructure failure: '+str(path)+' '+str(receipt))
  if receipt.get('status')!='completed':return None
  assert receipt.get('exit_code')==0,path
 found={}
 for shard in shards:
  c=root/f'shard{shard}_complete.json';p=root/f'shard{shard}_progress.json'
  if not c.exists() or not p.exists():return None
  complete=read(c);rows=read(p)
  assert complete['all_exit_zero'] and complete['postflight_hashes_pass']
  assert len(rows)==sum(len(x['method_order']) for x in spec['cases'] if int(x['shard'])==shard)==complete['completed']
  for entry in rows:
   key=(entry['case_id'],entry['method']);assert key in planned and key not in found and entry['exit_code']==0,key
   assert int(planned[key]['shard'])==shard
   assert int(entry['seed'])==int(planned[key]['seed']) and float(entry['endpoint_hz'])==float(planned[key]['endpoint_hz'])
   assert entry.get('frontend_overrides',{})==(frontend(spec,planned[key],entry['method']) or {})
   found[key]=entry
 assert set(found)==set(planned),'Missing or extra actual actors'
 return found,paths,shards,scheduling


def longest_contact_steps(trace):
 best=run=0;best_span=0.;start=None
 for row in trace:
  if row['hand_contact']:
   if not run:start=float(row['t_before'])
   run+=1
   if run>best:best=run;best_span=float(row['t'])-start
  else:run=0;start=None
 return dict(longest_consecutive_contact_control_steps=best,
  contact_controlstep_run_nominal_span_s=best*.04,contact_controlstep_run_recorded_interval_span_s=best_span,
  contact_controlstep_resolution_s=.04,continuous_physics_hold_not_inferred=True)


def precontact_metrics(trace,result):
 """Actual controller queries only; target labels are used after the run for scoring."""
 import numpy as np
 release=float(result['initial']['release_time_s'])
 first=next((i for i,r in enumerate(trace) if r['hand_contact']),None)
 indices=[i for i,r in enumerate(trace) if (first is None or i<=first) and r['t_before']>=release-1e-9]
 p_loss=[];v_loss=[];p_raw=[];v_raw=[];available=0;invalid=0
 for i in indices:
  state=trace[i].get('event_state')
  if state is not None:
   assert i>0
   ep=float(np.linalg.norm(np.asarray(state['position'])-trace[i]['scoring_only_GT_position_at_control']))
   ev=float(np.linalg.norm(np.asarray(state['velocity'])-np.asarray(trace[i-1]['qvel'][-6:-3])))
   if math.isfinite(ep) and math.isfinite(ev):
    available+=1;p_raw.append(ep);v_raw.append(ev);p_loss.append(min(ep,.5));v_loss.append(min(ev,2.));continue
   invalid+=1
  p_loss.append(.5);v_loss.append(2.)
 n=len(indices);p_mean=sum(p_loss)/n if n else .5;v_mean=sum(v_loss)/n if n else 2.
 return dict(queries=n,available_states=available,control_state_coverage=available/n if n else 0.,
  penalized_position_error_mean_m=p_mean,penalized_velocity_error_mean_m_s=v_mean,
  normalized_equal_weight_loss=.5*p_mean/.5+.5*v_mean/2.,
  conditional_position_error_mean_m=sum(p_raw)/len(p_raw) if p_raw else None,
  conditional_velocity_error_mean_m_s=sum(v_raw)/len(v_raw) if v_raw else None,
  nonfinite_states=invalid,zero_query_penalized=n==0,
  measurement_scope='All actual post-release/pre-first-real-contact controller states, including static predictions and false grasp propagation. No missing-state exclusion; reset release threshold is a scoring label, not exact substep release telemetry.')


def actor_record(root,spec,case,method,execution):
 import numpy as np
 run=Path(execution['output']);expected=root/'cases'/case['case_id']/method
 if run.resolve()!=expected.resolve():
  recovery=read(root/'RECOVERY_MAP.json');assert recovery[case['case_id']+'|'+method]['output']==str(run)
 protocol=read(run/'protocol.json');result=read(run/'episode_000/result.json');trace=read(run/'episode_000/trace.json')
 assert execution.get('episodes')==[result],'Progress receipt differs from actual closed result'
 assert int(result['seed'])==int(case['seed']) and protocol['seeds']==[int(case['seed'])]
 assert protocol['checkpoint_sha256']==spec['checkpoint_sha256']
 assert protocol['launch_override']==case['launch']
 assert protocol['asset_initialization_seed']==1702 and protocol['sphere_radius_m']==.04
 assert protocol['sensor_pitch_offset_degrees']==7 and float(protocol['event_endpoint_rate_hz'])==float(case['endpoint_hz'])
 assert abs(float(protocol['timestep'])-.002)<1e-12 and protocol['steps_per_policy']==20
 assert protocol['env_time']==2.5 and not protocol['gate_until_s'] and protocol['gate_after_release_s'] is None
 assert protocol['backend'].lower().find('genesis')>=0
 contact=read(run/'migration_audit/adapter_audit.json')
 assert contact['original_mj_step_calls']==0 and contact['contact_profile']['name']=='combined'
 assert contact['contact_profile']['substeps']==2
 c=contact['contact_calibration'];assert c['unselected_unchanged'] and c['hand_geom_ids']==list(range(33,50))
 assert all(abs(x['friction']-1.5)<1e-12 for x in c['after'] if x['selected_hand'])
 assert abs(c['after'][-1]['friction']-.95)<1e-12
 init_failed=result.get('failure_class')=='initialization_failure'
 if init_failed:
  assert method!='state_reference' and result['task_failure_no_physics_steps'] and not trace
  assert result['success'] is False and result['terminated'] is False
  assert result['steps']==result['sim_time']==result['hand_contact_steps']==result['valid_event_polls']==0
  assert result['first_touch'] is None and result['initial']
 else:
  assert trace and len(trace)==result['steps']
  assert bool(trace[-1]['truncated'])==bool(result['success']) and bool(trace[-1]['terminated'])==bool(result['terminated'])
  assert sum(bool(x['hand_contact']) for x in trace)==result['hand_contact_steps']
 verified=bool(result['success'] and not result['terminated'] and result['hand_contact_steps']>0 and trace and trace[-1]['hand_contact'])
 configured=frontend(spec,case,method)
 if method in PRIMARY:
  assert protocol['event_generator']==method and not protocol['initialization_only']
 if configured is not None and method in PRIMARY:
  assert float(protocol['event_observation_window_ms'])==float(configured['CATCHIT_EVENT_OBSERVATION_WINDOW_MS'])
  assert bool(protocol['event_flight_fit'])==(configured['CATCHIT_EVENT_FLIGHT_FIT']=='1')
  assert float(protocol.get('event_confirmation_max_gap_ms',50))==float(configured['CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS'])
 initialization=None;window_audit=None;packets=[];samples=[];events_count=0
 if method=='state_reference':
  assert not protocol['event_control'] and protocol['RGB_frame_budget']==protocol['control_depth_frame_budget']==0
 else:
  assert protocol['event_control'] and protocol['RGB_frame_budget']==protocol['control_depth_frame_budget']==1
  sensor=run/'migration_audit/sensor_episode_000';initialization=read(sensor/'initialization.json')
  assert initialization['RGB_frames_read']==initialization['control_depth_frames_read']==1
  assert initialization['time_s']==initialization['available_s']==0.
  loaded=read(run/'loaded_module_receipt.json')
  required={'genesis_adapter','sparse_event_sensor','event_sphere_tracker','event_sphere_tracker_kalman',
   'event_sphere_tracker_contour_q80_occlusion','rgb_sphere_initialization','rgbd_sphere_initialization',
   'event_sphere_tracker_rgbd_base','event_bearing_flight'}
  if float(protocol['event_observation_window_ms'])>0:required.add('event_observation_windows')
  if protocol.get('lighting')=='front_glare':required.add('light_schedule')
  assert required<=set(loaded),'Missing loaded module receipt: '+str(required-set(loaded))
  for name,entry in loaded.items():
   assert Path(entry['path']).parent==Path(spec['source']) and entry['sha256']==spec['runtime_files'][name+'.py'],name
  packets=read(sensor/'packet_audit.json');samples=read(sensor/'event_samples.json')
  raw=np.load(sensor/'raw_events.npz',allow_pickle=False)['events'];events_count=len(raw)
  assert raw.shape==(sum(int(p['events']) for p in packets),4)
  if init_failed or method=='init_only':
   assert len(raw)==len(packets)==len(samples)==0
   assert not (sensor/'worker_start.json').exists() and not (sensor/'sparse_hdr_endpoints.h5').exists()
   if init_failed:assert initialization['valid'] is False
   else:assert initialization['valid'] is True and protocol['initialization_only'] and protocol['event_updates_disabled']
   window_audit=dict(status='NOT_APPLICABLE_NO_EVENT_GENERATION',reason='initialization_failure' if init_failed else 'initialization_only',raw_events=0,packets=0,control_queries=len(trace))
  else:
   assert initialization['valid'] is True and (sensor/'sparse_hdr_endpoints.h5').exists()
   worker=read(sensor/'worker_start.json');assert worker['ready'] is True
   if method=='ours':assert worker['core_files_verified']==17 and worker['profile_sha256']==spec['profile_sha256']
   if method=='v2e':
    assert worker['privileged_scene_inputs'] is False
    native=worker['receipt'];assert native['checkpoint_sha256']==spec['v2e_checkpoint_sha256'] and native['source_sha256']==spec['v2e_source_files']
    assert native['auto_upsample'] is True and native['pos_thres']==native['neg_thres']==.2 and native['noise'] is False
    assert native['minimum_upsampling']==int(1000//float(case['endpoint_hz']))
    proof=read(worker['parity_receipt']);assert proof['status']=='PASS' and proof['minimum_upsampling']==native['minimum_upsampling']
    assert proof['adapter_sha256']==spec['runtime_files']['streaming_v2e.py']
   if float(protocol['event_observation_window_ms'])==10.:
    from audit_contact_observation_windows import audit_run
    window_audit=audit_run(run,protocol,spec)
   else:
    assert float(protocol['event_observation_window_ms'])==0.
    window_audit=dict(status='NOT_APPLICABLE_NATIVE_WINDOW0',raw_audit='established audit_fair_raw.py contract remains mandatory')
  for row in trace:
   state=row.get('event_state');now=float(row['t_before'])
   if state is None:
    assert row['policy_object_input'] is None and all(abs(float(x))<1e-10 for x in row['action'])
   else:
    assert row['policy_object_input'] is not None
    assert state.get('maximum_event_s') is None or state['maximum_event_s']<=now+1e-8
    assert state.get('RGB_available_s',0.)<=now+1e-8
 failure='none' if verified else ('initialization_failure' if init_failed else ('no_contact' if result['hand_contact_steps']==0 else ('contact_then_termination' if result['terminated'] else 'contact_not_retained_at_task_end')))
 row=dict(case_id=case['case_id'],physical_case_id=case['physical_case_id'],seed=int(case['seed']),method=method,
  endpoint_hz=float(case['endpoint_hz']) if method in PRIMARY else None,protocol_endpoint_hz=float(case['endpoint_hz']),
  frontend_id=frontend_id(configured) if method in PRIMARY else 'not_applicable_reference',frontend_overrides=configured,
  verified_catch=verified,raw_success=bool(result['success']),terminated=bool(result['terminated']),failure_class=failure,
  initialization_valid=None if initialization is None else bool(initialization['valid']),
  hand_contact_steps=int(result['hand_contact_steps']),final_contact=bool(trace and trace[-1]['hand_contact']),first_touch_s=result['first_touch'],
  sim_time_s=float(result['sim_time']),wall_s=float(execution['wall_s']),raw_events=events_count,packets=len(packets),
  trace_rows=len(trace),output=str(run),perception=precontact_metrics(trace,result) if method in PRIMARY else None,**longest_contact_steps(trace))
 legacy=dict(case_id=case['case_id'],physical_case_id=case['physical_case_id'],seed=int(case['seed']),mode=method,
  endpoint_hz=case['endpoint_hz'],result=result,protocol=protocol,output=str(run),initialization=initialization,
  verified_catch=verified,final_hand_contact=row['final_contact'],terminal_flag_conflict=bool(result['success'] and result['terminated']),
  raw_event_count=events_count,trace_rows=len(trace),trace_sha256=sha(run/'episode_000/trace.json'),window_audit=window_audit)
 return row,legacy


def wilson(k,n):
 if not n:return [None,None]
 z=1.959963984540054;p=k/n;den=1+z*z/n;centre=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [max(0.,centre-half),min(1.,centre+half)]


def group_rows(rows):
 groups={}
 for row in rows:groups.setdefault((row['method'],row['endpoint_hz'],row['frontend_id']),[]).append(row)
 output=[]
 for (method,hz,fid),values in sorted(groups.items(),key=lambda x:str(x[0])):
  units={(r['physical_case_id'],r['seed']) for r in values};assert len(units)==len(values),'Repeated physical blocks in one method/rate/frontend group'
  k=sum(r['verified_catch'] for r in values);n=len(values)
  output.append(dict(method=method,endpoint_hz=hz,frontend_id=fid,n_blocks=n,verified_catches=k,success_rate=k/n,
   wilson95=wilson(k,n),initialization_failures=sum(r['failure_class']=='initialization_failure' for r in values),
   contact_episodes=sum(r['hand_contact_steps']>0 for r in values),wall_s_sum=sum(r['wall_s'] for r in values)))
 return output


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['validation','blind'],required=True)
 p.add_argument('--wait',action='store_true');p.add_argument('--timeout-s',type=float,default=43200)
 p.add_argument('--job-result-file',action='append',help='Explicit currently selected normal-exit job receipt; omitted uses exactly one top-level result per shard')
 p.add_argument('--recovery-plan',help='Explicit preregistered OOM recovery manifest under the actual stage root; no automatic failed-worker exemption')
 a=p.parse_args();root=a.root.resolve();spec=read(root/'EVALUATION_PROTOCOL.json');planned=expected_rows(spec);start=time.monotonic()
 assert not (root/'FAIR_COLLECTION_COMPLETE.json').exists(),'Refuse to overwrite an already sealed completed stage'
 while True:
  done=finished_progress(root,spec,a.job_result_file,a.recovery_plan)
  if done is not None:break
  if not a.wait:raise RuntimeError('Actual planned stage actors/jobs are not all closed; no score emitted')
  if time.monotonic()-start>a.timeout_s:raise TimeoutError('Actual stage remains incomplete')
  time.sleep(5)
 executions,job_receipts,shards,scheduling=done;verify_inputs(spec);rows=[];legacy=[]
 for key,case in planned.items():
  row,record=actor_record(root,spec,case,key[1],executions[key]);rows.append(row);legacy.append(record)
 # Repeated frontend/rate trials must share the actual reset state and RGB-D init.
 initial_by_block={};rgbd_by_block={};launch_by_block={}
 for row,record in zip(rows,legacy):
  key=(row['physical_case_id'],row['seed']);initial=record['result']['initial'];launch=planned[(row['case_id'],row['method'])]['launch']
  if key in initial_by_block:assert initial_by_block[key]==initial,'Actual paired reset-state mismatch: '+str(key)
  initial_by_block[key]=initial
  if key in launch_by_block:assert launch_by_block[key]==launch,'Registered physical-case launch mismatch'
  launch_by_block[key]=launch
  if row['method']!='state_reference':
   if key in rgbd_by_block:assert rgbd_by_block[key]==record['initialization'],'Shared first RGB-D mismatch: '+str(key)
   rgbd_by_block[key]=record['initialization']
 if a.stage=='blind':
  for method in PRIMARY:assert len({r['frontend_id'] for r in rows if r['method']==method})==1,'Blind frontend changed by cadence/case'
 groups=group_rows(rows)
 metadata=dict(actual_planned_actors=len(planned),physical_seed_blocks=len(initial_by_block),
  sample_unit='explicit physical_case_id and episode-seed block; candidate/rate repeats are correlated conditions, not new independent cases',
  stage=a.stage,declared_old_expected_episodes=spec.get('expected_episodes'),declared_old_sample_unit=spec.get('sample_unit'))
 fields=['case_id','physical_case_id','seed','method','endpoint_hz','frontend_id','verified_catch','failure_class','initialization_valid','raw_success','terminated','hand_contact_steps','final_contact','first_touch_s','sim_time_s','wall_s','raw_events','packets','trace_rows','longest_consecutive_contact_control_steps','contact_controlstep_run_recorded_interval_span_s','contact_controlstep_resolution_s','output']
 for name,encoding in [('FAIR_EPISODES.csv','utf-8'),('公平前端逐集结果.csv','utf-8-sig')]:
  with (root/name).open('w',newline='',encoding=encoding) as stream:
   w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 receipt=dict(status='PASS',valid_completed_unique_episodes=len(rows),frozen_source_policy_profile_hashes_verified=True,all_shards_checked=shards,
  selected_normal_exit_job_receipts=[dict(path=str(x),sha256=sha(x)) for x in job_receipts],scheduling_audit=scheduling,all_task_failures_retained=True,**metadata)
 (root/'DATA_COMPLETION_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
 (root/'PAIRED_PORTABLE.json').write_text(json.dumps(dict(complete=True,violations=[],records=legacy,
  scope='Actual event-input control; noisy-GT reference is a separate control mode, not an event result or guaranteed upper bound',metadata=metadata),indent=2))
 (root/'FAIR_STAGE_RESULTS.json').write_text(json.dumps(dict(status='PASS',complete=True,metadata=metadata,rows=rows,groups=groups,
  primary_methods=list(PRIMARY),reference_methods=list(REFERENCES),references_not_pooled=True,
  primary_score='time-limit flag AND no native failure termination AND actual hand contact AND final40ms controlstep contains actual hand contact',
  supplementary_contact_note='Consecutive contact-containing40ms steps are not continuous0.5s physical hold; no score threshold changed'),indent=2))
 with (root/'audit_fair_raw.final.log').open('w') as log:
  subprocess.run([sys.executable,str(HERE/'audit_fair_raw.py'),'--root',str(root)],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert read(root/'FINAL_AUDIT.json')['status']=='PASS'
 marker=dict(status='PASS',actual_episodes=len(rows),physical_seed_blocks=len(initial_by_block),groups=groups,
  all_actual_jobs_normal_exit=not bool(scheduling.get('infrastructure_recovery')),all_final_actor_attempts_normal_exit=True,
  historical_infrastructure_failures_retained=bool(scheduling.get('infrastructure_recovery')),all_task_failures_retained=True,raw_manifest_sha256=sha(root/'SHA256SUMS'),
  final_audit_sha256=sha(root/'FINAL_AUDIT.json'),results_sha256=sha(root/'FAIR_STAGE_RESULTS.json'),
  note='Completion marker is written after successful raw audit; manifest covers its predecessor files, not this marker itself')
 (root/'FAIR_COLLECTION_COMPLETE.json').write_text(json.dumps(marker,indent=2));print(json.dumps(marker),flush=True)


if __name__=='__main__':main()
