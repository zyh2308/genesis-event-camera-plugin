"""Artifact-only contract tests. Synthetic fixtures are never experimental results."""
from pathlib import Path
import copy, datetime, hashlib, json, subprocess, sys, tempfile, unittest
import numpy as np
import collect_fair_stage as collect
import select_frontend as selection


def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value))


class Contracts(unittest.TestCase):
 def test_frontend_case_override_and_schema(self):
  spec={'frontend_by_method':{'ours':selection.CANONICAL}}
  env=dict(selection.CANONICAL,CATCHIT_EVENT_FLIGHT_FIT='0')
  case={'case_id':'x','frontend_overrides':env}
  self.assertEqual(collect.frontend(spec,case,'ours'),env)
  with self.assertRaises(ValueError):collect.frontend({}, {'case_id':'x'},'ours')
  with self.assertRaises(ValueError):collect.frontend({},dict(case,frontend_overrides={'extra':'1'}),'ours')

 def test_expected_counts_and_dynamic_shards(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);cases=[]
   for shard in [3,7]:
    c=dict(case_id='case'+str(shard),physical_case_id='p'+str(shard),seed=shard,endpoint_hz=25,
           shard=shard,method_order=['ours'],frontend_overrides=selection.CANONICAL)
    cases.append(c)
    write(root/f'job{shard}.result.json',dict(status='completed',exit_code=0))
    write(root/f'shard{shard}_complete.json',dict(all_exit_zero=True,postflight_hashes_pass=True,completed=1))
    write(root/f'shard{shard}_progress.json',[dict(case_id=c['case_id'],seed=shard,method='ours',endpoint_hz=25,exit_code=0,frontend_overrides=selection.CANONICAL)])
   spec=dict(cases=cases,expected_episodes=999)
   self.assertEqual(len(collect.expected_rows(spec)),2)
   self.assertEqual(collect.finished_progress(root,spec)[2],[3,7])
   (root/'shard7_complete.json').unlink()
   self.assertIsNone(collect.finished_progress(root,spec))

 def init_fixture(self,root):
  case=dict(case_id='sample',physical_case_id='block',seed=42,endpoint_hz=25,launch={'position':[1,2,3]},frontend_overrides=selection.CANONICAL)
  run=root/'cases/sample/ours';sensor=run/'migration_audit/sensor_episode_000';sensor.mkdir(parents=True)
  source=root/'source';source.mkdir();runtime={};loaded={}
  for name in ['genesis_adapter','sparse_event_sensor','event_sphere_tracker','event_sphere_tracker_kalman',
   'event_sphere_tracker_contour_q80_occlusion','rgb_sphere_initialization','rgbd_sphere_initialization',
   'event_sphere_tracker_rgbd_base','event_bearing_flight','event_observation_windows']:
   module=source/(name+'.py');module.write_text('fixture = True\n');h=hashlib.sha256(module.read_bytes()).hexdigest()
   runtime[module.name]=h;loaded[name]={'path':str(module),'sha256':h}
  spec=dict(source=str(source),runtime_files=runtime,checkpoint_sha256='policy')
  initial=dict(position=[1,2,3],velocity=[0]*6,gravity=[0,0,-9.81],release_time_s=.6)
  result=dict(seed=42,success=False,terminated=False,steps=0,sim_time=0.,hand_contact_steps=0,valid_event_polls=0,first_touch=None,
              initial=initial,failure_class='initialization_failure',task_failure_no_physics_steps=True)
  protocol=dict(checkpoint_sha256='policy',seeds=[42],launch_override=case['launch'],asset_initialization_seed=1702,
   sphere_radius_m=.04,sensor_pitch_offset_degrees=7,event_endpoint_rate_hz=25,timestep=.002,steps_per_policy=20,env_time=2.5,
   gate_until_s=0,gate_after_release_s=None,backend='Genesis',event_generator='ours',initialization_only=False,
   event_observation_window_ms=10,event_flight_fit=True,event_confirmation_max_gap_ms=50,event_control=True,RGB_frame_budget=1,control_depth_frame_budget=1)
  write(run/'protocol.json',protocol);write(run/'episode_000/result.json',result);write(run/'episode_000/trace.json',[])
  contact=dict(original_mj_step_calls=0,contact_profile={'name':'combined','substeps':2},
   contact_calibration=dict(unselected_unchanged=True,hand_geom_ids=list(range(33,50)),after=[{'selected_hand':True,'friction':1.5},{'selected_hand':False,'friction':.95}]))
  write(run/'migration_audit/adapter_audit.json',contact)
  write(run/'loaded_module_receipt.json',loaded)
  write(sensor/'initialization.json',dict(valid=False,RGB_frames_read=1,control_depth_frames_read=1,time_s=0.,available_s=0.))
  write(sensor/'packet_audit.json',[]);write(sensor/'event_samples.json',[])
  np.savez(sensor/'raw_events.npz',events=np.empty((0,4)))
  execution=dict(output=str(run),episodes=[result],wall_s=1.25)
  return case,spec,result,run,sensor,execution

 def test_queue_two_workers_cover_three_shards(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);queue=root/'workqueue';cases=[]
   for shard in range(3):
    case=dict(case_id='case'+str(shard),physical_case_id='p'+str(shard),seed=shard,endpoint_hz=25,shard=shard,
     method_order=['ours'],frontend_overrides=selection.CANONICAL);cases.append(case)
    write(root/f'shard{shard}_complete.json',dict(all_exit_zero=True,postflight_hashes_pass=True,completed=1))
    write(root/f'shard{shard}_progress.json',[dict(case_id=case['case_id'],seed=shard,method='ours',endpoint_hz=25,exit_code=0,frontend_overrides=selection.CANONICAL)])
   spec=dict(cases=cases);write(root/'EVALUATION_PROTOCOL.json',spec);sha=collect.sha(root/'EVALUATION_PROTOCOL.json')
   write(queue/'PLAN.json',dict(protocol_sha256=sha,ordered_shards=[0,1,2]))
   for worker,shards in [('workerA',[0,2]),('workerB',[1])]:
    write(root/(worker+'.job.json'),dict(name=worker,command=['python','fair_workqueue_worker.py','--protocol',str(root/'EVALUATION_PROTOCOL.json'),'--worker-id',worker]))
    write(root/(worker+'.result.json'),dict(status='completed',exit_code=0))
    own=[]
    for shard in shards:
     claim=dict(shard=shard,worker_id=worker,status='completed',exit_code=0,protocol_sha256=sha)
     own.append(claim);write(queue/'claims'/f'shard{shard:03d}.json',claim)
    write(queue/(worker+'.finished.json'),dict(status='PASS',shards=len(own)))
    write(queue/(worker+'.progress.json'),own)
   actual,paths,shards,proof=collect.finished_progress(root,spec)
   self.assertEqual(len(actual),3);self.assertEqual(len(paths),2);self.assertEqual(shards,[0,1,2])
   self.assertEqual(proof['mode'],'finite_registered_workqueue')
   (root/'workerB.result.json').unlink();self.assertIsNone(collect.finished_progress(root,spec))
   write(root/'workerB.result.json',dict(status='failed',error='synthetic fixture'))
   with self.assertRaises(RuntimeError):collect.finished_progress(root,spec)
   write(root/'workerB.result.json',dict(status='completed',exit_code=0))
   claim=collect.read(queue/'claims/shard001.json');claim['worker_id']='unregistered';write(queue/'claims/shard001.json',claim)
   with self.assertRaises(AssertionError):collect.finished_progress(root,spec)

 def test_initialization_failure_is_completed_task_failure(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);case,spec,result,run,sensor,execution=self.init_fixture(root)
   row,legacy=collect.actor_record(root,spec,case,'ours',execution)
   self.assertFalse(row['verified_catch']);self.assertEqual(row['failure_class'],'initialization_failure')
   self.assertEqual(row['sim_time_s'],0);self.assertEqual(row['wall_s'],1.25)
   self.assertEqual(row['perception']['control_state_coverage'],0.)
   self.assertEqual(row['perception']['normalized_equal_weight_loss'],1.)
   self.assertEqual(collect.group_rows([row])[0]['initialization_failures'],1)
   self.assertEqual(collect.group_rows([row])[0]['n_blocks'],1)
   write(sensor/'worker_start.json',dict(ready=True))
   with self.assertRaises(AssertionError):collect.actor_record(root,spec,case,'ours',execution)
   (sensor/'worker_start.json').unlink()
   np.savez(sensor/'raw_events.npz',events=np.array([[0.,2.,3.,1.]]))
   with self.assertRaises(AssertionError):collect.actor_record(root,spec,case,'ours',execution)

 def test_explicit_oom_recovery_preserves_history_and_retries_original_actors(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);queue=root/'workqueue';archive=root/'infrastructure_failures/attempt0/shard001';cases=[]
   for shard in range(3):
    case=dict(case_id='case'+str(shard),physical_case_id='block'+str(shard),seed=200+shard,endpoint_hz=25,shard=shard,
     method_order=['ours'],frontend_overrides=selection.CANONICAL);cases.append(case)
    write(root/f'shard{shard}_complete.json',dict(all_exit_zero=True,postflight_hashes_pass=True,completed=1))
    write(root/f'shard{shard}_progress.json',[dict(case_id=case['case_id'],seed=case['seed'],method='ours',endpoint_hz=25,exit_code=0,frontend_overrides=selection.CANONICAL)])
   spec=dict(cases=cases);write(root/'EVALUATION_PROTOCOL.json',spec);sha=collect.sha(root/'EVALUATION_PROTOCOL.json')
   write(queue/'PLAN.json',dict(protocol_sha256=sha,ordered_shards=[0,1,2]))
   def job(worker):return dict(name=worker,command=['python','fair_workqueue_worker.py','--protocol',str(root/'EVALUATION_PROTOCOL.json'),'--worker-id',worker])
   old=dict(shard=0,worker_id='old',status='completed',exit_code=0,protocol_sha256=sha)
   failed=dict(shard=1,worker_id='old',status='infrastructure_failure',exit_code=1,protocol_sha256=sha)
   normal=dict(shard=2,worker_id='normal',status='completed',exit_code=0,protocol_sha256=sha)
   retry=dict(shard=1,worker_id='retry',status='completed',exit_code=0,protocol_sha256=sha)
   for claim in [old,normal,retry]:write(queue/'claims'/('shard%03d.json'%claim['shard']),claim)
   write(root/'old.job.json',job('old'));write(root/'old.result.json',dict(status='failed',exit_code=1))
   write(queue/'old.progress.json',[old,failed]);write(archive/'claims/shard001.json',failed)
   original=copy.deepcopy(collect.read(root/'shard1_progress.json'));original[0]['exit_code']=1
   write(archive/'shard1_progress.json',original);(archive/'cases/case1/ours').mkdir(parents=True)
   (archive/'worker.log').write_text('synthetic OOM fixture; not an experiment')
   for worker,claim in [('normal',normal),('retry',retry)]:
    if worker=='normal':write(root/(worker+'.job.json'),job(worker))
    write(root/(worker+'.result.json'),dict(status='completed',exit_code=0))
    write(queue/(worker+'.progress.json'),[claim]);write(queue/(worker+'.finished.json'),dict(status='PASS',shards=1))
   entry=dict(worker_id='old',job_file='old.job.json',job_sha256=collect.sha(root/'old.job.json'),
    result_file='old.result.json',result_sha256=collect.sha(root/'old.result.json'),
    progress_file='workqueue/old.progress.json',progress_sha256=collect.sha(queue/'old.progress.json'),
    archived_failed_claims=[dict(shard=1,claim_file='infrastructure_failures/attempt0/shard001/claims/shard001.json',
     claim_sha256=collect.sha(archive/'claims/shard001.json'),archived_attempt_root='infrastructure_failures/attempt0/shard001')])
   plan=dict(schema='fair_workqueue_infra_recovery_v1',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    protocol_sha256=sha,reason='cuda_out_of_memory',historical_failed_workers=[entry],replacement_worker_ids=['retry'])
   write(root/'INFRASTRUCTURE_RECOVERY.json',plan)
   newjob=job('retry');newjob['infrastructure_recovery_sha256']=collect.sha(root/'INFRASTRUCTURE_RECOVERY.json');write(root/'retry.job.json',newjob)
   with self.assertRaises(RuntimeError):collect.finished_progress(root,spec)
   found,results,shards,proof=collect.finished_progress(root,spec,recovery_plan='INFRASTRUCTURE_RECOVERY.json')
   self.assertEqual(len(found),3);self.assertEqual(len(results),2)
   self.assertEqual(proof['infrastructure_recovery']['replacement_worker_ids'],['retry'])
   self.assertFalse((queue/'old.finished.json').exists())
   # Neither lost actors nor a different retry input may hide behind recovery.
   (root/'shard1_complete.json').unlink();self.assertIsNone(collect.finished_progress(root,spec,recovery_plan='INFRASTRUCTURE_RECOVERY.json'))
   write(root/'shard1_complete.json',dict(all_exit_zero=True,postflight_hashes_pass=True,completed=1))
   original[0]['seed']+=1;write(archive/'shard1_progress.json',original)
   with self.assertRaises(AssertionError):collect.finished_progress(root,spec,recovery_plan='INFRASTRUCTURE_RECOVERY.json')
 def test_missing_states_and_false_grasp_are_scored(self):
  def row(t,state,contact=False):return dict(t_before=t,t=t+.04,event_state=state,hand_contact=contact,
   scoring_only_GT_position_at_control=[0,0,0],qvel=[0]*9)
  trace=[row(0,None),row(.04,dict(position=[.1,0,0],velocity=[0,0,0],source='false_grasp_propagation')),row(.08,None,True),row(.12,None)]
  m=collect.precontact_metrics(trace,dict(initial={'release_time_s':.04}))
  self.assertEqual(m['queries'],2);self.assertEqual(m['available_states'],1)
  self.assertAlmostEqual(m['penalized_position_error_mean_m'],.3)
  self.assertAlmostEqual(m['penalized_velocity_error_mean_m_s'],1.)
  self.assertAlmostEqual(m['normalized_equal_weight_loss'],.55)
  span=collect.longest_contact_steps([row(0,None,True),row(.04,None,True),row(.08,None),row(.12,None,True)])
  self.assertEqual(span['longest_consecutive_contact_control_steps'],2)
  self.assertTrue(span['continuous_physics_hold_not_inferred'])

 def test_full_collector_raw_audit_keeps_zero_step_failure(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);case,spec,result,run,sensor,execution=self.init_fixture(root)
   case.update(shard=3,method_order=['ours'])
   spec.update(cases=[case],expected_episodes=999,core_root=str(root/'core'),core_files={})
   for index in range(17):
    path=root/'core/genesis_event_plugin'/f'fixture{index}.py';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b'# synthetic hash contract fixture\n')
    spec['core_files'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
   for field,key in [('checkpoint','checkpoint_sha256'),('profile_file','profile_sha256'),('v2e_checkpoint','v2e_checkpoint_sha256')]:
    path=root/field;path.write_bytes(b'fixture')
    spec[field]=str(path);spec[key]=hashlib.sha256(path.read_bytes()).hexdigest()
   protocol=collect.read(run/'protocol.json');protocol['checkpoint_sha256']=spec['checkpoint_sha256'];write(run/'protocol.json',protocol)
   execution.update(case_id=case['case_id'],method='ours',seed=42,endpoint_hz=25,exit_code=0,frontend_overrides=selection.CANONICAL)
   write(root/'EVALUATION_PROTOCOL.json',spec)
   write(root/'job3.result.json',dict(status='completed',exit_code=0))
   write(root/'shard3_complete.json',dict(all_exit_zero=True,postflight_hashes_pass=True,completed=1))
   write(root/'shard3_progress.json',[execution])
   process=subprocess.run([sys.executable,str(Path(collect.__file__)),'--root',str(root),'--stage','validation'],text=True,capture_output=True)
   self.assertEqual(process.returncode,0,process.stdout+process.stderr)
   marker=collect.read(root/'FAIR_COLLECTION_COMPLETE.json');self.assertEqual(marker['actual_episodes'],1)
   actual=collect.read(root/'FAIR_STAGE_RESULTS.json')
   self.assertEqual(actual['metadata']['actual_planned_actors'],1)
   self.assertFalse(actual['rows'][0]['verified_catch'])
   self.assertEqual(actual['groups'][0]['initialization_failures'],1)
   self.assertEqual(collect.read(root/'FINAL_AUDIT.json')['status'],'PASS')

 def selection_data(self):
  rows=[]
  envs=[selection.CANONICAL,dict(selection.CANONICAL,CATCHIT_EVENT_FLIGHT_FIT='0')]
  for source in ['ours','v2e']:
   for env in envs:
    for block in range(4):
     for rate in [25,12.5]:
      rows.append(dict(method=source,frontend_id=collect.frontend_id(env),frontend_overrides=env,
       physical_case_id='fixture'+str(block),seed=100+block,endpoint_hz=rate,verified_catch=block<2,
       perception=dict(control_state_coverage=.75,normalized_equal_weight_loss=.5,
        penalized_position_error_mean_m=.25,penalized_velocity_error_mean_m_s=1.)))
  return dict(status='PASS',complete=True,metadata={'stage':'validation'},rows=rows)

 def test_selection_equal_budget_and_prespecified_ties(self):
  data=self.selection_data();selected=selection.select(data)
  self.assertEqual(selected['frontend_by_method']['ours'],selection.CANONICAL)
  self.assertEqual(selected['actual_primary_episodes'],32)
  # Equal catch: availability on actual validation wins before canonical preference.
  for r in data['rows']:
   if r['method']=='ours' and r['frontend_id']=='w10_fit0_gap50':r['perception']['control_state_coverage']=.9
  self.assertEqual(selection.select(data)['frontend_by_method']['ours']['CATCHIT_EVENT_FLIGHT_FIT'],'0')
  broken=copy.deepcopy(data);broken['rows'].pop()
  with self.assertRaises(AssertionError):selection.select(broken)
  blind=copy.deepcopy(data);blind['metadata']['stage']='blind'
  with self.assertRaises(AssertionError):selection.select(blind)

 def test_selection_lock_has_real_utc_and_cannot_be_overwritten(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);write(root/'FAIR_STAGE_RESULTS.json',self.selection_data());write(root/'EVALUATION_PROTOCOL.json',dict(fixture_only=True))
   write(root/'FAIR_COLLECTION_COMPLETE.json',dict(status='PASS',results_sha256=collect.sha(root/'FAIR_STAGE_RESULTS.json')))
   command=[sys.executable,str(Path(selection.__file__)),'--root',str(root)]
   before=datetime.datetime.now(datetime.timezone.utc);p=subprocess.run(command,text=True,capture_output=True)
   self.assertEqual(p.returncode,0,p.stdout+p.stderr)
   lock=collect.read(root/'FRONTEND_SELECTION.json');actual=datetime.datetime.fromisoformat(lock['locked_utc'])
   self.assertLessEqual(before,actual);self.assertLessEqual(actual,datetime.datetime.now(datetime.timezone.utc))
   self.assertEqual(lock['status'],'LOCKED');self.assertEqual(lock['validation_results_sha256'],collect.sha(root/'FAIR_STAGE_RESULTS.json'))
   p=subprocess.run(command,text=True,capture_output=True);self.assertNotEqual(p.returncode,0)


if __name__=='__main__':unittest.main(verbosity=2)
