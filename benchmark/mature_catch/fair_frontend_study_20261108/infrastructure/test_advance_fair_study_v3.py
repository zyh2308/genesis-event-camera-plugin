"""CPU fixtures only; no SSH, renderer, physics, actual job or GPU is started."""
from pathlib import Path
from unittest import mock
import json, tempfile, types, unittest
import advance_fair_study_v3 as d


def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
def completion(stage,phase,count):
 write(stage/'FAIR_STAGE_RESULTS.json',dict(status='PASS',complete=True,metadata={'stage':phase,'actual_planned_actors':count},rows=[{}]*count))
 write(stage/'FINAL_AUDIT.json',dict(status='PASS',episodes=count));(stage/'SHA256SUMS').write_text('fixture manifest\n')
 write(stage/'FAIR_COLLECTION_COMPLETE.json',dict(status='PASS',actual_episodes=count,
  results_sha256=d.sha(stage/'FAIR_STAGE_RESULTS.json'),final_audit_sha256=d.sha(stage/'FINAL_AUDIT.json'),raw_manifest_sha256=d.sha(stage/'SHA256SUMS')))


class Contracts(unittest.TestCase):
 def test_existing_completion_and_lock_are_verified_without_rewrite(self):
  with tempfile.TemporaryDirectory() as temp:
   s=Path(temp);completion(s,'validation',48);write(s/'EVALUATION_PROTOCOL.json',{'fixture':True})
   env={'CATCHIT_EVENT_OBSERVATION_WINDOW_MS':'10','CATCHIT_EVENT_FLIGHT_FIT':'1','CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS':'50'}
   lock=dict(status='LOCKED',locked_utc=d.stamp(),validation_root=str(s),actual_primary_episodes=48,physical_seed_blocks=4,
    no_blind_outcome_used=True,single_global_configuration_per_source=True,frontend_by_method={'ours':env,'v2e':env},
    validation_results_sha256=d.sha(s/'FAIR_STAGE_RESULTS.json'),validation_completion_sha256=d.sha(s/'FAIR_COLLECTION_COMPLETE.json'),registered_protocol_sha256=d.sha(s/'EVALUATION_PROTOCOL.json'))
   write(s/'FRONTEND_SELECTION.json',lock);before=d.sha(s/'FRONTEND_SELECTION.json')
   d.verify_completion(s,'validation',48);self.assertEqual(d.verify_lock(s/'FRONTEND_SELECTION.json',s),lock)
   self.assertEqual(before,d.sha(s/'FRONTEND_SELECTION.json'))
   (s/'SHA256SUMS').write_text('changed')
   with self.assertRaises(AssertionError):d.verify_completion(s,'validation',48)

 def test_cost_recipe_uses_one_public_reference_and_fixed_source(self):
  steps=d.cost_commands(Path('/new'),Path('/locked'))
  self.assertEqual([x[0] for x in steps],['record','legacy','required','parity'])
  self.assertIn('--record-physics-reference',steps[0][1]);self.assertNotIn('--physics-reference',steps[0][1])
  for _,cmd in steps[1:3]:
   self.assertEqual(cmd[cmd.index('--physics-reference')+1],str(Path('/new/public_physics_recording/PHYSICS_REFERENCE.json')))
   self.assertEqual(cmd[cmd.index('--source-meta')+1],str(d.ROOT/'FAIR_RUNTIME.json'))
   self.assertEqual(cmd[cmd.index('--reference-run')+1],str(d.REFERENCE))

 def test_cost_failure_is_truthful_and_skips_remaining_cost_steps(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);write(root/'FAIR_RUNTIME.json',{});write(root/'frontend.json',{'env':{}});stage=root/'smoke';stage.mkdir()
   proc=types.SimpleNamespace(pid=999999,wait=lambda timeout:1)
   with mock.patch.object(d,'ROOT',root),mock.patch.object(d.subprocess,'Popen',return_value=proc),mock.patch.object(d,'terminate_step'),mock.patch.object(d.signal,'signal'):
    self.assertEqual(d.run_cost_sequence(stage,root/'frontend.json'),1)
   actual=d.read(stage/'COST_V3_SMOKE_SEQUENCE.json')
   self.assertEqual(actual['status'],'FAILED');self.assertEqual(len(actual['steps']),1);self.assertEqual(actual['steps'][0]['exit_code'],1)
   self.assertFalse(actual['formal_cost'])

 def test_release_requires_own_reservation_to_be_marked_released(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);stage=root/'stage';stage.mkdir();res=root/'reservations';res.mkdir()
   rec=dict(pid=22,process_start_ticks='33',gpu_uuid='fixture-only',gpu_index=0)
   write(stage/'job.allocation.json',rec);write(res/'job_gpu0.json',dict(rec,gpu_reserved=False,status='released_after_job'))
   with mock.patch.object(d,'RESERVATIONS',res):
    self.assertEqual(d.verify_cost_release(stage,'job')['status'],'RELEASE_CONFIRMED')
    write(res/'job_gpu0.json',dict(rec,gpu_reserved=True,status='running'))
    with self.assertRaises(AssertionError):d.verify_cost_release(stage,'job')

 def test_main_cost_failure_still_launches_gpu0_after_manager_exit(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);validation=root/'frontend_validation';completion(validation,'validation',48)
   write(validation/'EVALUATION_PROTOCOL.json',{'fixture':True});(validation/'workqueue').mkdir();(validation/'workqueue/fair_workqueue_worker.py').write_text('# fixture')
   env={'CATCHIT_EVENT_OBSERVATION_WINDOW_MS':'10','CATCHIT_EVENT_FLIGHT_FIT':'1','CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS':'50'}
   write(validation/'FRONTEND_SELECTION.json',dict(status='LOCKED',locked_utc=d.stamp(),validation_root=str(validation),actual_primary_episodes=48,physical_seed_blocks=4,
    no_blind_outcome_used=True,single_global_configuration_per_source=True,frontend_by_method={'ours':env,'v2e':env},
    validation_results_sha256=d.sha(validation/'FAIR_STAGE_RESULTS.json'),validation_completion_sha256=d.sha(validation/'FAIR_COLLECTION_COMPLETE.json'),registered_protocol_sha256=d.sha(validation/'EVALUATION_PROTOCOL.json')))
   source=root/'source_cae1ab9e7f48';meta=dict(source=str(source),code_commit=d.COMMIT,runtime_files={})
   write(root/'FAIR_RUNTIME.json',meta);write(root/'BASE_RUNTIME.json',meta)
   write(root/'blind_design/SEED_COLLISION_SERVER.json',{'status':'PASS'});write(root/'blind_design/BLIND32_BANK.json',{'fixture':True})
   write(validation/'catchit_fair_frontend_validation_g1_gpu0_dev_20261108b.job.json',dict(env={}))
   affinity=[];launched=[];waited=[];ran=[]
   def process(command,**kwargs):
    self.assertEqual(affinity[-1],list(range(128)))
    jobfile=Path(command[-1]);job=d.read(jobfile);launched.append(job['name'])
    def wait():
     waited.append(job['name']);write(jobfile.parent/(job['name']+'.result.json'),dict(status='failed',exit_code=1));return 0
    return types.SimpleNamespace(pid=1000+len(launched),wait=wait)
   def run(command,**kwargs):
    name=Path(command[6]).name;ran.append(name)
    if name=='make_blind_protocol.py':
     blind=root/'blind32';blind.mkdir();(blind/'workqueue').mkdir()
     write(blind/'EVALUATION_PROTOCOL.json',dict(frontend_selection_sha256=d.sha(validation/'FRONTEND_SELECTION.json'),source=str(source),cases=[{}]*320))
    elif name=='collect_fair_stage.py':completion(root/'blind32','blind',320)
    else:self.fail('Existing development/lock should be reused; unexpected command '+str(command))
   with (mock.patch.multiple(d,ROOT=root,SOURCE=source,TOOLS=root/'analysis_v1',BANK_SHA=d.sha(root/'blind_design/BLIND32_BANK.json')),
    mock.patch.object(d,'start_ticks',return_value='fixture'),mock.patch.object(d.os,'sched_getaffinity',return_value=set(range(128)),create=True),
    mock.patch.object(d.os,'sched_setaffinity',side_effect=lambda pid,v:affinity.append(list(v)),create=True),mock.patch.object(d.os,'cpu_count',return_value=128),
    mock.patch.object(d.subprocess,'Popen',side_effect=process),mock.patch.object(d.subprocess,'run',side_effect=run)):
    d.main()
   self.assertIn('gpu1_blind',launched[0]);self.assertIn('cost_v3',launched[1]);self.assertIn('gpu0_blind',launched[2])
   self.assertEqual(waited,[launched[1]])
   self.assertEqual(d.read(root/'ADVANCE_V3_COST_RECEIPT.json')['status'],'FAILED')
   self.assertEqual(d.read(root/'ADVANCE_V3_STATUS.json')['stage'],'BLIND_COMPLETE')
   self.assertEqual(ran,['make_blind_protocol.py','collect_fair_stage.py'])


if __name__=='__main__':unittest.main(verbosity=2)
