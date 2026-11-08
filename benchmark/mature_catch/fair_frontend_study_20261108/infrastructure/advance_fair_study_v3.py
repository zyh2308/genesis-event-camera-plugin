"""Independent coordinator: audited development -> locked blind + isolated cost smoke.

No actor, protocol, bank, estimator or frozen simulator changes. The cost process
owns GPU0 only through the original atomic launcher; GPU1 independently runs blind.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, shutil, signal, subprocess, sys, time, traceback

ROOT=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/fair_frontend_blind_20261108b')
PY='/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python'
TOOLS=ROOT/'analysis_v1'
COST_TOOLS=ROOT/'cost_tools_v3'
REFERENCE=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/hdr_utility_diagnosis_20261108a/contact_paired25_windows10/cases/v01_normal25/ours')
LAUNCHER='/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/visual_fast_paired_20261108a/run_gpu_stage_split_cpu.py'
COMMIT='4931807d1cb229a55a96961b5c4d99b054ef4fb0'
SOURCE=ROOT/'source_cae1ab9e7f48'
RESERVATIONS=Path('/home/zhaoyuhan/Eventbased_WAM/.gpu_reservations')
BANK_SHA='a38be34e36a70b710e711b91fabd4673b244b2107e7c9d2b8a14c5f0784e6b72'
STEP_CHILD=None


def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def fresh(p,value):
 with Path(p).open('x') as f:json.dump(value,f,indent=2)
def start_ticks():return Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]


def verify_completion(stage,phase,actors):
 marker=read(stage/'FAIR_COLLECTION_COMPLETE.json')
 assert marker['status']=='PASS' and marker['actual_episodes']==actors
 for key,name in [('results_sha256','FAIR_STAGE_RESULTS.json'),('final_audit_sha256','FINAL_AUDIT.json'),('raw_manifest_sha256','SHA256SUMS')]:
  assert marker[key]==sha(stage/name),'Completed stage bytes changed: '+name
 result=read(stage/'FAIR_STAGE_RESULTS.json');audit=read(stage/'FINAL_AUDIT.json')
 assert result['status']=='PASS' and result['complete'] and result['metadata']['stage']==phase
 assert result['metadata']['actual_planned_actors']==len(result['rows'])==actors
 assert audit['status']=='PASS' and audit['episodes']==actors
 return marker


def verify_lock(path,validation):
 lock=read(path)
 assert lock['status']=='LOCKED' and lock['locked_utc'] and Path(lock['validation_root']).resolve()==validation.resolve()
 assert lock['actual_primary_episodes']==48 and lock['physical_seed_blocks']==4
 for field,name in [('validation_results_sha256','FAIR_STAGE_RESULTS.json'),('validation_completion_sha256','FAIR_COLLECTION_COMPLETE.json'),('registered_protocol_sha256','EVALUATION_PROTOCOL.json')]:
  assert lock[field]==sha(validation/name),'Locked development bytes changed: '+name
 assert lock['no_blind_outcome_used'] and lock['single_global_configuration_per_source']
 assert set(lock['frontend_by_method'])=={'ours','v2e'}
 return lock


def terminate_step():
 global STEP_CHILD
 if STEP_CHILD is None:return
 try:os.killpg(STEP_CHILD.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 try:STEP_CHILD.wait(timeout=5)
 except subprocess.TimeoutExpired:
  try:os.killpg(STEP_CHILD.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  STEP_CHILD.wait()
 STEP_CHILD=None


def sequence_stop(signum,frame):
 terminate_step();raise SystemExit(128+signum)


def cost_commands(stage,frontend):
 public=stage/'public_physics_recording/PHYSICS_REFERENCE.json'
 common=[PY,'-B','-u',str(COST_TOOLS/'run_fixed_trajectory_cost.py'),
  '--source-meta',str(ROOT/'FAIR_RUNTIME.json'),'--reference-run',str(REFERENCE),
  '--frontend-json',str(frontend),'--method','ours','--rate','25']
 return [
  ('record',common+['--repeat-id','public_recording','--output',str(public.parent),'--acquisition','legacy','--retain-audit-io','--record-physics-reference']),
  ('legacy',common+['--physics-reference',str(public),'--repeat-id','validation_legacy','--output',str(stage/'ours25_legacy'),'--acquisition','legacy','--retain-audit-io']),
  ('required',common+['--physics-reference',str(public),'--repeat-id','validation_native','--output',str(stage/'ours25_required'),'--acquisition','required']),
  ('parity',[PY,'-B','-u',str(COST_TOOLS/'compare_cost_event_parity.py'),'--legacy',str(stage/'ours25_legacy'),
   '--native',str(stage/'ours25_required'),'--output',str(stage/'ours25_parity.json')])]


def run_cost_sequence(stage,frontend):
 """Runs inside an actual atomic GPU job. A failure skips remaining smoke steps."""
 global STEP_CHILD
 signal.signal(signal.SIGTERM,sequence_stop);signal.signal(signal.SIGINT,sequence_stop)
 receipt=dict(status='RUNNING',started_utc=stamp(),cost_smoke_only=True,formal_cost=False,
  frontend_sha256=sha(frontend),source_meta_sha256=sha(ROOT/'FAIR_RUNTIME.json'),steps=[])
 output=stage/'COST_V3_SMOKE_SEQUENCE.json'
 try:
  for label,command in cost_commands(stage,frontend):
   step=dict(label=label,command=command,started_utc=stamp());begin=time.monotonic();receipt['steps'].append(step)
   try:
    with (stage/(label+'.log')).open('x') as log:
     STEP_CHILD=subprocess.Popen(command,cwd=stage,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     step['pid']=STEP_CHILD.pid;step['exit_code']=STEP_CHILD.wait(timeout=1200 if label!='parity' else 180)
    terminate_step()
    assert step['exit_code']==0,'Actual cost smoke step failed: '+label
    resultpath={'record':stage/'public_physics_recording/COST_RESULT.json','legacy':stage/'ours25_legacy/COST_RESULT.json',
     'required':stage/'ours25_required/COST_RESULT.json','parity':stage/'ours25_parity.json'}[label]
    result=read(resultpath);step.update(result_path=str(resultpath),result_sha256=sha(resultpath),result_status=result['status'])
    expected={'record':'PHYSICS_REFERENCE_RECORDED','legacy':'MEASURED_FIXED_TRAJECTORY','required':'MEASURED_FIXED_TRAJECTORY','parity':'PASS'}[label]
    assert result['status']==expected
    if label=='record':assert result['cost_measurement'] is False and result['eligible_as_native_primary'] is False
    elif label=='parity':assert result['ordered_events_exact'] and result['physics_reference_sha256']==sha(stage/'public_physics_recording/PHYSICS_REFERENCE.json')
    else:assert result['physics_reference_sha256']==sha(stage/'public_physics_recording/PHYSICS_REFERENCE.json') and result['fixed_capture_budget_pass']
   finally:
    step.update(finished_utc=stamp(),wall_s=time.monotonic()-begin);output.write_text(json.dumps(receipt,indent=2))
  receipt['status']='PASS'
 except BaseException as exc:
  receipt.update(status='FAILED',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  terminate_step();receipt['finished_utc']=stamp();output.write_text(json.dumps(receipt,indent=2))
 return 0 if receipt['status']=='PASS' else 1


def verify_cost_release(stage,name):
 allocation=stage/(name+'.allocation.json')
 if not allocation.exists():return dict(status='NO_GPU_ALLOCATION_WAS_CREATED')
 actual=read(allocation)
 reservation=RESERVATIONS/(name+'_gpu'+str(actual['gpu_index'])+'.json')
 final=read(reservation)
 assert final['pid']==actual['pid'] and final['process_start_ticks']==actual['process_start_ticks'] and final['gpu_uuid']==actual['gpu_uuid']
 assert final['gpu_reserved'] is False and final['status'].startswith('released'),'Cost launcher did not confirm reservation release'
 return dict(status='RELEASE_CONFIRMED',reservation=str(reservation),sha256=sha(reservation))


def main():
 started=time.monotonic();state=ROOT/'ADVANCE_V3_STATUS.json'
 inherited=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,range(os.cpu_count()))
 full=sorted(os.sched_getaffinity(0));narrow=[8,9];assert set(narrow)<=set(full)
 pidfile=ROOT/'ADVANCE_V3_COORDINATOR.json'
 fresh(pidfile,dict(status='running',pid=os.getpid(),process_start_ticks=start_ticks(),
  started_utc=stamp(),script=str(Path(__file__).resolve()),script_sha256=sha(__file__),inherited_affinity=inherited,full_affinity=full,cpu_affinity=narrow))
 os.sched_setaffinity(0,narrow)
 def update(phase,**extra):state.write_text(json.dumps(dict(stage=phase,updated_utc=stamp(),elapsed_s=time.monotonic()-started,**extra),indent=2))
 def run(label,args,timeout=45000):
  update(label)
  with (ROOT/('v3_'+label+'.log')).open('x') as log:
   subprocess.run(['nice','-n','10',PY,'-B','-u']+args,env=os.environ,cwd=ROOT,stdin=subprocess.DEVNULL,
    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
 def manager(jobfile,stage,name):
  os.sched_setaffinity(0,full)
  try:
   with (stage/(name+'.supervisor.log')).open('x') as log:
    return subprocess.Popen(['/usr/bin/python3','-u',LAUNCHER,'--job',str(jobfile)],cwd=stage,
     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  finally:os.sched_setaffinity(0,narrow)
 try:
  meta=read(ROOT/'FAIR_RUNTIME.json');base=read(ROOT/'BASE_RUNTIME.json')
  assert meta['source']==base['source']==str(SOURCE) and meta['code_commit']==base['code_commit']==COMMIT
  assert meta['runtime_files']==base['runtime_files']
  validation=ROOT/'frontend_validation'
  if not (validation/'FAIR_COLLECTION_COMPLETE.json').exists():
   run('audit_development',[str(TOOLS/'collect_fair_stage.py'),'--root',str(validation),'--stage','validation','--wait',
    '--recovery-plan',str(validation/'INFRASTRUCTURE_RECOVERY.json')])
  verify_completion(validation,'validation',48)
  selection=validation/'FRONTEND_SELECTION.json'
  if not selection.exists():run('lock_frontends',[str(TOOLS/'select_frontend.py'),'--root',str(validation)],180)
  lock=verify_lock(selection,validation)
  assert read(ROOT/'blind_design/SEED_COLLISION_SERVER.json')['status']=='PASS'
  assert sha(ROOT/'blind_design/BLIND32_BANK.json')==BANK_SHA
  blind=ROOT/'blind32';assert not blind.exists(),'Blind stage already exists: inspect it; do not duplicate or overwrite actors'
  run('prepare_blind_protocol',[str(ROOT/'tools/make_blind_protocol.py'),'--base-runtime',str(ROOT/'BASE_RUNTIME.json'),
   '--selection',str(selection),'--bank',str(ROOT/'blind_design/BLIND32_BANK.json'),'--root',str(blind),'--output',str(blind)],180)
  shutil.copyfile(validation/'workqueue/fair_workqueue_worker.py',blind/'workqueue/fair_workqueue_worker.py')
  protocol=blind/'EVALUATION_PROTOCOL.json';spec=read(protocol)
  assert spec['frontend_selection_sha256']==sha(selection) and len(spec['cases'])==320 and spec['source']==str(SOURCE)
  fresh(blind/'BLIND_PREREQUISITES.json',dict(status='PASS',locked_utc=lock['locked_utc'],verified_utc=stamp(),
   selection_sha256=sha(selection),protocol_sha256=sha(protocol),bank_sha256=BANK_SHA,
   development_complete_sha256=sha(validation/'FAIR_COLLECTION_COMPLETE.json'),camera_original_policy_frozen_core_unchanged=True))
  template=read(validation/'catchit_fair_frontend_validation_g1_gpu0_dev_20261108b.job.json');jobs=[]
  def launch_blind(gpu):
   worker='g1_gpu'+str(gpu)+'_blind';name='catchit_fair_blind32_'+worker+'_20261108b'
   job=dict(name=name,gpu_candidates=[gpu],command=['/usr/bin/python3','-u',str(blind/'workqueue/fair_workqueue_worker.py'),
    '--protocol',str(protocol),'--worker-id',worker],env=template['env'],cwd=str(blind),timeout_s=86400)
   jobfile=blind/(name+'.job.json');fresh(jobfile,job);proc=manager(jobfile,blind,name)
   jobs.append(dict(worker_id=worker,supervisor_pid=proc.pid,gpu_candidate=gpu))
   (blind/'JOBS_LAUNCHED.json').write_text(json.dumps(dict(jobs=jobs,utc=stamp()),indent=2))
  update('launch_gpu1_blind',frontend_by_method=lock['frontend_by_method']);launch_blind(1)
  cost_receipt=dict(status='SETUP_FAILED',started_utc=stamp(),cost_smoke_only=True,formal_cost=False);coststage=ROOT/'cost_v3_preblind_smoke'
  costname='catchit_fair_cost_v3_preblind_gpu0_20261108b';manager_finished=False;manager_started=False
  try:
   coststage.mkdir(exist_ok=False)
   frontend=coststage/'locked_ours_frontend.json'
   fresh(frontend,dict(env=lock['frontend_by_method']['ours'],selection_sha256=sha(selection)))
   job=dict(name=costname,gpu_candidates=[0],command=[PY,'-B','-u',str(Path(__file__).resolve()),'--cost-sequence',
    '--stage-dir',str(coststage),'--frontend-json',str(frontend)],env=template['env'],cwd=str(coststage),timeout_s=3900)
   jobfile=coststage/(costname+'.job.json');fresh(jobfile,job)
   update('cost_v3_smoke_gpu0',gpu1_blind_already_launched=True)
   process=manager(jobfile,coststage,costname);manager_started=True
   cost_receipt['launcher_returncode']=process.wait();manager_finished=True
   receipt_path=coststage/(costname+'.result.json');actual=read(receipt_path)
   sequence=coststage/'COST_V3_SMOKE_SEQUENCE.json'
   cost_receipt.update(status='PASS' if actual['status']=='completed' and actual.get('exit_code')==0 and sequence.exists() and read(sequence)['status']=='PASS' else 'FAILED',
    launcher_result=actual,launcher_result_path=str(receipt_path),launcher_result_sha256=sha(receipt_path),
    sequence_receipt_path=str(sequence) if sequence.exists() else None,sequence_receipt_sha256=sha(sequence) if sequence.exists() else None)
  except BaseException as exc:cost_receipt.update(status='FAILED',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
  finally:
   cost_receipt['finished_utc']=stamp()
   release_error=None
   try:
    if manager_started:
     assert manager_finished,'GPU0 manager is still active; never overlap it with blind'
     cost_receipt['resource_release']=verify_cost_release(coststage,costname)
    else:cost_receipt['resource_release']=dict(status='NO_MANAGER_STARTED')
   except BaseException as exc:
    release_error=exc;cost_receipt['resource_release']=dict(status='UNCONFIRMED',error=str(exc))
   fresh(ROOT/'ADVANCE_V3_COST_RECEIPT.json',cost_receipt)
   if release_error is not None:raise release_error
  update('launch_gpu0_blind',cost_smoke_status=cost_receipt['status']);launch_blind(0)
  run('audit_blind',[str(TOOLS/'collect_fair_stage.py'),'--root',str(blind),'--stage','blind','--wait','--timeout-s','86400'],87000)
  verify_completion(blind,'blind',320)
  update('BLIND_COMPLETE',cost_smoke_status=cost_receipt['status'],frontend_by_method=lock['frontend_by_method'])
 except BaseException as exc:
  update('STOPPED',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
 finally:
  receipt=read(pidfile);receipt.update(status='finished',finished_utc=stamp());pidfile.write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--cost-sequence',action='store_true');parser.add_argument('--stage-dir',type=Path);parser.add_argument('--frontend-json',type=Path)
 args=parser.parse_args()
 if args.cost_sequence:
  assert args.stage_dir and args.frontend_json
  raise SystemExit(run_cost_sequence(args.stage_dir,args.frontend_json))
 main()
