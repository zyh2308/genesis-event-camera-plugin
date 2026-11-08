"""Pause only at complete queue blocks; run isolated cost, then hand GPU0 back.

Uses the existing finite workqueue and atomic GPU launcher. No actor is killed,
and no registered scientific case, source, parameter or ordering is rewritten.
"""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, time, traceback

ROOT=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/fair_frontend_blind_20261108b')
PY='/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python'
LAUNCHER='/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/visual_fast_paired_20261108a/run_gpu_stage_split_cpu.py'
RESERVATIONS=Path('/home/zhaoyuhan/Eventbased_WAM/.gpu_reservations')
COMMIT='4931807d1cb229a55a96961b5c4d99b054ef4fb0'
OWNER='robot closed-loop research chat'


def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def fresh(p,value):
 with Path(p).open('x') as f:json.dump(value,f,indent=2)
def start_ticks():return Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]


def released(stage,name,gpu,required=True):
 """Read the atomic launcher's own final reservation; do not alter it."""
 allocation=stage/(name+'.allocation.json')
 reservation=RESERVATIONS/(name+'_gpu'+str(gpu)+'.json')
 if not allocation.exists():
  if required:return None
  if reservation.exists():
   final=read(reservation)
   if final.get('gpu_reserved') is not False or not final.get('status','').startswith('released'):return None
   assert final['owner']==OWNER and final['run_id']==name and Path(final['output_path']).resolve()==stage.resolve()
   return dict(status='RELEASED_WITHOUT_LOCAL_ALLOCATION_RECEIPT',reservation=str(reservation),sha256=sha(reservation))
  return dict(status='NO_ALLOCATION_CREATED')
 actual=read(allocation);final=read(reservation)
 assert actual['gpu_index']==final['gpu_index']==gpu and final['owner']==OWNER and final['run_id']==name
 assert Path(final['output_path']).resolve()==stage.resolve()
 assert all(final[k]==actual[k] for k in ('pid','process_start_ticks','gpu_uuid'))
 if final.get('gpu_reserved') is not False or not final.get('status','').startswith('released'):return None
 return dict(status='RELEASE_CONFIRMED',allocation=str(allocation),allocation_sha256=sha(allocation),
  reservation=str(reservation),reservation_sha256=sha(reservation))


def closed_original_workers(blind,names):
 receipts=[]
 for gpu,name in enumerate(names):
  jobfile=blind/(name+'.job.json');job=read(jobfile);command=job['command']
  worker=command[command.index('--worker-id')+1]
  assert worker=='g1_gpu'+str(gpu)+'_blind' and job['gpu_candidates']==[gpu]
  assert Path(command[command.index('--protocol')+1]).resolve()==(blind/'EVALUATION_PROTOCOL.json').resolve()
  path=blind/(name+'.result.json')
  if not path.exists():return None
  result=read(path)
  if result.get('status')=='failed':raise RuntimeError('Original blind worker failed; retain it and stop checkpoint: '+str(path))
  if result.get('status')!='completed':return None
  assert result['exit_code']==0
  resource=released(blind,name,gpu)
  if resource is None:return None
  finished=blind/'workqueue'/(worker+'.finished.json');progress=blind/'workqueue'/(worker+'.progress.json')
  if not finished.exists():return None
  done=read(finished);rows=read(progress) if progress.exists() else []
  assert done['status']=='PASS' and done['shards']==len(rows)
  for row in rows:
   assert row['status']=='completed' and row['exit_code']==0
   assert row==read(blind/'workqueue/claims'/('shard%03d.json'%row['shard']))
  receipts.append(dict(worker_id=worker,job=str(jobfile),job_sha256=sha(jobfile),result=str(path),result_sha256=sha(path),
   finished_sha256=sha(finished),progress_sha256=sha(progress) if progress.exists() else None,resource_release=resource))
 return receipts


def main():
 blind=ROOT/'blind32';queue=blind/'workqueue';status=ROOT/'BLIND_COST_V4_CHECKPOINT_STATUS.json';begin=time.monotonic()
 os.sched_setaffinity(0,range(os.cpu_count()));full=sorted(os.sched_getaffinity(0));narrow=[8,9]
 assert set(narrow)<=set(full);os.sched_setaffinity(0,narrow)
 def update(phase,**extra):status.write_text(json.dumps(dict(stage=phase,utc=stamp(),elapsed_s=time.monotonic()-begin,**extra),indent=2))
 def manager(jobfile,stage,name):
  os.sched_setaffinity(0,full)
  try:
   with (stage/(name+'.supervisor.log')).open('x') as log:
    return subprocess.Popen(['/usr/bin/python3','-u',LAUNCHER,'--job',str(jobfile)],cwd=stage,
     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  finally:os.sched_setaffinity(0,narrow)
 try:
  assert not (blind/'FAIR_COLLECTION_COMPLETE.json').exists(),'Blind data already sealed; do not change its job directory'
  protocol=blind/'EVALUATION_PROTOCOL.json';digest=sha(protocol);spec=read(protocol);plan=read(queue/'PLAN.json')
  assert spec['code_commit']==COMMIT and spec['source']==str(ROOT/'source_cae1ab9e7f48')
  assert sum(len(c['method_order']) for c in spec['cases'])==320
  shards=plan['ordered_shards'];assert shards==list(range(32)) and plan['protocol_sha256']==digest
  assert {int(c['shard']) for c in spec['cases']}==set(shards)
  assert sha(Path(spec['frontend_selection_file']))==spec['frontend_selection_sha256']
  names=['catchit_fair_blind32_g1_gpu'+str(g)+'_blind_20261108b' for g in [0,1]]
  request=blind/'SCHEDULING_COST_V4_REQUEST.json'
  fresh(request,dict(schema='blind_cost_v4_block_checkpoint_v1',registered_utc=stamp(),pid=os.getpid(),process_start_ticks=start_ticks(),
   protocol_sha256=digest,actor_budget=320,original_manager_names=names,claim_status='deferred_for_cost_checkpoint',
   no_actor_termination=True,no_scientific_configuration_change=True))
  request_sha=sha(request);held=[]
  for shard in shards:
   path=queue/'claims'/('shard%03d.json'%shard)
   try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
   except FileExistsError:continue
   with os.fdopen(fd,'w') as f:json.dump(dict(shard=shard,status='deferred_for_cost_checkpoint',request_sha256=request_sha,created_utc=stamp()),f,indent=2)
   held.append(path)
  assert held,'All blocks were already claimed; this pre-completion checkpoint is no longer needed'
  update('WAITING_FOR_COMPLETE_BLOCKS',held_unexecuted_shards=len(held))
  while True:
   old=closed_original_workers(blind,names)
   if old is not None:break
   if time.monotonic()-begin>14400:raise TimeoutError('Original managers have not naturally closed; no process was killed')
   time.sleep(5)
  archive=blind/'scheduling_checkpoints/cost_v4/held_claims';archive.mkdir(parents=True,exist_ok=False);archived=[]
  for path in held:
   value=read(path);assert value['status']=='deferred_for_cost_checkpoint' and value['request_sha256']==request_sha
   target=archive/path.name;assert not target.exists();path.rename(target)
   archived.append(dict(shard=value['shard'],path=str(target),sha256=sha(target)))
  completed=[]
  for path in (queue/'claims').glob('shard*.json'):
   claim=read(path);assert claim['status']=='completed' and claim['exit_code']==0 and claim['protocol_sha256']==digest
   completed.append(dict(shard=claim['shard'],path=str(path),sha256=sha(path)))
  assert sha(protocol)==digest
  checkpoint=blind/'SCHEDULING_COST_V4_CHECKPOINT.json'
  fresh(checkpoint,dict(schema='blind_cost_v4_block_checkpoint_v1',registered_utc=stamp(),request_sha256=request_sha,
   protocol_sha256=digest,actor_budget=320,ordered_shards=shards,old_workers=old,completed_claims=completed,
   archived_unexecuted_claims=archived,no_actor_killed=True,no_scientific_configuration_changed=True,
   cost_output_root=str(ROOT/'cost_v4_matched'),cost_worker=str(ROOT/'tools/run_cost_v4_study.py'),
   replacement_worker_ids=['g1_gpu0_blind_costv4_resume1','g1_gpu1_blind_costv4_resume1']))
  checkpoint_sha=sha(checkpoint);template=read(blind/(names[0]+'.job.json'));newjobs={}
  # Register BOTH future workers before either resumes; collector cannot seal early.
  for gpu in [0,1]:
   worker='g1_gpu'+str(gpu)+'_blind_costv4_resume1';name='catchit_fair_blind32_'+worker+'_20261108b'
   job=dict(name=name,gpu_candidates=[gpu],command=['/usr/bin/python3','-u',str(queue/'fair_workqueue_worker.py'),
    '--protocol',str(protocol),'--worker-id',worker],env=template['env'],cwd=str(blind),timeout_s=86400,
    scheduling_checkpoint_sha256=checkpoint_sha)
   jobfile=blind/(name+'.job.json');fresh(jobfile,job);newjobs[gpu]=(jobfile,name)
  update('RESUME_GPU1',checkpoint_sha256=checkpoint_sha)
  first=manager(newjobs[1][0],blind,newjobs[1][1]);launches=[dict(gpu=1,worker=newjobs[1][1],manager_pid=first.pid,utc=stamp())]
  receipt=dict(status='SETUP_FAILED',started_utc=stamp(),scheduling_checkpoint_sha256=checkpoint_sha,blind_gpu1_running=True)
  coststage=ROOT/'cost_v4_gpu0_job';costname='catchit_fair_cost_v4_gpu0_checkpoint_20261108b';started=False;finished=False
  try:
   coststage.mkdir(exist_ok=False)
   job=dict(name=costname,gpu_candidates=[0],command=[PY,'-B','-u',str(ROOT/'tools/run_cost_v4_study.py'),
    '--root',str(ROOT/'cost_v4_matched')],env=template['env'],cwd=str(coststage),timeout_s=14400,
    scheduling_checkpoint_sha256=checkpoint_sha)
   jobfile=coststage/(costname+'.job.json');fresh(jobfile,job)
   update('COST_GPU0_GPU1_CONTINUES',checkpoint_sha256=checkpoint_sha)
   process=manager(jobfile,coststage,costname);started=True;receipt['manager_pid']=process.pid
   receipt['manager_returncode']=process.wait();finished=True
   resultpath=coststage/(costname+'.result.json');result=read(resultpath)
   receipt.update(status='PASS' if result['status']=='completed' and result.get('exit_code')==0 else 'FAILED',
    actual_job_result=result,result_path=str(resultpath),result_sha256=sha(resultpath))
  except BaseException as exc:receipt.update(status='FAILED',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
  finally:
   release_error=None
   try:
    if started:
     assert finished,'Actual cost manager is still running; do not overlap GPU0'
     release=released(coststage,costname,0,required=False);assert release is not None,'GPU0 release remains unconfirmed'
    else:release=dict(status='NO_COST_MANAGER_STARTED')
    receipt['resource_release']=release
   except BaseException as exc:release_error=exc;receipt['resource_release']=dict(status='UNCONFIRMED',error=str(exc))
   receipt['finished_utc']=stamp();fresh(ROOT/'BLIND_COST_V4_CHECKPOINT_COST_RECEIPT.json',receipt)
   if release_error is not None:raise release_error
  assert sha(protocol)==digest and sha(checkpoint)==checkpoint_sha
  second=manager(newjobs[0][0],blind,newjobs[0][1]);launches.append(dict(gpu=0,worker=newjobs[0][1],manager_pid=second.pid,utc=stamp()))
  fresh(ROOT/'BLIND_COST_V4_CHECKPOINT_RECEIPT.json',dict(status='HANDED_BACK',utc=stamp(),protocol_sha256=digest,
   scheduling_checkpoint_sha256=checkpoint_sha,cost_status=receipt['status'],launched_blind_workers=launches,
   note='Actual blind workers continue; the existing collector still requires all320 actors and all registered workers to close normally.'))
  update('HANDED_BACK',cost_status=receipt['status'],launched_blind_workers=launches)
 except BaseException as exc:update('STOPPED',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
