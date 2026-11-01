import os,sys,json,time,subprocess,traceback,fcntl
from pathlib import Path
R=Path(os.environ['EPOSE_REPAIR_RUN']);P=Path('/home/zhaoyuhan/Eventbased_WAM');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1');finish_started=False
def run(script,stem):
 with (R/f'{stem}.stdout').open('w') as a,(R/f'{stem}.stderr').open('w') as b:code=subprocess.run([sys.executable,str(R/'sources'/script)],env=env,stdout=a,stderr=b).returncode
 print(json.dumps({'phase':stem,'exit_code':code,'unix':time.time()}),flush=True);assert code==0,(stem,code)
try:
 pids=json.loads((R/'generation_workers.json').read_text())['pids']
 while any(Path(f'/proc/{p}').exists() and ') Z ' not in Path(f'/proc/{p}/stat').read_text() for p in pids):
  for j,p in enumerate(pids):
   if (not Path(f'/proc/{p}').exists() or ') Z ' in Path(f'/proc/{p}/stat').read_text()) and 'BANK_COMPLETE' not in (R/f'worker{j}.stdout').read_text():raise RuntimeError(f'worker{j} failed before bank complete')
  time.sleep(10)
 assert all('BANK_COMPLETE' in (R/f'worker{j}.stdout').read_text() for j in range(8))
 run('merge_diverse_bank.py','merge');run('v2e_diverse_bank.py','v2e_generation')
 sys.path.insert(0,str(R/'sources'));sys.path.insert(0,str(P/'runs/sim2real/20261101-epose-wood-short-research-v1/sources/benchmark/sim2real/scripts'));import numpy as np;import run_pose_stage2_pilot as pose
 from diverse_samples import collect_diverse
 reference=None;counts={}
 for arm in ['eventgenesis','v2e']:
  items=collect_diverse(pose,R,arm,'train')+collect_diverse(pose,R,arm,'validation');labels=[(s['sample_id'],s['label'].tolist(),s['event_start_s'],s['label_time_s']) for s in items];assert len(items)==256;counts[arm]=[s['event_count'] for s in items]
  if reference is None:reference=labels
  else:assert reference==labels
  assert all(np.isfinite(s['image']).all() and np.isfinite(s['label']).all() for s in items)
  del items
 (R/'bank_preflight.json').write_text(json.dumps({'passed':True,'paired_labels_equal':True,'train_windows':192,'validation_windows':64,'event_counts':counts,'no_real_data_access':True},indent=2))
 children=[]
 for arm in ['eventgenesis','v2e']:
  children.append(subprocess.Popen([sys.executable,str(R/'sources/train_diverse_pose.py'),arm],env=env,stdout=(R/f'{arm}_training.stdout').open('w'),stderr=(R/f'{arm}_training.stderr').open('w'),start_new_session=True))
 (R/'training_processes.json').write_text(json.dumps({'pids':[q.pid for q in children],'start_unix':time.time(),'device':'cpu','threads_per_arm':8},indent=2));print('PAIRED_TRAINING_STARTED',[q.pid for q in children],flush=True);finish_started=True;run('finish_diverse_pipeline.py','finish_pipeline')
except Exception:
 (R/'orchestration_error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
 if not finish_started:
  (R/'completion.json').write_text(json.dumps({'exit_code':1,'real_scoring_started':False,'finish_unix':time.time(),'scope':'bank preflight/orchestration failed before training'}))
  reg=P/'.gpu_reservations';lease=reg/(R.name+'.json')
  with (reg/'.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if lease.exists():
    v=json.loads(lease.read_text());v.update(status='released_cpu',finished_unix=time.time(),completion_exit_code=1);tmp=lease.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(lease);(R/'resource_reservation.json').write_text(json.dumps(v,indent=2))
   fcntl.flock(lock,fcntl.LOCK_UN)
 sys.exit(1)
print('PAIRED_PIPELINE_COMPLETE',flush=True)
