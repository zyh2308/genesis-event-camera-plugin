"""Finish and verify fixed paired training before opening the registered real test."""
import os,sys,json,time,subprocess,hashlib,traceback,fcntl
from pathlib import Path
R=Path(os.environ['EPOSE_REPAIR_RUN']);P=Path('/home/zhaoyuhan/Eventbased_WAM');OLD=P/'runs/sim2real/20261101-epose-wood-short-research-v1';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1')
def run(script,stem):
 with (R/f'{stem}.stdout').open('w') as a,(R/f'{stem}.stderr').open('w') as b:code=subprocess.run([sys.executable,str(R/'sources'/script)],env=env,stdout=a,stderr=b).returncode
 print(json.dumps({'phase':stem,'exit_code':code}),flush=True);assert code==0,(stem,code)
code=1;real=False
try:
 pids=json.loads((R/'training_processes.json').read_text())['pids']
 while any(Path(f'/proc/{pid}').exists() and ') Z ' not in Path(f'/proc/{pid}/stat').read_text() for pid in pids):time.sleep(10)
 assert all((R/f'training/{a}_result.json').exists() for a in ['eventgenesis','v2e']),'training process stopped without completed result'
 core=json.loads((OLD/'frozen_core_manifest.json').read_text());assert all(hashlib.sha256((OLD/'sources'/name).read_bytes()).hexdigest()==digest for name,digest in core['files'].items());(R/'frozen_core_verification.json').write_text(json.dumps({'passed':True,'files':len(core['files']),'frozen_commit':core['frozen_core']},indent=2))
 run('verify_diverse_pose.py','independent_synthetic_verification_run')
 gate=json.loads((R/'synthetic_gate.json').read_text());eligible=[a for a,v in gate['arms'].items() if v['passed']];fallback=json.loads((R/'protocol.json').read_text()).get('deadline_fallback_passed_arms_only',False)
 if not gate['passed'] and not (fallback and eligible):code=2;raise RuntimeError('no eligible localization-qualified real evaluation; fresh real evaluation remains closed')
 real=True;run('evaluate_diverse_epose.py','real_evaluation');run('verify_full_real.py','independent_real_verification_run');assert json.loads((R/'real_eval/independent_verification.json').read_text())['passed'];code=0
except Exception as e:
 (R/'completion_error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
finally:
 receipt={'exit_code':code,'real_scoring_started':real,'finish_unix':time.time(),'run':str(R)};(R/'completion.json').write_text(json.dumps(receipt,indent=2))
 reg=P/'.gpu_reservations';lease=reg/(R.name+'.json')
 with (reg/'.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if lease.exists():
   v=json.loads(lease.read_text());v.update(status='released_cpu',finished_unix=time.time(),completion_exit_code=code);tmp=lease.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(lease);(R/'resource_reservation.json').write_text(json.dumps(v,indent=2))
  fcntl.flock(lock,fcntl.LOCK_UN)
 print(json.dumps(receipt),flush=True)
sys.exit(code)
