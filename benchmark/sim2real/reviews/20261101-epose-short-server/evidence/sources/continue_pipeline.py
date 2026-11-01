import os,sys,json,subprocess,time,hashlib,socket,fcntl,signal
from pathlib import Path
from datetime import datetime,timezone
R=Path(os.environ['EPOSE_SHORT_RUN']);P=Path('/home/zhaoyuhan/Eventbased_WAM');S=R/'sources'
def now():return datetime.now(timezone.utc).isoformat()
state={'started_utc':now(),'host':socket.gethostname(),'run':str(R),'phases':{}};code=1
try:
 deadline=time.monotonic()+600
 while not (R/'generation_exit_code.txt').exists():
  if time.monotonic()>deadline:raise TimeoutError('generator did not finish within registered deadline')
  time.sleep(5)
 if (R/'generation_exit_code.txt').read_text().strip()!='0':raise RuntimeError('generation failed; inspect saved logs')
 for name,script,limit in [('v2e','run_v2e_bank.py',1200),('training','train_matched_pair.py',1200),('real_eval','evaluate_real_epose.py',300)]:
  record={'started_utc':now(),'script':str(S/script),'script_sha256':hashlib.sha256((S/script).read_bytes()).hexdigest(),'timeout_s':limit};state['phases'][name]=record;(R/'pipeline_status.json').write_text(json.dumps(state,indent=2))
  with (R/f'{name}.stdout.log').open('w') as a,(R/f'{name}.stderr.log').open('w') as b:
   proc=subprocess.Popen([sys.executable,'-u',str(S/script)],stdout=a,stderr=b,start_new_session=True)
   try:rc=proc.wait(timeout=limit)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise
  record.update({'exit_code':rc,'finished_utc':now()});(R/f'{name}_exit_code.txt').write_text(str(rc));(R/'pipeline_status.json').write_text(json.dumps(state,indent=2))
  if rc:raise RuntimeError(f'{name} exited {rc}')
 code=0;state['status']='completed'
except Exception as e:state.update({'status':'failed','error':repr(e)})
finally:
 state['finished_utc']=now();state['exit_code']=code;(R/'pipeline_status.json').write_text(json.dumps(state,indent=2));(R/'pipeline_exit_code.txt').write_text(str(code))
 paths=json.loads((R/'reservation_paths.json').read_text());reg=Path(paths['registry']);lock=reg.parent/'.gpu_reservations.lock'
 with lock.open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);d=json.loads(reg.read_text());assert d['run_id']==R.name;d.update({'status':'released_complete' if code==0 else 'released_failed','ended_at_utc':now(),'pipeline_exit_code':code});reg.write_text(json.dumps(d,indent=2));(R/'resource_reservation_finished.json').write_text(json.dumps(d,indent=2));legacy=Path(paths['legacy_lock']);owner=legacy/'resource_reservation.json'
  if owner.is_file() and json.loads(owner.read_text()).get('run_id')==R.name:owner.unlink();legacy.rmdir()
 print(json.dumps(state),flush=True)
sys.exit(code)
