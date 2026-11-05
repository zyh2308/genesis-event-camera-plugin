import datetime,json,os,signal,subprocess,sys
from pathlib import Path
T=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261106-pose-synthetic-diagnostic-v1');R=T.parent/'20261103-stage9-direct-v1';child=None
lease=int(os.environ['GPU_LEASE_FD'])
def stop(sig,frame):
 if child is not None and child.poll() is None:child.terminate();child.wait()
 raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
 for mode in ['baseline_ce','quality_confidence']:
  out=T/('fullbank500_'+mode);assert not out.exists()
  with (T/(mode+'_pilot.stdout.log')).open('xb',buffering=0) as log:
   child=subprocess.Popen([sys.executable,'-u',str(T/'pose_confidence_pilot.py'),'--campaign',str(R),'--out',str(out),'--fit-updates','500','--loss-mode',mode],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,pass_fds=(lease,));rc=child.wait();assert rc==0,(mode,rc)
  print('PILOT_MODE_COMPLETE',mode,flush=True)
 (T/'confidence_pilot_completion.json').write_text(json.dumps(dict(status='complete',runs=4,updates_per_run=500,real_scoring=False),indent=2))
finally:
 if child is not None and child.poll() is None:child.terminate();child.wait()
