import os,sys,time,json,subprocess
from pathlib import Path
R=Path(os.environ['EPOSE_REPAIR_RUN'])
while True:
 try:
  ready=all(json.loads((R/f'training/{a}_checkpoint_progress.json').read_text())['completed_update']>=100 for a in ['eventgenesis','v2e'])
 except (OSError,ValueError,KeyError):ready=False
 if ready:break
 time.sleep(5)
with (R/'early_checkpoint_audit.stdout').open('w') as a,(R/'early_checkpoint_audit.stderr').open('w') as b:code=subprocess.run([sys.executable,str(R/'audit_periodic_pose_checkpoint.py')],stdout=a,stderr=b).returncode
assert code==0
