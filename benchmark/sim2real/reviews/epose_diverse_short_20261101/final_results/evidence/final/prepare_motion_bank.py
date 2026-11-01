import os,sys,json,time,subprocess,traceback
from pathlib import Path
R=Path(os.environ['EPOSE_REPAIR_RUN'])
try:
 pids=json.loads((R/'generation_workers.json').read_text())['pids']
 while any(Path(f'/proc/{p}').exists() and ') Z ' not in Path(f'/proc/{p}/stat').read_text() for p in pids):time.sleep(5)
 assert all('BANK_COMPLETE' in (R/f'worker{j}.stdout').read_text() for j in range(8))
 for script,stem in [('merge_diverse_bank.py','merge'),('v2e_diverse_bank.py','v2e_generation'),('audit_motion_observability.py','observability')]:
  with (R/f'{stem}.stdout').open('w') as a,(R/f'{stem}.stderr').open('w') as b:code=subprocess.run([sys.executable,str(R/'sources'/script)],stdout=a,stderr=b).returncode
  print(stem,code,flush=True);assert code==0
 (R/'bank_ready_for_visual_review.json').write_text(json.dumps({'passed':True,'unix':time.time(),'no_training_launched':True}))
except Exception:(R/'generation_controller_error.txt').write_text(traceback.format_exc());raise
