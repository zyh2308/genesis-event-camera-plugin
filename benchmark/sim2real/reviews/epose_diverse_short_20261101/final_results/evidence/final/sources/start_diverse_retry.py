import os,sys,json,time,subprocess
from pathlib import Path
R=Path(os.environ['EPOSE_REPAIR_RUN']);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1')
assert json.loads((R/'bank_preflight.json').read_text())['passed']
children=[]
for arm in ['eventgenesis','v2e']:
 children.append(subprocess.Popen([sys.executable,str(R/'sources/train_diverse_pose.py'),arm],env=env,stdout=(R/f'{arm}_training.stdout').open('w'),stderr=(R/f'{arm}_training.stderr').open('w'),start_new_session=True))
(R/'training_processes.json').write_text(json.dumps({'pids':[p.pid for p in children],'start_unix':time.time(),'device':'cpu','threads_per_arm':8},indent=2));print('PAIRED_RETRY_STARTED',[p.pid for p in children],flush=True)
with (R/'finish_pipeline.stdout').open('w') as a,(R/'finish_pipeline.stderr').open('w') as b:code=subprocess.run([sys.executable,str(R/'sources/finish_diverse_pipeline.py')],env=env,stdout=a,stderr=b).returncode
print('FINISH_PIPELINE_EXIT',code,flush=True);sys.exit(code)
