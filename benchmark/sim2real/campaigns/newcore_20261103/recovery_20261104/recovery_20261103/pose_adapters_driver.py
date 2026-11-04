import subprocess,sys,os
from pathlib import Path
root=Path.cwd(); fix=root/'recovery_20261103'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
for method in ('eventgenesis','v2e','film_esim','rife_esim'):
    with (fix/f'pose_adapter_{method}.log').open('xb') as log:
        r=subprocess.run([sys.executable,'-u',str(fix/'build_pose_adapter.py'),'--campaign',str(root),'--kind','pose','--method',method],env=env,stdout=log,stderr=subprocess.STDOUT)
    print(method,r.returncode,flush=True)
    r.check_returncode()
