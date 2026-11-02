"""Install an isolated compatible SavedModel inference runtime."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(sys.argv[1]);prefix=R/'runtime/film';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PIP_CACHE_DIR=str(R/'runtime/pip_cache'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
subprocess.run([sys.executable,'-m','venv',str(prefix)],check=True)
cmd=[str(prefix/'bin/python'),'-m','pip','install','tensorflow[and-cuda]==2.15.1','tensorflow-addons==0.23.0','numpy<2']
started=time.monotonic()
with (R/'runtime/film_install.log').open('wb') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
(R/'runtime/film_install.json').write_text(json.dumps({'exit_code':result.returncode,'prefix':str(prefix),'command':cmd,'elapsed_s':time.monotonic()-started,'shared_env_modified':False},indent=2)+'\n');raise SystemExit(result.returncode)
