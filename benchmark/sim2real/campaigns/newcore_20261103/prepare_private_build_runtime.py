"""Isolated ESIM compiler toolchain; never install into the shared Torch env."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(sys.argv[1]);root=R/'runtime';root.mkdir(exist_ok=True);prefix=root/'esim_build_tools';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CONDA_PKGS_DIRS=str(root/'conda_pkgs'),CONDA_ENVS_PATH=str(root/'conda_envs'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
cmd=['/home/zhaoyuhan/miniconda3/bin/conda','create','--prefix',str(prefix),'--override-channels','-c','nvidia','-c','conda-forge','cuda-nvcc=12.6','cuda-cudart-dev=12.6','gxx_linux-64=12','ninja','-y']
started=time.monotonic()
with (root/'esim_build_tools_install.log').open('wb') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'esim_build_tools_install.json').write_text(json.dumps({'exit_code':result.returncode,'prefix':str(prefix),'command':cmd,'elapsed_s':time.monotonic()-started,'shared_env_modified':False},indent=2)+'\n')
raise SystemExit(result.returncode)
