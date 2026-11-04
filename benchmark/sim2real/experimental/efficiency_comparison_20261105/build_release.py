"""Private reproducible build; never installs into the shared environment."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
tools=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/runtime/esim_build_tools')
os.environ.update(CUDA_HOME=str(tools),CXX=str(tools/'bin/x86_64-conda-linux-gnu-c++'),CC=str(tools/'bin/x86_64-conda-linux-gnu-cc'),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.0',OMP_NUM_THREADS='2',EVENTGENESIS_GPU_BUILD_DIR=str(ROOT/'release_build'))
os.environ['PATH']=str(tools/'bin')+os.pathsep+os.environ['PATH']
sys.path.insert(0,str(ROOT/'release_candidate_static'))
from genesis_event_plugin.gpu_kernel import load_kernel
kernel=load_kernel()
print('BUILD_OK',kernel.__file__,flush=True)
