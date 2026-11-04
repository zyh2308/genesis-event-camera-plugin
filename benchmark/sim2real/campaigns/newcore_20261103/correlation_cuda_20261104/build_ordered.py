import os
from pathlib import Path
root=Path(__file__).resolve().parent;tools=root.parent/'runtime/esim_build_tools'
os.environ.update(CUDA_HOME=str(tools),CXX=str(tools/'bin/x86_64-conda-linux-gnu-c++'),CC=str(tools/'bin/x86_64-conda-linux-gnu-cc'),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.0;8.6',OMP_NUM_THREADS='2')
os.environ['PATH']=str(tools/'bin')+os.pathsep+os.environ['PATH']
from torch.utils.cpp_extension import load
build=root/'build_ordered';build.mkdir(exist_ok=True)
m=load(name='ordered_correlation_backward',sources=[str(root/'ordered_backward.cpp'),str(root/'ordered_backward.cu')],extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','--fmad=false'],extra_include_paths=[str(tools/'targets/x86_64-linux/include')],extra_ldflags=['-L'+str(tools/'targets/x86_64-linux/lib')],build_directory=str(build),verbose=True)
print('ORDERED_BUILD_OK',m.__file__,flush=True)
