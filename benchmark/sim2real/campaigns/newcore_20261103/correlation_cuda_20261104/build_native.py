"""Build only in a private directory; never install into the shared environment."""
import os
from pathlib import Path
root = Path(__file__).resolve().parent
campaign = root.parent
tools = campaign / 'runtime/esim_build_tools'
os.environ.update(CUDA_HOME=str(tools), CXX=str(tools/'bin/x86_64-conda-linux-gnu-c++'),
                  CC=str(tools/'bin/x86_64-conda-linux-gnu-cc'), MAX_JOBS='2',
                  TORCH_CUDA_ARCH_LIST='7.0;8.6', OMP_NUM_THREADS='2')
os.environ['PATH'] = str(tools/'bin') + os.pathsep + os.environ['PATH']
from torch.utils.cpp_extension import load
build = root/'build'; build.mkdir(exist_ok=True)
module = load(name='spatial_correlation_sampler_backend',
    sources=[str(root/'upstream/Correlation_Module'/f) for f in
             ['correlation.cpp','correlation_sampler.cpp','correlation_cuda_kernel.cu']],
    # CPU OpenMP pragmas precede modern AT_DISPATCH lambdas and fail GCC 12;
    # this CUDA-only build leaves those unused CPU loops serial.
    extra_cflags=['-O3','-std=c++17','-DUSE_CUDA'], extra_cuda_cflags=['-O3'],
    extra_include_paths=[str(tools/'targets/x86_64-linux/include')],
    extra_ldflags=['-lgomp','-L'+str(tools/'targets/x86_64-linux/lib')],
    build_directory=str(build), verbose=True)
print('BUILD_OK', module.__file__, flush=True)
