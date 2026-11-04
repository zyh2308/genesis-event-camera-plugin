"""Load a private extension, built without fast math or fused multiply-add."""
import os,threading
from pathlib import Path
import torch
_kernel=None
_lock=threading.Lock()
def load_kernel():
    global _kernel
    with _lock:
        if _kernel is None:
            from torch.utils.cpp_extension import load
            source=Path(__file__).resolve().parent/'csrc'
            build=os.environ.get('EVENTGENESIS_GPU_BUILD_DIR')
            if build:Path(build).mkdir(parents=True,exist_ok=True)
            toolkit=Path(os.environ['CUDA_HOME']) if os.environ.get('CUDA_HOME') else None
            includes=[str(toolkit/'targets/x86_64-linux/include')] if toolkit and (toolkit/'targets/x86_64-linux/include').exists() else []
            links=['-L'+str(toolkit/'targets/x86_64-linux/lib')] if includes else []
            _kernel=load(name='eventgenesis_precision_sensor',sources=[str(source/'fused_sensor.cpp'),str(source/'fused_sensor.cu')],
                extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','--fmad=false','-std=c++17'],
                extra_include_paths=includes,extra_ldflags=links,build_directory=build,verbose=False)
        return _kernel
