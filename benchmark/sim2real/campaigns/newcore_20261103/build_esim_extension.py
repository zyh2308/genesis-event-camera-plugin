"""Compile an isolated copy of official ESIM, with API-only Torch compatibility."""
import difflib,hashlib,json,os,sys
from pathlib import Path
R=Path(sys.argv[1]);tools=R/'runtime/esim_build_tools';src=R/'baseline_sources/vid2e_ssm/esim_torch/esim_cuda_kernel.cu';out=R/'runtime/esim_extension';out.mkdir(parents=True,exist_ok=True)
os.environ.update(CUDA_HOME=str(tools),CXX=str(tools/'bin/x86_64-conda-linux-gnu-c++'),CC=str(tools/'bin/x86_64-conda-linux-gnu-cc'),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.0;7.5',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
os.environ['PATH']=str(tools/'bin')+os.pathsep+os.environ['PATH']
assert (tools/'bin/nvcc').exists()
original=src.read_text();updated=original.replace('.type().is_cuda()','.is_cuda()').replace('.data<','.data_ptr<');copy=out/'esim_cuda_kernel.cu';copy.write_text(updated)
(out/'torch28_api_only.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile='official',tofile='private_compatibility_copy')))
import torch
from torch.utils.cpp_extension import load
targets=tools/'targets/x86_64-linux';module=load(name='esim_cuda',sources=[str(copy)],extra_include_paths=[str(targets/'include')],extra_ldflags=['-L'+str(targets/'lib')],build_directory=str(out),verbose=True)
rows={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*') if p.is_file()}
(out/'build_receipt.json').write_text(json.dumps({'passed':True,'module_path':str(Path(module.__file__).resolve()),'torch_version':torch.__version__,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'compatibility_source_sha256':hashlib.sha256(copy.read_bytes()).hexdigest(),'changes':'Tensor CUDA validation and data_ptr API names only; unchanged event kernels','files':rows,'core_simulator_modified':False},indent=2)+'\n')
