"""Existing Mesa runtime, valid Genesis options and established HDR stress setup."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'final_metrology.json').read_text())['passed']
tools=ROOT.parent/'sim2real/20261103-stage9-direct-v1/runtime/esim_build_tools'
mesa=ROOT.parent/'sim2real/20261101-epose-diverse-bank-v8/runtime/osmesa/usr/lib/x86_64-linux-gnu'
env=dict(os.environ,CUDA_HOME=str(tools),CXX=str(tools/'bin/x86_64-conda-linux-gnu-c++'),CC=str(tools/'bin/x86_64-conda-linux-gnu-cc'),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.0',EVENTGENESIS_GPU_BUILD_DIR=str(ROOT/'release_build'),
    PYOPENGL_PLATFORM='osmesa',GALLIUM_DRIVER='llvmpipe',LIBGL_ALWAYS_SOFTWARE='1',MESA_GL_VERSION_OVERRIDE='4.5',LD_LIBRARY_PATH=str(mesa),LP_NUM_THREADS='1',QD_NUM_THREADS='2',QD_OFFLINE_CACHE='1',
    GS_CACHE_FILE_PATH=str(ROOT/'accepted_render_cache/genesis'),QD_OFFLINE_CACHE_FILE_PATH=str(ROOT/'accepted_render_cache/quad'))
env['PATH']=str(tools/'bin')+os.pathsep+env['PATH'];begin=time.perf_counter()
with (ROOT/'native_accepted.log').open('x') as log:r=subprocess.run([sys.executable,str(ROOT/'genesis_smoke_accepted.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
(ROOT/'native_acceptance.json').write_text(json.dumps(dict(passed=r.returncode==0,exit_code=r.returncode,wall_s=time.perf_counter()-begin,
    physics_render_backend='same CPU + private OSMesa as the formal physical bank',event_backend='g1 GPU2; static full-profile entry',renderer_sources_modified=False),indent=2)+'\n')
raise SystemExit(r.returncode)
