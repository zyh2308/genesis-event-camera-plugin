"""Reuse completed controls; fix only private render-runtime linkage, then final suite."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'release_acceptance_final.json').read_text())['passed']
prior=json.loads((ROOT/'post_acceptance_final.json').read_text())
assert next(x for x in prior['jobs'] if x['name']=='controls')['exit_code']==0
tools=ROOT.parent/'sim2real/20261103-stage9-direct-v1/runtime/esim_build_tools'
mesa=ROOT.parent/'sim2real/20261101-epose-diverse-bank-v8/runtime/osmesa/usr/lib/x86_64-linux-gnu'
assert mesa.exists()
env=dict(os.environ,CUDA_HOME=str(tools),CXX=str(tools/'bin/x86_64-conda-linux-gnu-c++'),CC=str(tools/'bin/x86_64-conda-linux-gnu-cc'),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.0',EVENTGENESIS_GPU_BUILD_DIR=str(ROOT/'release_build'),
    PYOPENGL_PLATFORM='osmesa',GALLIUM_DRIVER='llvmpipe',LIBGL_ALWAYS_SOFTWARE='1',MESA_GL_VERSION_OVERRIDE='4.5',LD_LIBRARY_PATH=str(mesa),LP_NUM_THREADS='1',QD_NUM_THREADS='2',QD_OFFLINE_CACHE='1',
    GS_CACHE_FILE_PATH=str(ROOT/'render_cache/genesis'),QD_OFFLINE_CACHE_FILE_PATH=str(ROOT/'render_cache/quad'))
env['PATH']=str(tools/'bin')+os.pathsep+env['PATH']
report=dict(passed=False,completed_controls_receipt='post_acceptance_final.json',jobs=[],render_runtime='unchanged private OSMesa used by formal physical-bank generation; native CPU physics/render, CUDA event generation')
commands=[['genesis_smoke_verified.py'],['compare_static.py','--mode','suite','--intervals','20','--trials','3','--out',str(ROOT/'matched_suite_static')],['analyze_comparison.py']]
for args in commands:
    name=Path(args[0]).stem;start=time.perf_counter()
    with (ROOT/('verified_'+name+'.log')).open('x') as log:r=subprocess.run([sys.executable,str(ROOT/args[0]),*args[1:]],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=5400)
    report['jobs'].append(dict(name=name,exit_code=r.returncode,wall_s=time.perf_counter()-start))
    (ROOT/'post_acceptance_verified.json').write_text(json.dumps(report,indent=2)+'\n')
    if r.returncode:raise SystemExit(r.returncode)
report['passed']=True;(ROOT/'post_acceptance_verified.json').write_text(json.dumps(report,indent=2)+'\n')
