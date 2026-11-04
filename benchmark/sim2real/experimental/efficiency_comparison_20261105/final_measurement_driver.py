"""Final repeated metrology is independent of renderer smoke setup."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'release_acceptance_final.json').read_text())['passed']
tools=ROOT.parent/'sim2real/20261103-stage9-direct-v1/runtime/esim_build_tools'
env=dict(os.environ,CUDA_HOME=str(tools),CXX=str(tools/'bin/x86_64-conda-linux-gnu-c++'),CC=str(tools/'bin/x86_64-conda-linux-gnu-cc'),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.0',EVENTGENESIS_GPU_BUILD_DIR=str(ROOT/'release_build'))
env['PATH']=str(tools/'bin')+os.pathsep+env['PATH'];jobs=[]
for args in [['compare_static.py','--mode','suite','--intervals','20','--trials','3','--out',str(ROOT/'matched_suite_static')],['analyze_comparison.py']]:
    name=Path(args[0]).stem;start=time.perf_counter()
    with (ROOT/('measurement_'+name+'.log')).open('x') as log:r=subprocess.run([sys.executable,str(ROOT/args[0]),*args[1:]],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=5400)
    jobs.append(dict(name=name,exit_code=r.returncode,wall_s=time.perf_counter()-start))
    (ROOT/'final_metrology.json').write_text(json.dumps(dict(passed=False,jobs=jobs),indent=2)+'\n')
    if r.returncode:raise SystemExit(r.returncode)
(ROOT/'final_metrology.json').write_text(json.dumps(dict(passed=True,jobs=jobs),indent=2)+'\n')
