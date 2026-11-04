"""Final controls and an independent rerun against the integrated static package."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'release_acceptance.json').read_text())['passed']
tools=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/runtime/esim_build_tools')
env=dict(os.environ,CUDA_HOME=str(tools),CXX=str(tools/'bin/x86_64-conda-linux-gnu-c++'),CC=str(tools/'bin/x86_64-conda-linux-gnu-cc'),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.0',EVENTGENESIS_GPU_BUILD_DIR=str(ROOT/'release_build'))
env['PATH']=str(tools/'bin')+os.pathsep+env['PATH']
report=dict(passed=False,jobs=[])
commands=[['controls.py'],['genesis_smoke.py'],['compare_static.py','--mode','suite','--intervals','20','--trials','3','--out',str(ROOT/'matched_suite_static')],['analyze_comparison.py']]
for args in commands:
    name=Path(args[0]).stem;start=time.perf_counter()
    with (ROOT/('post_'+name+'.log')).open('x') as log:r=subprocess.run([sys.executable,str(ROOT/args[0]),*args[1:]],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=5400)
    report['jobs'].append(dict(name=name,exit_code=r.returncode,wall_s=time.perf_counter()-start))
    (ROOT/'post_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    if r.returncode:raise SystemExit(r.returncode)
report['passed']=True;(ROOT/'post_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
