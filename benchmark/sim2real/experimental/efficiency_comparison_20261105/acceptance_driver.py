"""Sequential GPU acceptance following the exclusive comparison job."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'02_matched_suite.done.json').read_text())['exit_code']==0
tools=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/runtime/esim_build_tools')
env=dict(os.environ,CUDA_HOME=str(tools),CXX=str(tools/'bin/x86_64-conda-linux-gnu-c++'),CC=str(tools/'bin/x86_64-conda-linux-gnu-cc'),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.0',OMP_NUM_THREADS='2',EVENTGENESIS_GPU_BUILD_DIR=str(ROOT/'release_build'))
env['PATH']=str(tools/'bin')+os.pathsep+env['PATH']
report=dict(passed=False,checks=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
commands=[['build_release.py'],['acceptance_scripts/test_sensor_v2.py'],['acceptance_scripts/test_holes_v2.py'],['acceptance_scripts/test_deferred_final.py'],['acceptance_scripts/test_actual_sensor_final.py'],['acceptance_scripts/test_process_resume_v2.py'],['release_integration.py']]
for i,args in enumerate(commands):
    name=Path(args[0]).stem;start=time.perf_counter()
    with (ROOT/('acceptance_'+name+'.log')).open('x') as log:
        value=subprocess.run([sys.executable,str(ROOT/args[0]),*args[1:]],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    report['checks'].append(dict(name=name,exit_code=value.returncode,wall_s=time.perf_counter()-start))
    (ROOT/'release_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ACCEPTANCE',name,value.returncode,flush=True)
    if value.returncode:raise SystemExit(value.returncode)
report['passed']=True
report['source_hashes']={str(x.relative_to(ROOT/'release_candidate_static')):hashlib.sha256(x.read_bytes()).hexdigest() for x in (ROOT/'release_candidate_static').rglob('*') if x.is_file() and '__pycache__' not in str(x)}
(ROOT/'release_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
