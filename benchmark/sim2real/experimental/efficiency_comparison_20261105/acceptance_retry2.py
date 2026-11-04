"""Fix profile numeric-type equality; rerun actual fast-path and integration gates."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
old=json.loads((ROOT/'release_acceptance_retry1.json').read_text())
assert len(old['checks'])==7 and all(x['exit_code']==0 for x in old['checks'][:6])
previous_hashes=json.loads((ROOT/'05_static_acceptance_retry1.done.json').read_text())['job']['sha256']
for rel in ['gpu_sensor.py','gpu_interpolator.py','gpu_geometry.py','gpu_kernel.py']:
    path=ROOT/'release_candidate_static/genesis_event_plugin'/rel
    assert hashlib.sha256(path.read_bytes()).hexdigest()==previous_hashes[str(path)]
tools=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/runtime/esim_build_tools')
env=dict(os.environ,CUDA_HOME=str(tools),CXX=str(tools/'bin/x86_64-conda-linux-gnu-c++'),CC=str(tools/'bin/x86_64-conda-linux-gnu-cc'),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.0',EVENTGENESIS_GPU_BUILD_DIR=str(ROOT/'release_build'))
env['PATH']=str(tools/'bin')+os.pathsep+env['PATH']
report=dict(passed=False,reused_unchanged_component_checks=old['checks'][:4],checks=[],prior_failed_attempts_preserved=True)
for name in ['acceptance_scripts/test_actual_sensor_active.py','release_integration_retry2.py']:
    start=time.perf_counter()
    with (ROOT/('retry2_'+Path(name).stem+'.log')).open('x') as log:r=subprocess.run([sys.executable,str(ROOT/name)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    report['checks'].append(dict(name=name,exit_code=r.returncode,wall_s=time.perf_counter()-start))
    (ROOT/'release_acceptance_final.json').write_text(json.dumps(report,indent=2)+'\n')
    if r.returncode:raise SystemExit(r.returncode)
report['passed']=True
report['source_hashes']={str(x.relative_to(ROOT/'release_candidate_static')):hashlib.sha256(x.read_bytes()).hexdigest() for x in (ROOT/'release_candidate_static').rglob('*') if x.is_file() and '__pycache__' not in str(x)}
(ROOT/'release_acceptance_final.json').write_text(json.dumps(report,indent=2)+'\n')
