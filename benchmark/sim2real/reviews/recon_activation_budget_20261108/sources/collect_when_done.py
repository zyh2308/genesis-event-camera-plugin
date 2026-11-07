"""CPU-only, bounded collector for this one experiment."""
import datetime,hashlib,json,os,pathlib,subprocess,sys,time,traceback
here=pathlib.Path(__file__).parent
roots=[here.parent/'20261108-recon-activation-budget-eventgenesis-v3',here.parent/'20261108-recon-activation-budget-v2e-v4']
try:
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    deadline=time.monotonic()+2*3600+600
    while not all((r/'resource_release.json').exists() and (r/'completion.json').exists() for r in roots):
        for r in roots:
            if (r/'failure.json').exists():raise RuntimeError('Preserved study failure: '+str(r))
        if time.monotonic()>deadline:raise TimeoutError('Bounded diagnostic did not finish')
        time.sleep(10)
    with (here/'collect.log').open('x') as log:
        code=subprocess.run([sys.executable,str(here/'collect_results.py')],stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
    assert code==0,'Collector failed; preserve log and original completed studies'
    target=here.parent/'20261108-recon-activation-budget-summary.json'
    (here/'completion.json').write_text(json.dumps(dict(status='complete',exit_code=0,summary=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
except BaseException:
    (here/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
    raise
