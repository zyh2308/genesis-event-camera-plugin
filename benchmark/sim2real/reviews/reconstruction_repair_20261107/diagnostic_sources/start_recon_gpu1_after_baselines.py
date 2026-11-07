"""Wait without a GPU; add a formal worker only after the baseline short checks release GPU1."""
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.partial');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);args=p.parse_args()
    F=args.root.resolve();R=args.campaign.resolve();baseline=R.parent/'20261107-recon-sigmoid-baselines-v1'
    assert F==R.parent/'20261107-recon-sigmoid-formal-v1'
    save(F/'gpu1_waiting_baselines.json',dict(status='waiting',GPU_reserved=False,upstream=str(baseline),utc=now()))
    deadline=time.monotonic()+4*3600
    while not (baseline/'resource_release.json').exists():
        if (baseline/'failure.json').exists():raise RuntimeError('Baseline short study failed')
        if time.monotonic()>deadline:raise TimeoutError('Baseline study did not release')
        time.sleep(10)
    assert json.loads((baseline/'completion.json').read_text())['status']=='complete'
    for name,digest in json.loads((baseline/'sources/source_manifest.json').read_text()).items():
        assert hashlib.sha256((baseline/'sources'/name).read_bytes()).hexdigest()==digest
    gate=json.loads((baseline/'baseline_gate.json').read_text())
    if not gate['passed']:
        with (F/'.queue.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);q=json.loads((F/'queue_state.json').read_text())
            q['halt_reason']='Supplemental baseline synthetic stability check failed; no new formal job may start'
            save(F/'queue_state.json',q)
        raise RuntimeError('Baseline synthetic stability check did not pass')
    command=[sys.executable,'-u',str(F/'sources/run_recon_repair_queue.py'),'--root',str(F),'--campaign',str(R),
        '--gpu-index','1','--gpu-uuid','GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460']
    with (F/'gpu1_launcher.log').open('x') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    save(F/'gpu1_submission.json',dict(pid=child.pid,argv=command,utc=now(),baseline_checks_passed=True,
        baseline_gate_sha256=hashlib.sha256((baseline/'baseline_gate.json').read_bytes()).hexdigest(),
        next='worker still requires the full primary synthetic gate and a fresh atomic GPU reservation'))
    print('GPU1_FORMAL_WORKER_SUBMITTED',child.pid,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1]);save(root/'gpu1_waiter_failure.json',dict(error=traceback.format_exc(),utc=now()))
        raise
