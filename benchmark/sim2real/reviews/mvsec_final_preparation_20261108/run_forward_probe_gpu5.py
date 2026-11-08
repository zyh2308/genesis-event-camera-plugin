"""Own one bounded GPU5 lease for synthetic numerical checks; never real data."""
import datetime,fcntl,hashlib,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path

P=Path('/home/zhaoyuhan/Eventbased_WAM');RUNS=P/'runs/sim2real'
ROOT=RUNS/'20261108-mvsec-forward-probe-gpu5-v1'
PREP=RUNS/'20261108-mvsec-final-preparation-v1'
CAMPAIGN=RUNS/'20261103-stage9-direct-v1'
UUID='GPU-06ee62ba-c171-40d4-b4ae-ea63391dc885'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def stop(child):
    if child is None or child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=20)
def interrupted(signum,frame):raise SystemExit('Own probe interrupted '+str(signum))

def main():
    assert socket.gethostname()=='g1' and ROOT.resolve()==ROOT and PREP.resolve()==PREP
    assert ROOT.is_dir() and not (ROOT/'probe.log').exists() and not (ROOT/'failure.json').exists()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    state=None;child=None;gpu_lock=None;record=P/'.gpu_reservations'/(ROOT.name+'_gpu5.json')
    try:
        binding=json.loads((ROOT/'worker_binding.json').read_text())
        worker=ROOT/'probe_synthetic_forward_equivalence.py';assert sha(worker)==binding['worker_sha256']
        ready=json.loads((PREP/'cpu_preparation/PREPARATION_COMPLETE_NO_SCORING.json').read_text())
        manifest=PREP/'cpu_preparation/matrix_preparation/manifest_candidate.json'
        assert sha(manifest)==ready['manifest_sha256']
        adapter=PREP/'source_snapshot/scripts/run_mvsec_candidate_inference.py'
        compat=PREP/'source_snapshot/compat/steflow_align_corners.py'
        reference=CAMPAIGN/'sources/benchmark/sim2real/scripts/run_steflow_stage2_pilot.py'
        assert reference.is_file()
        snapshot=json.loads((PREP/'deployment_manifest.json').read_text())
        for name in [adapter,compat]:
            item=next(x for x in snapshot['files'] if x['path']==str(name.relative_to(PREP)))
            assert sha(name)==item['sha256']
        observed=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,name','--format=csv,noheader'],text=True,timeout=15)
        assert UUID in observed and 'TITAN RTX' in observed
        registry=P/'.gpu_reservations';gpu_lock=(registry/('.gpu-'+UUID+'.lock')).open('a+')
        with (registry/'.gpu_reservations.lock').open('a+') as global_lock:
            fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=15)
            assert UUID not in apps
            for f in registry.glob('*.json'):
                old=json.loads(f.read_text())
                if old.get('gpu_uuid')==UUID:assert str(old.get('status','')).startswith(('released','complete')),str(f)
            assert not record.exists()
            state=dict(owner='sim-to-real research chat',run_id=ROOT.name,pid=os.getpid(),gpu_index=5,gpu_uuid=UUID,
                output_path=str(ROOT),status='reserved',started_utc=now(),purpose='synthetic-only complete-forward numerical preflight; max600s; no optimizer or real scoring')
            save(record,state)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NVIDIA_TF32_OVERRIDE='0',PYTHONDONTWRITEBYTECODE='1')
        cmd=[sys.executable,'-u',str(worker),'--manifest',str(manifest),'--manifest-sha256',sha(manifest),
             '--reference-source',str(reference),'--reference-sha256',sha(reference),
             '--adapter-source',str(adapter),'--adapter-sha256',sha(adapter),
             '--compat-source',str(compat),'--compat-sha256',sha(compat),
             '--model-root',str(P/'external/sources/downstream/STE-FlowNet'),'--out',str(ROOT/'numerical_preflight.json')]
        save(ROOT/'invocation.json',dict(argv=cmd,worker_sha256=sha(worker),started_utc=now(),real_data_loaded=False))
        with (ROOT/'probe.log').open('x') as log:
            child=subprocess.Popen(cmd,env=env,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            state.update(status='running',child_pid=child.pid);save(record,state)
            try:rc=child.wait(timeout=600)
            except subprocess.TimeoutExpired:stop(child);raise TimeoutError('Own synthetic probe exceeded600s')
        save(ROOT/'worker_exit.json',dict(exit_code=rc,finished_utc=now()))
        assert rc==0
        result=json.loads((ROOT/'numerical_preflight.json').read_text())
        assert result['status']=='passed' and len(result['checkpoints'])==15
        save(ROOT/'completion.json',dict(status='complete',numerical_preflight_sha256=sha(ROOT/'numerical_preflight.json'),real_scores_computed=False,finished_utc=now()))
    except BaseException:
        save(ROOT/'failure.json',dict(status='failed',error=traceback.format_exc(),utc=now()));raise
    finally:
        stop(child)
        if state is not None:
            state.update(status='released',released_utc=now());save(record,state);save(ROOT/'resource_release.json',state)
        if gpu_lock is not None:fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()

if __name__=='__main__':main()
