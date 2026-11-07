"""Claim one unstarted, unchanged formal job; release the GPU when it finishes."""
import datetime,fcntl,hashlib,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path

P=Path('/home/zhaoyuhan/Eventbased_WAM')
R=P/'runs/sim2real/20261103-stage9-direct-v1'
ROOT=P/'runs/sim2real/20261107-flow-rife-seed2-gpu2-v1'
UUID='GPU-a02affe0-7598-579e-ca39-f551f35ad6e9'
JOB='82_2_flow_rife_esim_seed2'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def main():
    assert socket.gethostname()=='g1'
    registry=P/'.gpu_reservations'
    lock=(registry/('.gpu-'+UUID+'.lock')).open('a+')
    global_lock=(registry/'.gpu_reservations.lock').open('a+')
    record=registry/(ROOT.name+'_gpu2.json')
    pending=R/'gpu4/queue'/(JOB+'.json')
    claimed=ROOT/'claimed_original_spec.json'
    child=None;started=False;state=None
    fcntl.flock(global_lock,fcntl.LOCK_EX)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        observed=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        assert observed[0].strip()==UUID and int(observed[1])>=30000
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        assert UUID not in apps
        for file in registry.glob('*.json'):
            data=json.loads(file.read_text())
            if data.get('gpu_uuid')==UUID:assert str(data.get('status','')).startswith(('released','complete')),str(file)
        assert not record.exists() and not ROOT.joinpath('completion.json').exists()
        assert pending.is_file() and not list(R.glob('gpu*/'+JOB+'.running_spec.json'))
        spec=json.loads(pending.read_text());assert spec['job_id']==JOB and not spec.get('after')
        for path,digest in spec['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        assert all(Path(f).is_file() for f in spec['ready_files'])
        assert not (R/'training/flow/seed2/rife_esim/training_metrics.jsonl').exists()
        state=dict(owner='sim-to-real research chat',host='g1',gpu_index=2,gpu_uuid=UUID,run_id=ROOT.name,
            status='reserved',pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
            started_utc=now(),output_path=str(ROOT),purpose='unchanged queued formal RIFE flow seed2')
        pending.rename(claimed)
        try:save(record,state)
        except BaseException:
            claimed.rename(pending)
            raise
    finally:fcntl.flock(global_lock,fcntl.LOCK_UN)
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        env.update(spec.get('env',{}));env['CUDA_VISIBLE_DEVICES']=UUID
        env['CUDA_CACHE_PATH']=str(ROOT/'cuda_cache')
        output=Path(spec['output_dir']).resolve();assert output.is_relative_to(R)
        output.mkdir(parents=True,exist_ok=True)
        (R/'gpu2').mkdir(exist_ok=True)
        receipt=dict(utc=now(),original_pending=str(pending),original_spec_sha256=hashlib.sha256(claimed.read_bytes()).hexdigest(),
            original_spec=spec,argv_unchanged=True,changed_only=['GPU assignment','CUDA cache path'],
            launcher_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        save(ROOT/'launch.json',receipt)
        with (output/'stdout.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(spec['argv'],cwd=spec.get('cwd',str(R)),env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started=True
            state.update(status='formal_synthetic_training',current_job=dict(job_id=JOB,pid=child.pid,
                process_start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21],argv=spec['argv']))
            deadline=time.monotonic()+24*3600
            while child.poll() is None:
                state['heartbeat_utc']=now();save(record,state);save(ROOT/'live.json',state)
                if time.monotonic()>deadline:raise TimeoutError('Own job exceeded 24 hour bound')
                time.sleep(15)
            completion=dict(job_id=JOB,status='complete' if child.returncode==0 else 'failed',exit_code=child.returncode,
                finished_utc=now(),original_spec_sha256=receipt['original_spec_sha256'],argv=spec['argv'])
            save(ROOT/'completion.json',completion)
            save(R/'gpu2'/(JOB+'.done.json'),completion)
    except BaseException:
        save(ROOT/'failure.json',dict(utc=now(),error=traceback.format_exc(),child_started=started))
        if not started and claimed.exists() and not pending.exists():claimed.rename(pending)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(status='released',released_utc=now())
        save(record,state);save(ROOT/'resource_release.json',state)
        fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':main()
