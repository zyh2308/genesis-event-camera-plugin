"""Bounded diagnostic lease; no intervention in existing jobs or environments."""
import datetime,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
registry=Path('/home/zhaoyuhan/Eventbased_WAM/.gpu_reservations')
uuid='GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460'
record=registry/'correlation_cuda_audit_20261104_gpu1_attempt2.json'
lock=(registry/('.gpu-'+uuid+'.lock')).open('a+')
global_lock=(registry/'.gpu_reservations.lock').open('a+')
fcntl.flock(global_lock,fcntl.LOCK_EX)
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert uuid not in apps,'GPU acquired by another process'
    # Inspect any existing lease mentioning this UUID; fail closed for live owners.
    for p in registry.glob('*.json'):
        if p==record:continue
        text=p.read_text()
        if uuid not in text:continue
        d=json.loads(text)
        if str(d.get('status','')).startswith(('released','complete')):continue
        raise RuntimeError('Existing non-released registry: '+str(p))
    state=dict(status='diagnostic_running',gpu_uuid=uuid,gpu_index=1,node='g1',pid=os.getpid(),
               proc_start_ticks=Path('/proc/self/stat').read_text().split()[21],
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='bounded native correlation correctness and performance audit')
    record.write_text(json.dumps(state,indent=2)+'\n')
finally:fcntl.flock(global_lock,fcntl.LOCK_UN)
os.environ.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2')
try:
    for name in ['build_native.py','audit_operator.py','audit_model.py']:
        with (root/(name+'.attempt2.log')).open('x') as out:
            result=subprocess.run([sys.executable,'-u',str(root/name)],stdout=out,stderr=subprocess.STDOUT,timeout=1800)
        assert result.returncode==0,(name,result.returncode)
    state['audit_passed']=True
except Exception:
    state['audit_passed']=False;state['error']=traceback.format_exc()
finally:
    state.update(status='released_complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    record.write_text(json.dumps(state,indent=2)+'\n')
    fcntl.flock(lock,fcntl.LOCK_UN)
print(json.dumps(state),flush=True)
