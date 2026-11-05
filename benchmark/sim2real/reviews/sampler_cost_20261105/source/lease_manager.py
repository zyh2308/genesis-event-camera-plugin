"""Isolated GPU2 test queue with explicit ownership and a bounded lease."""
import datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
P=Path('/home/zhaoyuhan/Eventbased_WAM');reg=P/'.gpu_reservations'
uuid='GPU-a02affe0-7598-579e-ca39-f551f35ad6e9'
record=reg/'simulator_sampler_optimization_20261105e_gpu2.json'
lock=(reg/('.gpu-'+uuid+'.lock')).open('a+');global_lock=(reg/'.gpu_reservations.lock').open('a+')
fcntl.flock(global_lock,fcntl.LOCK_EX)
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert uuid not in apps,'GPU2 is no longer idle'
    for p in reg.glob('*.json'):
        if p==record:continue
        s=p.read_text()
        if uuid in s:
            data=json.loads(s)
            assert str(data.get('status','')).startswith(('released','complete')),str(p)
    state=dict(status='reserved_efficiency_comparison_tests',gpu_uuid=uuid,gpu_index=2,host='g1',pid=os.getpid(),
               process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
               owner='sim2real research; user explicitly assigned g1 GPU2',
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),output_path=str(root))
    record.write_text(json.dumps(state,indent=2)+'\n')
finally:fcntl.flock(global_lock,fcntl.LOCK_UN)
os.environ.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
queue=root/'queue';queue.mkdir(exist_ok=True)
try:
    import torch
    torch.set_num_threads(2)
    assert 'V100' in torch.cuda.get_device_name()
    hold=torch.empty(1024**2,dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline and not (root/'FINISH').exists():
        pending=sorted(queue.glob('*.json'))
        if not pending:time.sleep(5);continue
        path=pending[0];job=json.loads(path.read_text());path.rename(path.with_suffix('.running'))
        assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in job.get('sha256',{}).items())
        state.update(status='running_efficiency_comparison_test',job=job['id'],heartbeat_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        record.write_text(json.dumps(state,indent=2)+'\n')
        started=time.monotonic()
        with (root/(job['id']+'.log')).open('x') as log:
            try:result=subprocess.run(job['argv'],stdout=log,stderr=subprocess.STDOUT,timeout=job.get('timeout_s',1800));code=result.returncode
            except subprocess.TimeoutExpired:code=124
        (root/(job['id']+'.done.json')).write_text(json.dumps(dict(exit_code=code,wall_s=time.monotonic()-started,job=job),indent=2)+'\n')
        state.update(status='reserved_efficiency_comparison_tests',last_job=job['id'],last_exit_code=code)
        record.write_text(json.dumps(state,indent=2)+'\n')
except Exception:state['error']=traceback.format_exc()
finally:
    state.update(status='released_complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    record.write_text(json.dumps(state,indent=2)+'\n');fcntl.flock(lock,fcntl.LOCK_UN)
