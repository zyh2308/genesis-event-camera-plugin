"""One real-evaluation job with an atomic GPU lease; releases on completion."""
import datetime,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path('/home/zhaoyuhan/Eventbased_WAM')
R=ROOT/'runs/sim2real/20261103-stage9-direct-v1'
A=ROOT/'runs/sim2real/20261106-pose-synthetic-diagnostic-v1'
UUID='GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460'
PY='/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
regroot=ROOT/'.gpu_reservations'
lease=(regroot/('.gpu-'+UUID+'.lock')).open('a+')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['hostname'],text=True).strip()=='g1'
status=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
target=[x for x in status.splitlines() if UUID in x]
assert len(target)==1 and target[0].split(',')[0].strip()=='1'
assert int(target[0].split(',')[2].strip())>20000
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
assert not any(UUID in line for line in compute.splitlines()),'GPU acquired by another workload'
job=A/'gpu1_job'
job.mkdir(exist_ok=False)
registry=regroot/'stage10_pose_synthetic_diagnostic_20261106_gpu1.json'
assert not registry.exists()
record=dict(pid=os.getpid(),host='g1',gpu_index=1,gpu_uuid=UUID,status='real_evaluation',
            process_start_ticks=Path('/proc/self/stat').read_text().split()[21],started_utc=utc(),
            task='Stage10 synthetic-only frozen-model diagnostic and200-step paired fit',output=str(job))
registry.write_text(json.dumps(record,indent=2)+'\n')
cmd=[PY,'-u',str(A/'pose_synthetic_diagnostic.py'),'--campaign',str(R),'--out',str(A/'results'),'--fit-updates','200']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
    OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',NVIDIA_TF32_OVERRIDE='0')
rc=None
child=None
def interrupted(signum, frame):
    if child is not None and child.poll() is None:
        child.terminate()
        child.wait()
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
try:
    with (job/'stdout.log').open('xb',buffering=0) as log:
        child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
        record.update(child_pid=child.pid,argv=cmd);registry.write_text(json.dumps(record,indent=2)+'\n')
        rc=child.wait()
finally:
    if child is not None and child.poll() is None:
        child.terminate()
        child.wait()
    if rc is None and child is not None:rc=child.returncode
    record.update(status='released_complete' if rc==0 else 'released_failed',exit_code=rc,finished_utc=utc())
    registry.write_text(json.dumps(record,indent=2)+'\n')
    (job/'exit.json').write_text(json.dumps(record,indent=2)+'\n')
sys.exit(rc if rc is not None else 1)
