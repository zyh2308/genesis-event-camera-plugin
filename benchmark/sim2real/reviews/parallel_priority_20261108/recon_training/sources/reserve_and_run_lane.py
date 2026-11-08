"""g3 CPU-only lease admission, then one real reconstruction lane; no dummy CUDA."""
import argparse,datetime,fcntl,hashlib,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path

P=Path('/home/zhaoyuhan/Eventbased_WAM');ROOT=P/'runs/sim2real/20261108-recon-selected-full-v1'
CAMPAIGN=P/'runs/sim2real/20261103-stage9-direct-v1'
LAYOUT={0:(2,'GPU-b89ca413-eb9c-61a5-a89b-9c244d032fff'),1:(5,'GPU-dd93b58d-df3d-9d33-bffc-2203bb503e87'),2:(6,'GPU-48385047-4365-b814-dac9-7d0d2c63d9b0')}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def members(group):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==group and fields[0]!='Z':result.append(int(p.name))
        except (OSError,ValueError,IndexError):pass
    return result
def stop_group(child):
    if child is None:return
    group=child.pid
    if members(group):
        try:os.killpg(group,signal.SIGTERM)
        except ProcessLookupError:pass
        deadline=time.monotonic()+45
        while members(group) and time.monotonic()<deadline:time.sleep(.5)
        if members(group):
            try:os.killpg(group,signal.SIGKILL)
            except ProcessLookupError:pass
            deadline=time.monotonic()+15
            while members(group) and time.monotonic()<deadline:time.sleep(.2)
    try:child.wait(timeout=1)
    except subprocess.TimeoutExpired:pass
    if members(group):raise RuntimeError('Own worker process group did not stop')
def interrupted(signum,frame):raise SystemExit('Own lane parent interrupted '+str(signum))
def checked_ready(root):
    ready=json.loads((root/'DEPLOYMENT_READY.json').read_text())
    assert ready['schema']=='recon_selected_deployment_ready_v1' and ready['passed'] is True
    assert ready['source_manifest_sha256']==sha(root/'sources/source_manifest.json')
    manifest=json.loads((root/'sources/source_manifest.json').read_text())
    for name,digest in manifest.items():
        assert Path(name).name==name and sha(root/'sources'/name)==digest,name
    assert ready['cpu_admission_sha256']==sha(root/'cpu_admission.json')
    assert json.loads((root/'cpu_admission.json').read_text())['passed'] is True
    assert ready['registered_plan_sha256']==sha(root/'sources/registered_plan.json')
    plan=json.loads((root/'sources/registered_plan.json').read_text())
    assert plan['methods']==['eventgenesis','v2e','vid2e','film_esim','rife_esim'] and plan['seeds']==[0,1,2]
    assert plan['training_budget_updates']==12000 and plan['epochs']==200
    assert plan['real_sequences']==['boxes','reflective_materials','still_life']
    assert plan['real_result_scope']=='posthoc_selected_hqf3_development'
    return ready
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args()
    assert socket.gethostname()=='g3' and ROOT.is_dir() and ROOT.resolve()==ROOT
    idx,uuid=LAYOUT[a.seed];lane=ROOT/'lanes'/('seed'+str(a.seed));lane.mkdir(parents=True,exist_ok=False)
    registry=P/'.gpu_reservations';record=registry/(ROOT.name+'_g3_seed'+str(a.seed)+'_gpu'+str(idx)+'.json')
    state=None;lock=None;child=None;entered=time.monotonic();hard_deadline=entered+48*3600
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        observed=subprocess.check_output(['nvidia-smi','-i',str(idx),'--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
        assert observed[0].strip()==uuid and observed[1].strip()=='NVIDIA GeForce RTX 4090' and int(observed[2])>=21000
        lock=(registry/('.gpu-'+uuid+'.lock')).open('a+')
        with (registry/'.gpu_reservations.lock').open('a+') as global_lock:
            until=time.monotonic()+30
            while True:
                try:fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    assert time.monotonic()<until,'Reservation registry lock timeout';time.sleep(.2)
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=15)
            assert uuid not in apps,'Existing compute process; do not interfere'
            for path in registry.glob('*.json'):
                old=json.loads(path.read_text())
                if old.get('gpu_uuid')==uuid:assert str(old.get('status','')).startswith(('released','complete')),str(path)
            assert not record.exists()
            state=dict(owner='sim-to-real research chat',host='g3',gpu_index=idx,gpu_uuid=uuid,gpu_model=observed[1].strip(),seed=a.seed,
                       run_id=ROOT.name,pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                       output_path=str(lane),status='preparing',CUDA_context_created_by_parent=False,started_utc=now(),
                       staging_timeout_seconds=1800,overall_timeout_seconds=48*3600,current_job=None,
                       purpose='authorized selected-HQF3 reconstruction full matrix; preparing metadata only until CPU admission')
            save(record,state)
        staging_deadline=time.monotonic()+1800
        while not (ROOT/'DEPLOYMENT_READY.json').exists():
            assert not (ROOT/'GLOBAL_FAILURE.json').exists(),'Preparation/global failure'
            assert time.monotonic()<staging_deadline,'Staging did not complete within30min'
            state['heartbeat_utc']=now();save(record,state);save(lane/'live.json',state);time.sleep(5)
        ready=checked_ready(ROOT);save(lane/'ready_verification.json',ready)
        command=[sys.executable,'-B','-u',str(ROOT/'sources/run_selected_lane.py'),'--root',str(ROOT),'--seed',str(a.seed)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,STAGE9_CAMPAIGN=str(CAMPAIGN),STAGE9_RECON_REPAIR_ROOT=str(ROOT),
                 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NVIDIA_TF32_OVERRIDE='0',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (lane/'worker.log').open('x') as log:
            child=subprocess.Popen(command,env=env,cwd=str(ROOT/'sources'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running',current_job=dict(argv=command,pid=child.pid,source_manifest_sha256=ready['source_manifest_sha256']))
            while child.poll() is None:
                assert time.monotonic()<hard_deadline,'Own lane exceeded48h bound'
                state['heartbeat_utc']=now();save(record,state);save(lane/'live.json',state);time.sleep(5)
        save(lane/'worker_exit.json',dict(exit_code=child.returncode,finished_utc=now()))
        assert child.returncode==0,'Own worker failed; no automatic retry'
        assert not members(child.pid),'Worker exited while own group children still running'
        save(lane/'parent_completion.json',dict(status='complete',seed=a.seed,finished_utc=now()))
    except BaseException:
        save(lane/'parent_failure.json',dict(status='failed',error=traceback.format_exc(),utc=now()));raise
    finally:
        cleanup_error=None
        try:stop_group(child)
        except BaseException:cleanup_error=traceback.format_exc()
        stopped=child is None or not members(child.pid)
        if state is not None:
            state.update(status='released' if stopped else 'cleanup_failed',current_job=None if stopped else state.get('current_job'),
                         released_utc=now() if stopped else None,cleanup_error=cleanup_error)
            save(record,state);save(lane/'resource_release.json',state)
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        if cleanup_error:raise RuntimeError(cleanup_error)

if __name__=='__main__':main()
