"""Bounded, shared-claim reconstruction queue. No idle CUDA holder and no real scoring."""
import argparse,datetime,fcntl,hashlib,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.partial')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def check_sources(root):
    manifest=json.loads((root/'sources/source_manifest.json').read_text())
    for name,digest in manifest.items():assert sha(root/'sources'/name)==digest,name
def stop_own_child(child):
    if child is None or child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=45)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def gate(root,campaign):
    sys.path.insert(0,str(root/'sources'))
    from recon_repair_gate import decide
    config=json.loads((root/'sources/training_recipe.json').read_text());study=Path(config['reconstruction_repair']['synthetic_adoption_study'])
    assert study.parent==campaign.parent
    payload=json.loads((study/'mixed/completion.json').read_text());reported=json.loads((study/'synthetic_gate.json').read_text())
    result=decide(payload['results']);assert result['checks']==reported['checks']
    assert result['passed'] and reported['passed']
    assert reported['completion_sha256']==sha(study/'mixed/completion.json')
    assert reported['gate_source_sha256']==sha(root/'sources/recon_repair_gate.py')
    return result,study

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--gpu-index',type=int,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--wait-flow-root',type=Path)
    args=p.parse_args();root=args.root.resolve();R=args.campaign.resolve();assert root.parent==R.parent and root!=R
    assert socket.gethostname()=='g1'
    lane=root/f'gpu{args.gpu_index}';lane.mkdir(exist_ok=False);registry=R.parents[2]/'.gpu_reservations'
    record=registry/(root.name+f'_gpu{args.gpu_index}.json');state=None;lock=None;child=None
    try:
        check_sources(root);config=json.loads((root/'sources/training_recipe.json').read_text())
        study=Path(config['reconstruction_repair']['synthetic_adoption_study'])
        waiting=dict(status='waiting_for_synthetic_adoption',GPU_reserved=False,upstream=str(study),utc=now())
        save(lane/'waiting.json',waiting);deadline=time.monotonic()+5*3600
        while not (study/'resource_release.json').exists():
            if (study/'failure.json').exists():raise RuntimeError('Synthetic adoption study failed')
            if time.monotonic()>deadline:raise TimeoutError('Synthetic gate did not complete')
            time.sleep(10)
        adopted,_=gate(root,R);save(lane/'adoption_verification.json',adopted)
        if args.wait_flow_root:
            upstream=args.wait_flow_root.resolve();assert upstream==R.parent/'20261107-flow-rife-seed2-gpu2-v1'
            save(lane/'waiting.json',dict(status='waiting_for_existing_flow_release',GPU_reserved=False,upstream=str(upstream),utc=now()))
            deadline=time.monotonic()+36*3600
            while not (upstream/'resource_release.json').exists():
                if time.monotonic()>deadline:raise TimeoutError('Existing flow resource did not release')
                time.sleep(30)
            release=json.loads((upstream/'resource_release.json').read_text())
            assert release['status'].startswith('released')
            assert json.loads((upstream/'completion.json').read_text())['status']=='complete'
        check_sources(root)
        observed=subprocess.check_output(['nvidia-smi','-i',str(args.gpu_index),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        assert observed[0].strip()==args.gpu_uuid and int(observed[1])>20000
        lock=(registry/('.gpu-'+args.gpu_uuid+'.lock')).open('a+')
        with (registry/'.gpu_reservations.lock').open('a+') as global_lock:
            fcntl.flock(global_lock,fcntl.LOCK_EX);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            assert args.gpu_uuid not in apps,'GPU has another active compute process'
            for path in registry.glob('*.json'):
                existing=json.loads(path.read_text())
                if existing.get('gpu_uuid')==args.gpu_uuid:assert str(existing.get('status','')).startswith(('released','complete')),str(path)
            assert not record.exists()
            state=dict(owner='sim-to-real research chat',host='g1',gpu_index=args.gpu_index,gpu_uuid=args.gpu_uuid,
                run_id=root.name,pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                status='reserved',purpose='gated paired reconstruction repair; original flow/pose unchanged',output_path=str(root),started_utc=now())
            save(record,state)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu_uuid,STAGE9_CAMPAIGN=str(R),STAGE9_RECON_REPAIR_ROOT=str(root),
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NVIDIA_TF32_OVERRIDE='0',PYTHONDONTWRITEBYTECODE='1')
        queue_lock=root/'.queue.lock';queue_state=root/'queue_state.json'
        started=time.monotonic()
        while True:
            assert time.monotonic()-started<72*3600,'Own lane exceeded bounded runtime'
            check_sources(root)
            with queue_lock.open('a+') as qlock:
                fcntl.flock(qlock,fcntl.LOCK_EX);queue=json.loads(queue_state.read_text())
                if queue.get('halt_reason'):break
                job=next((job for job in queue['jobs'] if job['status']=='pending'),None)
                if job is None:break
                job.update(status='running',gpu_index=args.gpu_index,manager_pid=os.getpid(),claimed_utc=now())
                save(queue_state,queue);job=dict(job)
            method,seed=job['method'],job['seed'];tag=f'{method}_seed{seed}';output=root/f'training/seed{seed}'/method
            output.mkdir(parents=True,exist_ok=False)
            base=[sys.executable,'-u',str(root/'sources/train_e2vid_repaired.py'),'--model-root',str(R.parents[2]/'external/sources/downstream/event_cnn_minimal'),
                '--data-root',str(R/'adapters'/method/'reconstruction'),'--config',str(root/'sources/training_recipe.json'),
                '--output-root',str(output),'--generator',method,'--seed',str(seed),'--device','cuda:0']
            receipts=[]
            for phase,extra in [('update20',['--max-updates','20']),('resume21',['--max-updates','21','--resume',str(output/'last.pt')]),
                                ('full',['--resume',str(output/'last.pt')])]:
                check_sources(root);command=base+extra;receipt=dict(job=tag,phase=phase,argv=command,started_utc=now())
                with (output/(phase+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=str(root/'sources'),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    receipt.update(pid=child.pid,process_start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21])
                    state.update(status='running',current_job=receipt,heartbeat_utc=now());save(record,state);save(lane/'live.json',state)
                    phase_deadline=time.monotonic()+(24 if phase=='full' else 1)*3600
                    while child.poll() is None:
                        if time.monotonic()>phase_deadline:stop_own_child(child);raise TimeoutError(tag+'/'+phase+' exceeded runtime')
                        state['heartbeat_utc']=now();save(record,state);save(lane/'live.json',state);time.sleep(10)
                    receipt.update(exit_code=child.returncode,finished_utc=now());save(output/(phase+'_exit.json'),receipt)
                    assert child.returncode==0,(tag,phase,child.returncode)
                receipts.append(receipt);child=None
                summary=json.loads((output/'run_summary.json').read_text())
                if phase=='update20':
                    assert summary['optimizer_update']==20 and (output/'update_000020.pt').is_file()
                if phase=='resume21':
                    oracle=json.loads((output/'resume_oracle.json').read_text());assert oracle['status']=='pass'
                    assert (oracle['source_update'],oracle['observed_next_update'])==(20,21)
                    assert set(oracle['restored_optimizer_step_counts'])=={20} and set(oracle['optimizer_step_counts_after_next_update'])=={21}
                    save(root/'preflight'/f'{tag}_update20_resume21.json',dict(passed=True,oracle=oracle,checkpoint_update20_sha256=sha(output/'update_000020.pt')))
                    (output/'training_metrics_updates_1_21.jsonl').write_bytes((output/'training_metrics.jsonl').read_bytes())
                if phase=='full':
                    assert summary['status']=='complete' and summary['optimizer_update']==12000
                    selection=json.loads((output/'synthetic_best.json').read_text());assert sha(output/'synthetic_best.pt')==selection['sha256']
                    save(output/'completion.json',dict(status='complete',method=method,seed=seed,optimizer_update=12000,
                        initial_state_sha256=summary['initial_model_sha256'],recipe_sha256=sha(root/'sources/training_recipe.json'),
                        selected_checkpoint=selection,summary=summary,real_data_used=False,real_scoring_enabled=False,phase_receipts=receipts))
            with queue_lock.open('a+') as qlock:
                fcntl.flock(qlock,fcntl.LOCK_EX);queue=json.loads(queue_state.read_text())
                row=next(r for r in queue['jobs'] if (r['method'],r['seed'])==(method,seed));row.update(status='complete',finished_utc=now())
                save(queue_state,queue)
                if all(r['status']=='complete' for r in queue['jobs']):save(root/'completion.json',dict(status='complete',jobs=queue['jobs'],real_data_used_in_training=False,real_scoring_enabled=False,utc=now()))
        final_queue=json.loads(queue_state.read_text())
        save(lane/'completion.json',dict(status='stopped_due_to_global_failure' if final_queue.get('halt_reason') else 'complete',real_scoring_enabled=False,utc=now()))
    except BaseException:
        error=traceback.format_exc();save(lane/'failure.json',dict(error=error,utc=now()))
        if state is not None:
            with (root/'.queue.lock').open('a+') as qlock:
                fcntl.flock(qlock,fcntl.LOCK_EX);queue=json.loads((root/'queue_state.json').read_text());queue['halt_reason']=error
                for row in queue['jobs']:
                    if row.get('status')=='running' and row.get('manager_pid')==os.getpid():row.update(status='failed',finished_utc=now(),failure_receipt=str(lane/'failure.json'))
                save(root/'queue_state.json',queue)
        raise
    finally:
        stop_own_child(child)
        if state is not None:
            state.update(status='released',current_job=None,released_utc=now());save(record,state);save(lane/'resource_release.json',state)
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_UN)
if __name__=='__main__':main()
