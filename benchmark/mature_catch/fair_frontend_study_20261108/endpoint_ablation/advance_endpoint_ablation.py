"""After complete blind+cost acceptance, run the registered endpoint-only actors."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, socket, subprocess, sys, time, traceback
import xml.etree.ElementTree as ET

PY='/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python'
LAUNCHER='/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/visual_fast_paired_20261108a/run_gpu_stage_split_cpu.py'
RESERVATIONS=Path('/home/zhaoyuhan/Eventbased_WAM/.gpu_reservations')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def fresh(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2)


def main_ready(blind,study):
    marker=blind/'FAIR_COLLECTION_COMPLETE.json'
    if not marker.exists():return None
    main=read(marker)
    assert main['status']=='PASS' and main['actual_episodes']==320 and main['physical_seed_blocks']==32
    assert main['all_final_actor_attempts_normal_exit'] and main['all_task_failures_retained']
    for key,name in [('raw_manifest_sha256','SHA256SUMS'),('final_audit_sha256','FINAL_AUDIT.json'),('results_sha256','FAIR_STAGE_RESULTS.json')]:
        assert sha(blind/name)==main[key]
    cost=study/'cost_v4_matched';complete=cost/'COMPLETE.json';outer=study/'BLIND_COST_V4_CHECKPOINT_COST_RECEIPT.json'
    if (cost/'STOPPED.json').exists():raise RuntimeError('Cost study stopped; preserve it and resolve before endpoint-only execution')
    if not complete.exists() or not outer.exists():return None
    c=read(complete);job=read(outer)
    assert c['status']=='PASS' and c['validation_cells']==8 and c['formal_primary_runs']==24
    assert job['status']=='PASS' and job['actual_job_result']['status']=='completed' and job['actual_job_result']['exit_code']==0
    assert sha(cost/'FAIR_COST_RUNS.csv')==c['cost_csv_sha256']
    assert sha(cost/'BUDGET_RECEIPT.json')==c['budget_receipt_sha256']
    return dict(blind_marker_sha256=sha(marker),cost_marker_sha256=sha(complete),outer_cost_sha256=sha(outer))


def available_cards():
    tree=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=30))
    reserved=set()
    for path in RESERVATIONS.glob('*.json'):
        try:value=read(path)
        except (OSError,ValueError):continue
        if value.get('host','').split('.')[0]!='g1':continue
        if value.get('gpu_reserved') is False or str(value.get('status','')).startswith(('released','failed','cancelled','completed')):continue
        reserved.add(value.get('gpu_uuid'))
    result=[]
    for index,gpu in enumerate(tree.findall('gpu')):
        if index not in (0,1):continue
        used=int(gpu.findtext('fb_memory_usage/used','999999 MiB').split()[0])
        if gpu.findall('processes/process_info') or used>128 or gpu.findtext('uuid') in reserved:continue
        result.append(dict(index=index,uuid=gpu.findtext('uuid'),used_mib=used))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--tools',type=Path,required=True);p.add_argument('--timeout-s',type=float,default=86400)
    a=p.parse_args();assert socket.gethostname().split('.')[0]=='g1'
    protocol=a.protocol.resolve();spec=read(protocol);root=Path(spec['root']);blind=Path(spec['paired_main_root']);study=blind.parent
    assert spec['actual_actor_budget']==128 and spec['event_reconstruction_mode']=='endpoint_only'
    assert spec['required_job_env']=={'CATCHIT_EVENT_RECONSTRUCTION_MODE':'endpoint_only'}
    digest=sha(protocol);started=time.monotonic();status=study/(root.name+'_ADVANCE_ENDPOINT_STATUS.json')
    assert not status.exists(),'Refuse a duplicate coordinator; inspect its actual PID and stage instead'
    full=range(os.cpu_count());os.sched_setaffinity(0,full);full=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[10,11])
    def update(stage,**extra):
        temporary=study/(root.name+'_ADVANCE_ENDPOINT_STATUS.pending.json')
        temporary.write_text(json.dumps(dict(stage=stage,utc=stamp(),elapsed_s=time.monotonic()-started,**extra),indent=2))
        temporary.replace(status)
    try:
        fresh(root/'ENDPOINT_COORDINATOR.json',dict(pid=os.getpid(),host=socket.gethostname(),started_utc=stamp(),
            process_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            driver_sha256=sha(__file__),protocol_sha256=digest,no_gpu_reserved_while_waiting=True))
        update('WAIT_MAIN_BLIND_AND_COST')
        while True:
            assert sha(protocol)==digest
            readiness=main_ready(blind,study)
            if readiness is not None:break
            if time.monotonic()-started>a.timeout_s:raise TimeoutError('Main blind/cost did not complete within the registered waiting bound')
            time.sleep(30)
        fresh(root/'MAIN_STUDY_COMPLETION_GATE.json',dict(status='PASS',verified_utc=stamp(),**readiness))
        sys.path.insert(0,str(study/'analysis_v1'))
        from collect_fair_stage import verify_inputs
        verify_inputs(spec)
        update('WAIT_AVAILABLE_G1_RESOURCE')
        while True:
            cards=available_cards()
            if cards:break
            if time.monotonic()-started>a.timeout_s:raise TimeoutError('No unreserved original g1 GPU0/1 available')
            time.sleep(60)
        # Select the actually available cards once. One card is sufficient; no idle holder.
        template=read(blind/'catchit_fair_blind32_g1_gpu1_blind_costv4_resume1_20261108b.job.json')
        env=dict(template['env']);old=read(Path(spec['paired_main_protocol']))['source']
        paths=env['PYTHONPATH'].split(':');assert old in paths
        env['PYTHONPATH']=':'.join(spec['source'] if value==old else value for value in paths)
        env.update(spec['required_job_env'])
        worker=blind/'workqueue/fair_workqueue_worker.py';worker_sha=sha(worker)
        jobs=[]
        for card in cards:
            index=card['index'];identity='g1_gpu'+str(index)+'_endpoint_only32_v1'
            name='catchit_'+identity+'_20261109a'
            job=dict(name=name,gpu_candidates=[index],command=['/usr/bin/python3','-u',str(worker),
                '--protocol',str(protocol),'--worker-id',identity],env=env,cwd=str(root),timeout_s=86400,
                ablation_protocol_sha256=digest,main_completion_gate_sha256=sha(root/'MAIN_STUDY_COMPLETION_GATE.json'),
                shared_queue_worker_sha256=worker_sha)
            path=root/(name+'.job.json');fresh(path,job);jobs.append((path,name))
        managers=[]
        for path,name in jobs:
            assert sha(protocol)==digest and sha(worker)==worker_sha
            os.sched_setaffinity(0,full)
            try:
                with (root/(name+'.supervisor.log')).open('x') as log:
                    manager=subprocess.Popen(['/usr/bin/python3','-u',LAUNCHER,'--job',str(path)],cwd=root,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            finally:os.sched_setaffinity(0,[10,11])
            managers.append((manager,name))
        fresh(root/'ENDPOINT_WORKERS_LAUNCHED.json',dict(status='STARTED',utc=stamp(),cards=cards,
            managers=[dict(pid=manager.pid,name=name) for manager,name in managers],worker_script_sha256=worker_sha))
        update('RUN_ENDPOINT_ONLY128',registered_workers=len(managers))
        while any(manager.poll() is None for manager,name in managers):
            if time.monotonic()-started>a.timeout_s:raise TimeoutError('Actor managers remain active; no actor was terminated by the coordinator')
            time.sleep(10)
        for manager,name in managers:
            value=read(root/(name+'.result.json'))
            assert value['status']=='completed' and value['exit_code']==0,(name,value)
        update('AUDIT_ENDPOINT_ONLY128')
        with (root/'collect_endpoint.final.log').open('x') as log:
            subprocess.run([PY,'-B',str(a.tools/'collect_endpoint_ablation.py'),'--root',str(root),
                '--main-blind-root',str(blind),'--analysis-dir',str(study/'analysis_v1')],cwd=root,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
        marker=read(root/'FAIR_COLLECTION_COMPLETE.json')
        assert marker['status']=='PASS' and marker['actual_episodes']==128
        update('COMPLETE',actual_episodes=128,completion_marker_sha256=sha(root/'FAIR_COLLECTION_COMPLETE.json'))
    except BaseException as error:
        update('STOPPED',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
