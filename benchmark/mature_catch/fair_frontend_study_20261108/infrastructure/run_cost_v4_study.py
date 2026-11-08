"""Serial matched-cost study, inside an already allocated GPU0 atomic job.

No allocation, SSH, tuning, physics-reference recording, or utility scoring.
All eight validation cells must pass before any of the24 formal repeats.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

COMMIT='4931807d1cb229a55a96961b5c4d99b054ef4fb0'
PHYSICS_SHA='41458da631174aad39846d96e0ab7b753aa45c79bfcc0046658ba913d02b32a9'
TOOLS_SHA='902af6a574f74b285457aad2bdfeecb933f438e6776b17430e6ba29354c4c7f7'
REFERENCE=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/hdr_utility_diagnosis_20261108a/contact_paired25_windows10/cases/v01_normal25/ours')
PY='/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python'
RATES=(100.,50.,25.,12.5)
CURRENT=None
STOP_SIGNAL=None
STOP_TIME=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def text_rate(rate):return format(rate,'g')
def token(rate):return text_rate(rate).replace('.','p')


def fresh(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)


def progress(root,value):
    tmp=root/'PROGRESS.pending.json'
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(str(tmp),str(root/'PROGRESS.json'))


def stop(signum,frame):
    global STOP_SIGNAL,STOP_TIME
    if STOP_SIGNAL is None:STOP_SIGNAL,STOP_TIME=signum,time.monotonic()
    if CURRENT is not None and CURRENT.poll() is None:
        try:os.killpg(CURRENT.pid,signum)
        except ProcessLookupError:pass


def run_child(root,name,command,history,timeout):
    global CURRENT
    if STOP_SIGNAL is not None:raise InterruptedError('Study received signal '+str(STOP_SIGNAL))
    log=root/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
    entry=dict(name=name,command=command,started_utc=stamp(),log=str(log))
    started=time.monotonic();cause=None;kill_at=None;output=None
    try:
        with log.open('x',encoding='utf-8') as f:
            CURRENT=subprocess.Popen(command,cwd=str(root),stdout=f,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
            entry['pid']=CURRENT.pid
            while True:
                try:output=CURRENT.wait(timeout=.25);break
                except subprocess.TimeoutExpired:pass
                now=time.monotonic()
                if STOP_SIGNAL is not None and cause is None:
                    cause='signal '+str(STOP_SIGNAL);kill_at=STOP_TIME+30
                    try:os.killpg(CURRENT.pid,STOP_SIGNAL)
                    except ProcessLookupError:pass
                if now-started>timeout and cause is None:
                    cause='child timeout';kill_at=now+30
                    try:os.killpg(CURRENT.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                if kill_at is not None and now>=kill_at:
                    try:os.killpg(CURRENT.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    kill_at=None
        if cause or STOP_SIGNAL is not None:raise InterruptedError(cause or 'Study interrupted')
        if output!=0:raise RuntimeError(name+' failed with exit '+str(output)+'; preserve '+str(log))
    finally:
        # Every launched command has its own process group, including its worker.
        # Kill only that owned group if Python exits before the normal wait path.
        if CURRENT is not None:
            # The leader may already have died/OOMed while its worker remains.
            # Its private process group is ours even after the leader exits.
            try:os.killpg(CURRENT.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            if CURRENT.poll() is None:
                try:CURRENT.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:os.killpg(CURRENT.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    CURRENT.wait()
            try:os.killpg(CURRENT.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        entry.update(exit_code=(CURRENT.returncode if CURRENT is not None else None),finished_utc=stamp(),wall_s=time.monotonic()-started)
        CURRENT=None;history.append(entry)
        progress(root,dict(status='RUNNING',finished_commands=len(history),history=history))
    return entry


def frontend_id(env):
    return 'w'+env['CATCHIT_EVENT_OBSERVATION_WINDOW_MS']+'_fit'+env['CATCHIT_EVENT_FLIGHT_FIT']+'_gap'+env['CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS']


def formal_order():
    result=[]
    for rep in range(3):
        for ri,rate in enumerate(RATES):
            methods=('ours','v2e') if (rep+ri)%2==0 else ('v2e','ours')
            for method in methods:result.append((rep+1,method,rate))
    return result


def validate_actual(path,method,rate,env,identity,primary=False,receipt=None):
    v=read(path/'COST_RESULT.json')
    expected=dict(status='MEASURED_FIXED_TRAJECTORY',method=method,source_rate_hz=rate,frontend_env=env,
        frontend_id=frontend_id(env),source_code_commit=COMMIT,physics_reference_sha256=PHYSICS_SHA,
        physics_reference_kind='public_recorded_fixed_archived_actions',equivalence_kind='task_functional',
        host=identity['host'],source_meta_sha256=identity['source_meta_sha256'],
        actual_matched_control_steps=28,actor_query_count=28,fixed_capture_budget_pass=True,
        reference_seed=2026110831,cost_only_shadow=True,utility_result=False,cost_measurement=True,
        warmup_simulated_s=.32,horizon_simulated_s=1.12,measured_simulated_duration_s=.8,
        expected_original_packets=int(round(1.12*rate))+1,actual_original_packets=int(round(1.12*rate))+1,
        profiled_worker=False)
    for key,value in expected.items():
        if v.get(key)!=value:raise ValueError(str(path)+': actual result mismatch '+key)
    if v['worker_start']['gpu_uuid']!=identity['gpu_uuid']:raise ValueError('Actual GPU changed')
    if v.get('cpu_affinity')!=identity['cpu_affinity']:raise ValueError('Actual CPU affinity changed')
    checks=v['physics_qpos_qvel_time_checks']
    if len(checks)!=28 or any(not math.isfinite(c['qpos_max_abs_error']) or not math.isfinite(c['qvel_max_abs_error']) or
        c['qpos_max_abs_error']>1e-6 or c['qvel_max_abs_error']>1e-5 or
        abs(c['t_before']-i*.04)>1e-8 or abs(c['t']-(i+1)*.04)>1e-8 for i,c in enumerate(checks)):
        raise ValueError('Physical reference contract failed')
    witness=path/'ACTOR_QUERY_WITNESS.json'
    if sha(witness)!=v['actor_query_witness_sha256'] or read(witness)['status']!='COMPLETE':
        raise ValueError('Actual policy witness incomplete or changed')
    if not isinstance(v.get('actor_query_payload_sha256'),str) or len(v['actor_query_payload_sha256'])!=64:
        raise ValueError('Missing actual policy-query fingerprint')
    if not v['timing']['hot_wall_s'] or not math.isfinite(v['timing']['hot_wall_s']):
        raise ValueError('No actual complete-window wall measurement')
    if primary:
        for key in ('eligible_as_native_primary','acquisition_parity_verified','functional_query_exact','audit_hdf_and_demo_output_suppressed'):
            if v.get(key) is not True:raise ValueError('Formal primary validation failed: '+key)
        if v['acquisition_mode']!='required' or v['native_parity_receipt_sha256']!=sha(receipt):
            raise ValueError('Formal acquisition or receipt differs')
        if v['actor_query_payload_sha256']!=read(receipt)['actor_query_payload_sha256']:
            raise ValueError('Formal query fingerprint differs from validated actual inputs')
    return v


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--call-timeout-s',type=float,default=1800)
    args=p.parse_args();root=args.root.resolve();study=root.parent
    if os.name!='posix' or os.environ.get('EGL_DEVICE_ID')!='0':p.error('Run only inside the already allocated Linux GPU0 atomic job')
    if args.call_timeout_s<=0:p.error('Positive child timeout required')
    root.mkdir(parents=True,exist_ok=True)
    if any((root/name).exists() for name in ('COST_STUDY_PLAN.json','validation','formal','FAIR_COST_RUNS.csv','COMPLETE.json','STOPPED.json')):
        p.error('Fresh cost study required; existing evidence is never overwritten or resumed silently')
    history=[];formal=[];validation=[];phase='preflight';gpu_uuid=None
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        tools=study/'cost_tools_v4';meta_path=study/'FAIR_RUNTIME.json';selection_path=study/'frontend_validation/FRONTEND_SELECTION.json'
        physics=study/'cost_v3_preblind_smoke/public_physics_recording/PHYSICS_REFERENCE.json'
        meta=read(meta_path);selection=read(selection_path)
        if meta['code_commit']!=COMMIT or selection.get('status')!='LOCKED':raise ValueError('Source or frontend selection is not locked')
        if sha(physics)!=PHYSICS_SHA or sha(tools/'SOURCE_SHA256.json')!=TOOLS_SHA:raise ValueError('Registered physics/tool seal differs')
        tool_files=read(tools/'SOURCE_SHA256.json')['files']
        for name,digest in tool_files.items():
            if sha(tools/name)!=digest:raise ValueError('v4 tool bytes differ: '+name)
        envs=selection['frontend_by_method']
        if set(envs)!=set(('ours','v2e')):raise ValueError('Two locked source frontends required')
        for method in ('ours','v2e'):
            if frontend_id(envs[method])!=selection['rankings'][method]['selected_frontend']:raise ValueError('Locked frontend ID mismatch')
        parity_paths={rate:study/'native_parity'/('shard0' if rate in (100.,50.) else 'shard1')/text_rate(rate)/'V2E_PARITY_RECEIPT.json' for rate in RATES}
        for rate,path in parity_paths.items():
            proof=read(path)
            if proof.get('status')!='PASS' or proof.get('minimum_upsampling')!=int(round(1000/rate)):
                raise ValueError('Official V2E parity receipt missing/wrong cadence: '+str(path))
        # Read identity only; this never claims or probes another GPU.
        run_child(root,'000_gpu_identity',['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],history,30)
        gpu_uuid=(root/'logs/000_gpu_identity.log').read_text().strip()
        if not gpu_uuid.startswith('GPU-') or '\n' in gpu_uuid:raise ValueError('Ambiguous allocated GPU identity')
        identity=dict(host=socket.gethostname(),gpu_uuid=gpu_uuid,source_meta_sha256=sha(meta_path),
            cpu_affinity=sorted(os.sched_getaffinity(0)))
        inputs={str(meta_path):sha(meta_path),str(selection_path):sha(selection_path),str(physics):sha(physics),
            str(tools/'SOURCE_SHA256.json'):sha(tools/'SOURCE_SHA256.json')}
        inputs.update({str(path):sha(path) for path in parity_paths.values()})
        locked={}
        for method in ('ours','v2e'):
            locked[method]=root/'locked_frontends'/(method+'.json')
            fresh(locked[method],dict(frontend_id=frontend_id(envs[method]),env=envs[method],selection_sha256=sha(selection_path)))
        cells=[('ours',25.),('v2e',25.)]+[(m,r) for r in (100.,50.,12.5) for m in ('ours','v2e')]
        repeats=formal_order()
        plan=dict(schema='serial_cost_v4_predeclared_study_v1',registered_utc=stamp(),driver_sha256=sha(__file__),
            source_commit=COMMIT,tool_manifest_sha256=TOOLS_SHA,physics_reference_sha256=PHYSICS_SHA,
            physics_reference=str(physics),reference_run=str(REFERENCE),reference_seed=2026110831,
            frontend_by_method=envs,identity=identity,input_sha256=inputs,
            equivalence_kind='task_functional',measurement_window_s=[.32,1.12],simulated_s=.8,
            source_rates_hz=RATES,validation_order=[dict(method=m,source_hz=r) for m,r in cells],
            formal_order=[dict(repeat_id='r%02d'%rep,method=m,source_hz=r) for rep,m,r in repeats],
            equal_source_budget=dict(validation_physical_runs_per_source=8,formal_runs_per_source=12,formal_repeats_per_cell=3),
            expected_validation_cells=8,expected_formal_runs=24,expected_queries_per_run=28,
            native_v2e_receipts={text_rate(r):dict(path=str(path),sha256=sha(path)) for r,path in parity_paths.items()},
            child_timeout_s=args.call_timeout_s,resource='Already allocated GPU0 only; no allocation or blind-actor interruption',
            stopping='Any validation/formal mismatch stops the whole study; no re-recording, tuning, threshold relaxation or formal CSV')
        fresh(root/'COST_STUDY_PLAN.json',plan)
        def unchanged():
            if STOP_SIGNAL is not None:raise InterruptedError('Stop signal received')
            for path,digest in inputs.items():
                if sha(path)!=digest:raise ValueError('Registered study input changed: '+path)
            for name,digest in tool_files.items():
                if sha(tools/name)!=digest:raise ValueError('Sealed cost tool changed: '+name)
        def command(method,rate,dest,repeat,legacy=False,receipt=None):
            cmd=[PY,'-B',str(tools/'run_fixed_trajectory_cost.py'),'--source-meta',str(meta_path),
                '--reference-run',str(REFERENCE),'--physics-reference',str(physics),'--frontend-json',str(locked[method]),
                '--method',method,'--rate',text_rate(rate),'--repeat-id',repeat,'--output',str(dest),
                '--acquisition','legacy' if legacy else 'required','--equivalence-kind','task_functional']
            if legacy:cmd.append('--retain-audit-io')
            if method=='v2e':cmd+=['--v2e-parity-receipt',str(parity_paths[rate])]
            if receipt is not None:cmd+=['--cost-equivalence-receipt',str(receipt)]
            return cmd
        receipts={};phase='validation'
        for method,rate in cells:
            label=method+'_hz'+token(rate);base=root/'validation';left=base/(label+'_legacy');right=base/(label+'_required')
            for dest,legacy in ((left,True),(right,False)):
                unchanged();name='validation_'+dest.name
                run_child(root,name,command(method,rate,dest,'validation_'+('legacy' if legacy else 'required'),legacy),history,args.call_timeout_s)
                validate_actual(dest,method,rate,envs[method],identity)
            receipt=root/'receipts'/(label+'_task_functional.json')
            unchanged();run_child(root,'compare_'+label,[PY,'-B',str(tools/'compare_cost_event_parity.py'),
                '--legacy',str(left),'--native',str(right),'--kind','task_functional','--output',str(receipt)],history,120)
            proof=read(receipt)
            if proof['status']!='PASS_TASK_FUNCTIONAL_EQUIVALENCE' or proof['functional_query_exact'] is not True or proof['actual_query_count']!=28:
                raise ValueError('Actual functional equivalence failed: '+label)
            receipts[(method,rate)]=receipt
            validation.append(dict(method=method,source_hz=rate,legacy=str(left),required=str(right),receipt=str(receipt),receipt_sha256=sha(receipt),raw_exact=proof['raw_exact']))
        fresh(root/'VALIDATION_COMPLETE.json',dict(status='PASS',cells=validation,count=8,formal_runs_started=0))
        phase='formal'
        for rep,method,rate in repeats:
            unchanged();dest=root/'formal'/('r%02d'%rep)/(method+'_hz'+token(rate));receipt=receipts[(method,rate)]
            run_child(root,'formal_r%02d_%s_hz%s'%(rep,method,token(rate)),
                command(method,rate,dest,'r%02d'%rep,receipt=receipt),history,args.call_timeout_s)
            result=validate_actual(dest,method,rate,envs[method],identity,True,receipt)
            formal.append(dict(method=method,source_hz=rate,repeat_id='r%02d'%rep,path=str(dest),
                cost_result_sha256=sha(dest/'COST_RESULT.json'),actor_query_payload_sha256=result['actor_query_payload_sha256'],
                raw_event_exact=result['raw_event_exact']))
        if len(formal)!=24:raise ValueError('Formal study incomplete')
        phase='export';unchanged();pending=root/'FAIR_COST_RUNS.pending.csv'
        cmd=[PY,'-B',str(tools/'build_cost_table.py')]
        for result in formal:cmd+=['--run',result['path']]
        cmd+=['--output',str(pending)]
        run_child(root,'export_formal_cost',cmd,history,120)
        with pending.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        if len(rows)!=24 or any(row['primary_validated']!='true' or row['functional_query_exact']!='true' for row in rows):
            raise ValueError('Export is not the full24 actual eligible primary runs')
        for method in ('ours','v2e'):
            for rate in RATES:
                group=[r for r in rows if r['method']==method and float(r['source_hz'])==rate]
                if len(group)!=3 or {r['repeat_id'] for r in group}!={'r01','r02','r03'}:
                    raise ValueError('Equal source/rate/repeat budget incomplete')
        fresh(root/'BUDGET_RECEIPT.json',dict(status='PASS',validation_cells=validation,formal_runs=formal,
            sources={method:dict(validation_physical_runs=8,formal_primary_runs=12,
                formal_simulated_measured_s=9.6,formal_full_endpoint_packets=sum((int(round(1.12*r))+1)*3 for r in RATES)) for method in ('ours','v2e')},
            identity=identity,physics_reference_sha256=PHYSICS_SHA,source_commit=COMMIT,
            selection_sha256=sha(selection_path),driver_sha256=sha(__file__),
            scope='Actual fixed-trajectory cost only; no utility success claims; raw differences retained'))
        unchanged()
        os.replace(str(pending),str(root/'FAIR_COST_RUNS.csv'))
        os.replace(str(pending.with_suffix('.manifest.json')),str(root/'FAIR_COST_RUNS.manifest.json'))
        progress(root,dict(status='COMPLETE',history=history,validation_cells=8,formal_primary_runs=24))
        fresh(root/'COMPLETE.json',dict(status='PASS',finished_utc=stamp(),validation_cells=8,formal_primary_runs=24,
            cost_csv_sha256=sha(root/'FAIR_COST_RUNS.csv'),budget_receipt_sha256=sha(root/'BUDGET_RECEIPT.json'),
            plan_sha256=sha(root/'COST_STUDY_PLAN.json'),physics_reference_sha256=PHYSICS_SHA,
            identity=identity,equivalence_kind='task_functional',gpu_release='Owned outer atomic job handles normal/failure release'))
        owned=[]
        for name in ('logs','locked_frontends','validation','receipts','formal'):
            owned.extend(p for p in (root/name).rglob('*') if p.is_file())
        owned.extend(root/name for name in ('COST_STUDY_PLAN.json','PROGRESS.json','VALIDATION_COMPLETE.json',
            'BUDGET_RECEIPT.json','FAIR_COST_RUNS.csv','FAIR_COST_RUNS.manifest.json','COMPLETE.json'))
        with (root/'SHA256SUMS').open('x',encoding='utf-8') as f:
            for path in sorted(owned):f.write(sha(path)+'  '+str(path.relative_to(root))+'\n')
        print(json.dumps(dict(status='PASS',root=str(root),formal_runs=24),indent=2))
    except BaseException as exc:
        # A partial export is never left under the formal publication filename.
        for name in ('FAIR_COST_RUNS.csv','FAIR_COST_RUNS.manifest.json','COMPLETE.json'):
            path=root/name
            if path.exists():path.rename(root/('REJECTED_'+name))
        value=dict(status='STOPPED',phase=phase,error=type(exc).__name__+': '+str(exc),finished_utc=stamp(),
            history=history,validation_cells_completed=len(validation),formal_runs_completed=len(formal),
            no_formal_cost_csv=True,traceback=traceback.format_exc(),signal=STOP_SIGNAL)
        fresh(root/'STOPPED.json',value);progress(root,value)
        print(json.dumps({k:v for k,v in value.items() if k not in ('history','traceback')},indent=2))
        raise SystemExit(128+STOP_SIGNAL if STOP_SIGNAL else 1)


if __name__=='__main__':main()
