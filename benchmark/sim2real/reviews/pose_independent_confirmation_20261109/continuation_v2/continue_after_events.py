"""After the successful fixed job172, run the already frozen C1 evaluation.

This CPU-only launcher never trains, generates events, or initializes CUDA. The
existing GPU4 parent owns resource admission; the frozen evaluator retains its
original GPU4 guard. Only this launcher's own child process group is cleaned up.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path

R=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1')
N=R.parent/'20261109-pose-advantage-and-failure-v1'
ROOT=N/'advantage_confirmation'
PYTHON=Path('/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python')
JOB='172_pose_advantage_events_20261109'
DONE_SHA='d14259d8112333381c3416da5979bb55e9668706ee628d5844e57ee54e70444d'
SPEC_SHA='3fb8531fd6e4dbc06d28ca25a2d50fa4da4f7c8f2d1c2b00d71ad0e9de8f1eb3'
REG=N/'PREPARATION_CONTRACT.json'
REG_SHA='8acbe3f76b841e908814b8c1d53c2c6ebabafa8e6dd78a5a69f8729d6e3540ef'
PHYSICAL=ROOT/'physical_completion.json'
PHYSICAL_SHA='a8a3f0c3bc04657d38196e0575072116284da028e8bf3f8af18902afcd4580fd'
PREP=N/'advantage_preparation_sources_v1/prepare_confirmation.py'
PREP_SHA='3a44394f510182bd3950cac011248692a11a67fc94ed7406e470536527d435f4'
EDIR=N/'advantage_evaluation_sources_v1'
EMANIFEST_SHA='8a285c7343edc3be18f20d4650fb6e7cc2bc94489f08b73663da2711298f255d'
EVENT=ROOT/'event_completion.json'
EVENT_SHA='ed9257f02abeeca975aa8d5563eb4704e87980c1412c87538acd8c53540d4f7c'
GPU4='GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3'
PARENT=335186
PARENT_START_TICKS=53273702
OUT=N/'advantage_continuation_v2'
METHODS=('eventgenesis','v2e')
CONDITIONS=('normal','half','eighth')

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def record(p):
    p=Path(p);return dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p))
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def verify(r):
    digest=r['sha256']
    need(isinstance(digest,str) and len(digest)==64 and all(ch in '0123456789abcdef' for ch in digest),'Invalid SHA256 for '+r['path'])
    p=Path(r['path']);need(p.is_file() and ('size_bytes' not in r or p.stat().st_size==r['size_bytes']) and sha(p)==r['sha256'],'Pinned bytes changed: '+str(p));return p

def verify_queue_parent():
    need(os.name=='posix' and os.getppid()==PARENT,'Only the existing GPU4 queue may launch this worker')
    fields=(Path('/proc')/str(PARENT)/'stat').read_text().rsplit(')',1)[-1].split()
    need(len(fields)>19 and fields[0] not in ('Z','X') and int(fields[19])==PARENT_START_TICKS,'Existing GPU4 queue process identity changed')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU4 and Path(sys.executable).resolve()==PYTHON.resolve(),'Existing interpreter/GPU4 visibility required')

def verify_job(done,spec):
    need(done['job_id']==spec['job_id']==JOB and done['status']=='complete' and done['exit_code']==0,'Job172 is not successfully complete')
    need(done['argv']==spec['argv'] and done['output_dir']==spec['output_dir'] and done['phase']==spec['phase'],'Queue done/spec lineage differs')
    argv=done['argv'];need(isinstance(argv,list) and all(isinstance(v,str) for v in argv) and argv[0]==str(PYTHON),'Unknown producer interpreter/argv')
    tail=argv[1:]
    while tail and tail[0] in ('-B','-u'):tail=tail[1:]
    need(tail and Path(tail[0]).is_absolute() and Path(tail[0]).is_relative_to(N) and Path(tail[0]).name=='prepare_confirmation.py','Unregistered producer source location')
    expected=['events','--registration',str(REG),'--registration-sha256',REG_SHA,'--physical-receipt-sha256',PHYSICAL_SHA]
    need(tail[1:]==expected,'Job172 did not execute exactly the fixed events command')
    need(Path(done['output_dir']).is_relative_to(R),'Producer output escaped its existing queue campaign')
    return Path(tail[0])

def verify_event_inventory(event,physical,root=ROOT):
    """Rehash all event/physical artifacts and require the entire fixed inventory."""
    need(event['schema']=='pose_confirm32_events_v1' and event['status']=='complete' and event['optimizer_updates']==0 and event['real_data_used'] is False,'Event phase incomplete/isolation differs')
    need(event['registration_sha256']==REG_SHA and physical['registration_sha256']==REG_SHA,'Preparation contract binding differs')
    need(physical['schema']=='pose_confirm32_physical_v1' and physical['status']=='complete' and physical['optimizer_updates']==0 and physical['real_data_used'] is False and physical['count']==32 and physical['physical_seed']==202611090137 and physical['cuda_initialized'] is False,'Physical phase scope differs')
    physical_path=root/'physical_completion.json';need(event['physical_receipt']==record(physical_path),'Event-to-physical receipt binding differs')
    rids=[f'validation_{i:03d}' for i in range(32)]
    need(len(physical['physical_files'])==32 and [Path(r['path']).stem for r in physical['physical_files']]==rids,'Physical32 ordered inventory differs')
    need(set(event['banks'])==set(CONDITIONS) and all(event['banks'][k]==str(root/k/'pose_bank') for k in CONDITIONS),'Event bank roots differ')
    records=event['artifacts'];by={r['path']:r for r in records};need(len(by)==len(records),'Duplicate event artifact')
    for r in records:
        need(Path(r['path']).is_relative_to(root),'Event artifact escaped confirmation root');verify(r)
    for r in physical['physical_files']+[physical['bank_manifest']]:verify(r)
    for condition in CONDITIONS:
        bank=root/condition/'pose_bank';bm=bank/'bank_manifest.json';need(str(bm) in by,'Missing physical bank manifest')
        for rid in rids:need(str(bank/'physical'/(rid+'.h5')) in by,'Missing physical trajectory artifact')
        for method in METHODS:
            mp=bank/'events'/method/'manifest.json';need(str(mp) in by,'Missing method manifest');m=read(mp)
            need(m['physical_manifest_sha256']==sha(bm) and m['real_data_used'] is False and len(m['rollouts'])==32,'Generator physical pairing/isolation differs')
            items={r['rollout_id']:r for r in m['rollouts']};need(len(items)==32 and set(items)==set(rids),'Missing/duplicate method rollout')
            for rid in rids:
                p=bank/'events'/method/(rid+'.npy');need(str(p) in by and items[rid]['events_sha256']==by[str(p)]['sha256'],'Event manifest/file SHA differs')
    return dict(physical_rollouts=32,conditions=list(CONDITIONS),methods=list(METHODS),required_event_streams=192,all_recorded_artifacts_verified=len(records))

def build_argv(event_sha):
    need(len(event_sha)==64 and all(x in '0123456789abcdef' for x in event_sha),'Bad actual event receipt digest')
    return [str(PYTHON),'-B','-u',str(EDIR/'evaluate_confirmation.py'),'--registration',str(REG),'--registration-sha256',REG_SHA,
        '--event-receipt-sha256',event_sha,'--out',str(ROOT/'evaluation')]

def cleanup_child(child):
    if child is None or child.poll() is not None:return
    need(os.getpgid(child.pid)==child.pid,'Child is not its own process-group leader')
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=10)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-manifest-sha256',required=True);a=ap.parse_args()
    verify_queue_parent()
    need(not OUT.exists() and not (ROOT/'evaluation').exists(),'Fresh continuation/evaluation output required');OUT.mkdir()
    child=None;started=time.monotonic();before=[]
    def interrupted(signum,frame):raise InterruptedError('Bounded launcher received signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,interrupted)
    signal.alarm(4200)
    try:
        own=Path(__file__).resolve().parent;manifest=own/'source_manifest.json';need(sha(manifest)==a.source_manifest_sha256,'Launcher source manifest differs');before.append(record(manifest))
        for name,digest in read(manifest).items():need(Path(name).name==name,'Unsafe source name');before.append(dict(path=str(own/name),sha256=digest));verify(before[-1])
        em=EDIR/'evaluation_source_manifest.json';need(sha(em)==EMANIFEST_SHA,'Frozen evaluation package manifest differs');before.append(record(em))
        emap=read(em);need(set(emap)=={'prepare_confirmation.py','evaluate_confirmation.py','test_evaluate_confirmation.py','EVALUATION_RUNBOOK.md'},'Unexpected evaluator source scope')
        for name,digest in emap.items():r=dict(path=str(EDIR/name),sha256=digest);verify(r);before.append(r)
        lane=R/'gpu4';dp=lane/(JOB+'.done.json');sp=lane/(JOB+'.running_spec.json')
        for p,digest in ((REG,REG_SHA),(PHYSICAL,PHYSICAL_SHA),(EVENT,EVENT_SHA),(dp,DONE_SHA),(sp,SPEC_SHA)):
            r=dict(path=str(p),sha256=digest);verify(r);before.append(record(p))
        done=read(dp);spec=read(sp);producer=verify_job(done,spec)
        need(producer==PREP and spec['source_sha256'][str(producer)]==PREP_SHA and spec['env']['CUDA_VISIBLE_DEVICES']==GPU4,'Frozen producer source or resource binding differs')
        verify(dict(path=str(producer),sha256=PREP_SHA));before.append(record(producer))
        for p,digest in spec.get('source_sha256',{}).items():verify(dict(path=p,sha256=digest))
        module_path=EDIR/'prepare_confirmation.py';ms=importlib.util.spec_from_file_location('continuation_pinned_preparation',module_path);P=importlib.util.module_from_spec(ms);ms.loader.exec_module(P)
        contract=read(REG);P.validate_registration(contract);P.check_science_and_core(contract);P.verify_records(contract['source_records'])
        for name in ('fresh_source','relight_source','export_source','base_contract','baseline_scores','seed_inventory'):P.verify_records([contract[name]])
        physical=read(PHYSICAL);need(physical['source_records']==contract['source_records'] and physical['seed_inventory']==contract['seed_inventory'],'Physical receipt source/inventory binding differs')
        event=read(EVENT);inventory=verify_event_inventory(event,physical);event_record=record(EVENT);need(event_record['sha256']==EVENT_SHA,'Event receipt changed during admission')
        need(time.monotonic()-started<600,'Preparation admission exceeded600s')
        argv=build_argv(event_record['sha256']);write(OUT/'admission.json',dict(status='passed',job_id=JOB,producer_exit_code=0,inventory=inventory,event_receipt=event_record,
            argv=argv,input_records=before,optimizer_updates=0,real_data_used=False,evaluation_source_manifest_sha256=EMANIFEST_SHA,launcher_source_manifest_sha256=a.source_manifest_sha256))
        with (OUT/'evaluation_stdout.log').open('xb') as log:
            child=subprocess.Popen(argv,cwd=str(EDIR),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            code=child.wait(timeout=3600)
        need(code==0,'Frozen evaluator failed with exit '+str(code))
        verify_queue_parent()
        for r in before:verify(r)
        P.check_science_and_core(contract);P.verify_records(contract['source_records']);completion=ROOT/'evaluation/completion.json';c=read(completion)
        need(c['schema']=='pose_confirm32_frozen_evaluation_v1' and c['status']=='complete' and c['optimizer_updates']==0 and c['real_data_used'] is False and c['GT_forward_inputs'] is False,'Evaluator completion scope differs')
        need(c['registration_sha256']==REG_SHA and c['event_completion']==event_record and c['score_rows']==18 and c['all_baseline_parities_passed'] and len(c['models'])==6 and all(r['unchanged'] for r in c['models']),'Frozen evaluation lineage/inventory differs')
        write(OUT/'completion.json',dict(schema='pose_confirm32_automatic_continuation_v2',status='complete',job_id=JOB,producer_exit_code=0,evaluator_exit_code=code,evaluation_completion=record(completion),
            event_receipt=event_record,conditional_advantage_passed=c['conditional_advantage']['passed'],launcher_input_records_unchanged=True,frozen_evaluator_completed_its_input_hash_checks=True,optimizer_updates=0,real_data_used=False,
            no_further_job_started=True,elapsed_seconds=time.monotonic()-started))
    except BaseException as e:
        cleanup_child(child)
        write(OUT/'failure.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),optimizer_updates=0,real_data_used=False,elapsed_seconds=time.monotonic()-started));raise
    finally:
        cleanup_child(child);signal.alarm(0)

if __name__=='__main__':main()
