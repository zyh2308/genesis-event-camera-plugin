"""Strict128-actor endpoint-only companion; existing blind320 collector is unchanged."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,sys,time
sys.dont_write_bytecode=True
RATES=(12.5,25.,50.,100.)
VARIANT='endpoint_only'
FRONTEND='w10_fit0_gap100'
FIELDS=['case_id','physical_case_id','seed','method','variant','endpoint_hz','frontend_id','verified_catch',
    'failure_class','initialization_valid','raw_success','terminated','hand_contact_steps','final_contact',
    'first_touch_s','sim_time_s','wall_s','raw_events','packets','trace_rows','output']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def key(row):return str(row['physical_case_id']),int(row['seed']),float(row['endpoint_hz'])
def write(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def frontend_id(env):
    return 'w%s_fit%s_gap%s'%tuple(str(env[name]) for name in
        ('CATCHIT_EVENT_OBSERVATION_WINDOW_MS','CATCHIT_EVENT_FLIGHT_FIT','CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS'))


def main_reference(root,portable=False):
    root=Path(root);marker=read(root/'FAIR_COLLECTION_COMPLETE.json');data=read(root/'FAIR_STAGE_RESULTS.json')
    require(marker.get('status')=='PASS' and marker.get('actual_episodes')==320 and marker.get('all_task_failures_retained') is True,
        'Main reference must be the complete actual320-actor blind stage')
    require(marker.get('results_sha256')==sha(root/'FAIR_STAGE_RESULTS.json') and data.get('status')=='PASS'
        and data.get('complete') is True and data['metadata']['stage']=='blind' and len(data['rows'])==320,
        'Main blind results/completion disagree')
    audit=read(root/'FINAL_AUDIT.json')
    require(sha(root/'FINAL_AUDIT.json')==marker['final_audit_sha256'] and audit.get('status')=='PASS'
        and audit.get('episodes')==320 and audit.get('raw_event_contract_and_packet_counts_pass') is True,
        'Main reference lacks its complete raw audit')
    rows=[r for r in data['rows'] if r['method']=='ours'];indexed={key(r):r for r in rows}
    blocks={(k[0],k[1]) for k in indexed};expected={(p,s,rate) for p,s in blocks for rate in RATES}
    require(len(blocks)==32 and len(rows)==len(indexed)==128 and set(indexed)==expected,
        'Main Ours reference is not exactly32 blocks by four registered rates')
    actual={(str(r['physical_case_id']),int(r['seed']),None if r['endpoint_hz'] is None else float(r['endpoint_hz']),r['method']) for r in data['rows']}
    full_expected={(p,s,r,m) for p,s in blocks for r in RATES for m in ('ours','v2e')}
    full_expected|={(p,s,None,m) for p,s in blocks for m in ('init_only','state_reference')}
    require(len(actual)==320 and actual==full_expected,'Main blind must retain its full320-source/rate/reference grid')
    spec=read(root/'EVALUATION_PROTOCOL.json')
    cases={key(c):c for c in spec['cases'] if 'ours' in c['method_order']}
    require(set(cases)==expected,'Main registered physical grid differs from actual Ours outcomes')
    result=dict(root=root,marker=marker,data=data,spec=spec,rows=indexed,cases=cases,blocks=blocks,
        hashes=dict(results_sha256=sha(root/'FAIR_STAGE_RESULTS.json'),completion_sha256=sha(root/'FAIR_COLLECTION_COMPLETE.json'),
            protocol_sha256=sha(root/'EVALUATION_PROTOCOL.json')))
    if portable:
        records=read(root/'PAIRED_PORTABLE.json');require(records.get('complete') is True and not records.get('violations'), 'Main portable source record is incomplete')
        result['records']={(str(r['physical_case_id']),int(r['seed']),float(r['endpoint_hz'])):r for r in records['records'] if r['mode']=='ours'}
        require(set(result['records'])==expected,'Main source traces do not cover128 full Ours actors')
    return result


def verify_grid(spec,main):
    require(spec.get('event_reconstruction_mode',spec.get('variant'))==VARIANT,'Register endpoint_only before running actors')
    require(len(spec.get('cases',[]))==128,'Endpoint-only protocol must register exactly128 actors')
    cases={}
    for case in spec['cases']:
        require(case.get('method_order')==['ours'],'Endpoint-only stage contains exactly one Ours actor per cell')
        logical=key(case);require(logical not in cases,'Repeated endpoint-only physical seed/rate cell');cases[logical]=case
    require(set(cases)==set(main['rows']),'Endpoint-only must contain the full same32-seed by four-rate grid; no dropped/extra cells')
    baseline=main['spec']
    for field in ['core_files','checkpoint_sha256','profile_sha256','frontend_selection_sha256','bank_sha256']:
        require(spec.get(field) is not None and spec[field]==baseline.get(field),'Frozen/main input changed: '+field)
    require(len(spec['core_files'])==17,'Keep all17 frozen core hashes')
    require(frontend_id(spec['frontend_by_method']['ours'])==FRONTEND
        and spec['frontend_by_method']['ours']==baseline['frontend_by_method']['ours'], 'Keep main Ours lockedB frontend without retuning')
    changed={name for name,digest in baseline['runtime_files'].items() if spec['runtime_files'].get(name)!=digest}
    added=set(spec['runtime_files'])-set(baseline['runtime_files'])
    require(changed=={'sparse_event_sensor.py','run_original.py'} and added=={'endpoint_event_worker.py'},
        'Only the two reviewed benchmark hooks and one new worker may differ; frozen/shared runtime remains unchanged')
    for logical,case in cases.items():
        reference=main['cases'][logical]
        require(case['launch']==reference['launch'],'Endpoint-only launch differs from its paired main actor')
        require(case.get('lighting','normal')==reference.get('lighting','normal'),'Endpoint-only lighting differs from its paired actor')
        override=case.get('frontend_overrides',spec['frontend_by_method']['ours'])
        require(override==baseline['frontend_by_method']['ours'],'Per-case endpoint-only frontend tuning is forbidden')
    return cases


def source_delta_proof(root,spec,main):
    path=Path(spec['endpoint_source_delta_receipt']);path=path if path.is_absolute() else root/path
    require(sha(path)==spec['endpoint_source_delta_sha256'],'Endpoint source delta receipt hash mismatch')
    receipt=read(path)
    require(set(receipt['changed_files'])=={'sparse_event_sensor.py','run_original.py'}
        and set(receipt['added_files'])=={'endpoint_event_worker.py'},'Source delta receipt declares unreviewed changed files')
    require(receipt.get('review',{}).get('allowed_hunks_only') is True
        and receipt.get('review',{}).get('control_policy_scoring_unchanged') is True,'Specific benchmark hunks lack the required root review')
    names={'sparse_event_sensor.py','run_original.py','endpoint_event_worker.py'}
    require(set(receipt['files'])==names,'Source delta receipt must cover exactly the three reviewed files')
    for name,entry in receipt['files'].items():
        previous=main['spec']['runtime_files'].get(name)
        require(entry.get('old_sha256')==previous and entry['new_sha256']==spec['runtime_files'][name]
            and sha(Path(spec['source'])/name)==entry['new_sha256'],'Actual source delta hashes disagree: '+name)
        if previous:require(sha(Path(main['spec']['source'])/name)==previous,'Original shared runtime changed after main freeze')
    diff=Path(receipt['unified_diff_file']);diff=diff if diff.is_absolute() else path.parent/diff
    require(sha(diff)==receipt['unified_diff_sha256'],'Reviewed unified diff hash mismatch')
    paths=set()
    for line in diff.read_text(encoding='utf-8').splitlines():
        if line.startswith(('--- ','+++ ')):
            label=line[4:].split('\t')[0].strip()
            if label!='/dev/null':paths.add(Path(label).name)
    require(paths==names,'Reviewed unified diff changes files outside the three allowed hooks')
    original=(Path(main['spec']['source'])/'run_original.py').read_bytes();candidate=(Path(spec['source'])/'run_original.py').read_bytes()
    def control_loop(content):
        begin=content.index(b'                while True:');end=content.index(b'            row=dict(episode=episode',begin)
        return content[begin:end]
    require(control_loop(original)==control_loop(candidate),'Original control/action/termination loop bytes changed')
    return dict(status='PASS',receipt_path=str(path),receipt_sha256=sha(path),unified_diff_path=str(diff),
        unified_diff_sha256=sha(diff),changed_files=receipt['changed_files'],added_files=receipt['added_files'],
        control_loop_byte_identical=True,control_loop_sha256=hashlib.sha256(control_loop(original)).hexdigest())


def variant_proof(run,protocol,result):
    require(protocol.get('event_reconstruction_mode')==VARIANT,'Actual actor protocol did not select endpoint_only')
    sensor=Path(run)/'migration_audit/sensor_episode_000';packets=read(sensor/'packet_audit.json')
    if result.get('failure_class')=='initialization_failure':
        require(not packets and not (sensor/'worker_start.json').exists(),'Zero-step initialization failure must not invent worker evidence')
        return dict(status='NO_GENERATION_INITIALIZATION_FAILURE',variant=VARIANT,packets=0)
    worker=read(sensor/'worker_start.json')
    require(worker.get('reconstruction_mode')==VARIANT and worker.get('intermediate_frames')==0,
        'Actual worker did not prove endpoint-only/zero intermediate frames')
    require(packets and all(p.get('reconstruction_mode')==VARIANT and p.get('intermediate_frames')==0 for p in packets),
        'Every actual parent packet must retain endpoint-only/zero-intermediate evidence')
    require(all(p.get('dvs_update_calls')==i and p.get('endpoint_reads')==i+1 for i,p in enumerate(packets)),
        'Endpoint-only packet counters must show exactly one DVS update per new actual endpoint after reset')
    return dict(status='PASS',variant=VARIANT,packets=len(packets),worker_start_sha256=sha(sensor/'worker_start.json'),
        packet_audit_sha256=sha(sensor/'packet_audit.json'),intermediate_frames=0,
        endpoint_reads=len(packets),dvs_update_calls=len(packets)-1,per_endpoint_dvs_counter_pass=True,worker=worker)


def load_base(directory):
    directory=Path(directory).resolve();path=directory/'collect_fair_stage.py';require(path.is_file(),'Missing original read-only collector helpers')
    sys.path.insert(0,str(directory));spec=importlib.util.spec_from_file_location('endpoint_base_collector',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def collect(args):
    root=args.root.resolve();main_root=args.main_blind_root.resolve()
    require(root!=main_root and main_root not in root.parents and root not in main_root.parents,'Use an independent sibling ablation root')
    require(not (root/'FAIR_COLLECTION_COMPLETE.json').exists(),'Endpoint-only stage is already sealed')
    main=main_reference(main_root,portable=True);spec=read(root/'EVALUATION_PROTOCOL.json');verify_grid(spec,main)
    base=load_base(args.analysis_dir);planned=base.expected_rows(spec);begin=time.monotonic()
    require(len(planned)==128,'Original queue helper must see exactly128 registered actual actors')
    while True:
        done=base.finished_progress(root,spec,args.job_result_file,args.recovery_plan)
        if done is not None:break
        if not args.wait:raise RuntimeError('Not all128 planned actors/jobs are closed; no ablation score emitted')
        if time.monotonic()-begin>args.timeout_s:raise TimeoutError('Endpoint-only stage remains incomplete')
        time.sleep(5)
    executions,jobs,shards,scheduling=done;base.verify_inputs(spec);delta=source_delta_proof(root,spec,main);rows=[];records=[];proofs=[]
    for pair,case in planned.items():
        row,record=base.actor_record(root,spec,case,pair[1],executions[pair])
        logical=key(row);reference=main['records'][logical]
        require(record['result']['initial']==reference['result']['initial'],'Actual endpoint-only reset state differs from paired main actor')
        require(record['initialization']==reference['initialization'],'Actual one-frameRGB-D initialization differs from paired main actor')
        proof=variant_proof(record['output'],record['protocol'],record['result'])
        row['variant']=VARIANT;record['variant']=VARIANT;rows.append(row);records.append(record)
        proofs.append(dict(case_id=row['case_id'],physical_case_id=row['physical_case_id'],seed=row['seed'],endpoint_hz=row['endpoint_hz'],**proof))
    require(len(rows)==128 and {key(r) for r in rows}==set(main['rows']) and all(r['method']=='ours' and r['frontend_id']==FRONTEND for r in rows),
        'Actual endpoint-only actor set is not the exact full registered grid')
    metadata=dict(stage='ablation',variant=VARIANT,actual_planned_actors=128,physical_seed_blocks=32,
        main_blind_root=str(main_root),main_blind_reference=main['hashes'],new_independent_throwing_blocks=0,
        sample_unit='The same32 held-out physical seed blocks reused across four rates;128 new actors are not128 independent blocks')
    groups=base.group_rows(rows)
    for name,encoding in [('FAIR_EPISODES.csv','utf-8'),('消融逐集结果.csv','utf-8-sig')]:
        with (root/name).open('w',encoding=encoding,newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    write(root/'ENDPOINT_VARIANT_PROOFS.json',dict(status='PASS',variant=VARIANT,actual_actors=128,records=proofs))
    write(root/'ENDPOINT_SOURCE_DELTA_PROOF.json',delta)
    write(root/'DATA_COMPLETION_RECEIPT.json',dict(status='PASS',valid_completed_unique_episodes=128,
        frozen_source_policy_profile_hashes_verified=True,all_shards_checked=shards,
        selected_normal_exit_job_receipts=[dict(path=str(path),sha256=sha(path)) for path in jobs],
        scheduling_audit=scheduling,all_task_failures_retained=True,**metadata))
    write(root/'PAIRED_PORTABLE.json',dict(complete=True,violations=[],records=records,metadata=metadata,
        scope='Independent endpoint-only ablation; main blind320 is referenced unchanged, not rerun or weakened'))
    write(root/'FAIR_STAGE_RESULTS.json',dict(status='PASS',complete=True,metadata=metadata,rows=rows,groups=groups,
        primary_methods=['ours'],variants=[VARIANT],reference_methods=[],all_task_failures_retained=True))
    import subprocess
    with (root/'audit_fair_raw.final.log').open('w') as log:
        subprocess.run([sys.executable,str(Path(args.analysis_dir)/'audit_fair_raw.py'),'--root',str(root)],
            cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    require(read(root/'FINAL_AUDIT.json')['status']=='PASS','Original raw event audit did not pass')
    marker=dict(status='PASS',actual_episodes=128,physical_seed_blocks=32,stage='ablation',variant=VARIANT,
        all_actual_jobs_normal_exit=not bool(scheduling.get('infrastructure_recovery')),
        all_final_actor_attempts_normal_exit=True,historical_infrastructure_failures_retained=bool(scheduling.get('infrastructure_recovery')),
        all_task_failures_retained=True,raw_manifest_sha256=sha(root/'SHA256SUMS'),
        final_audit_sha256=sha(root/'FINAL_AUDIT.json'),results_sha256=sha(root/'FAIR_STAGE_RESULTS.json'),
        variant_proofs_sha256=sha(root/'ENDPOINT_VARIANT_PROOFS.json'),
        source_delta_proof_sha256=sha(root/'ENDPOINT_SOURCE_DELTA_PROOF.json'),main_blind_reference=main['hashes'])
    write(root/'FAIR_COLLECTION_COMPLETE.json',marker);print(json.dumps(marker))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--main-blind-root',type=Path,required=True)
    parser.add_argument('--analysis-dir',type=Path,default=Path(__file__).resolve().parents[1]/'analysis')
    parser.add_argument('--wait',action='store_true');parser.add_argument('--timeout-s',type=float,default=86400)
    parser.add_argument('--job-result-file',action='append');parser.add_argument('--recovery-plan')
    args=parser.parse_args()
    try:collect(args)
    except (ValueError,AssertionError,KeyError,OSError,RuntimeError,TimeoutError) as error:
        parser.exit(2,'NOT_READY_ENDPOINT_ABLATION: '+str(error)+'\nNo incomplete ablation is marked complete.\n')


if __name__=='__main__':main()
