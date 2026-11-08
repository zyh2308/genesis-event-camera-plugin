"""Export measured fixed-window cost runs; never use episode lifetime as cost.

Audit/profile/equivalence passes remain identifiable, but only actual native
primary-validated runs can be used for the formal effectiveness/cost plot.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from cost_ledger import sha256_file


SCOPE = 'native_capture+generator+frontend+actor+common_physics'


def row_from_result(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if (value.get('status') != 'MEASURED_FIXED_TRAJECTORY' or
            value.get('cost_only_shadow') is not True or value.get('utility_result') is not False):
        raise ValueError(str(path)+': expected an actual measured cost-only shadow run')
    begin, end = value['warmup_simulated_s'], value['horizon_simulated_s']
    duration = value['measured_simulated_duration_s']
    if abs(begin-.32)>1e-8 or abs(end-1.12)>1e-8 or abs(duration-.8)>1e-8:
        raise ValueError(str(path)+': formal table requires the registered .32..1.12s window')
    timing = value['timing']
    wall = timing['hot_wall_s']
    checks = value['physics_qpos_qvel_time_checks']
    complete = bool(value['fixed_capture_budget_pass'] and checks and
        abs(checks[-1]['t']-end)<1e-8 and value['actual_matched_control_steps']==len(checks) and
        all(x['qpos_max_abs_error']<=1e-6 and x['qvel_max_abs_error']<=1e-5 for x in checks))
    if not complete or not isinstance(wall,(int,float)) or wall<=0:
        raise ValueError(str(path)+': incomplete/invalid fixed physical window')
    worker = value.get('worker_start') or {}
    if not worker.get('gpu_uuid') or not value.get('host'):
        raise ValueError(str(path)+': missing actual host/GPU identity')
    audit_suppressed = value['audit_hdf_and_demo_output_suppressed']
    native = value['acquisition_mode']=='required' and audit_suppressed
    instrumented = value['profiled_worker']
    valid = bool(value['eligible_as_native_primary'] and native and
        value['acquisition_parity_verified'] and value['functional_query_exact'] is True and
        value['actor_query_payload_sha256'] and value['equivalence_kind'] in ('raw_exact','task_functional') and
        (value['equivalence_kind']=='task_functional' or value['raw_event_exact'] is True) and not instrumented)
    if valid and value['actor_query_count']!=len(checks):
        raise ValueError(str(path)+': primary query witness does not cover every physical control')
    stages = timing['hot_stages']
    def exclusive(*names):
        return sum(stages.get(name,{}).get('exclusive_wall_s',0.) for name in names)
    row = dict(run_id=path.parent.name,method=value['method'],source_hz=value['source_rate_hz'],
        frontend_id=value['frontend_id'],repeat_id=value['repeat_id'],
        measurement_start_s=begin,measurement_end_s=end,simulated_s=duration,measured_wall_s=wall,
        wall_per_simulated_s=wall/duration,rtf=duration/wall,cost_scope=SCOPE,
        complete_window=complete,native_only=native,
        audit_included=(not audit_suppressed or (value['method']=='v2e' and value['acquisition_mode']=='legacy')),
        primary_validated=valid,instrumented=instrumented,
        equivalence_kind=value['equivalence_kind'],functional_query_exact=value['functional_query_exact'],
        actor_query_payload_sha256=value['actor_query_payload_sha256'],actor_query_count=value['actor_query_count'],
        raw_event_exact=value['raw_event_exact'] if value['raw_event_exact'] is not None else '',
        equivalence_receipt_sha256=value['native_parity_receipt_sha256'],
        reference_seed=value['reference_seed'],reference_trace_sha256=value['reference_trace_sha256'],
        physics_reference_sha256=value['physics_reference_sha256'],physics_reference_kind=value['physics_reference_kind'],
        source_snapshot_sha256=value['source_meta_sha256'],source_commit=value['source_code_commit'],
        policy_sha256=value['original_policy_sha256'],host=value['host'],gpu_uuid=worker['gpu_uuid'],
        common_physics_wall_s=exclusive('common_physics_driver_inclusive'),
        common_pid_wall_s=exclusive('common_pid_control'),
        acquisition_HDR_depth_seg_wall_s=exclusive('ours_HDR_depth_seg_acquisition'),
        acquisition_LDR_wall_s=exclusive('v2e_native_LDR_acquisition'),
        audit_HDR_depth_seg_wall_s=exclusive('v2e_audit_only_HDR_depth_seg'),
        geometry_pose_wall_s=exclusive('ours_geometry_pose_acquisition'),
        audit_geometry_pose_wall_s=exclusive('v2e_audit_only_geometry_poses'),
        array_readback_wall_s=exclusive('capture_array_readback','robot_array_readback'),
        worker_RPC_wall_s=exclusive('worker_RPC_send_serialization','worker_RPC_receive_wait',
            'worker_RPC_array_wire_encode','worker_RPC_array_wire_decode'),
        frontend_wall_s=exclusive('frontend_observe'),actor_wall_s=exclusive('original_actor_inference'),
        shared_robot_proprioception_wall_s=exclusive('shared_robot_proprioception'),
        parent_unattributed_wall_s=timing['unattributed_parent_hot_wall_s'],
        all_parent_exclusive_spans_wall_s=timing['measured_parent_exclusive_sum_s'],
        cold_setup_wall_s=value['cold_setup_until_hot_s'],
        process_wall_with_setup_and_close_s=value['process_wall_including_setup_and_close_s'],
        gpu_cumulative_time_s='',gpu_time_status='unknown; no activity profiler collected',
        cost_result_path=str(path.resolve()),cost_result_sha256=sha256_file(path))
    return {k:('true' if v else 'false') if isinstance(v,bool) else v for k,v in row.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True,help='Actual cost directory; repeat for each run')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.output.with_suffix('.manifest.json').exists():
        p.error('Use a fresh table output path')
    rows=[]
    seen=set()
    primary_keys=set()
    primary_references=set()
    primary_equivalence_kinds=set()
    primary_query_fingerprints={}
    for directory in args.run:
        path=directory/'COST_RESULT.json'
        if path.resolve() in seen:
            p.error('Same measured result listed twice: '+str(path))
        seen.add(path.resolve())
        try:
            row=row_from_result(path)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            p.error(str(exc))
        if row['primary_validated']=='true':
            primary_equivalence_kinds.add(row['equivalence_kind'])
            if len(primary_equivalence_kinds)>1:
                p.error('Predeclare one equivalence kind for the whole formal cost comparison')
            point=tuple(row[k] for k in ('method','source_hz','frontend_id'))
            fingerprint=row['actor_query_payload_sha256']
            if point in primary_query_fingerprints and primary_query_fingerprints[point]!=fingerprint:
                p.error('Formal repeats at one operating point must match one actual policy-query fingerprint')
            primary_query_fingerprints[point]=fingerprint
            primary_references.add((row['physics_reference_kind'],row['physics_reference_sha256']))
            if len(primary_references)>1:
                p.error('Formal primary rows must share one public physical trajectory; split different references into separate tables')
            key=tuple(row[k] for k in ('method','source_hz','frontend_id','repeat_id',
                'reference_trace_sha256','physics_reference_sha256','source_snapshot_sha256','host','gpu_uuid'))
            if key in primary_keys:
                p.error('Repeated primary measurement ID at one operating point; provide unique repeat_id')
            primary_keys.add(key)
        rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest=dict(table_sha256=sha256_file(args.output),rows=len(rows),
        primary_rows=sum(r['primary_validated']=='true' for r in rows),
        window_s=[.32,1.12],simulated_duration_s=.8,
        cost_scope=SCOPE,source='Actual COST_RESULT.json only; no episode lifetime extrapolation',
        stage_accounting='Exclusive parent host-wall spans; worker compute is already inside RPC wait. Not all exclusive stages are separate CSV columns.',
        detailed_profile='instrumented means optional detailed worker profile, not the minimal parent stopwatch instrumentation shared by primary runs',
        functional_acceptance='Every primary repeat matches a validated full actual policy-query byte witness; raw_event_exact is separately reported, never silently assumed',
        sources={r['cost_result_path']:r['cost_result_sha256'] for r in rows})
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(table=str(args.output),rows=len(rows),primary_rows=manifest['primary_rows']),indent=2))


if __name__=='__main__':
    main()
