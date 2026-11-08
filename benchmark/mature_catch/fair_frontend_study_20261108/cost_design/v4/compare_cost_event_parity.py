"""Validate actual task-policy queries on one strictly matched cost trajectory.

raw_exact is the default and additionally requires identical raw event bytes.
task_functional retains raw differences while requiring every actual policy
query to match exactly. This is cost instrumentation acceptance, not utility
success or generator fidelity scoring.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cost_ledger import sha256_file
from query_witness import compare_raw_events,canonical_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['raw_exact','task_functional'],default='raw_exact')
    args=p.parse_args()
    if args.output.exists():
        p.error('Parity receipt already exists')
    left_path=args.legacy/'COST_RESULT.json'
    right_path=args.native/'COST_RESULT.json'
    left=json.loads(left_path.read_text())
    right=json.loads(right_path.read_text())
    for value in (left,right):
        if value['status']!='MEASURED_FIXED_TRAJECTORY' or not value['fixed_capture_budget_pass']:
            p.error('Only physically matched, complete-horizon actual cost runs are admissible')
        if value['profiled_worker']:
            p.error('Validate the original-worker path before a detailed profile')
    if left['acquisition_mode']!='legacy' or right['acquisition_mode']!='required':
        p.error('Compare legacy actual acquisition with the required-only native path')
    if left['audit_hdf_and_demo_output_suppressed'] or not right['audit_hdf_and_demo_output_suppressed']:
        p.error('Legacy must retain audit/presentation; native must suppress those explicit nonalgorithm outputs')
    fields=('method','source_rate_hz','frontend_env','reference_trace_sha256','physics_reference_sha256',
            'physics_reference_kind','source_meta_sha256',
            'original_policy_sha256','frozen_core_files','frozen_sensor_profile_sha256',
            'warmup_simulated_s','horizon_simulated_s','host','equivalence_kind','instrumentation_sha256')
    for field in fields:
        if left[field]!=right[field]:
            p.error('Mismatch outside acquisition/audit suppression: '+field)
    if left['equivalence_kind']!=args.kind:
        p.error('Requested equivalence kind must be predeclared in both actual runs')
    if left['worker_start']['gpu_uuid']!=right['worker_start']['gpu_uuid']:
        p.error('Output-equivalence validation must use the same actual GPU')
    left_raw=args.legacy/'runner_output/migration_audit/sensor_episode_000/raw_events.npz'
    right_raw=args.native/'runner_output/migration_audit/sensor_episode_000/raw_events.npz'
    with np.load(left_raw) as f:
        a=f['events']
    with np.load(right_raw) as f:
        b=f['events']
    raw=compare_raw_events(a,b,np)
    witness_paths=[directory/'ACTOR_QUERY_WITNESS.json' for directory in (args.legacy,args.native)]
    witnesses=[json.loads(path.read_text()) for path in witness_paths]
    for witness,result,path in zip(witnesses,(left,right),witness_paths):
        if sha256_file(path)!=result['actor_query_witness_sha256']:
            p.error('Actual policy-query witness file hash mismatch')
        if witness['status']!='COMPLETE' or witness['error'] is not None or not witness['payload']:
            p.error('Actual full policy query witness is incomplete')
        payload=witness['payload']
        if payload['query_payload_sha256']!=canonical_sha256(payload['queries']):
            p.error('Actual query witness payload hash mismatch')
        if payload['query_payload_sha256']!=result['actor_query_payload_sha256']:
            p.error('Cost result does not describe this actual query witness')
        if payload['query_count']!=len(payload['queries']) or payload['query_count']!=result['actual_matched_control_steps']:
            p.error('Actual query count does not match the physical control count')
        expected=result['physics_qpos_qvel_time_checks']
        if [q['query_index'] for q in payload['queries']]!=list(range(len(expected))):
            p.error('Actual query order is incomplete')
        if [q['time_s'] for q in payload['queries']]!=[step['t_before'] for step in expected]:
            p.error('Actual query timestamps differ from the physical control schedule')
    functional=(witnesses[0]['payload']['queries']==witnesses[1]['payload']['queries'])
    runner_hash=left['instrumentation_sha256']['run_fixed_trajectory_cost.py']
    if runner_hash!=right['instrumentation_sha256']['run_fixed_trajectory_cost.py']:
        p.error('Cost launcher code changed between equivalence runs')
    accepted=functional and (args.kind=='task_functional' or raw['raw_exact'])
    status=('PASS_TASK_FUNCTIONAL_EQUIVALENCE' if args.kind=='task_functional' else 'PASS_RAW_AND_TASK_EQUIVALENCE') if accepted else 'FAIL_COST_EQUIVALENCE'
    result=dict(status=status,equivalence_kind=args.kind,
        equivalence_protocol='actual_query_bytes_v1',equivalence_checker_sha256=sha256_file(__file__),
        comparison='Actual full per-query task-policy equivalence on the fixed public cost trajectory; raw-event equality separately recorded',
        functional_query_exact=functional,
        actor_query_payload_sha256=witnesses[0]['payload']['query_payload_sha256'] if functional else None,
        actual_query_count=witnesses[0]['payload']['query_count'],
        raw_exact=raw['raw_exact'],raw_comparison=raw,
        raw_reference_content_sha256=raw['left']['content_sha256'],
        method=right['method'],rate_hz=right['source_rate_hz'],frontend_env=right['frontend_env'],
        reference_trace_sha256=right['reference_trace_sha256'],source_meta_sha256=right['source_meta_sha256'],
        physics_reference_sha256=right['physics_reference_sha256'],physics_reference_kind=right['physics_reference_kind'],
        horizon_s=right['horizon_simulated_s'],warmup_s=right['warmup_simulated_s'],
        cost_runner_sha256=runner_hash,instrumentation_sha256=right['instrumentation_sha256'],
        event_shape=list(a.shape),event_dtype=str(a.dtype),ordered_events_exact=raw['raw_exact'],
        same_gpu_uuid=right['worker_start']['gpu_uuid'],same_host=right['host'],
        source_results={str(f):sha256_file(f) for f in (left_path,right_path,left_raw,right_raw,*witness_paths)},
        scope='Cost-only fixed-trajectory instrumentation acceptance; not a generator-fidelity or real-camera claim')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=status,events=len(a),method=right['method'],rate_hz=right['source_rate_hz'],
        functional_query_exact=functional,raw_exact=raw['raw_exact']),indent=2))
    if not accepted:
        raise SystemExit(1)


if __name__=='__main__':
    main()
