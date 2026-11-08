"""Validate native/no-audit cost execution against a retained legacy run.

This is an output-equivalence check, not a task success or event-quality score.
Both runs must have matched every archived qpos/qvel/time point.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cost_ledger import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
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
            'original_policy_sha256','warmup_simulated_s','horizon_simulated_s','host')
    for field in fields:
        if left[field]!=right[field]:
            p.error('Mismatch outside acquisition/audit suppression: '+field)
    if left['worker_start']['gpu_uuid']!=right['worker_start']['gpu_uuid']:
        p.error('Output-equivalence validation must use the same actual GPU')
    left_raw=args.legacy/'runner_output/migration_audit/sensor_episode_000/raw_events.npz'
    right_raw=args.native/'runner_output/migration_audit/sensor_episode_000/raw_events.npz'
    with np.load(left_raw) as f:
        a=f['events']
    with np.load(right_raw) as f:
        b=f['events']
    if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b):
        p.error('Raw event dtype/shape/ordered rows differ; do not publish the required-only native cost')
    runner_hash=left['instrumentation_sha256']['run_fixed_trajectory_cost.py']
    if runner_hash!=right['instrumentation_sha256']['run_fixed_trajectory_cost.py']:
        p.error('Cost launcher code changed between equivalence runs')
    result=dict(status='PASS',comparison='legacy with audit vs native required-only without presentation/HDF audit',
        method=right['method'],rate_hz=right['source_rate_hz'],frontend_env=right['frontend_env'],
        reference_trace_sha256=right['reference_trace_sha256'],source_meta_sha256=right['source_meta_sha256'],
        physics_reference_sha256=right['physics_reference_sha256'],physics_reference_kind=right['physics_reference_kind'],
        horizon_s=right['horizon_simulated_s'],warmup_s=right['warmup_simulated_s'],
        cost_runner_sha256=runner_hash,event_shape=list(a.shape),event_dtype=str(a.dtype),ordered_events_exact=True,
        same_gpu_uuid=right['worker_start']['gpu_uuid'],same_host=right['host'],
        source_results={str(f):sha256_file(f) for f in (left_path,right_path,left_raw,right_raw)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',events=len(a),method=right['method'],rate_hz=right['source_rate_hz']),indent=2))


if __name__=='__main__':
    main()
