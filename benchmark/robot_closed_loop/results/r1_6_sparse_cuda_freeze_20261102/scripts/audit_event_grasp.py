"""Read-only causal/state/geometry audit after the physical run has finished."""
from pathlib import Path
import json,csv,hashlib,math,sys,argparse
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_R2_kinematic_fit_20261102e'))
parser.add_argument('--episode-dir',default='R2_kinematic_fit')
args=parser.parse_args();root=args.root
plan=json.loads((root/'R2_plan.json').read_text())
progress=json.loads((root/'R2_progress.json').read_text());assert progress['status']=='completed' and progress['exit_code']==0
ep=root/args.episode_dir/'episodes/episode_000'
with (ep/'episode.csv').open() as f:row=next(csv.DictReader(f))
strict=json.loads(row['strict_score']);frames=[json.loads(l) for l in (ep/'state_control_trace.jsonl').open()]
anchor=np.asarray(strict['grasp_relative_anchor_m']) if strict['grasp_relative_anchor_m'] is not None else None
violations=0;future=0;wrong_source=0;fit_rows=0;qualification_mismatch=0;max_clearance_residual=0.;max_slip_residual=0.
qualified=[];all_source=set()
for frame in frames:
 p=np.asarray(frame['object']['position_m']);w,x,y,z=frame['object']['orientation_wxyz']
 rz=np.asarray([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)])
 clearance=float(p[2]-.02*np.abs(rz).sum())
 score=frame['score'];max_clearance_residual=max(max_clearance_residual,abs(clearance-score['bottom_clearance_m']))
 relative=p-np.asarray(frame['tcp_pad_center_m'])
 slip=None if anchor is None else float(np.linalg.norm(relative-anchor))
 if score['relative_slip_m'] is not None:
  max_slip_residual=max(max_slip_residual,abs(slip-score['relative_slip_m']))
 qualified_here=bool(frame['contacts']['bilateral'] and not frame['contacts']['support'] and clearance>=.08 and slip is not None and slip<=.01)
 qualification_mismatch+=int(qualified_here!=score['qualified'])
 if qualified_here:qualified.append({'time_s':frame['time_s'],'clearance_m':clearance,'slip_m':slip,'hold_s':score['continuous_hold_s']})
 obs=frame.get('controller_observation_state')
 if obs:
  future+=int(float(obs['time_s'])>float(frame['time_s'])-.01+1e-8)
 state_source=frame.get('state_source');all_source.add(state_source)
 wrong_source+=int(state_source!='direct_hdr_events_controller')
 status=frame.get('event_observation') or {};diag=status.get('estimator_diagnostics') or {}
 if diag.get('position_source')=='causal_fixed_deceleration_fit':fit_rows+=1
assert future==0 and wrong_source==0
assert max_clearance_residual<1e-10 and max_slip_residual<1e-10 and qualification_mismatch==0
accepted=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_r1_5_sparse_time_repair_20261101a/source/benchmark/robot_closed_loop')
snapshot=Path(plan['benchmark_source'])/'benchmark/robot_closed_loop'
unchanged={}
for name in ['scoring.py','grip_gate.py','robot_env.py','predictor.py','event_state_model.py']:
 a=hashlib.sha256((accepted/name).read_bytes()).hexdigest();b=hashlib.sha256((snapshot/name).read_bytes()).hexdigest();assert a==b,name;unchanged[name]=a
modified={}
for name,digest in plan['modified_files'].items():
 measured=hashlib.sha256((snapshot/name).read_bytes()).hexdigest();assert measured==digest,name;modified[name]=digest
core_expected=json.loads(Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_sparse_cuda_acceptance_20261102c/local_cuda_20261102a/acceptance_summary.json').read_text())['core_canonical_LF_sha256']
for name,digest in core_expected.items():
 assert hashlib.sha256((Path(plan['core_source'])/'genesis_event_plugin'/name).read_text().encode()).hexdigest()==digest,name
checkpoint=Path(plan['command'][plan['command'].index('--event-model-checkpoint')+1]);assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==plan['checkpoint_sha256']
success=row['success_strict_r0_5']=='True'
if success:
 assert float(row['hold_s'])>=.5-1e-12 and strict['success'] and qualified
 assert min(q['clearance_m'] for q in qualified)>=.08 and max(q['slip_m'] for q in qualified)<=.01
 assert max(q['hold_s'] for q in qualified)>=.499-1e-9
with np.load(ep/'direct_hdr_events_1khz_2khz.npz') as data:
 counts={k:len(data[k]) for k in data.files}
result=dict(status='PASS_strict_event_grasp' if success else 'R2_failed_audit_consistent',seed=int(row['seed']),
 speed_m_s=float(row['launch_speed_m_s']),y_m=float(row['start_y_m']),strict_success=success,
 close_start_s=float(row['close_start_s']),first_contact_s=float(row['first_contact_s']) if row['first_contact_s'] else None,
 lift_start_s=float(row['lift_start_s']) if row['lift_start_s'] else None,hold_s=float(row['hold_s']),failure=row['failure_primary'],
 strict_score=strict,qualified_trace_rows=len(qualified),qualified_min_clearance_m=min((q['clearance_m'] for q in qualified),default=None),
 qualified_max_slip_m=max((q['slip_m'] for q in qualified),default=None),causal_time_violations=future,state_sources=sorted(all_source),
 motion_fit_trace_rows=fit_rows,qualified_flag_mismatch=qualification_mismatch,clearance_recomputation_max_error_m=max_clearance_residual,
 slip_recomputation_max_error_m=max_slip_residual,checkpoint_sha256=plan['checkpoint_sha256'],unchanged_task_and_model_files=unchanged,
 event_counts=counts,progress=progress,GT_role='Evaluator geometry/contact/score only; no GT into event estimator or control',
 source_delta=plan.get('only_change_from_F',plan.get('estimator_change','causal benchmark adapter')),
 modified_benchmark_files=modified,sparse_core_hashes_match=True,trace_sha256=hashlib.sha256((ep/'state_control_trace.jsonl').read_bytes()).hexdigest())
(root/'strict_event_grasp_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
