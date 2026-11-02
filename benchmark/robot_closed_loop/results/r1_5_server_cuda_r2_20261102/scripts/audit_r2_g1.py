"""Audit completed episode scores and causal observation times, without rerunning."""
import pathlib,json,csv,hashlib,math
root=pathlib.Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_cuda_r2_server_20261102a')
out=root/'R2_events_seed2026110200'
plan=json.loads((root/'R2_plan.json').read_text())
report={'episodes':[],'plan':plan,'score_source':'original 0.5ms strict scorer; trace audit at stored 1ms cadence'}
def value(x):
 if x=='':return None
 if x=='True':return True
 if x=='False':return False
 try:return json.loads(x)
 except Exception:return x
for folder in sorted((out/'episodes').glob('episode_*')):
 if not (folder/'episode.csv').is_file():continue
 with (folder/'episode.csv').open() as f:row={k:value(v) for k,v in next(csv.DictReader(f)).items()}
 summary=row['strict_score'];episode={'episode':folder.name,'seed':row['seed'],'speed_m_s':row['launch_speed_m_s'],
  'y_m':row['start_y_m'],'success':row['success_strict_r0_5'],'hold_s':row['hold_s'],
  'first_contact_s':row['first_contact_s'],'first_bilateral_contact_s':row['first_bilateral_contact_s'],
  'lift_start_s':row['lift_start_s'],'failure':row['failure_primary'],'maximum_clearance_m':row['maximum_target_bottom_clearance_m'],
  'strict_score':summary,'trace_rows':0,'observation_rows':0,'fresh_rows':0,'causal_time_violations':0,
  'qualified_trace_rows':0,'qualified_clearance_min_m':None,'qualified_slip_max_m':None,'qualified_support_rows':0}
 trace=folder/'state_control_trace.jsonl';minimum=float('inf');maximum=0.
 with trace.open() as f:
  for line in f:
   frame=json.loads(line);episode['trace_rows']+=1
   obs=frame.get('controller_observation_state')
   if obs:
    episode['observation_rows']+=1
    if float(obs['time_s'])>float(frame['time_s'])-.01+1e-8:episode['causal_time_violations']+=1
   status=frame.get('event_observation') or {}
   if status.get('state_status')=='fresh':episode['fresh_rows']+=1
   score=frame.get('score') or {}
   if score.get('qualified'):
    episode['qualified_trace_rows']+=1
    minimum=min(minimum,float(score['bottom_clearance_m']))
    maximum=max(maximum,float(score['relative_slip_m']))
    episode['qualified_support_rows']+=int(bool(score['support_contact']))
 episode['qualified_clearance_min_m']=minimum if math.isfinite(minimum) else None
 episode['qualified_slip_max_m']=maximum if episode['qualified_trace_rows'] else None
 assert episode['causal_time_violations']==0,episode
 if episode['success']:
  assert episode['hold_s']>=.5-1e-12
  assert episode['qualified_trace_rows']>0 and episode['qualified_clearance_min_m']>=.08-1e-10
  assert episode['qualified_slip_max_m']<=.01+1e-10 and episode['qualified_support_rows']==0
 episode['trace_sha256']=hashlib.sha256(trace.read_bytes()).hexdigest()
 report['episodes'].append(episode)
report['strict_successes']=sum(bool(e['success']) for e in report['episodes'])
report['completed_episodes']=len(report['episodes'])
report['R2_exit']=(root/'R2_events.exit').read_text().strip() if (root/'R2_events.exit').is_file() else None
(root/'R2_independent_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='plan'}))
