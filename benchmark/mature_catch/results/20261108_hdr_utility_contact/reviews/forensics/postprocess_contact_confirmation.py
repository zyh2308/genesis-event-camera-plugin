"""Four-method independent confirmation, preserving the original physical score.

Ours/V2E are the primary comparison. Init-only and native noisy-GT observations
are separate input-ablation and feasibility references, never pooled trials.
"""
from pathlib import Path
import argparse,csv,hashlib,json,subprocess,sys,time

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def method_contract(method,protocol,expected_window_ms,expected_fit):
 assert method in ('ours','v2e','init_only','state_reference')
 if method=='state_reference':
  assert protocol['event_control'] is False,'GT reference must use original observations'
  assert protocol['RGB_frame_budget']==protocol['control_depth_frame_budget']==0,'GT reference has no RGB-D control input'
  assert not protocol.get('initialization_only',False)
  return dict(role='native policy with original noisy GT observations; feasibility reference only',event_frontend_active=False,RGB_frame_budget=0,control_depth_frame_budget=0)
 assert protocol['event_control'] is True
 assert protocol['RGB_frame_budget']==protocol['control_depth_frame_budget']==1
 if method=='init_only':
  assert protocol['initialization_only'] is True and protocol['event_updates_disabled'] is True
  assert float(protocol.get('event_observation_window_ms',0.)) in (0.,expected_window_ms)
  return dict(role='one shared RGB-D initialization, no later events; causal input ablation',event_frontend_active=False,RGB_frame_budget=1,control_depth_frame_budget=1)
 assert not protocol['initialization_only'] and not protocol['event_updates_disabled']
 assert protocol['event_generator']==method
 assert bool(protocol['event_flight_fit'])==expected_fit
 assert float(protocol.get('event_observation_window_ms',0.))==expected_window_ms
 return dict(role='primary event-generator utility comparison',event_frontend_active=True,RGB_frame_budget=1,control_depth_frame_budget=1)


def audit_init_only(run,protocol,expected_window_ms):
 import numpy as np
 sensor=Path(run)/'migration_audit/sensor_episode_000'
 assert read(sensor/'packet_audit.json')==[],'init-only must not generate event packets'
 raw=np.load(sensor/'raw_events.npz')['events'];assert raw.shape==(0,4)
 assert read(sensor/'event_samples.json')==[],'init-only must not append event detections'
 window_file=sensor/'event_observation_windows.json'
 if window_file.exists():
  doc=read(window_file);assert doc['windows']==[]
  assert float(doc['observation_window_ms'])==float(protocol.get('event_observation_window_ms',0.))
 assert not (sensor/'sparse_hdr_endpoints.h5').exists(),'init-only must not acquire a later HDR event stream'
 return dict(status='PASS',mode='init_only',window_audit_applicable=False,original_packets=0,raw_events=0,
             window_artifact='present_empty' if window_file.exists() else 'not_required',output=str(run))


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--analyzer',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);a=p.parse_args();r=a.root;s=read(r/'EVALUATION_PROTOCOL.json');start=time.monotonic()
 shards=sorted({c['shard'] for c in s['cases']})
 while True:
  for f in r.glob('*.result.json'):
   x=read(f)
   if x['status']=='failed':raise RuntimeError('Resource or actor failure retained: '+str(f)+' '+str(x))
  if all((r/f'shard{q}_complete.json').exists() for q in shards) and len(list(r.glob('*.result.json')))==len(shards):break
  if time.monotonic()-start>5400:raise TimeoutError('Planned data incomplete, no final score')
  time.sleep(10)
 required={'ours','v2e','init_only','state_reference'}
 assert all(set(c['method_order'])==required and len(c['method_order'])==4 for c in s['cases'])
 assert len({c['seed'] for c in s['cases']})==len(s['cases']),'confirmation seed repeated'
 expected={(c['case_id'],m) for c in s['cases'] for m in c['method_order']};found={}
 for q in shards:
  assert read(r/f'shard{q}_complete.json')['postflight_hashes_pass']
  for x in read(r/f'shard{q}_progress.json'):
   key=x['case_id'],x['method'];assert key in expected and key not in found and x['exit_code']==0;found[key]=x
 assert set(found)==expected and len(found)==s['expected_episodes']
 expected_window_ms=float(s.get('shared_estimator_overrides',{}).get('CATCHIT_EVENT_OBSERVATION_WINDOW_MS','0'))
 assert expected_window_ms in (0.,10.)
 window_audits=[];method_audits=[]
 for x in found.values():
  run=Path(x['output']);audit=read(run/'migration_audit/adapter_audit.json');protocol=read(run/'protocol.json');c=audit['contact_calibration'];assert audit['original_mj_step_calls']==0 and audit['contact_profile']['name']=='combined';assert c['unselected_unchanged'] and c['hand_geom_ids']==list(range(33,50));assert all(z['friction']==1.5 for z in c['after'] if z['selected_hand']);assert c['after'][-1]['friction']==.95
  method=x['method'];contract=method_contract(method,protocol,expected_window_ms,s.get('shared_estimator_overrides',{}).get('CATCHIT_EVENT_FLIGHT_FIT','0')=='1')
  method_audits.append(dict(case_id=x['case_id'],method=method,**contract))
  if method=='init_only':
   window_audits.append(dict(case_id=x['case_id'],method=method,**audit_init_only(run,protocol,expected_window_ms)))
  elif method=='state_reference':
   window_audits.append(dict(case_id=x['case_id'],method=method,status='NOT_APPLICABLE',window_audit_applicable=False,
       reason='original noisy-GT observations; no event sensor or RGB-D control input',output=str(run)))
  elif expected_window_ms:
   from audit_contact_observation_windows import audit_run
   window_audits.append(dict(case_id=x['case_id'],method=method,window_audit_applicable=True,**audit_run(run,protocol,s)))
 if expected_window_ms:
  assert len(window_audits)==len(found) and all(x['status'] in ('PASS','NOT_APPLICABLE') for x in window_audits)
  (r/'OBSERVATION_WINDOW_AUDIT.json').write_text(json.dumps(dict(status='PASS',episodes=len(window_audits),observation_window_ms=expected_window_ms,records=window_audits),indent=2))
 (r/'CONFIRMATION_METHOD_CONTRACT_AUDIT.json').write_text(json.dumps(dict(status='PASS',records=method_audits,
     primary_methods=['ours','v2e'],reference_methods=['init_only','state_reference'],roles_not_pooled=True),indent=2))
 receipt={'status':'PASS','valid_completed_unique_episodes':len(found),'frozen_source_policy_profile_hashes_verified':True,'all_shards_checked':shards,'contact_profile_readback':'combined on all actors; hand-only1.5, two substeps, unchanged2ms outer step','all_successes_and_failures_retained':True,'event_observation_window_ms':expected_window_ms}
 if expected_window_ms:receipt.update(observation_window_audit='PASS',observation_window_audit_sha256=sha(r/'OBSERVATION_WINDOW_AUDIT.json'))
 (r/'DATA_COMPLETION_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
 with (r/'analyze_cadence.final.log').open('w') as log:subprocess.run([sys.executable,'-u',str(a.analyzer),'--root',str(r)],stdout=log,stderr=subprocess.STDOUT,check=True)
 data=read(r/'PAIRED_PORTABLE.json');assert data['complete'] and not data['violations']
 assert {x['mode'] for x in data['records']}==required
 assert all(x.get('matched_initial_state') is True for x in data['records'])
 data['scope']='Preregistered independent four-method confirmation; all outcomes retained. Ours/V2E/init-only use target GT only for offline scoring. The separately reported state_reference runs the original policy with original noisy-GT observations as a feasibility reference. Primary all_precontact_state metrics retain every available precontact state; separately named precontact metrics exclude grasp propagation.'
 (r/'PAIRED_PORTABLE.json').write_text(json.dumps(data,indent=2))
 with (r/'audit_cadence.final.log').open('w') as log:subprocess.run([sys.executable,'-u',str(a.audit),'--root',str(r)],stdout=log,stderr=subprocess.STDOUT,check=True)
 assert read(r/'FINAL_AUDIT.json')['status']=='PASS'
 rows=[]
 for x in data['records']:
  e=x['result'];rows.append(dict(case_id=x['case_id'],seed=x['seed'],method=x['mode'],verified_catch=x['verified_catch'],raw_success=e['success'],terminated=e['terminated'],hand_contact_steps=e['hand_contact_steps'],final_contact=x['final_hand_contact'],first_touch_s=e['first_touch'],raw_events=x.get('raw_event_count'),output=x['output']))
 for name,encoding in [('CONTACT_PAIRED_EPISODE_RESULTS.csv','utf-8'),('事件闭环逐条结果.csv','utf-8-sig')]:
  with (r/name).open('w',newline='',encoding=encoding) as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 methods=[m for m in ['ours','v2e','init_only','state_reference'] if any(x['method']==m for x in rows)]
 comparison={'status':'PASS','scope':s['selection_scope'],'rows':rows,'scores':{m:dict(n=sum(x['method']==m for x in rows),verified_catches=sum(x['verified_catch'] for x in rows if x['method']==m),contact_episodes=sum(x['hand_contact_steps']>0 for x in rows if x['method']==m)) for m in methods},'fixed_parameters':dict(camera_pitch_degrees=7,endpoint_hz=25,policy='unchanged original pretrained',contact='combined selected by GT only',event_core='unchanged frozen sparse path',init='one rendered RGB-D frame then events/proprioception',causal_flight_fit_enabled=s.get('shared_estimator_overrides',{}).get('CATCHIT_EVENT_FLIGHT_FIT','0')=='1',observation_window_ms=expected_window_ms),'all_planned_successes_and_failures_retained':True,'offline_perception_note':'Actual closed-loop trajectories diverge with actions; per-trajectory perception diagnostics do not use common query states.'}
 comparison.update(primary_comparison_methods=['ours','v2e'],reference_methods=['init_only','state_reference'],
   method_roles={m:next({k:v for k,v in x.items() if k not in ('case_id','method')} for x in method_audits if x['method']==m) for m in methods},
   independent_confirmation=True,all_reference_outcomes_retained=True)
 comparison['fixed_parameters']['init']='Ours/V2E/init-only share one rendered RGB-D frame; noisy-GT reference uses no image input'
 (r/'CONTACT_PAIRED_COMPARISON.json').write_text(json.dumps(comparison,indent=2))
 completion={'status':'PASS','actual_episodes':len(rows),'all_planned_cases_retained':True,'scores':comparison['scores'],'raw_audit_sha256':sha(r/'FINAL_AUDIT.json'),'comparison_sha256':sha(r/'CONTACT_PAIRED_COMPARISON.json')};(r/'CONTACT_PAIR_COMPLETE.json').write_text(json.dumps(completion,indent=2))
 paths=[p for p in sorted(r.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and not p.name.endswith('.final.log') and p.name!='postprocess_coordinator.log'];lines=[]
 for path in paths:
  h=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
  lines.append(h.hexdigest()+'  '+path.relative_to(r).as_posix())
 (r/'SHA256SUMS').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(completion,full_manifest_sha256=sha(r/'SHA256SUMS'),hashed_files=len(paths))),flush=True)

if __name__=='__main__':main()
