"""Supplemental timing only; never changes raw/score closeout or controller state."""
from pathlib import Path
import argparse,hashlib,json,sys

def read(p):return json.loads(Path(p).read_text())

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reconstruct_confirmation(spec,initial,K,radius,samples):
 """Replay only the existing static-to-motion decision from public samples.

 No association, event generation, fitting, physics, or release GT is used.
 The saved accepted samples are fixed. Stop at the first motion confirmation.
 """
 import importlib
 import numpy as np
 source=Path(spec['source']).resolve()
 if str(source) not in sys.path:sys.path.insert(0,str(source))
 module=importlib.import_module('event_sphere_tracker_rgbd_base')
 for name in ['event_sphere_tracker_rgbd_base','event_sphere_tracker_kalman','event_sphere_tracker_contour_q80_occlusion']:
  loaded=sys.modules[name];path=Path(loaded.__file__).resolve()
  assert path.parent==source and sha(path)==spec['runtime_files'][name+'.py'],'timing replay source mismatch '+name
 tracker=module.EventSphereTracker(K,(640,480),radius)
 tracker.gravity=np.asarray(initial['public_world_gravity_m_s2'],dtype=float)
 tracker.initialize_rgb(initial,time_s=float(initial['time_s']),available_s=float(initial['available_s']))
 for index,sample in enumerate(samples):
  tracker.samples.append(sample)
  tracker._predict(float(sample['t']))
  if tracker.motion_onset_s is not None:
   indices=[];cursor=0
   for pending in tracker.motion_suspicion:
    found=next((i for i in range(cursor,index+1) if abs(samples[i]['t']-pending['t'])<=1e-9 and
       np.allclose(np.asarray(samples[i]['position'])-tracker.static_position,pending['innovation'],rtol=0,atol=1e-8)),None)
    assert found is not None,'Cannot map actual confirmation observation'
    indices.append(found);cursor=found+1
   return dict(motion_onset_event_s=float(tracker.motion_onset_s),confirmation_sample_indices=indices,
       source='sealed RGBD-base static-to-motion rule replayed on stored public event detections only')
 return dict(motion_onset_event_s=None,confirmation_sample_indices=[],
     source='sealed RGBD-base rule: no motion confirmed from stored detections')

def timing_episode(run,protocol,spec,case_id,method):
 sensor=run/'migration_audit/sensor_episode_000'
 packets=read(sensor/'packet_audit.json');window=read(sensor/'event_observation_windows.json')
 parents=window['windows'];samples=read(sensor/'event_samples.json')
 trace=read(run/'episode_000/trace.json')
 query_onsets=[r['event_state']['motion_onset_s'] for r in trace if r.get('event_state') and r['event_state'].get('motion_onset_s') is not None]
 actual_onset=float(query_onsets[0]) if query_onsets else None
 flat=[];by_parent={int(p['parent_packet_id']):p for p in parents}
 for parent in parents:
  for child in parent['windows']:
   if child['detected'] is not None:
    s=child['detected'];flat.append(dict(sample_index=len(flat),parent_packet_id=int(parent['parent_packet_id']),
      subwindow_index=int(child['subwindow_index']),event_measurement_s=float(s['t']),
      maximum_event_s=float(s['maximum_event_s']),original_parent_available_s=float(s['available_s']),
      original_parent_processed_s=float(parent['processing_now_s'])))
 assert len(flat)==len(samples)
 for actual,s in zip(flat,samples):assert abs(actual['event_measurement_s']-s['t'])<1e-8
 confirmation=[];details_status='UNKNOWN';details_error=None;reconstructed=dict(motion_onset_event_s=actual_onset,source='UNKNOWN')
 try:
  import h5py
  with h5py.File(sensor/'sparse_hdr_endpoints.h5','r') as h:K=h['intrinsics'][:]
  reconstructed=reconstruct_confirmation(spec,read(sensor/'initialization.json'),K,protocol['sphere_radius_m'],samples)
  if query_onsets:assert reconstructed['motion_onset_event_s'] is not None and abs(actual_onset-reconstructed['motion_onset_event_s'])<1e-8
  confirmation=[flat[i] for i in reconstructed['confirmation_sample_indices']]
  details_status='RECONSTRUCTED' if confirmation else 'NO_MOTION_CONFIRMED'
 except Exception as error:
  details_error=type(error).__name__+': '+str(error)
  reconstructed=dict(motion_onset_event_s=actual_onset,source='UNKNOWN: basic trace/packet timing remains available')
 gaps=[b['event_measurement_s']-a['event_measurement_s'] for a,b in zip(confirmation,confirmation[1:])]
 try:
  if confirmation:
   assert len(confirmation)>=2 and max(gaps)<=.05+1e-9
   assert confirmation[-1]['event_measurement_s']-confirmation[0]['event_measurement_s']>=.005-1e-9
   assert abs(confirmation[-1]['event_measurement_s']-reconstructed['motion_onset_event_s'])<1e-8
 except Exception as error:
  details_status='UNKNOWN';details_error=type(error).__name__+': '+str(error);confirmation=[];gaps=[]
 is_moving=lambda row:bool(row.get('event_state') and row['event_state'].get('motion_onset_s') is not None)
 first_state=next((float(r['t_before']) for r in trace if is_moving(r)),None)
 first_policy=next((float(r['t_before']) for r in trace if is_moving(r) and r.get('policy_object_input') is not None and not r.get('held_before_observation',False)),None)
 available=max((c['original_parent_available_s'] for c in confirmation),default=None)
 earliest_poll=next((float(r['t_before']) for r in trace if available is not None and r['t_before']>=available-1e-9),None)
 anomalies=[]
 if first_state is not None and available is not None and first_state<available-1e-9:anomalies.append('Published moving state precedes reconstructed confirmation arrival')
 if first_policy is not None and (first_state is None or first_policy<first_state-1e-9):anomalies.append('Policy adoption precedes first moving state')
 period=1./float(protocol['event_endpoint_rate_hz']);parent_rows=[]
 for i,p in enumerate(packets):
  consumed=by_parent.get(i);children=consumed['windows'] if consumed else None
  parent_rows.append(dict(parent_packet_id=i,original_endpoint_s=float(p['time_s']),
    nominal_capture_interval_s=None if i==0 else float(p['time_s']-packets[i-1]['time_s']),
    original_available_s=float(p['available_s']),raw_events=int(p['events']),consumed=consumed is not None,
    original_processed_control_s=None if consumed is None else float(consumed['processing_now_s']),
    total_subwindows=None if children is None else len(children),
    nonempty_subwindows=None if children is None else sum(c['events']>0 for c in children),
    detected_subwindows=None if children is None else sum(c['detected'] is not None for c in children),
    confirmation_observations=[c for c in confirmation if c['parent_packet_id']==i],
    carry_in_subwindows=None if consumed is None else int(consumed['carry_in_window_count'])))
 return dict(case_id=case_id,method=method,endpoint_hz=float(protocol['event_endpoint_rate_hz']),
  captured_original_packets=len(packets),consumed_original_packets=len(parents),
  total_detected_subwindows=len(flat),total_nonempty_subwindows=sum(c['events']>0 for p in parents for c in p['windows']),
  original_parent_period_s=period,observation_window_ms=10.,confirmation_max_adjacent_gap_limit_s=.05,
  original_controller_poll_period_s=float(protocol['timestep'])*int(protocol['steps_per_policy']),
  motion_onset_event_s=reconstructed['motion_onset_event_s'],confirmation_observations=confirmation,
  confirmation_span_s=None if not confirmation else confirmation[-1]['event_measurement_s']-confirmation[0]['event_measurement_s'],
  confirmation_max_adjacent_gap_s=max(gaps) if gaps else None,
  confirmation_from_single_arrival_parent=bool(confirmation and len({c['parent_packet_id'] for c in confirmation})==1),
  confirmation_information_available_s=available,earliest_control_poll_after_confirmation_arrival_s=earliest_poll,
  first_control_published_moving_state_s=first_state,first_control_policy_adopted_moving_state_s=first_policy,
  moving_state_observed=first_state is not None,policy_adopted_moving_state=first_policy is not None,
  confirmation_details_status=details_status,confirmation_details_error=details_error,timing_anomalies=anomalies,
  subwindow_confirmation_observed_with_parent_period_longer_than_gap=None if details_status=='UNKNOWN' else bool(confirmation and period>.05),
  confirmation_reconstruction_source=reconstructed['source'],original_packets=parent_rows,
  timing_interpretation='Historical event confirmation is distinct from information arrival, original control query, and unheld policy adoption; no earlier controller updates or independent-render claim')


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 spec=read(r/'EVALUATION_PROTOCOL.json');recovery=read(r/'RECOVERY_MAP.json') if (r/'RECOVERY_MAP.json').exists() else {}
 records=[]
 for case in spec['cases']:
  for method in case['method_order']:
   run=Path(recovery.get(case['case_id']+'|'+method,{}).get('output',str(r/'cases'/case['case_id']/method)))
   try:record=timing_episode(run,read(run/'protocol.json'),spec,case['case_id'],method);record['status']='EXTRACTED'
   except Exception as error:record=dict(case_id=case['case_id'],method=method,status='UNAVAILABLE',error=type(error).__name__+': '+str(error))
   records.append(record)
 complete=len(records)==spec['expected_episodes'] and all(x['status']=='EXTRACTED' for x in records)
 payload=dict(status='PASS' if complete else 'PARTIAL',complete=complete,episodes=len(records),records=records,
  supplement_only=True,raw_and_score_closeout_modified=False,no_target_GT_read=True,
  confirmation_unknown_does_not_block_primary_results=True,
  interpretation='Actual trace polls and parent availability are primary; historical confirmation times are supplemental and may be UNKNOWN; no earlier online updates claimed')
 path=r/'TIMING_ACQUISITION_SUMMARY.json';path.write_text(json.dumps(payload,indent=2))
 (r/'TIMING_ACQUISITION_SUMMARY.SHA256').write_text(sha(path)+'  '+path.name+'\n')
 print(json.dumps(dict(status=payload['status'],complete=complete,
   moving_state_episodes=sum(x.get('moving_state_observed',False) for x in records),
   policy_adoption_episodes=sum(x.get('policy_adopted_moving_state',False) for x in records),
   confirmation_details_unknown=sum(x.get('confirmation_details_status')=='UNKNOWN' for x in records),output=str(path))),flush=True)


if __name__=='__main__':main()
