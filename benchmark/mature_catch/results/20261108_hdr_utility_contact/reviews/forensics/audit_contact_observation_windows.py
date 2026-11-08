"""Strict artifact-only audit for the preselected 10 ms paired benchmark.

No target GT, renderer, physics or estimator is invoked. Unconsumed packets
after the last actual controller query are expected and explicitly retained.
"""
from pathlib import Path
import hashlib,json


def audit_data(protocol,packets,raw,camera_times,camera_poses,document,trace,samples):
 import numpy as np
 from scipy.spatial.transform import Rotation,Slerp
 def check(ok,message):
  if not ok:raise AssertionError('Observation-window audit: '+message)
 def close(x,y):return abs(float(x)-float(y))<=1e-8
 check(float(protocol.get('event_observation_window_ms',0.))==10.,'actor protocol does not enable 10 ms')
 check(float(document['observation_window_ms'])==10.,'sensor artifact disagrees with protocol')
 parents=document['windows'];raw=np.asarray(raw);times=np.asarray(camera_times,dtype=float);poses=np.asarray(camera_poses,dtype=float)
 check(raw.shape==(sum(int(p['events']) for p in packets),4),'raw array/count mismatch')
 queries=np.asarray([float(r['t_before']) for r in trace],dtype=float)
 check(not len(queries) or np.all(np.diff(queries)>0),'controller queries are not strictly ordered')
 last_query=float(queries[-1]) if len(queries) else None
 due=[i for i,p in enumerate(packets) if last_query is not None and p['available_s']<=last_query+1e-9]
 check([int(p['parent_packet_id']) for p in parents]==due,'consumed parents differ from packets due at final control query')
 check(due==list(range(len(due))),'arrival queue is not a chronological prefix')
 check(len(times)==len(packets),'camera endpoint / original packet count mismatch')
 if len(times):
  check(poses.shape==(len(times),4,4) and np.isfinite(poses).all(),'invalid archived camera transforms')
  check(np.all(np.diff(times)>0),'camera endpoints not chronological')
  check(all(close(t,p['time_s']) for t,p in zip(times,packets)),'camera times differ from original packets')
 offsets=np.cumsum([0]+[int(p['events']) for p in packets]);detections=[];seen_ends={};previous_max=None
 total_children=carry_count=carry_events=repeated_count=0;previous_processing=None
 for parent in parents:
  pid=int(parent['parent_packet_id']);packet=packets[pid];chunk=raw[offsets[pid]:offsets[pid+1]]
  end=float(packet['time_s']);available=float(packet['available_s']);now=parent['processing_now_s']
  check(parent['status']=='PASS' and parent['raw_event_order_check']=='PASS','parent did not finish successfully')
  check(parent['processing_clock_measured'] is True and now is not None,'online processing clock not measured')
  now=float(now);eligible=queries[queries>=available-1e-9]
  check(len(eligible)>0 and close(now,eligible[0]),'parent not consumed at first available control poll')
  check(available<=now+1e-9,'information exposed before original arrival')
  check(previous_processing is None or now>=previous_processing-1e-9,'processing clock went backwards');previous_processing=now
  check(close(parent['original_end_s'],end) and close(parent['original_available_s'],available),'parent timestamp mismatch')
  check(int(parent['original_events'])==len(chunk),'parent count differs from raw packet')
  check((parent['previous_parent_raw_max_s'] is None)==(previous_max is None),'previous event support mismatch')
  if previous_max is not None:check(close(parent['previous_parent_raw_max_s'],previous_max),'wrong prior event maximum')
  if len(chunk):
   check(np.isfinite(chunk).all() and np.all(np.diff(chunk[:,0])>=-1e-12),'invalid/unordered raw events')
   check(previous_max is None or float(chunk[:,0].min())>=previous_max-1e-12,'out-of-sequence arrival')
   check(close(parent['actual_event_min_s'],chunk[:,0].min()) and close(parent['actual_event_max_s'],chunk[:,0].max()),'actual timestamp bounds mismatch')
   ties=0 if previous_max is None else int(np.sum(chunk[:,0]==previous_max))
   check(int(parent['raw_events_equal_previous_parent_max'])==ties,'equal-timestamp accounting mismatch')
   previous_max=max(previous_max if previous_max is not None else -float('inf'),float(chunk[:,0].max()))
  else:check(parent['actual_event_min_s'] is None and parent['actual_event_max_s'] is None,'empty parent has event times')
  children=parent['windows'];check(int(parent['subwindows'])==len(children),'child count mismatch')
  if pid==0:check(close(end,0.) and len(chunk)==0 and children==[],'reset parent is not empty')
  else:check(bool(children) and close(children[-1]['end_s'],end),'partition misses parent endpoint')
  cursor=0;previous_hi=None;n_detected=n_carry=carry_n=0
  for index,child in enumerate(children):
   total_children+=1;lo=float(child['start_s']);hi=float(child['end_s']);start=float(times[pid-1])
   check(int(child['parent_packet_id'])==pid and int(child['subwindow_index'])==index,'child identity mismatch')
   check(hi>lo and hi-lo<=.01000001,'invalid child duration')
   check(close(hi-lo,.01) or (index==len(children)-1 and hi-lo<.01) or close(lo,times[0]),'unexpected child cadence')
   check(previous_hi is None or close(lo,previous_hi),'gap/overlap in child time support');previous_hi=hi
   check(close(child['nominal_parent_start_s'],start),'nominal interval mismatch')
   carry=hi<=start+1e-10;check(bool(child['carry_in_historical_window'])==carry,'carry classification mismatch')
   check(close(child['available_s'],available),'child availability changed')
   begin,stop=map(int,child['input_event_index_range'])
   check(begin==cursor and begin<=stop<=len(chunk),'overlapping or dropped original event indices');cursor=stop
   part=chunk[begin:stop];check(int(child['events'])==len(part),'child event count mismatch')
   if len(part):
    check(float(part[:,0].min())>=lo-1e-8 and float(part[:,0].max())<=hi+1e-8,'event outside actual historical child support')
    if index<len(children)-1:check(float(part[:,0].max())<hi+1e-12,'non-final upper-boundary ownership mismatch')
   if carry:carry_count+=1;carry_events+=len(part);n_carry+=1;carry_n+=len(part)
   key=round(hi,12);prior=seen_ends.get(key,[])
   check(child['same_window_end_previous_parent_ids']==prior,'repeated endpoint provenance mismatch')
   repeated_count+=bool(prior);seen_ends.setdefault(key,[]).append(pid)
   check(close(child['camera_pose_time_s'],hi),'camera time violates fixed child-end convention')
   support=list(map(float,child['camera_endpoint_times_s']));support_available=list(map(float,child['camera_endpoint_available_s']))
   nearest=int(np.argmin(abs(times-hi)))
   if abs(times[nearest]-hi)<1e-9:
    indices=[nearest];expected_T=poses[nearest]
    check(child['pose_source']=='exact_available_historical_camera_endpoint','exact camera source label mismatch')
   else:
    upper=int(np.searchsorted(times,hi,side='right'));lower=upper-1
    check(0<=lower<upper<len(times),'no archived camera bracket')
    indices=[lower,upper];alpha=(hi-times[lower])/(times[upper]-times[lower]);expected_T=np.eye(4)
    expected_T[:3,:3]=Slerp([0.,1.],Rotation.from_matrix(np.stack([poses[lower,:3,:3],poses[upper,:3,:3]])))(alpha).as_matrix()
    expected_T[:3,3]=(1-alpha)*poses[lower,:3,3]+alpha*poses[upper,:3,3]
    check(child['pose_source']=='approx_available_camera_history_SE3_slerp_translation_linear_at_subwindow_end','interpolated source label mismatch')
   check(len(support)==len(indices)==len(support_available),'camera support length mismatch')
   for t,arrival,k in zip(support,support_available,indices):
    check(close(t,times[k]) and close(arrival,packets[k]['available_s']),'camera support differs from recorded endpoint')
    check(arrival<=available+1e-9,'future camera endpoint used')
   T=np.asarray(child['camera_to_world'],dtype=float)
   check(T.shape==(4,4) and np.isfinite(T).all(),'invalid child transform')
   np.testing.assert_allclose(T,expected_T,rtol=0,atol=1e-8,err_msg='camera interpolation differs from archived endpoints')
   sample=child['detected'];same=False
   if sample is not None:
    check(len(part)>0 and close(sample['available_s'],available),'detection without events or changed arrival')
    check(float(part[:,0].min())-1e-8<=sample['t']<=float(part[:,0].max())+1e-8,'measurement time outside child events')
    check(close(sample['maximum_event_s'],part[:,0].max()),'detector support differs from child events')
    check(sample['maximum_event_s']<=available+1e-8,'detector used a future event')
    check(not detections or sample['t']>=detections[-1]['t']-1e-12,'measurement order went backwards')
    same=bool(detections and abs(sample['t']-detections[-1]['t'])<=1e-12)
    np.testing.assert_allclose(sample['camera_origin'],T[:3,3],rtol=0,atol=1e-8,err_msg='world / camera origin mismatch')
    detections.append(sample);n_detected+=1
   check(bool(child['same_measurement_timestamp_as_previous'])==same,'equal measurement time accounting mismatch')
  check(cursor==len(chunk),'original event suffix dropped')
  check(int(parent['added_samples'])==n_detected,'sample append count mismatch')
  check(int(parent['carry_in_window_count'])==n_carry and int(parent['carry_in_event_count'])==carry_n,'carry totals mismatch')
  check(int(parent['event_partition_count'])==len(chunk) and parent['event_partition_exact'] is True,'exact partition record mismatch')
 check(len(detections)==len(samples),'event_samples contain missing or duplicate appends')
 for got,want in zip(detections,samples):
  for key in ['t','position','uv','radius_px','quality','available_s','maximum_event_s','camera_origin']:
   np.testing.assert_allclose(got[key],want[key],rtol=0,atol=1e-8,err_msg='child / event_samples mismatch: '+key)
 for row in trace:
  state=row.get('event_state');now=float(row['t_before'])
  if state is not None and state.get('maximum_event_s') is not None:
   known=[s['maximum_event_s'] for s in detections if s['available_s']<=now+1e-9]
   check(bool(known) and close(state['maximum_event_s'],max(known)),'controller support differs from arrived detections')
   check(state['maximum_event_s']<=now+1e-8,'controller used a future event')
 return dict(status='PASS',observation_window_ms=10.,original_packets=len(packets),consumed_parents=len(parents),
  pending_parents_at_final_control=len(packets)-len(parents),last_actual_control_query_s=last_query,
  consumed_events=int(offsets[len(parents)]),pending_events=int(offsets[-1]-offsets[len(parents)]),
  child_windows=total_children,child_detections=len(detections),carry_in_windows=carry_count,
  carry_in_events=carry_events,repeated_child_endpoints=repeated_count,
  pending_contract='Packets arriving after the final actual control query may remain pending, even if physics ends later')


def audit_run(run,protocol,spec):
 import numpy as np
 import h5py
 def read(p):return json.loads(Path(p).read_text())
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 run=Path(run);sensor=run/'migration_audit/sensor_episode_000'
 packets=read(sensor/'packet_audit.json');document=read(sensor/'event_observation_windows.json')
 samples=read(sensor/'event_samples.json');trace=read(run/'episode_000/trace.json')
 raw=np.load(sensor/'raw_events.npz')['events']
 if packets:
  with h5py.File(sensor/'sparse_hdr_endpoints.h5','r') as h:times=h['time_s'][:];poses=h['camera_to_world_cv'][:]
 else:times=np.empty(0);poses=np.empty((0,4,4))
 loaded=read(run/'loaded_module_receipt.json')
 for module in ['event_observation_windows','event_sphere_tracker','sparse_event_sensor']:
  expected=spec['runtime_files'][module+'.py'];entry=loaded[module]
  assert entry['sha256']==expected and Path(entry['path']).name==module+'.py',module+' loaded source mismatch'
 result=audit_data(protocol,packets,raw,times,poses,document,trace,samples)
 result.update(output=str(run),sensor_window_audit_sha256=sha(sensor/'event_observation_windows.json'),
  loaded_module_receipt_sha256=sha(run/'loaded_module_receipt.json'),runtime_loaded_hashes='PASS')
 return result
