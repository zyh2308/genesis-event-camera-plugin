from pathlib import Path
import argparse,json
import numpy as np,h5py

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();spec=json.loads((a.root/'PREREGISTERED_PROTOCOL.json').read_text());rows=[];allK=[]
for case in spec['cases']:
 folder=a.root/'cases'/case['case_id']/'event';sensor=folder/'migration_audit/sensor_episode_000';init=json.loads((sensor/'initialization.json').read_text());audit=json.loads((sensor/'packet_audit.json').read_text());result=json.loads((folder/'episodes.json').read_text())[0]
 assert init['RGB_frames_read']==1 and init['control_depth_frames_read']==1
 events=np.load(sensor/'raw_events.npz')['events'];assert events.ndim==2 and events.shape[1]==4
 assert len(events)==sum(x['events'] for x in audit)
 assert np.isfinite(events).all()
 for packet in audit:assert abs(packet['available_s']-packet['time_s']-.01)<1e-8 and packet['dense_HDR_reads']==0
 if len(events):
  assert events[:,0].min()>=-1e-8 and events[:,0].max()<=audit[-1]['time_s']+1e-8
  assert ((events[:,1]>=0)&(events[:,1]<640)&(events[:,2]>=0)&(events[:,2]<480)).all()
  assert np.array_equal(events[:,1:3],np.round(events[:,1:3]))
 frame_count=0
 if (sensor/'sparse_hdr_endpoints.h5').exists():
  with h5py.File(sensor/'sparse_hdr_endpoints.h5','r') as h:
   times=h['time_s'][:];frame_count=len(times);assert frame_count==len(audit)
   assert h['radiance'].dtype==np.dtype('float32') and h['radiance'].shape[1:3]==(480,640)
   assert np.allclose(np.diff(times),.01,atol=1e-8,rtol=0)
   allK.append(h['intrinsics'][:])
 trace=json.loads((folder/'episode_000/trace.json').read_text()) if (folder/'episode_000/trace.json').exists() else []
 for row in trace:
  state=row['event_state']
  if state is not None and state.get('maximum_event_s') is not None:assert state['maximum_event_s']<=row['t_before']-.01+1e-8
 rows.append(dict(case_id=case['case_id'],raw_events=len(events),hdr_endpoints=frame_count,control_queries=len(trace),polarity_values=np.unique(events[:,3]).tolist() if len(events) else [],success=result['success']))
assert all(np.array_equal(K,allK[0]) for K in allK)
output=dict(checked_episodes=len(rows),all_checks_passed=True,raw_events=sum(x['raw_events'] for x in rows),same_intrinsics_all_cases=True,intrinsics=allK[0].tolist() if allK else None,rows=rows)
(a.root/'RAW_INPUT_CAUSALITY_VERIFICATION.json').write_text(json.dumps(output,indent=2));print(json.dumps(output))
