"""Score archived perception before contact; object poses are scoring only."""
import argparse,json
from pathlib import Path
import numpy as np,h5py

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 trace=json.loads((a.run/'episode_000/trace.json').read_text());result=json.loads((a.run/'episode_000/result.json').read_text());rows=[]
 with h5py.File(a.run/'migration_audit/sensor_episode_000/sparse_hdr_endpoints.h5','r') as h:
  times=h['time_s'][:];ids=h['segmentation_pose_ids'][:];object_index=int(np.where(ids==34)[0][0]);positions=h['object_poses'][:,object_index,:3,3]
  for row in trace:
   s=row['event_state'];t=row['t_before']
   if s is None or s['source'].startswith('event_acquisition_then_'):continue
   if result['first_touch'] is not None and t>=result['first_touch']-.04:continue
   k=int(np.argmin(abs(times-t)))
   if k==0 or k+1>=len(times) or abs(times[k]-t)>1e-7:continue
   truth_v=(positions[k+1]-positions[k-1])/(times[k+1]-times[k-1]);truth_p=np.array(row['scoring_only_GT_position_at_control'])
   state_p=np.array(s['position']);state_v=np.array(s['velocity'])
   rows.append(dict(time_s=t,pixel_error=row['scoring_only_pixel_error'],position_error_mm=float(np.linalg.norm(state_p-truth_p)*1000),
                    velocity_error_m_s=float(np.linalg.norm(state_v-truth_v)),event_position=state_p.tolist(),GT_position=truth_p.tolist(),
                    event_velocity=state_v.tolist(),GT_velocity=truth_v.tolist(),raw_action_base=row['action'][:2],fresh=s['fresh_event'],
                    event_age_ms=(t-s['maximum_event_s'])*1000))
 def summarize(rs):
  return dict(polls=len(rs),position_median_mm=float(np.median([r['position_error_mm'] for r in rs])) if rs else None,
              position_p95_mm=float(np.percentile([r['position_error_mm'] for r in rs],95)) if rs else None,
              velocity_median_m_s=float(np.median([r['velocity_error_m_s'] for r in rs])) if rs else None,
              velocity_p95_m_s=float(np.percentile([r['velocity_error_m_s'] for r in rs],95)) if rs else None)
 output=dict(scope='archived pre-contact scoring only; derivative GT never fed into controller',summary=summarize(rows),phases={},rows=rows)
 for name,start,end in [('early',.58,.80),('mid',.8,1.1),('late',1.1,1.4)]:output['phases'][name]=summarize([r for r in rows if start<=r['time_s']<end])
 a.output.write_text(json.dumps(output,indent=2));print(json.dumps(dict(summary=output['summary'],phases=output['phases'],early_rows=rows[:4],last_rows=rows[-2:])),flush=True)
if __name__=='__main__':main()
