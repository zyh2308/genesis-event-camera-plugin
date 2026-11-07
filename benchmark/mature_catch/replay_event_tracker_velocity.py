"""Replay a completed event pilot with no rendering; GT is scoring only."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import cv2
cv2.setNumThreads(2)
import h5py
import numpy as np

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.EventSphereTracker

def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    sensor=args.episode.parent/'migration_audit/sensor_episode_000';events=np.load(sensor/'raw_events.npz')['events'];audit=json.loads((sensor/'packet_audit.json').read_text())
    trace=json.loads((args.episode/'trace.json').read_text()) if (args.episode/'trace.json').is_file() else None
    episode=json.loads((args.episode/'result.json').read_text()) if (args.episode/'result.json').is_file() else dict(first_touch=None)
    results={};start=time.monotonic()
    with h5py.File(sensor/'sparse_hdr_endpoints.h5','r') as h:
        times=h['time_s'][:];K=h['intrinsics'][:];poses=h['object_poses'][:];cam=h['camera_to_world_cv'][:];ids=h['segmentation_pose_ids'][:]
        hand_index=int(np.where(ids==17)[0][0]);object_index=int(np.where(ids==34)[0][0]);truth_positions=poses[:,object_index,:3,3]
        if trace is None:
            object_index=int(np.where(ids==34)[0][0]);trace=[]
            for i in range(0,len(times),4):
                world=poses[i,object_index,:3,3];pc=(np.linalg.inv(cam[i])@np.r_[world,1.])[:3];q=K@pc
                trace.append(dict(t_before=float(times[i]),qpos=np.zeros(44).tolist(),scoring_only_GT_position_at_control=world.tolist(),scoring_only_GT_uv_at_control=(q[:2]/q[2]).tolist()))
        for key,path in [('old',args.old),('candidate',args.new)]:
            tracker=load(path,'tracker_'+key)(K,(640,480),.04);offset=0
            for i,row in enumerate(audit):
                count=row['events'];tracker.push(events[offset:offset+count],row['available_s'],row['time_s'],cam[i]);offset+=count
            assert offset==len(events)
            rows=[]
            for i,row in enumerate(trace):
                t=row['t_before'];index=int(np.argmin(abs(times-t)));assert abs(times[index]-t)<1e-7
                q=np.asarray(trace[i-1]['qpos'][21:37]) if i else np.zeros(16)
                hand=q[[0,2,3,4,6,7,8,10,11,13,14,15]]
                state=tracker.observe(t,poses[index,hand_index],hand)
                item=dict(time_s=t,state_available=state is not None)
                if state is not None:
                    world=np.asarray(state['position']);pc=(np.linalg.inv(cam[index])@np.r_[world,1.])[:3];uv=K@pc;uv=uv[:2]/uv[2]
                    truth=np.asarray(row['scoring_only_GT_position_at_control']);truth_uv=np.asarray(row['scoring_only_GT_uv_at_control'])
                    item.update(pixel_error=float(np.linalg.norm(uv-truth_uv)),position_error_m=float(np.linalg.norm(world-truth)),
                        source=state['source'],maximum_event_s=float(state['maximum_event_s']),position=world.tolist(),velocity=np.asarray(state['velocity']).tolist())
                    if 0<index<len(times)-1:
                        truth_velocity=(truth_positions[index+1]-truth_positions[index-1])/(times[index+1]-times[index-1])
                        item['velocity_error_m_s']=float(np.linalg.norm(np.asarray(state['velocity'])-truth_velocity))
                    assert state['maximum_event_s']<=t+1e-8
                rows.append(item)
            before=[r for r in rows if episode['first_touch'] is None or r['time_s']<episode['first_touch']-.04]
            valid=[r for r in before if r['state_available']]
            summary=dict(polls=len(before),available=len(valid),detections=len(tracker.samples),first_detection=tracker.samples[0] if tracker.samples else None,
                first_valid_time_s=next((r['time_s'] for r in rows if r['state_available']),None),
                pixel_median=float(np.median([r['pixel_error'] for r in valid])) if valid else None,
                pixel_p95=float(np.percentile([r['pixel_error'] for r in valid],95)) if valid else None,
                position_median_mm=float(np.median([r['position_error_m'] for r in valid])*1000) if valid else None,
                velocity_median_m_s=float(np.median([r['velocity_error_m_s'] for r in valid if 'velocity_error_m_s' in r])) if valid else None,
                velocity_p95_m_s=float(np.percentile([r['velocity_error_m_s'] for r in valid if 'velocity_error_m_s' in r],95)) if valid else None)
            results[key]=dict(summary=summary,rows=rows,samples=tracker.samples)
    args.output.write_text(json.dumps(dict(scope='offline perception replay, not a new closed-loop outcome',wall_s=time.monotonic()-start,results=results),indent=2))
    print(json.dumps({key:value['summary'] for key,value in results.items()}),flush=True)

if __name__=='__main__':main()
