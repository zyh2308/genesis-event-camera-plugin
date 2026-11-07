"""Offline actual-physics scoring and audit; no input to controller."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import h5py

def stats(values):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    return dict(n=len(values),median=float(np.median(values)) if len(values) else None,p95=float(np.percentile(values,95)) if len(values) else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    spec=json.loads((a.root/'PREREGISTERED_PROTOCOL.json').read_text());records=[];perception=[];violations=[]
    for case in spec['cases']:
        pair={}
        for mode in ['event','state_reference']:
            folder=a.root/'cases'/case['case_id']/mode
            f=folder/'episode_000/result.json'
            if not f.exists():
                if a.allow_partial:continue
                raise FileNotFoundError(f)
            result=json.loads(f.read_text());tracefile=f.with_name('trace.json');trace=json.loads(tracefile.read_text()) if tracefile.exists() else []
            protocol=json.loads((folder/'protocol.json').read_text());assert protocol['checkpoint_sha256']==spec['checkpoint_sha256'];assert protocol['sensor_pitch_offset_degrees']==7
            assert protocol['asset_initialization_seed']==1702 and protocol['horizontal_speed_m_s']==case['horizontal_speed_m_s']
            record=dict(case,mode=mode,result=result,protocol=protocol,trace=trace,output=str(folder))
            if result['success'] and result.get('terminated',False):violations.append(case['case_id']+': both success and termination flags')
            if trace:
                # Native object free joint is last six DOF, used for scoring only.
                record['final_hand_contact']=trace[-1]['hand_contact']
                record['actual_contact_consistency']=sum(x['hand_contact'] for x in trace)==result['hand_contact_steps']
                assert record['actual_contact_consistency']
                if result['success'] and result['hand_contact_steps']==0:violations.append(case['case_id']+':success without actual contact')
            if mode=='event':
                sensor=folder/'migration_audit/sensor_episode_000'
                modules=json.loads((folder/'loaded_module_receipt.json').read_text())
                for name,loaded in modules.items():
                    assert Path(loaded['path']).parent==Path(spec['source'])
                    assert loaded['sha256']==spec['runtime_files'][name+'.py']
                record['initialization']=json.loads((sensor/'initialization.json').read_text())
                audit=json.loads((sensor/'packet_audit.json').read_text()) if (sensor/'packet_audit.json').exists() else []
                record['packets']=len(audit);record['raw_event_count']=sum(x['events'] for x in audit)
                record['dense_reads']=sum(x.get('dense_HDR_reads',0) for x in audit)
                assert record['dense_reads']==0
                for row in trace:
                    state=row['event_state']
                    if state is not None:
                        assert state.get('maximum_event_s') is None or state['maximum_event_s']<=row['t_before']+1e-8
                        assert state.get('RGB_available_s',0)<=row['t_before']+1e-8
                        assert row['policy_object_input'] is not None
                    else:assert row['policy_object_input'] is None and all(abs(x)<1e-10 for x in row['action'])
                release=result.get('initial',{}).get('release_time_s')
                stop=result['first_touch'] if result.get('first_touch') is not None else result['sim_time']
                # The state at the start of the first contact-containing step was
                # issued before that contact; retain this critical controller input.
                camera_stop=next((x['t_before'] for x in trace if x['hand_contact']),result['sim_time'])
                record['precontact_includes_first_contact_action_input']=True
                record['precontact_last_certainly_collision_free_endpoint_s']=camera_stop
                flight=[x for x in trace if release is not None and release<=x['t_before']<stop-1e-9]
                available=[x for x in flight if x['event_state'] is not None]
                eligible=[x for x in flight if x['event_state'] is not None and not x['event_state']['source'].startswith('event_acquisition_then_')]
                attached=[x for x in trace if x['event_state'] is not None and x['event_state']['source'].startswith('event_acquisition_then_')]
                record['flight_polls']=len(flight);record['available_flight_polls']=sum(x['event_state'] is not None for x in flight);record['non_attached_flight_polls']=len(eligible)
                record['fresh_flight_polls']=sum(bool(x['event_state'] and x['event_state'].get('fresh_event')) for x in flight)
                record['static_prediction_flight_polls']=sum(bool(x['event_state'] and x['event_state'].get('motion_phase')=='static_prediction') for x in flight)
                onsets=[x['event_state'].get('motion_onset_s') for x in trace if x['event_state'] and x['event_state'].get('motion_onset_s') is not None]
                record['motion_onset_delay_from_release_s']=onsets[0]-release if onsets and release is not None else None
                record['assumed_grasp_first_time']=attached[0]['t_before'] if attached else None
                record['all_precontact_state_pixel_error']=stats([x['scoring_only_pixel_error'] for x in available])
                record['all_precontact_state_position_error_mm']=stats([np.linalg.norm(np.array(x['event_state']['position'])-x['scoring_only_GT_position_at_control'])*1000 for x in available])
                record['precontact_pixel_error']=stats([x['scoring_only_pixel_error'] for x in eligible])
                record['precontact_position_error_mm']=stats([np.linalg.norm(np.array(x['event_state']['position'])-x['scoring_only_GT_position_at_control'])*1000 for x in eligible])
                # Exact current velocity is in the preceding archived physical row.
                velocities=[];all_velocities=[]
                for i,row in enumerate(trace):
                    if row not in available or i==0:continue
                    true_v=np.array(trace[i-1]['qvel'][36:39]);state=row['event_state']
                    velocity_error=float(np.linalg.norm(np.array(state['velocity'])-true_v));all_velocities.append(velocity_error)
                    if row in eligible:velocities.append(velocity_error)
                    perception.append(dict(case_id=case['case_id'],condition=case['condition'],seed=case['seed'],time_s=row['t_before'],pixel_error=row['scoring_only_pixel_error'],position_error_mm=float(np.linalg.norm(np.array(state['position'])-row['scoring_only_GT_position_at_control'])*1000),velocity_error_m_s=velocity_error,fresh=state.get('fresh_event'),assumed_grasp=state['source'].startswith('event_acquisition_then_'),state_source=state['source'],event_age_ms=(row['t_before']-state['maximum_event_s'])*1000 if state.get('maximum_event_s') is not None else None))
                record['precontact_velocity_error_m_s']=stats(velocities)
                record['all_precontact_state_velocity_error_m_s']=stats(all_velocities)
                ep=sensor/'sparse_hdr_endpoints.h5'
                if ep.exists() and result.get('initial'):
                    with h5py.File(ep,'r') as h:
                        times=h['time_s'][:];poses=h['object_poses'];ids=h['segmentation_pose_ids'][:];initial=np.array(result['initial']['position'])
                        j=int(np.argmin(np.linalg.norm(poses[0,:,:3,3]-initial,axis=1)));assert np.linalg.norm(poses[0,j,:3,3]-initial)<1e-5
                        K=h['intrinsics'][:];fov=[];fov0=[];visible=[];uvs=[]
                        angle=np.deg2rad(7.);R=np.array([[1.,0.,0.],[0.,np.cos(angle),-np.sin(angle)],[0.,np.sin(angle),np.cos(angle)]])
                        for k,t in enumerate(times):
                            if not(release<=t<=camera_stop+1e-9):continue
                            pc=(np.linalg.inv(h['camera_to_world_cv'][k])@np.r_[poses[k,j,:3,3],1])[:3];q=K@pc;uv=q[:2]/q[2]
                            inside=bool(pc[2]>0 and 0<=uv[0]<640 and 0<=uv[1]<480);fov.append(inside)
                            pc0=R@pc;q0=K@pc0;uv0=q0[:2]/q0[2];inside0=bool(pc0[2]>0 and 0<=uv0[0]<640 and 0<=uv0[1]<480);fov0.append(inside0)
                            uvs.append(dict(t=float(t),u=float(uv[0]),v=float(uv[1]),center_inside=inside,depth_m=float(pc[2]),u_at_zero_pitch=float(uv0[0]),v_at_zero_pitch=float(uv0[1]),center_inside_zero_pitch=inside0))
                            visible.append(int(np.count_nonzero(h['segmentation'][k]==ids[j])))
                        image_speeds=[np.hypot(b['u']-a['u'],b['v']-a['v'])/(b['t']-a['t']) for a,b in zip(uvs,uvs[1:]) if a['center_inside'] and b['center_inside']]
                        record['camera_fov']=dict(image_speed_px_s=stats(image_speeds),scoring_only=True,object_pose_id=int(ids[j]),samples=len(fov),center_inside=sum(fov),zero_pitch_center_inside_same_robot_trajectory=sum(fov0),counterfactual_scope='geometry only, same archived camera-body and ball states; not a 0-degree closed-loop run or occlusion test',last_captured_time_s=float(times[-1]),visible_samples=sum(x>0 for x in visible),visible_pixels=stats(visible),uv=uvs)
            pair[mode]=record;records.append(record)
        if len(pair)==2 and all('initial' in x['result'] for x in pair.values()):
            e=pair['event']['result']['initial'];g=pair['state_reference']['result']['initial'];equal=e==g
            pair['event']['matched_initial_state']=equal;pair['state_reference']['matched_initial_state']=equal
            if not equal:violations.append(case['case_id']+': initial state mismatch')
    output=dict(scope='preregistered locked configuration; all outcomes; ground truth for offline scoring only; primary all_precontact_state metrics include every available controller state after actual release and before real contact, including false assumed grasps; separately named precontact metrics exclude proprioceptive grasp propagation',records=records,perception_rows=perception,violations=violations,complete=len(records)==36)
    (a.root/'GENERALIZATION_PORTABLE.json').write_text(json.dumps(output,indent=2));print(json.dumps(dict(records=len(records),complete=output['complete'],violations=violations,results=[dict(case=r['case_id'],mode=r['mode'],success=r['result']['success'],contact=r['result'].get('hand_contact_steps'),pixel=r.get('precontact_pixel_error')) for r in records])))

if __name__=='__main__':main()
