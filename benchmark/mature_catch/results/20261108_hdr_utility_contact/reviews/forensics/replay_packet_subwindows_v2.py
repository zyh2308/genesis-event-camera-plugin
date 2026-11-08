"""CPU diagnostic: exact archive replay, native whole packets, native subwindows.

No production writes, physics, rendering, GPU, new HDR, or GT estimator input.
All subwindows retain their enclosing real packet's availability timestamp.
Camera SE(3) at each subwindow end is an explicitly approximate interpolation
of the two enclosing camera endpoints, both already available at ingestion.
The common replay grades target GT only after the estimator has returned.
"""
from pathlib import Path
import argparse,importlib.util,json,math,os,sys,types
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


def pose_between(t,t0,t1,T0,T1):
    assert t1>t0 and t0-1e-8<=t<=t1+1e-8
    if abs(t-t0)<1e-10:return np.asarray(T0,dtype=float).copy()
    if abs(t-t1)<1e-10:return np.asarray(T1,dtype=float).copy()
    alpha=(float(t)-t0)/(t1-t0)
    result=np.eye(4)
    result[:3,:3]=Slerp([0.,1.],Rotation.from_matrix(np.stack([T0[:3,:3],T1[:3,:3]])))(alpha).as_matrix()
    result[:3,3]=(1-alpha)*T0[:3,3]+alpha*T1[:3,3]
    return result


def history_camera_pose(t,times,poses,endpoint_available,parent_available):
    """Use camera history only; never extrapolate beyond available endpoints."""
    nearest=int(np.argmin(abs(times-t)))
    if abs(times[nearest]-t)<1e-9:
        assert endpoint_available[nearest]<=parent_available+1e-9,'Camera endpoint not yet available'
        return poses[nearest].copy(),[float(times[nearest])],[float(endpoint_available[nearest])]
    upper=int(np.searchsorted(times,t,side='right'));lower=upper-1
    assert 0<=lower<upper<len(times),'Camera history does not bracket event subwindow'
    assert endpoint_available[lower]<=parent_available+1e-9 and endpoint_available[upper]<=parent_available+1e-9,'Camera interpolation would use an unavailable endpoint'
    T=pose_between(t,float(times[lower]),float(times[upper]),poses[lower],poses[upper])
    return T,[float(times[lower]),float(times[upper])],[float(endpoint_available[lower]),float(endpoint_available[upper])]


def scoring_only_motion_publication(states,release_s):
    first=next((float(s['time_s']) for s in states if s['position'] is not None and s['motion_onset_s'] is not None),None)
    return dict(scoring_only_first_control_moving_time_s=first,
        scoring_only_pre_release_moving_control=bool(first is not None and first<release_s-1e-9),
        scoring_only_first_control_moving_delay_from_release_s=None if first is None else first-release_s)


def split_packet(packet,times,poses,endpoint_available,window_s,parent_packet_ids=None):
    """Keep actual event time and arrival parent, including carry-in history.

    Native V2E may emit events just before its nominal renderer interval start.
    Such events get a historical child window, never an earlier availability.
    """
    end=float(packet['end_s']);available=float(packet['available_s'])
    k=int(np.argmin(abs(times-end)))
    assert abs(times[k]-end)<1e-8,(times[k],end)
    if k==0:
        assert abs(end)<1e-8 and len(packet['events'])==0,'Only the empty reset packet may lack a previous endpoint'
        assert endpoint_available[0]<=available+1e-9
        return []
    start=float(times[k-1]);parent_id=int(parent_packet_ids[k]) if parent_packet_ids is not None else k-1
    assert parent_id>=0
    assert start<end<=available+1e-8
    assert endpoint_available[k-1]<=available+1e-9 and endpoint_available[k]<=available+1e-9
    np.testing.assert_allclose(packet['T'],poses[k],rtol=0,atol=1e-10)
    events=np.asarray(packet['events'])
    if len(events):
        assert np.isfinite(events).all()
        assert float(events[:,0].min())>=float(times[0])-1e-8,'Events precede available camera history'
        assert float(events[:,0].max())<=end+1e-8,'Events later than their nominal parent endpoint need separate review'
        assert np.all(np.diff(events[:,0])>=-1e-12)
    carry_windows=max(0,int(math.ceil((start-float(events[:,0].min()))/window_s-1e-8))) if len(events) else 0
    back=[max(float(times[0]),start-i*window_s) for i in range(carry_windows,0,-1)]
    n=max(1,int(math.ceil((end-start)/window_s-1e-8)))
    cuts=back+[start]+[min(end,start+i*window_s) for i in range(1,n)]+[end]
    assert all(hi>lo+1e-12 for lo,hi in zip(cuts[:-1],cuts[1:]))
    cursor=0;parts=[]
    for i,(lo,hi) in enumerate(zip(cuts[:-1],cuts[1:])):
        last=i==len(cuts)-2
        stop=len(events) if last else int(np.searchsorted(events[:,0],hi,side='left'))
        assert stop>=cursor
        first_index=cursor;part=events[cursor:stop];cursor=stop
        T,pose_times,pose_available=history_camera_pose(hi,times,poses,endpoint_available,available)
        parts.append(dict(events=part,available_s=available,end_s=float(hi),T=T,
            parent_packet_id=parent_id,
            subwindow_index=i,input_event_index_range=[int(first_index),int(stop)],
            diagnostic_start_s=float(lo),diagnostic_original_end_s=end,
            diagnostic_nominal_parent_start_s=start,diagnostic_carry_in=bool(hi<=start+1e-10),
            diagnostic_pose_source=('exact_available_historical_camera_endpoint' if len(pose_times)==1 else 'approx_available_camera_history_SE3_slerp_translation_linear_at_subwindow_end'),
            diagnostic_pose_endpoint_times_s=pose_times,
            diagnostic_pose_endpoint_available_s=pose_available))
    assert cursor==len(events) and sum(len(x['events']) for x in parts)==len(events)
    if len(events):
        np.testing.assert_array_equal(np.concatenate([x['events'] for x in parts]),events)
    return parts


def install_subwindows(tracker,native_class,times,poses,endpoint_available,parent_packet_ids,window_s,serial):
    tracker._subwindow_history=[]
    tracker._subwindow_last_raw_event_s=-np.inf
    tracker._subwindow_seen_end_s={}
    tracker._subwindow_temporal_error=None
    def detect(self,packet):
        records=[];before=len(self.samples)
        raw=np.asarray(packet['events']);previous_raw_max=float(self._subwindow_last_raw_event_s)
        if len(raw) and float(raw[:,0].min())<previous_raw_max-1e-12:
            self._subwindow_temporal_error=dict(kind='out_of_sequence_raw_events_across_arrival_parents',
                original_end_s=float(packet['end_s']),original_available_s=float(packet['available_s']),
                minimum_event_s=float(raw[:,0].min()),previous_consumed_raw_max_s=previous_raw_max)
            raise RuntimeError(json.dumps(self._subwindow_temporal_error))
        parts=split_packet(packet,times,poses,endpoint_available,window_s,parent_packet_ids)
        for sub in parts:
            same_measurement_time=False
            # Original detector/filter code; chronological observed-event times.
            # We do not predict to the enclosing availability timestamp here:
            # doing so would advance state ahead of still-unprocessed subwindows.
            sample=native_class._detect(self,sub)
            if sample is not None:
                assert sample['available_s']==packet['available_s']
                assert sample['t']<=sub['end_s']+1e-8
                assert sample['maximum_event_s']<=sub['end_s']+1e-8
                assert sample['maximum_event_s']<=sample['available_s']+1e-8
                filter_time=getattr(self,'filter_time',None)
                latest_sample_time=self.samples[-1]['t'] if self.samples else None
                same_measurement_time=bool(latest_sample_time is not None and abs(sample['t']-latest_sample_time)<=1e-12)
                if ((filter_time is not None and sample['t']<filter_time-1e-12) or
                    (latest_sample_time is not None and sample['t']<latest_sample_time-1e-12)):
                    self._subwindow_temporal_error=dict(kind='measurement_older_than_already_processed_filter_or_sample',
                        parent_packet_id=sub['parent_packet_id'],subwindow_index=sub['subwindow_index'],
                        measurement_t=float(sample['t']),filter_time_s=filter_time,latest_appended_measurement_s=latest_sample_time)
                    raise RuntimeError(json.dumps(serial(self._subwindow_temporal_error)))
                self.samples.append(sample)
                self.last_event_time=max(self.last_event_time,sample['maximum_event_s'])
            end_key=round(sub['end_s'],12)
            previous_end_parents=self._subwindow_seen_end_s.get(end_key,[])
            records.append(dict(start_s=sub['diagnostic_start_s'],end_s=sub['end_s'],
                subwindow_index=sub['subwindow_index'],
                input_event_index_range=sub['input_event_index_range'],
                carry_in_historical_window=sub['diagnostic_carry_in'],
                nominal_parent_start_s=sub['diagnostic_nominal_parent_start_s'],
                same_window_end_previous_parent_ids=list(previous_end_parents),
                same_measurement_timestamp_as_previous=same_measurement_time,
                parent_packet_id=sub['parent_packet_id'],camera_pose_time_s=sub['end_s'],
                available_s=sub['available_s'],events=len(sub['events']),
                pose_source=sub['diagnostic_pose_source'],
                camera_endpoint_times_s=sub['diagnostic_pose_endpoint_times_s'],
                camera_endpoint_available_s=sub['diagnostic_pose_endpoint_available_s'],
                camera_to_world=sub['T'].tolist(),detected=serial(sample)))
            self._subwindow_seen_end_s.setdefault(end_key,[]).append(sub['parent_packet_id'])
        parent_id=parts[0]['parent_packet_id'] if parts else int(parent_packet_ids[0])
        if len(raw):self._subwindow_last_raw_event_s=max(previous_raw_max,float(raw[:,0].max()))
        self._subwindow_history.append(dict(original_end_s=float(packet['end_s']),
            parent_packet_id=parent_id,
            original_available_s=float(packet['available_s']),original_events=len(packet['events']),
            previous_parent_raw_max_s=previous_raw_max if np.isfinite(previous_raw_max) else None,
            actual_event_min_s=float(raw[:,0].min()) if len(raw) else None,
            actual_event_max_s=float(raw[:,0].max()) if len(raw) else None,
            raw_events_equal_previous_parent_max=int(np.sum(raw[:,0]==previous_raw_max)) if len(raw) else 0,
            raw_event_order_check='PASS',
            subwindows=len(parts),added_samples=len(self.samples)-before,windows=records))
        # All samples have already been appended. A None return prevents the
        # existing one-result-per-packet common runner from appending duplicates.
        return None
    tracker._detect=types.MethodType(detect,tracker)


def read_geometry(run):
    import h5py
    sensor=run/'migration_audit/sensor_episode_000'
    audit=json.loads((sensor/'packet_audit.json').read_text())
    with h5py.File(sensor/'sparse_hdr_endpoints.h5','r') as h:
        times=h['time_s'][:];poses=h['camera_to_world_cv'][:]
    assert times.ndim==1 and np.all(np.diff(times)>0)
    assert abs(float(times[0]))<1e-8
    availability=np.full(len(times),np.inf);availability[0]=0.
    parent_ids=np.full(len(times),-1,dtype=int)
    for j,a in enumerate(audit):
        k=int(np.argmin(abs(times-a['time_s'])))
        assert abs(times[k]-a['time_s'])<1e-8
        availability[k]=min(availability[k],float(a['available_s']))
        assert parent_ids[k]==-1,'Ambiguous duplicate endpoint in packet audit'
        parent_ids[k]=j
    return times,poses,availability,parent_ids


def run_mode(run,mode,source,output,until,common,module,native_class,flight_fit,
             archive_association,window_s):
    geometry=read_geometry(run) if mode.startswith('production_subwindow') else None
    made=[]
    def factory(*args,**kwargs):
        tracker=native_class(*args,**kwargs)
        if mode=='original' and archive_association=='public':
            tracker._association_prediction=types.MethodType(lambda self,t:self._predict(t),tracker)
        if geometry is not None:
            install_subwindows(tracker,native_class,*geometry,window_s,common.serial)
        made.append(tracker);return tracker
    module.EventSphereTracker=factory
    try:
        result=common.replay(run,mode,source,output,until,flight_fit=flight_fit)
    except Exception as error:
        if made and getattr(made[0],'_subwindow_temporal_error',None) is not None:
            (output/(run.parent.name+'_'+run.name+'_TEMPORAL_SUPPORT_FAILURE.json')).write_text(json.dumps(dict(
                status='STOP_OUT_OF_SEQUENCE',error=str(error),details=common.serial(made[0]._subwindow_temporal_error),
                previous_parent_history=made[0]._subwindow_history),indent=2),encoding='utf-8')
        raise
    finally:module.EventSphereTracker=native_class
    assert len(made)==1;tracker=made[0]
    path=output/result['file'];payload=json.loads(path.read_text(encoding='utf-8'))
    # Refuse any input-time/availability leak in every mode, not only candidate.
    samples=payload['samples'];states=payload['states']
    assert all(x['maximum_event_s']<=x['available_s']+1e-8 for x in samples)
    for state in states:
        eligible=[s for s in samples if s['available_s']<=state['time_s']+1e-9]
        assert len(eligible)==state['detected_samples']
        assert state.get('maximum_event_s') is None or state['maximum_event_s']<=state['time_s']+1e-8
    payload['causal_availability_checks']='PASS'
    # Retain the original scalar's meaning: a retrospectively estimated event
    # time can precede release even if the first released control state did not.
    release_s=float(json.loads((run/'episode_000/result.json').read_text())['initial']['release_time_s'])
    publication_grading=scoring_only_motion_publication(states,release_s)
    payload.update(publication_grading)
    payload['onset_timestamp_vs_published_control_note']='motion_onset_s is the retrospectively estimated event time; first_control_moving_time_s is the first original controller query that could observe a moving state after real parent arrival'
    result.update(publication_grading)
    payload['archive_association_mode']=archive_association if mode=='original' else 'native production hook'
    payload['pose_source']=('available_camera_history_exact_endpoints_or_approx_SE3_interpolation' if geometry is not None else 'exact_archived_camera_endpoint_pose')
    if geometry is not None:
        payload['candidate_change']='10ms nonoverlapping original-event subwindows including true carry-in historical windows; same original parent availability; available camera history exact endpoints/approximate SE3 interpolation; native detector/filter; no event-time rewrite'
        payload['subwindow_history']=tracker._subwindow_history
        payload['subwindow_count']=sum(x['subwindows'] for x in tracker._subwindow_history)
        windows=[s for parent in tracker._subwindow_history for s in parent['windows']]
        payload['carry_in_window_count']=sum(s['carry_in_historical_window'] for s in windows)
        payload['carry_in_event_count']=sum(s['events'] for s in windows if s['carry_in_historical_window'])
        payload['same_window_end_across_parents_count']=sum(bool(s['same_window_end_previous_parent_ids']) for s in windows)
        payload['global_event_and_measurement_order_checks']='PASS; older-than-consumed events or measurements cause explicit stop'
        payload['packet_detected_field_note']='packets[].detected is None because multi-output samples are appended inside the wrapper; actual detections are samples[] and subwindow_history[].windows[].detected'
        result['subwindow_count']=payload['subwindow_count']
        result.update(carry_in_window_count=payload['carry_in_window_count'],carry_in_event_count=payload['carry_in_event_count'],
            same_window_end_across_parents_count=payload['same_window_end_across_parents_count'])
    else:payload['candidate_change']='none; whole original packets with '+payload['archive_association_mode']
    if mode=='original':
        assert result['baseline_parity']['status']=='PASS'
        archived=json.loads((run/'migration_audit/sensor_episode_000/event_samples.json').read_text())
        expected=[s for s in archived if s['available_s']<=states[-1]['time_s']+1e-9]
        assert len(expected)==len(samples)
        for got,want in zip(samples,expected):
            for key in ['t','position','uv','radius_px','quality','available_s','maximum_event_s','camera_origin']:
                np.testing.assert_allclose(got[key],want[key],rtol=0,atol=1e-8,err_msg='archived exact '+key)
        payload['baseline_parity']['all_original_sample_fields_checked']=True
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--until',type=float,default=1.45)
    p.add_argument('--flight-fit',action='store_true',help='Same existing fit1 in BOTH native whole-packet baseline and subwindow candidate; archived parity always retains original fit0')
    p.add_argument('--archived-association',choices=['public','native'],default='public',help='Explicit archived production version, not chosen using outcomes; old normal25 eight archives used public/static association')
    p.add_argument('--window-ms',type=float,default=10.)
    a=p.parse_args();assert 0<a.window_ms<=12.
    a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.source))
    os.environ.update(CATCHIT_REJECT_PARTIAL_CONTOURS='0',CATCHIT_EVENT_FLIGHT_FIT='0',CATCHIT_REVOCABLE_GRASP='0')
    spec=importlib.util.spec_from_file_location('subwindow_common_replay',Path(__file__).with_name('replay_static_association.py'))
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    import event_sphere_tracker as module
    native_class=module.EventSphereTracker;results=[]
    fixed_fit='fit1' if a.flight_fit else 'fit0'
    modes=[('original',False),(f'production_whole_packet_{fixed_fit}',a.flight_fit),(f'production_subwindow10ms_{fixed_fit}',a.flight_fit)]
    for run in a.run:
        for mode,usefit in modes:
            results.append(run_mode(run,mode,a.source,a.output,a.until,common,module,native_class,usefit,a.archived_association,a.window_ms/1000.))
            (a.output/'PARTIAL_SUMMARY.json').write_text(json.dumps({'status':'RUNNING','results':results},indent=2),encoding='utf-8')
    assert 'genesis' not in sys.modules and 'torch' not in sys.modules
    receipt=dict(status='PASS',physics_steps=0,renderer_calls=0,CUDA_calls=0,new_HDR_samples=0,
        scope='cached original physical trajectories only; GT offline scoring only; not closed-loop counterfactual success',
        comparison='native whole packets vs native 10ms subwindows; both same fit flag; archive-original parity is a separate prerequisite',
        confirmation_gap_s=.05,association_gate_m=.30,confirmations=2,subwindow_ms=a.window_ms,
        all_subwindow_availability='unchanged enclosing original packet availability',
        interpolated_pose='available historical camera endpoints only; exact endpoints or approximate Slerp + linear translation; carry-in windows keep current parent availability',
        diagnostic_version='v2 actual timestamp support with carry-in windows; not a float32 exception',
        subwindow_measurements_are_independent_renders=False,controller_query_times_not_added=True,
        results=results)
    (a.output/'SUMMARY.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    files=[Path(__file__),Path(__file__).with_name('replay_static_association.py')]+list(a.source.glob('event_sphere_tracker*.py'))+[a.source/'event_bearing_flight.py']
    (a.output/'SOURCE_HASHES.json').write_text(json.dumps({str(x):common.sha(x) for x in files},indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
