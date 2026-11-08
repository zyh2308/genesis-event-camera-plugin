"""CPU-only cached-event isolation. No renderer, physics, GT input, or core edits.

Use --run <actual actor output> (repeatable), --source <sealed actor snapshot>,
--output <fresh directory>. Original mode must reproduce archived detector
samples and flight states before tentative association is interpreted.
"""
from pathlib import Path
import argparse,copy,hashlib,inspect,json,os,sys,textwrap
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import h5py
import numpy as np

def serial(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [serial(v) for v in value]
 return value

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()

def instrument(output):
 import event_sphere_tracker_contour_q80_occlusion as contour
 import event_sphere_tracker_rgbd_base as rgbd
 original=textwrap.dedent(inspect.getsource(contour.EventSphereTracker._detect))
 needle='candidates=[];prediction=self._predict(t)'
 assert original.count(needle)==1
 replacement='''candidates=[];prediction=self._predict(t)
    original_prediction=prediction
    tentative_used=False
    if (self._diag_mode=='tentative' and prediction is not None and
            self.motion_onset_s is None and self.motion_suspicion):
        last=self.motion_suspicion[-1];dt=max(0.,t-last['t'])
        center=(self.static_position+last['innovation']+
                self.initial_velocity_prior*dt+.5*self.gravity*dt*dt)
        prediction=dict(prediction,position=center)
        tentative_used=True
    self._diag_packet_prediction={'end_s':t,'tentative_used':tentative_used,
        'original':None if original_prediction is None else np.asarray(original_prediction['position']).tolist(),
        'association':None if prediction is None else np.asarray(prediction['position']).tolist()}
'''.rstrip()
 patched=original.replace(needle,replacement)
 needle='if distance>.30:continue'
 assert patched.count(needle)==1
 patched=patched.replace(needle,'''self._diag_candidates.append(dict(end_s=float(t),measurement_t=float(measurement_t),
                    world=world.tolist(),uv=candidate['uv'].tolist(),radius_px=float(candidate['radius_px']),
                    quality_before_association=float(quality),distance_m=distance,
                    static_original_distance_m=(None if original_prediction is None else float(np.linalg.norm(world-original_prediction['position']))),
                    gate_m=.30,rejected=bool(distance>.30),tentative_used=tentative_used))
            if distance>.30:continue''')
 ns=dict(vars(contour));exec(compile(patched,'<diagnostic_detector_only>','exec'),ns)
 contour.EventSphereTracker._detect=ns['_detect']
 (output/'instrumented_detector.py.txt').write_text(patched,encoding='utf-8')
 onset=textwrap.dedent(inspect.getsource(rgbd.EventSphereTracker._predict))
 # This function is compiled outside its original class body, so preserve
 # zero-argument super semantics explicitly rather than losing __class__ cell.
 onset=onset.replace('super()._predict(time_s)','super(EventSphereTracker,self)._predict(time_s)')
 needle='suspicious = mahal > 25 and transverse > .010'
 assert onset.count(needle)==1
 onset=onset.replace(needle,needle+'''
            self._diag_onset.append(dict(sample_t=float(sample['t']),available_s=float(sample['available_s']),
                position=np.asarray(sample['position']).tolist(),mahalanobis=mahal,transverse_m=transverse,
                suspicious=bool(suspicious),previous_suspicion_times=[float(x['t']) for x in self.motion_suspicion],
                gap_limit_s=.05,confirmation_required=2))''')
 ns=dict(vars(rgbd));exec(compile(onset,'<diagnostic_onset_logging_only>','exec'),ns)
 rgbd.EventSphereTracker._predict=ns['_predict']
 (output/'instrumented_onset.py.txt').write_text(onset,encoding='utf-8')

def replay(run,mode,source,output,until,flight_fit=False):
 from event_sphere_tracker import EventSphereTracker
 sensor=run/'migration_audit/sensor_episode_000';protocol=json.loads((run/'protocol.json').read_text())
 assert not protocol['event_flight_fit'] and not protocol['reject_partial_contours']
 trace=json.loads((run/'episode_000/trace.json').read_text());initial=json.loads((sensor/'initialization.json').read_text())
 audit=json.loads((sensor/'packet_audit.json').read_text());raw=np.load(sensor/'raw_events.npz')['events']
 assert len(raw)==sum(x['events'] for x in audit)
 archived=json.loads((sensor/'event_samples.json').read_text());result=json.loads((run/'episode_000/result.json').read_text())
 first_touch=result.get('first_touch')
 precontact_limit=until if first_touch is None else min(until,first_touch-.04)
 indexed_rows=[(i,x) for i,x in enumerate(trace) if x['t_before']<=precontact_limit+1e-9]
 control_rows=[x for i,x in indexed_rows]
 # This isolation is deliberately before contact/attachment; no hand GT needed.
 assert all(not (x.get('event_state') or {}).get('source','').startswith('event_acquisition_then_') for x in control_rows)
 with h5py.File(sensor/'sparse_hdr_endpoints.h5','r') as h:
  times=h['time_s'][:];poses=h['camera_to_world_cv'][:];K=h['intrinsics'][:]
  # This block only establishes labels for the output after detector decisions.
  body_poses=h['object_poses'];p0=np.asarray(result['initial']['position'])
  target_index=int(np.argmin(np.linalg.norm(body_poses[0,:,:3,3]-p0,axis=1)))
  assert np.linalg.norm(body_poses[0,target_index,:3,3]-p0)<1e-5
  gt_end_positions=body_poses[:,target_index,:3,3]
 os.environ['CATCHIT_EVENT_FLIGHT_FIT']='1' if flight_fit else '0'
 tracker=EventSphereTracker(K,(640,480),protocol['sphere_radius_m']);tracker.gravity=np.asarray(initial['public_world_gravity_m_s2'])
 assert bool(tracker.use_flight_fit)==bool(flight_fit)
 tracker.initialize_rgb(initial,time_s=0.,available_s=0.)
 tracker._diag_mode=mode;tracker._diag_candidates=[];tracker._diag_onset=[];tracker._diag_packet_prediction=None
 offsets=np.cumsum([0]+[x['events'] for x in audit]);j=0;states=[];packets=[]
 for trace_index,row in indexed_rows:
  now=float(row['t_before'])
  while j<len(audit) and audit[j]['available_s']<=now+1e-9:
   a=audit[j];k=int(np.argmin(abs(times-a['time_s'])));assert abs(times[k]-a['time_s'])<1e-8
   packet=dict(events=raw[offsets[j]:offsets[j+1]],available_s=a['available_s'],end_s=a['time_s'],T=poses[k])
   unconstrained=None
   if mode=='original' and .65<=a['time_s']<=.90:
    shadow=copy.deepcopy(tracker);shadow._predict=lambda t:None
    unconstrained=shadow._detect(packet)
   tracker._diag_packet_prediction=None
   before=len(tracker._diag_candidates);sample=tracker._detect(packet)
   if sample is not None:
    # Kalman._detect already adds this in the sealed source; retain the actual
    # packet geometry explicitly, matching live RGBD ingestion if its caller
    # also supplies it. This is camera calibration, never object truth.
    sample['camera_origin']=np.asarray(packet['T'])[:3,3].tolist()
    tracker.samples.append(sample);tracker.last_event_time=max(tracker.last_event_time,sample['maximum_event_s'])
   candidates=copy.deepcopy(tracker._diag_candidates[before:])
   # GT is appended exclusively after detection. It is never passed to tracker.
   for candidate in candidates:
    candidate['scoring_only_error_to_GT_at_packet_end_m']=float(np.linalg.norm(np.asarray(candidate['world'])-gt_end_positions[k]))
   packets.append(dict(end_s=a['time_s'],available_s=a['available_s'],consumed_at_s=now,
       detected=serial(sample),prediction=copy.deepcopy(tracker._diag_packet_prediction),
       unconstrained_shadow_detection=serial(unconstrained),candidates=candidates))
   j+=1
  predicted=tracker._predict(now)
  state=dict(time_s=now,position=None if predicted is None else serial(predicted['position']),
      velocity=None if predicted is None else serial(predicted['velocity']),
      motion_onset_s=tracker.motion_onset_s,detected_samples=len(tracker.samples),
      suspicion_times=[float(s['t']) for s in tracker.motion_suspicion])
  if predicted is not None:
   state.update(latest_measurement_s=float(predicted['latest_s']),measurement_age_s=now-float(predicted['latest_s']),
       fresh_at_40ms=bool(now-float(predicted['latest_s'])<=.04),
       maximum_event_s=float(tracker.last_event_time) if np.isfinite(tracker.last_event_time) else None)
   state['scoring_only_position_error_m']=float(np.linalg.norm(np.asarray(predicted['position'])-row['scoring_only_GT_position_at_control']))
   if trace_index>0:
    # The logged native object's last six DOF are world linear/angular velocity.
    # Previous post-action state is this control query's actual physical state.
    gt_velocity=np.asarray(trace[trace_index-1]['qvel'][-6:-3],dtype=float)
    state['scoring_only_velocity_error_m_s']=float(np.linalg.norm(np.asarray(predicted['velocity'])-gt_velocity))
   camera_index=int(np.argmin(abs(times-now)))
   if abs(times[camera_index]-now)<1e-8:
    origin=poses[camera_index,:3,3];gt_position=np.asarray(row['scoring_only_GT_position_at_control'])
    gt_ray=gt_position-origin;gt_range=float(np.linalg.norm(gt_ray));gt_ray/=gt_range
    delta=np.asarray(predicted['position'])-gt_position
    state['scoring_only_signed_range_error_m']=float(np.linalg.norm(np.asarray(predicted['position'])-origin)-gt_range)
    state['scoring_only_absolute_range_error_m']=abs(state['scoring_only_signed_range_error_m'])
    state['scoring_only_transverse_position_error_m']=float(np.linalg.norm(delta-float(delta@gt_ray)*gt_ray))
   if flight_fit and tracker.flight_solution is not None:
    state['estimated_release_time_s']=float(tracker.flight_solution['release_time'])
    state['flight_fit_observations']=int(tracker.flight_solution['observations'])
    state['flight_fit_bearing_median_px']=float(tracker.flight_solution['median_bearing_error_px'])
  states.append(state)
 parity=None
 if mode=='original':
  expected=[x for x in archived if x['available_s']<=control_rows[-1]['t_before']+1e-9]
  assert len(expected)==len(tracker.samples),(len(expected),len(tracker.samples))
  max_sample=max([float(np.linalg.norm(np.asarray(a['position'])-np.asarray(b['position']))) for a,b in zip(expected,tracker.samples)]+[0.])
  max_state=max([float(np.linalg.norm(np.asarray(s['position'])-r['event_state']['position'])) for s,r in zip(states,control_rows) if s['position'] is not None]+[0.])
  max_velocity=max([float(np.linalg.norm(np.asarray(s['velocity'])-r['event_state']['velocity'])) for s,r in zip(states,control_rows) if s['velocity'] is not None]+[0.])
  assert max(max_sample,max_state,max_velocity)<1e-8,(max_sample,max_state,max_velocity)
  parity=dict(status='PASS',samples=len(expected),max_sample_position_difference_m=max_sample,max_state_position_difference_m=max_state,max_state_velocity_difference_m_s=max_velocity)
 graded=[x for x in states if x['time_s']>=result['initial']['release_time_s']-1e-9]
 grading={}
 for metric in ['scoring_only_position_error_m','scoring_only_velocity_error_m_s','scoring_only_absolute_range_error_m','scoring_only_transverse_position_error_m']:
  values=[x[metric] for x in graded if metric in x]
  grading[metric]=dict(n=len(values),median=float(np.median(values)) if values else None,p95=float(np.percentile(values,95)) if values else None)
 payload=dict(mode=mode,run=str(run),until_s=until,actual_precontact_limit_s=precontact_limit,source=str(source),baseline_parity=parity,
   scope='flight-only exact cached events and original packet availability; frozen event generator and policy untouched; GT output scoring only; no hand/physics/render calls',
   thresholds=dict(association_m=.30,confirmation_measurements=2,confirmation_max_gap_s=.05),
   candidate_change=('native production association hook; no monkeypatch' if mode.startswith('production') else ('tentative association center only; public output remains static until original two confirmations' if mode=='tentative' else 'none')),
   flight_fit_enabled=bool(flight_fit),scoring_only_postrelease_precontact=grading,
   motion_onset_s=tracker.motion_onset_s,
   scoring_only_pre_release_false_onset=bool(tracker.motion_onset_s is not None and tracker.motion_onset_s<result['initial']['release_time_s']-1e-9),
   samples=serial(tracker.samples),onset_checks=tracker._diag_onset,states=states,packets=packets)
 name=run.parent.name+'_'+protocol['event_generator']+'_'+mode;dest=output/(name+'.json');assert not dest.exists();dest.write_text(json.dumps(serial(payload),indent=2),encoding='utf-8')
 return dict(mode=mode,case=run.parent.name,method=protocol['event_generator'],motion_onset_s=tracker.motion_onset_s,samples=len(tracker.samples),baseline_parity=parity,flight_fit_enabled=bool(flight_fit),scoring_only_postrelease_precontact=grading,
   scoring_only_pre_release_false_onset=payload['scoring_only_pre_release_false_onset'],file=name+'.json')

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--until',type=float,default=.96);p.add_argument('--production',action='store_true',help='Load native source hook without diagnostic monkeypatching; skip old-state parity requirement');p.add_argument('--flight-fit',action='store_true',help='Production-only diagnostic: enable existing causal first-RGBD/bearing/gravity fit; no source/core/weight edits');a=p.parse_args()
 if a.flight_fit and not a.production:p.error('--flight-fit requires --production')
 a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.source))
 os.environ.update(CATCHIT_REJECT_PARTIAL_CONTOURS='0',CATCHIT_EVENT_FLIGHT_FIT='0',CATCHIT_REVOCABLE_GRASP='0')
 if not a.production:instrument(a.output)
 results=[]
 for run in a.run:
  for mode in (['production_flight_fit' if a.flight_fit else 'production'] if a.production else ['original','tentative']):results.append(replay(run,mode,a.source,a.output,a.until,flight_fit=a.flight_fit))
 assert 'genesis' not in sys.modules and 'torch' not in sys.modules
 (a.output/'SUMMARY.json').write_text(json.dumps({'status':'PASS','physics_steps':0,'renderer_calls':0,'CUDA_calls':0,'results':results},indent=2),encoding='utf-8')
 inputs=[__file__]+[str(x) for x in a.source.glob('event_sphere_tracker*.py')]+[str(a.source/'event_bearing_flight.py')]
 (a.output/'SOURCE_HASHES.json').write_text(json.dumps({x:sha(x) for x in inputs},indent=2),encoding='utf-8')
 print(json.dumps({'status':'PASS','results':results}),flush=True)
if __name__=='__main__':main()
