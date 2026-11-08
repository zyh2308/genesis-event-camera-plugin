"""Event-state candidates: full silhouette checks and optional causal flight fit."""
import os
import numpy as np
from event_sphere_tracker_rgbd_base import EventSphereTracker as Baseline
from event_bearing_flight import fit_event_flight,flight_state

class EventSphereTracker(Baseline):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.reject_partial=os.environ.get('CATCHIT_REJECT_PARTIAL_CONTOURS')=='1'
        self.use_flight_fit=os.environ.get('CATCHIT_EVENT_FLIGHT_FIT')=='1'
        self.partial_rejections=0;self.flight_solution=None;self.flight_fit_count=-1;self.release_bracket=None
        self.observation_window_ms=float(os.environ.get('CATCHIT_EVENT_OBSERVATION_WINDOW_MS','0'))
        self._observation_windows=None
        self.observation_window_audit=[]
        if self.observation_window_ms!=0:
            from event_observation_windows import EventObservationWindows
            self._observation_windows=EventObservationWindows(self.observation_window_ms)
            self.observation_window_audit=self._observation_windows.audit

    def push(self,events,available_s,packet_time_s,camera_to_world,parent_packet_id=None):
        result=super().push(events,available_s,packet_time_s,camera_to_world)
        if self._observation_windows is not None:
            packet=self.pending[-1]
            if parent_packet_id is not None:packet['parent_packet_id']=int(parent_packet_id)
            self._observation_windows.register_parent(packet)
        return result

    def observe(self,now_s,hand_pose,hand_joint_positions):
        if self._observation_windows is None:
            return super().observe(now_s,hand_pose,hand_joint_positions)
        previous=self._observation_windows.processing_now_s
        self._observation_windows.processing_now_s=float(now_s)
        try:return super().observe(now_s,hand_pose,hand_joint_positions)
        finally:self._observation_windows.processing_now_s=previous

    def _detect(self,packet):
        if self._observation_windows is not None and not packet.get('_observation_window_child',False):
            return self._observation_windows.consume(self,packet,self._detect)
        sample=super()._detect(packet)
        if sample is not None and self.reject_partial:
            u,v=sample['uv'];r=sample['radius_px']
            if not (r+2<=u<=self.W-r-2 and r+2<=v<=self.H-r-2):
                self.partial_rejections+=1
                return None
        return sample

    def _start_motion(self,sample):
        self.release_bracket=(max(0.,self.static_latest_s-.015),self.motion_suspicion[0]['t'])
        return super()._start_motion(sample)

    def _predict(self,time_s):
        state=super()._predict(time_s)
        if not self.use_flight_fit or self.motion_onset_s is None or state is None:return state
        eligible=[s for s in self.samples if self.motion_suspicion[0]['t']<=s['t']<=time_s+1e-8]
        if len(eligible)!=self.flight_fit_count:
            candidate=fit_event_flight(eligible,self.static_position,self.release_bracket,self.gravity,self.K[0,0],self.initial_velocity_prior,self.flight_solution,self.drag)
            if candidate is not None:self.flight_solution=candidate
            self.flight_fit_count=len(eligible)
        fit=self.flight_solution
        if fit is None or fit['latest_s']>time_s+1e-8:return state
        position,velocity=flight_state(self.static_position,fit['initial_velocity'],time_s-fit['release_time'],self.gravity,self.drag)
        state.update(position=position,velocity=velocity,acceleration=self.gravity-self.drag*velocity,
                     latest_s=fit['latest_s'],n=fit['observations'],fit_error_m=fit['median_bearing_error_px']*max(.1,np.linalg.norm(position))/self.K[0,0],
                     fit_kind='causal first-RGBD anchored event bearing/range flight fit',
                     estimated_release_time_s=fit['release_time'],estimated_initial_velocity=fit['initial_velocity'].tolist(),
                     fitting_bearing_error_px=fit['median_bearing_error_px'],partial_contour_rejections=self.partial_rejections)
        return state
