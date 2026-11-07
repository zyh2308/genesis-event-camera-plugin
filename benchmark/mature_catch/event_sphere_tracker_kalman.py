"""Event sphere filter with range uncertainty and a public incoming-ball prior.
No episode position/velocity, depth, segmentation or target GT is read here.
"""
from collections import deque
import numpy as np
from event_sphere_tracker_contour_q80_occlusion import EventSphereTracker as Contour

class EventSphereTracker(Contour):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.filter_mean=None;self.filter_cov=None;self.filter_time=None;self.filter_first_time=None
        self.filter_processed=0;self.filter_accepted=0;self.filter_rejected=0;self.innovations=deque(maxlen=30)
        self.minimum_filter_measurements=3;self.minimum_filter_span_s=.02
        self.drag=.30;self.gravity=np.array([0.,0.,-9.81])
        self.initial_velocity_prior=np.array([0.,-1.8,2.5])

    def _detect(self,packet):
        s=super()._detect(packet)
        if s is not None:
            s['camera_origin']=packet['T'][:3,3].tolist()
        return s

    def _transition(self,dt):
        e=np.exp(-self.drag*dt);a=(1-e)/self.drag;I=np.eye(3)
        F=np.block([[I,a*I],[np.zeros((3,3)),e*I]])
        b=np.r_[self.gravity*(dt-a)/self.drag,self.gravity*a]
        G=np.vstack((.5*dt*dt*I,dt*I));Q=.75**2*(G@G.T)+np.eye(6)*1e-10
        return F,b,Q

    def _measurement_cov(self,s):
        ray=np.array(s['position'])-np.array(s['camera_origin']);distance=np.linalg.norm(ray);n=ray/distance
        transverse=max(.0015,distance/float(self.K[0,0])*.85)
        radial=max(.025,distance/max(s['radius_px'],2)*.65)
        return transverse**2*(np.eye(3)-n[:,None]*n[None,:])+radial**2*n[:,None]*n[None,:],n

    def _predict(self,time_s):
        eligible=[s for s in self.samples if s['t']<=time_s+1e-8]
        for s in eligible[self.filter_processed:]:
            self.filter_processed+=1
            if 'camera_origin' not in s:continue
            R,n=self._measurement_cov(s);measurement=np.array(s['position']);t=s['t']
            if self.filter_mean is None:
                self.filter_mean=np.r_[measurement,self.initial_velocity_prior]
                self.filter_cov=np.zeros((6,6));self.filter_cov[:3,:3]=R;self.filter_cov[3:,3:]=np.diag([.75**2,.45**2,3.**2])
                self.filter_time=t;self.filter_first_time=t;self.filter_accepted=1;continue
            dt=t-self.filter_time
            if dt<=0:continue
            F,b,Q=self._transition(dt);pred=F@self.filter_mean+b;P=F@self.filter_cov@F.T+Q
            innovation=measurement-pred[:3];S=P[:3,:3]+R
            mahal=float(innovation@np.linalg.solve(S,innovation))
            if mahal>25:
                self.filter_rejected+=1;continue
            gain=np.linalg.solve(S,P[:3,:]).T
            self.filter_mean=pred+gain@innovation
            # Joseph form preserves positive covariance at small pixel noise.
            H=np.zeros((3,6));H[:3,:3]=np.eye(3);A=np.eye(6)-gain@H
            self.filter_cov=A@P@A.T+gain@R@gain.T
            self.filter_time=t;self.filter_accepted+=1
            self.innovations.append(float(np.linalg.norm(np.cross(innovation,n))))
        if self.filter_mean is None or self.filter_accepted<self.minimum_filter_measurements or self.filter_time-self.filter_first_time<self.minimum_filter_span_s-1e-8:return None
        F,b,Q=self._transition(max(0.,time_s-self.filter_time));state=F@self.filter_mean+b
        return dict(time_s=float(time_s),position=state[:3],velocity=state[3:],acceleration=self.gravity-self.drag*state[3:],latest_s=float(self.filter_time),
                    n=self.filter_accepted,fit_error_m=float(np.median(self.innovations)) if self.innovations else 0.,
                    fit_kind='causal anisotropic sphere Kalman, nominal gravity and linear drag',
                    fixed_initial_velocity_prior_m_s=self.initial_velocity_prior.tolist(),accepted=self.filter_accepted,rejected=self.filter_rejected)
