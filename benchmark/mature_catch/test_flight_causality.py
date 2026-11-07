import os,numpy as np,json
os.environ['CATCHIT_EVENT_FLIGHT_FIT']='1'
from event_sphere_tracker import EventSphereTracker
from event_bearing_flight import flight_state
K=np.array([[433.,0,320],[0,433.,240],[0,0,1.]]);p=np.array([.2,2.4,1.5]);v=np.array([-.15,-2.2,2.3]);g=np.array([0.,0.,-9.81])
def make(with_future):
 f=EventSphereTracker(K,(640,480),.04);f.initialize_rgb(dict(valid=True,position=p.tolist(),camera_origin=[0.,0.,0.],radius_px=6.,range_std_m=.01,initialization_mode='single_frame_RGBD'))
 for t in np.arange(.4,.601,.01):f.samples.append(dict(t=float(t),position=p.tolist(),camera_origin=[0.,0.,0.],radius_px=6.,available_s=float(t+.01),maximum_event_s=float(t)))
 for t in np.arange(.62,.701,.01):
  pos,_=flight_state(p,v,t-.6,g);f.samples.append(dict(t=float(t),position=pos.tolist(),camera_origin=[0.,0.,0.],radius_px=433*.04/np.linalg.norm(pos),available_s=float(t+.01),maximum_event_s=float(t)))
 if with_future:f.samples.append(dict(t=1.0,position=[50.,50.,50.],camera_origin=[0.,0.,0.],radius_px=50.,available_s=1.01,maximum_event_s=1.))
 return f._predict(.70)
a=make(False);b=make(True);assert a is not None and b is not None;assert np.allclose(a['position'],b['position'],atol=1e-12);assert np.allclose(a['velocity'],b['velocity'],atol=1e-12);assert a['latest_s']<=.70+1e-8
print(json.dumps(dict(status='PASS',future_target_observation_cannot_change_past_state=True)))
