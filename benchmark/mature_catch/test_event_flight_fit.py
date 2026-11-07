import numpy as np,json
from event_bearing_flight import flight_state,fit_event_flight
p=np.array([.2,2.4,1.5]);v=np.array([-.15,-2.2,2.3]);g=np.array([0.,0.,-9.8]);samples=[]
for t in np.arange(.62,1.10,.01):
 pos,_=flight_state(p,v,t-.6,g);origin=np.array([.01*t,.3*t,1.4]);distance=np.linalg.norm(pos-origin)
 samples.append(dict(t=float(t),position=pos.tolist(),camera_origin=origin.tolist(),radius_px=433*.04/distance))
f=fit_event_flight(samples,p,(.58,.61),g,433);assert f is not None
pos,vel=flight_state(p,f['initial_velocity'],1.1-f['release_time'],g);gt,gtv=flight_state(p,v,.5,g)
assert np.linalg.norm(pos-gt)<.01,(pos,gt,f);assert np.linalg.norm(vel-gtv)<.04
print(json.dumps(dict(status='PASS',position_error_m=float(np.linalg.norm(pos-gt)),velocity_error_m_s=float(np.linalg.norm(vel-gtv)))))
