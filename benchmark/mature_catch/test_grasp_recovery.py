import os,json,numpy as np
os.environ['CATCHIT_REVOCABLE_GRASP']='1'
from event_sphere_tracker_contour_q80_occlusion import EventSphereTracker
f=EventSphereTracker(np.diag([200.,200.,1.]),(640,480),.04)
f.attached_offset=np.array([0.,0.,1.]);f.hold_checked_sample_t=0.
f._predict=lambda t:dict(position=np.array([0.,0.,1.]),velocity=np.zeros(3),latest_s=t,fit_error_m=0.)
T=np.eye(4);q=np.zeros(12)
def sample(t,p):return dict(t=t,position=p,camera_origin=[0.,0.,0.],quality=20.,radius_px=8.,maximum_event_s=t)
f.samples=[sample(.01,[0.,0.,1.])];f.observe(.02,T,q);assert f.attached_offset is not None
f.samples.append(sample(.03,[.2,0.,1.]));f.observe(.04,T,q);assert f.attached_offset is not None
f.samples.append(sample(.05,[.2,0.,1.]));f.observe(.06,T,q);assert f.attached_offset is None
assert f.hold_events[-1]['kind']=='assumed_grasp_revoked';assert f.hold_events[-1]['GT_contact_used'] is False
# A consistent hand-relative observation never revokes an assumed grasp.
g=EventSphereTracker(np.diag([200.,200.,1.]),(640,480),.04);g.attached_offset=np.array([0.,0.,1.]);g.hold_checked_sample_t=0.;g._predict=f._predict
for k in range(1,5):g.samples.append(sample(k*.02,[.001,0.,1.]));g.observe(k*.02+.01,T,q)
assert g.attached_offset is not None;assert not g.hold_events
print(json.dumps(dict(status='PASS',single_outlier_ignored=True,contradictory_event_grasp_revoked=True,consistent_hold_preserved=True)))
