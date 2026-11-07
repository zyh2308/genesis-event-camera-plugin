"""Analytic sphere silhouettes validate off-axis metric reconstruction."""
import json
import numpy as np
from event_sphere_tracker_contour_q80_occlusion import sphere_from_edge_pixels

K=np.array([[433.,0.,320.],[0.,433.,240.],[0.,0.,1.]])
radius=.04;results=[]
for center in (np.array([.0,.0,2.]),np.array([.35,-.25,1.7]),np.array([-.2,.15,.65])):
    n=center/np.linalg.norm(center);e=np.cross(n,[0.,1.,0.]);e/=np.linalg.norm(e);f=np.cross(n,e)
    alpha=np.arcsin(radius/np.linalg.norm(center));angles=np.linspace(0,2*np.pi,120,endpoint=False)
    directions=np.cos(alpha)*n+np.sin(alpha)*(np.cos(angles)[:,None]*e+np.sin(angles)[:,None]*f)
    q=directions@K.T;points=q[:,:2]/q[:,2,None]
    fit=sphere_from_edge_pixels(points,K,radius);assert fit is not None
    error=float(np.linalg.norm(fit['camera_position']-center));assert error<1e-5,error
    results.append(dict(center=center.tolist(),error_m=error))
print(json.dumps(dict(status='PASS',cases=results)))
