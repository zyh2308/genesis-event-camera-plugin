"""Causal known-initial-position flight fitting from event bearings.

No target truth, target release flag, segmentation or ongoing depth input.
"""
import numpy as np
from scipy.optimize import least_squares


def flight_state(position0, velocity0, dt, gravity, drag=.30):
    dt=np.maximum(np.asarray(dt,dtype=float),0.)
    if drag>1e-8:
        a=-np.expm1(-drag*dt)/drag
        position=position0+a[...,None]*velocity0+((dt-a)/drag)[...,None]*gravity
        velocity=np.exp(-drag*dt)[...,None]*velocity0+a[...,None]*gravity
    else:
        position=position0+dt[...,None]*velocity0+.5*dt[...,None]**2*gravity
        velocity=velocity0+dt[...,None]*gravity
    return position,velocity


def fit_event_flight(samples, position0, release_bracket, gravity, focal_length,
                     initial_velocity=None, previous=None, drag=.30):
    if len(samples)<4 or samples[-1]['t']-samples[0]['t']<.025:return None
    times=np.array([s['t'] for s in samples]);origins=np.array([s['camera_origin'] for s in samples]);rays=np.array([s['position'] for s in samples])-origins
    rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    low,high=map(float,release_bracket);high=min(high,float(times[0]));low=min(low,high-.001)
    prior=np.array([0.,-1.8,2.5]) if initial_velocity is None else np.asarray(initial_velocity,dtype=float)
    tmid=.5*(low+high);x0=np.r_[prior,tmid] if previous is None else np.r_[previous['initial_velocity'],previous['release_time']]
    lo=np.array([-4.5,-4.5,-3.,low]);hi=np.array([4.5,1.,7.,high]);x0=np.clip(x0,lo+1e-8,hi-1e-8)
    def residual(x):
        pos,_=flight_state(position0,x[:3],times-x[3],gravity,drag);rel=pos-origins;depth=np.einsum('ij,ij->i',rel,rays)
        perp=rel-depth[:,None]*rays
        image=perp*focal_length/np.maximum(depth[:,None],.2)
        # Weak velocity prior only regularizes initial near-degenerate prefixes.
        observed_range=np.linalg.norm(np.array([s['position'] for s in samples])-origins,axis=1)
        radius=np.array([s['radius_px'] for s in samples]);sigma=np.maximum(.025,observed_range/np.maximum(radius,2.)*.65)
        radial=(np.linalg.norm(rel,axis=1)-observed_range)/sigma
        return np.r_[image.ravel(),radial,(x[:3]-prior)/5.,(x[3]-tmid)/.03]
    result=least_squares(residual,x0,bounds=(lo,hi),loss='soft_l1',f_scale=1.,max_nfev=45,xtol=1e-7,ftol=1e-7,gtol=1e-7)
    errors=residual(result.x)[:3*len(samples)].reshape(-1,3);error=float(np.median(np.linalg.norm(errors,axis=1)))
    if not np.isfinite(result.x).all() or error>4.:return None
    return dict(initial_velocity=result.x[:3],release_time=float(result.x[3]),median_bearing_error_px=error,
                observations=len(samples),latest_s=float(times[-1]),function_evaluations=int(result.nfev))
