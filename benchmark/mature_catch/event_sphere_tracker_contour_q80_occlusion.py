"""Event-only known-radius sphere geometry. No RGB, depth, segmentation or GT.

The first pipeline demo uses this non-learned front end, preserving the
published learned catching policy. It is not a learned-perception comparison.
"""
from collections import deque
import math
import os

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def sphere_from_edge_pixels(points,K,radius):
    points=np.asarray(points,dtype=float)
    matrix=np.column_stack((2*points,np.ones(len(points))))
    fit=np.linalg.lstsq(matrix,np.sum(points**2,axis=1),rcond=None)[0]
    center=fit[:2];r2=fit[2]+float(center@center)
    if r2<=0:return None
    image_radius=math.sqrt(r2)
    if image_radius<2 or image_radius>120:return None
    circle=least_squares(lambda x:np.linalg.norm(points-x[:2],axis=1)-x[2],np.r_[center,image_radius],loss='soft_l1',f_scale=.7,max_nfev=25).x
    if not 2<=circle[2]<=120:return None
    angles=np.arctan2(points[:,1]-circle[1],points[:,0]-circle[0])
    coverage=len(np.unique(np.floor((angles+np.pi)/(2*np.pi)*16).astype(int)))
    if coverage<6:return None
    residual=float(np.median(np.abs(np.linalg.norm(points-circle[:2],axis=1)-circle[2])))
    if residual>max(1.2,.15*circle[2]):return None
    rays=np.column_stack((points,np.ones(len(points))))@np.linalg.inv(K).T
    rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    z=float(K[0,0])*radius/circle[2];initial=np.linalg.solve(K,np.r_[circle[:2],1.])*z
    if not (radius*1.01<initial[2]<6. and np.all(np.abs(initial[:2])<5.)):return None
    def tangent_residual(c):
        distances=np.linalg.norm(np.cross(rays,c),axis=1)
        return (distances-radius)*float(K[0,0])/max(np.linalg.norm(c),radius*1.1)
    refined=least_squares(tangent_residual,initial,bounds=([-5.,-5.,radius*1.01],[5.,5.,6.]),loss='soft_l1',f_scale=.65,max_nfev=25)
    if not refined.success and refined.nfev>=25:return None
    if not np.isfinite(refined.x).all():return None
    return dict(camera_position=refined.x,uv=circle[:2],radius_px=float(circle[2]),residual_px=residual,angle_bins=coverage)

class EventSphereTracker:
    def __init__(self,K,resolution_wh,radius_m):
        self.K=np.asarray(K,dtype=float);self.W,self.H=map(int,resolution_wh);self.radius=float(radius_m)
        self.pending=deque();self.samples=[];self.diagnostics=[];self.last_event_time=-np.inf
        self.attached_offset=None;self.near_hand_count=0
        self.revocable_grasp=os.environ.get("CATCHIT_REVOCABLE_GRASP")=="1"
        self.hand_history=deque(maxlen=64);self.hold_checked_sample_t=-np.inf
        self.hold_contradictions=[];self.hold_events=[];self.hold_rearm_s=-np.inf

    def push(self,events,available_s,packet_time_s,camera_to_world):
        self.pending.append(dict(events=np.asarray(events),available_s=float(available_s),end_s=float(packet_time_s),T=np.asarray(camera_to_world,dtype=float)))

    def _predict(self,time_s):
        eligible=[s for s in self.samples if s['t']<=time_s+1e-8]
        if not eligible:return None
        latest=eligible[-1]['t'];recent=[s for s in eligible if latest-.14<=s['t']]
        if len(recent)<3:return None
        ts=np.asarray([s['t'] for s in recent]);positions=np.asarray([s['position'] for s in recent])
        if ts[-1]-ts[0]<.02:return None
        dt=ts-latest;acc=np.array([0.,0.,-9.81]);corrected=positions-.5*dt[:,None]**2*acc
        design=np.column_stack((np.ones(len(dt)),dt));coef=np.linalg.lstsq(design,corrected,rcond=None)[0]
        for _ in range(3):
            errors=np.linalg.norm(corrected-design@coef,axis=1);weights=1/np.maximum(errors,.025)
            coef=np.linalg.lstsq(design*weights[:,None],corrected*weights[:,None],rcond=None)[0]
        horizon=time_s-latest
        return dict(time_s=float(time_s),position=coef[0]+coef[1]*horizon+.5*acc*horizon*horizon,velocity=coef[1]+acc*horizon,latest_s=float(ts[-1]),
                    n=len(recent),fit_error_m=float(np.median(np.linalg.norm(corrected-design@coef,axis=1))))

    def _detect(self,packet):
        events=packet['events'];t=packet['end_s']
        events=events[(events[:,0]>=t-.012)&(events[:,0]<=t+1e-8)] if len(events) else events
        if len(events)<12:return None
        x=np.clip(events[:,1].astype(int),0,self.W-1);y=np.clip(events[:,2].astype(int),0,self.H-1)
        mask=np.zeros((self.H,self.W),np.uint8);mask[y,x]=1
        mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        candidates=[];prediction=self._predict(t)
        for label in range(1,n):
            bx,by,bw,bh,area=stats[label]
            if not (4<=bw<=200 and 4<=bh<=200 and .4<=bw/bh<=2.5):continue
            keep=labels[y,x]==label
            if int(keep.sum())<12:continue
            points=np.column_stack((x[keep],y[keep])).astype(float)+.5
            event_t=events[keep,0];measurement_t=float(np.median(event_t))
            recent=[v for v in self.samples if v['t']>=t-.08]
            if len(recent)>=3:
                times=np.array([v['t'] for v in recent]);uvs=np.array([v['uv'] for v in recent]);dt=times-times[-1]
                coef=np.linalg.lstsq(np.column_stack((np.ones(len(dt)),dt)),uvs,rcond=None)[0]
                flow=coef[1]
                if np.linalg.norm(flow)<2000:points+=(measurement_t-event_t)[:,None]*flow
            if len(points)<10:continue
            # Use an angular contour, rather than treating interior events as
            # exact silhouette points or accepting the farthest outlier.
            center=np.median(points,axis=0)
            for _ in range(2):
                angle=np.arctan2(points[:,1]-center[1],points[:,0]-center[0]);bins=np.floor((angle+np.pi)/(2*np.pi)*16).astype(int)%16
                radii=np.linalg.norm(points-center,axis=1);contour=[]
                for k in range(16):
                    idx=np.where(bins==k)[0]
                    if len(idx):
                        target=np.quantile(radii[idx],0.8);chosen=idx[np.argmin(abs(radii[idx]-target))];contour.append(points[chosen])
                if len(contour)<6:break
                contour=np.asarray(contour);mat=np.column_stack((2*contour,np.ones(len(contour))));c=np.linalg.lstsq(mat,np.sum(contour**2,axis=1),rcond=None)[0];center=c[:2]
            if len(contour)<6:continue
            points=np.asarray(contour)
            candidate=sphere_from_edge_pixels(points,self.K,self.radius)
            if candidate is None:continue
            world=(packet['T']@np.r_[candidate['camera_position'],1.])[:3]
            if not (-1.2<world[0]<1.2 and .05<world[1]<3.2 and .15<world[2]<2.5):continue
            quality=math.sqrt(len(points))*candidate['angle_bins']/(.5+candidate['residual_px'])
            if prediction is not None:
                distance=float(np.linalg.norm(world-prediction['position']))
                if distance>.30:continue
                quality/=1+distance/.025
            else:
                # Public launch region, not the episode's actual release state.
                quality/=1+max(0.,1.5-world[1])*5
            candidates.append((quality,world,candidate,float(np.median(events[keep,0]))))
        if not candidates:return None
        quality,world,candidate,time_s=max(candidates,key=lambda q:q[0])
        return dict(t=time_s,position=world.tolist(),uv=candidate['uv'].tolist(),radius_px=candidate['radius_px'],
                    quality=float(quality),available_s=packet['available_s'],maximum_event_s=float(events[:,0].max()))

    def _check_assumed_grasp(self,now_s):
        if not self.revocable_grasp or self.attached_offset is None:return
        for sample in self.samples:
            if sample['t']<=self.hold_checked_sample_t+1e-9:continue
            self.hold_checked_sample_t=sample['t']
            if now_s-sample['t']>.08 or sample['quality']<8:continue
            poses=list(self.hand_history)
            points=np.array([(T@np.r_[self.attached_offset,1.])[:3] for t,T in poses])
            times=np.array([t for t,T in poses])
            expected=np.array([np.interp(sample['t'],times,points[:,axis]) for axis in range(3)])
            origin=np.asarray(sample['camera_origin']);ray=np.asarray(sample['position'])-origin;ray/=np.linalg.norm(ray)
            rel=expected-origin
            error_px=float(np.linalg.norm(np.cross(rel,ray))*self.K[0,0]/max(np.linalg.norm(rel),.1))
            limit=max(8.,.75*sample['radius_px'])
            if error_px>limit:self.hold_contradictions.append(sample['t'])
            else:self.hold_contradictions=[]
            if len(self.hold_contradictions)>=2 and self.hold_contradictions[-1]-self.hold_contradictions[-2]<=.05:
                self.hold_events.append(dict(time_s=float(now_s),kind='assumed_grasp_revoked',bearing_error_px=error_px,GT_contact_used=False))
                self.attached_offset=None;self.near_hand_count=0;self.hold_rearm_s=now_s+.12;self.hold_contradictions=[]
                return

    def observe(self,now_s,hand_pose,hand_joint_positions):
        self.hand_history.append((float(now_s),np.asarray(hand_pose).copy()))
        while self.pending and self.pending[0]['available_s']<=now_s+1e-9:
            packet=self.pending.popleft();sample=self._detect(packet)
            if sample is not None:
                self.samples.append(sample);self.last_event_time=max(self.last_event_time,sample['maximum_event_s'])
        state=self._predict(now_s)
        self._check_assumed_grasp(now_s)
        if self.attached_offset is not None:
            T=np.asarray(hand_pose);world=(T@np.r_[self.attached_offset,1.])[:3]
            result=dict(time_s=float(now_s),position=world,velocity=np.zeros(3),source='event_acquisition_then_proprioceptive_grasp_propagation',
                        latest_s=self.samples[-1]['t'],maximum_event_s=self.last_event_time,fresh_event=False)
            self.diagnostics.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in result.items()})
            return result
        if state is None or now_s-state['latest_s']>.40:return None
        if state['fit_error_m']>.12:return None
        T=np.asarray(hand_pose);distance=float(np.linalg.norm(state['position']-T[:3,3]))
        closing=float(np.linalg.norm(hand_joint_positions))>.8
        self.near_hand_count=self.near_hand_count+1 if distance<.18 and closing and now_s>=self.hold_rearm_s else 0
        if self.near_hand_count>=2:
            self.attached_offset=(np.linalg.inv(T)@np.r_[state['position'],1.])[:3]
            self.hold_checked_sample_t=self.samples[-1]['t'] if self.samples else now_s
            self.hold_contradictions=[]
            self.hold_events.append(dict(time_s=float(now_s),kind='assumed_grasp_started',GT_contact_used=False))
        state.update(source='known_radius_event_contour_and_causal_motion_fit',maximum_event_s=self.last_event_time,fresh_event=now_s-state['latest_s']<=.04)
        self.diagnostics.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in state.items()})
        return state
