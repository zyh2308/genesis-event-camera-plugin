from pathlib import Path
import json,sys,h5py,numpy as np
R=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real');sys.path.insert(0,str(R))
from genesis_event_plugin.physics_warp import project_endpoint_to_alpha
P=Path('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v1_torch_cuda_32x2s/shared_trajectory.h5')
out=[]
with h5py.File(P,'r') as h:
    for eid in ['s0_06','s1_06','s2_06','s3_06']:
        e=h[f'episodes/{eid}'];ids=e['object_segmentation_link_ids'][:];K=e['camera_intrinsics_K_px'][:];cams=e['camera_to_world_cv'][:];poses=e['object_to_world'][:];stats=[]
        for i in [0,99,195]:
            seg=e['segmentation_link_idx'][i];unique,counts=np.unique(seg,return_counts=True);flow=e['flow_forward_30ms_px'][i//3];valid=e['flow_valid_30ms'][i//3].astype(bool);cam_inv=np.linalg.inv(cams[i]);centroids=[]
            for col,oid in enumerate(ids):
                v=cam_inv@poses[i,col]@np.array([0,0,0,1.]);uv=(K@v[:3])[:2]/v[2]
                x,y=np.rint(uv).astype(int)
                centroids.append({'pose_link_id':int(oid),'world_position':poses[i,col,:3,3].tolist(),'projected_centroid_uv':uv.tolist(),'rendered_seg_at_centroid':int(seg[y,x]) if 0<=x<seg.shape[1] and 0<=y<seg.shape[0] else None,'object_pose_delta_norm':float(np.max(np.abs(poses[i+3,col]-poses[i,col])))})
            per_seg=[]
            for sid,n in zip(unique,counts):
                m=(seg==sid)&valid
                per_seg.append({'seg':int(sid),'pixels':int(n),'valid_pixels':int(m.sum()),'mean_GT_flow':np.mean(flow[:,m],axis=1).tolist() if m.any() else None,'mean_GT_magnitude':float(np.hypot(flow[0],flow[1])[m].mean()) if m.any() else None})
            # Static-pose reproject is an identity regardless of depth.
            uv,_,good=project_endpoint_to_alpha(e['depth_m'][i],seg,K,cams[i],cams[i],1.,object_world0={int(k):poses[i,c] for c,k in enumerate(ids)},object_world1={int(k):poses[i,c] for c,k in enumerate(ids)})
            yy,xx=np.mgrid[:seg.shape[0],:seg.shape[1]];err=np.hypot(uv[0]-xx,uv[1]-yy)
            stats.append({'endpoint':i,'camera_delta_max':float(np.abs(cams[i+3]-cams[i]).max()),'segmentation':per_seg,'centroids':centroids,'identity_reprojection_max_error':float(err[good].max())})
        out.append({'rollout':eid,'pose_link_ids':ids.tolist(),'frame_checks':stats})
print(json.dumps(out))
