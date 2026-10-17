from pathlib import Path
import json,sys,cv2,h5py,numpy as np
R=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real');sys.path.insert(0,str(R));from genesis_event_plugin.physics_warp import project_endpoint_to_alpha
root=R/'runtime/sim2real/genesis_hdr_v2/genesis';hits=[]
for p in root.rglob('*.py'):
    if not any(s in str(p) for s in ['vis','render','camera']):continue
    lines=p.read_text(errors='replace').splitlines()
    for i,line in enumerate(lines):
        if any(term in line for term in ['seg_idx','segmentation_idx','link_idx + 1','link.idx + 1','link_idx+1','link.idx+1']):
            hits.append({'file':str(p),'line':i+1,'context':'\n'.join(lines[max(0,i-2):min(len(lines),i+3)])})
out={'renderer_source_hits':hits,'controlled_label_checks':[]}
bank=Path('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v1_torch_cuda_32x2s/shared_trajectory.h5')
with h5py.File(bank,'r') as h:
    for eid in ['s1_06','s1_07']:
        e=h[f'episodes/{eid}'];ids=e['object_segmentation_link_ids'][:];K=e['camera_intrinsics_K_px'][:];cams=e['camera_to_world_cv'][:];poses=e['object_to_world'][:]
        for i in [99,195]:
            raw=e['segmentation_link_idx'][i];seg=np.where(raw>0,raw-1,-1);uv,z,good=project_endpoint_to_alpha(e['depth_m'][i],seg,K,cams[i],cams[i+3],1.,source_index=0,object_world0={int(k):poses[i,c] for c,k in enumerate(ids)},object_world1={int(k):poses[i+3,c] for c,k in enumerate(ids)})
            yy,xx=np.mgrid[:raw.shape[0],:raw.shape[1]];correct=np.stack([uv[0]-xx,uv[1]-yy]);old=e['flow_forward_30ms_px'][i//3];oldvalid=e['flow_valid_30ms'][i//3].astype(bool);m=(raw==1)&oldvalid
            out['controlled_label_checks'].append({'rollout':eid,'endpoint':i,'camera_delta_max':float(np.max(np.abs(cams[i+3]-cams[i]))),'plane_pose_delta_max':float(np.max(np.abs(poses[i+3,0]-poses[i,0]))),'plane_old_valid_pixels':int(m.sum()),'old_plane_GT_magnitude_mean':float(np.hypot(old[0],old[1])[m].mean()),'canonical_id_plane_displacement_mean':float(np.hypot(correct[0],correct[1])[m].mean()),'raw_seg_values':np.unique(raw).tolist(),'pose_link_ids':ids.tolist()})
print(json.dumps(out,ensure_ascii=False))
