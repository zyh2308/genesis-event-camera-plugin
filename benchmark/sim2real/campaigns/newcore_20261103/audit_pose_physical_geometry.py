"""Independent raster/label and GT-PnP checks, using synthetic states only."""
import json,sys
from pathlib import Path
import cv2,h5py,numpy as np
R=Path(sys.argv[1]);rows=[];cv2.setNumThreads(1)
for file in sorted((R/'pose_bank/physical').glob('*.h5')):
 with h5py.File(file,'r') as h:
  g=h['episodes'][file.stem];keys=np.asarray(g['cad_keypoints']);K=np.asarray(g['camera_intrinsics_K_px']);target=int(g.attrs['target_link_id'])
  for i in [0,5,10,15,20]:
   C=np.asarray(g['camera_to_world_cv'][i]);T=np.asarray(g['target_pose'][i]);Tco=np.linalg.inv(C)@T;xyz=(Tco@np.c_[keys,np.ones(9)].T).T[:,:3];q=(K@xyz.T).T;uv=q[:,:2]/q[:,2:];seg=np.asarray(g['segmentation_link_idx'][i]);ys,xs=np.where(seg==target);assert len(xs)>0,(file.stem,i,'target not rendered')
   lo=uv[1:].min(0);hi=uv[1:].max(0);outside=((xs<lo[0]-2)|(xs>hi[0]+2)|(ys<lo[1]-2)|(ys>hi[1]+2)).mean()
   assert outside<.01,(file.stem,i,'raster outside CAD projection',outside,lo,hi,[xs.min(),ys.min(),xs.max(),ys.max()])
   bbox=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1]);projected=np.array([max(0,lo[0]),max(0,lo[1]),min(346,hi[0]),min(260,hi[1])]);inter=np.maximum(0,np.minimum(bbox[2:],projected[2:])-np.maximum(bbox[:2],projected[:2])).prod();union=(bbox[2:]-bbox[:2]).prod()+(projected[2:]-projected[:2]).prod()-inter;iou=inter/union
   ok,rvec,tvec=cv2.solvePnP(keys[1:],uv[1:],K,None,flags=cv2.SOLVEPNP_EPNP);assert ok
   rot=cv2.Rodrigues(rvec)[0];terr=float(np.linalg.norm(tvec[:,0]-Tco[:3,3]));rerr=float(np.linalg.norm(rot-Tco[:3,:3]));assert terr<1e-4 and rerr<1e-3,(file.stem,i,terr,rerr)
   rows.append({'rollout':file.stem,'frame':i,'rendered_pixels':len(xs),'raster_outside_projected_box_fraction':float(outside),'raster_box_projected_box_iou':float(iou),'GT_PnP_translation_error_m':terr,'GT_PnP_rotation_matrix_error':rerr})
assert len(rows)==640
result={'passed':True,'frames':640,'rollouts':128,'raster_vs_CAD_bbox_min_iou':min(r['raster_box_projected_box_iou'] for r in rows),'raster_vs_CAD_bbox_median_iou':float(np.median([r['raster_box_projected_box_iou'] for r in rows])),'max_GT_PnP_translation_error_m':max(r['GT_PnP_translation_error_m'] for r in rows),'real_data_used':False,'rows':rows}
(R/'pose_physical_geometry_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
