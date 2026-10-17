from pathlib import Path
import cv2,h5py,json,numpy as np
P=Path('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v1_torch_cuda_32x2s/shared_trajectory.h5')
rows=[]
with h5py.File(P,'r') as h:
 for eid in ['s0_00','s1_06','s2_00','s3_06']:
  e=h[f'episodes/{eid}'];K=e['camera_intrinsics_K_px'][:]
  for i in [0,99,195]:
   raw=e['segmentation_link_idx'][i];depth=e['depth_m'][i];T=e['camera_to_world_cv'][i];yy,xx=np.mgrid[:raw.shape[0],:raw.shape[1]]
   ground=(raw==1).astype(np.uint8);inside=cv2.erode(ground,np.ones((9,9),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
   results=[]
   for offset in [0.,.5,.75,.875,1.0]:
    rays=np.stack([(xx+offset-K[0,2])/K[0,0],(yy+offset-K[1,2])/K[1,1],np.ones_like(depth)])
    camera_point=rays*depth;world=(T[:3,:3]@camera_point.reshape(3,-1)+T[:3,3,None]).reshape(3,*raw.shape)
    results.append({'pixel_center_offset':offset,'plane_z_abs_quantiles_all':np.quantile(abs(world[2,ground.astype(bool)]),[.5,.9,.99]).tolist(),'plane_z_abs_quantiles_eroded':np.quantile(abs(world[2,inside]),[.5,.9,.99]).tolist(),'depth_range_eroded':np.quantile(depth[inside],[0,.5,1]).tolist()})
   rows.append({'rollout':eid,'endpoint':i,'K':K.tolist(),'checks':results})
source=[]
root=Path('/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2/genesis')
for name in ['ext/pyrender/jit_render.py','ext/pyrender/renderer.py']:
 p=root/name;lines=p.read_text().splitlines();terms=['dpscale','def read_depth_buf','def _resize_image','depth = 2.0','z_near *','z_far *']
 for n,line in enumerate(lines):
  if any(t in line for t in terms):source.append({'file':str(p),'line':n+1,'text':'\n'.join(str(i+1)+': '+lines[i] for i in range(max(0,n-3),min(len(lines),n+21)))})
print(json.dumps({'depth_checks':rows,'source':source}))
