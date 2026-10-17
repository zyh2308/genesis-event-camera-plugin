"""One static plane, two rendering paths; no training and no source edits."""
from pathlib import Path
import json,os,subprocess,sys,time
import cv2,numpy as np
check=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits','-i','0'],capture_output=True,text=True,check=True)
used,util=map(int,check.stdout.strip().split(','))
if used>1024 or util>5:raise RuntimeError(f'GPU0 not idle: {check.stdout}')
import genesis as gs
from genesis_event_plugin.genesis_adapter import camera_to_world_cv
t0=time.monotonic()
gs.init(backend=gs.gpu,logging_level='error')
try:
 s=gs.Scene(sim_options=gs.options.SimOptions(dt=.001,gravity=(0,0,-9.81)),vis_options=gs.options.VisOptions(ambient_light=(.38,.38,.38),segmentation_level='link'),show_viewer=False)
 plane=s.add_entity(gs.morphs.Plane(),surface=gs.surfaces.Default(color=(.38,.40,.43,1)))
 c=s.add_camera(res=(346,260),pos=(1.05,1.12,.78),lookat=(0,0,.24),fov=58,GUI=False,near=.03,far=8.)
 s.build();K=np.array(c.intrinsics);T=camera_to_world_cv(c)
 rgb,combined,seg,_=c.render(rgb=True,depth=True,segmentation=True,radiance=True)
 _,depth_only,seg_only,_=c.render(rgb=False,depth=True,segmentation=True)
 yy,xx=np.mgrid[:260,:346]
 rows=[]
 for name,depth,labels in [('radiance_plus_depth',np.asarray(combined),np.asarray(seg)),('depth_only',np.asarray(depth_only),np.asarray(seg_only))]:
  inside=cv2.erode((labels==1).astype(np.uint8),np.ones((9,9),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
  for offset in [0.,.5,.875]:
   rays=np.stack([(xx+offset-K[0,2])/K[0,0],(yy+offset-K[1,2])/K[1,1],np.ones_like(depth)])
   points=(T[:3,:3]@(rays*depth).reshape(3,-1)+T[:3,3,None]).reshape(3,260,346)
   rows.append({'render_path':name,'offset':offset,'plane_z_abs_quantiles_eroded_m':np.quantile(abs(points[2,inside]),[.5,.9,.99]).tolist()})
 out={'K':K.tolist(),'plane_link':int(plane.links[0].idx),'seg_ids':np.unique(seg).tolist(),'rows':rows,'depth_path_max_abs_difference':float(np.max(abs(combined-depth_only))),'source_runtime':gs.__file__,'runtime_seconds':time.monotonic()-t0}
 print('AUDIT_JSON='+json.dumps(out),flush=True)
finally:gs.destroy()
