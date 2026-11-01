import json
from pathlib import Path
import numpy as np,h5py
def collect_diverse(pose,R,arm,split):
 cfg=json.loads((R/'bank_protocol.json').read_text());keys=np.load(R/'cad_keypoints.npy');rows=[]
 with h5py.File(R/'bank/shared_trajectory.h5','r') as h:
  for name in sorted(h['episodes']):
   g=h['episodes'][name]
   if g.attrs['split']!=split:continue
   times=np.asarray(g['frame_timestamps_s']);K=np.asarray(g['camera_intrinsics_K_px']);C=np.asarray(g['camera_to_world_cv']);T=np.asarray(g['target_pose']);ts,x,y,p,schema=pose.read_event_table(arm,name,h,R/'bank/v2e_events')
   for start in range(cfg['warmup_ms'],cfg['duration_ms'],cfg['label_window_ms']):
    end=start+cfg['label_window_ms'];uv,xyz,ct=pose.project_keypoints(keys,C[end],T[end],K,346,260);label=pose.build_label(uv,346,260);mask=(ts>times[start])&(ts<=times[end]);im=pose.encode_event_window(ts,x,y,p,float(times[start]),float(times[end]),346,260,8.)
    rows.append({'generator':arm,'episode_id':name,'rollout_id':name,'sample_id':f'{name}/ms_{start:03d}_{end:03d}','interval_index':start,'event_start_s':float(times[start]),'event_end_s':float(times[end]),'label_time_s':float(times[end]),'event_count':int(mask.sum()),'event_schema':schema,'image':im,'label':label,'uv':uv.astype(np.float32),'uv_normalized':(uv/[346,260]).astype(np.float32),'keypoints_obj':keys.astype(np.float32),'xyz_cam':xyz.astype(np.float32),'camera_matrix_416':np.diag([416/346,416/260,1.])@K,'camera_from_object':ct})
 return rows
