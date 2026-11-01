"""Full-bank synthetic input audit; GT boxes are diagnostics, never model inputs."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np,h5py,cv2
cv2.setNumThreads(1)
R=Path(os.environ['EPOSE_REPAIR_RUN']);P=Path('/home/zhaoyuhan/Eventbased_WAM')
sys.path[:0]=[str(R/'sources'),str(P/'runs/sim2real/20261101-epose-wood-short-research-v1/sources/benchmark/sim2real/scripts')]
import run_pose_stage2_pilot as pose
from diverse_samples import collect_diverse
cfg=json.loads((R/'bank_protocol.json').read_text());keys=np.load(R/'cad_keypoints.npy');rows=[]
with h5py.File(R/'bank/shared_trajectory.h5','r') as h:
 for name in sorted(h['episodes']):
  g=h['episodes'][name];K=np.asarray(g['camera_intrinsics_K_px']);C=np.asarray(g['camera_to_world_cv']);T=np.asarray(g['target_pose']);times=np.asarray(g['frame_timestamps_s'])
  tables={arm:pose.read_event_table(arm,name,h,R/'bank/v2e_events') for arm in ['eventgenesis','v2e']}
  for start in range(cfg['warmup_ms'],cfg['duration_ms'],cfg['label_window_ms']):
   end=start+cfg['label_window_ms'];uv0=pose.project_keypoints(keys,C[start],T[start],K,346,260)[0];uv1=pose.project_keypoints(keys,C[end],T[end],K,346,260)[0];uv=np.vstack([uv0,uv1]);lo=uv.min(0)-2;hi=uv.max(0)+2
   row={'rollout':name,'start_ms':start,'centroid_motion_px_per10ms':float(np.linalg.norm(uv1[0]-uv0[0])),'arms':{}}
   for arm,(ts,x,y,p,schema) in tables.items():
    w=(ts>times[start])&(ts<=times[end]);inside=w&(x>=lo[0])&(x<=hi[0])&(y>=lo[1])&(y<=hi[1]);n=int(w.sum());ni=int(inside.sum());row['arms'][arm]={'events':n,'bbox_events_upper_bound':ni,'bbox_fraction_upper_bound':ni/max(1,n)}
   rows.append(row)
summary={'windows':len(rows),'centroid_motion_median_px_per10ms':float(np.median([r['centroid_motion_px_per10ms'] for r in rows])),'centroid_motion_min_px_per10ms':float(min(r['centroid_motion_px_per10ms'] for r in rows)),'arms':{},'GT_bounding_box_used_only_for_diagnostic_not_inference':True}
for arm in ['eventgenesis','v2e']:
 summary['arms'][arm]={key:float(np.median([r['arms'][arm][key] for r in rows])) for key in ['events','bbox_events_upper_bound','bbox_fraction_upper_bound']}
assert summary['centroid_motion_min_px_per10ms']>1,summary
(R/'foreground_observability_audit.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2))
reference=None;counts={};physical={}
for arm in ['eventgenesis','v2e']:
 train=collect_diverse(pose,R,arm,'train');val=collect_diverse(pose,R,arm,'validation');items=train+val;labels=[(s['sample_id'],s['label'].tolist(),s['event_start_s'],s['label_time_s']) for s in items]
 assert len(train)==cfg['train_rollouts']*4 and len(val)==cfg['validation_rollouts']*4
 if reference is None:reference=labels
 else:assert labels==reference
 assert all(np.isfinite(s['image']).all() and np.isfinite(s['label']).all() for s in items)
 counts[arm]=[s['event_count'] for s in items]
 out=R/'input_previews';out.mkdir(exist_ok=True)
 for name in ['train_00','train_04','train_08','validation_00']:
  sample=next(s for s in items if s['episode_id']==name and s['interval_index']==60)
  image=sample['image'];image=image.numpy() if hasattr(image,'numpy') else image
  if image.shape[0]==3:image=np.transpose(image,(1,2,0))
  display=np.full((image.shape[0],image.shape[1],3),255.,dtype=float);on=image[:,:,0];off=image[:,:,1];display[:,:,1]-=np.clip(on+off,0,1)*255;display[:,:,0]-=np.clip(off,0,1)*255;display[:,:,2]-=np.clip(on,0,1)*255
  cv2.imwrite(str(out/f'{name}_{arm}.png'),cv2.cvtColor(np.uint8(np.clip(display,0,255)),cv2.COLOR_RGB2BGR))
 physical[arm]={'train_windows':len(train),'validation_windows':len(val)}
(R/'bank_preflight.json').write_text(json.dumps({'passed':True,'paired_labels_equal':True,'train_windows':len(train),'validation_windows':len(val),'event_counts':counts,'no_real_data_access':True},indent=2))
files={}
for p in [R/'bank/shared_trajectory.h5']+sorted((R/'bank/v2e_events').glob('*.h5')):files[str(p.relative_to(R))]={'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(R/'dataset_artifact_manifest.json').write_text(json.dumps({'files':files},indent=2))
print(json.dumps(summary),flush=True)
