"""Post-test diagnostic only: reads existing predictions, never trains or selects weights."""
import json,gzip
from pathlib import Path
import numpy as np
O=Path('/mnt/c/Users/zhaoyuhan/Documents/ChatGPT/New project/outputs/epose_diverse_short_20261101');F=O/'evidence/final'
samples=json.loads((F/'training/eventgenesis_samples.json').read_text());train=[r for r in samples if r['sample_id'].startswith('train_')];uv=np.array([r['uv_normalized'] for r in train]);train_center=uv[:,0];train_size=np.ptp(uv[:,1:],axis=1)
real=json.loads((F/'real_eval/eventgenesis_predictions.json').read_text());truth=np.array([r['gt_uv_ideal'] for r in real])/[346,260];centers=truth[:,0];sizes=np.ptp(truth[:,1:],axis=1)
out={'scope':'post-test analysis of frozen predictions; r05 is now seen; no training, preprocessing, checkpoint or main metric changes','training_center_min_normalized':train_center.min(0).tolist(),'training_center_max_normalized':train_center.max(0).tolist(),'real_center_min_normalized':centers.min(0).tolist(),'real_center_max_normalized':centers.max(0).tolist(),'real_center_outside_training_axis_aligned_range_fraction':float(np.mean(np.any((centers<train_center.min(0))|(centers>train_center.max(0)),axis=1))),'training_median_projected_bbox_width_height_normalized':np.median(train_size,axis=0).tolist(),'real_median_projected_bbox_width_height_normalized':np.median(sizes,axis=0).tolist(),'real_gt_all_corners_in_frame_fraction':float(np.mean(np.all((truth[:,1:]>=0)&(truth[:,1:]<=1),axis=(1,2)))),'real_events_rectified_median':float(np.median([r['events_rectified'] for r in real])),'real_fixed_image_center_epe_native_px':float(np.linalg.norm((centers-.5)*[346,260],axis=1).mean()),'arms':{}}
augmented=[]
with gzip.open(F/'training/eventgenesis_updates.jsonl.gz','rt') as f:
 for line in f:
  r=json.loads(line)
  for index,(scale,dx,dy) in zip(r['sample_indices'],r['affine']):augmented.append((train_center[index]-.5)*scale+.5+np.array([dx,dy])/416)
augmented=np.asarray(augmented);out.update(actual_augmented_training_examples=len(augmented),actual_augmented_center_min_normalized=augmented.min(0).tolist(),actual_augmented_center_max_normalized=augmented.max(0).tolist(),real_center_outside_actual_augmented_training_axis_aligned_range_fraction=float(np.mean(np.any((centers<augmented.min(0))|(centers>augmented.max(0)),axis=1))))
for arm in ['eventgenesis','v2e']:
 records=json.loads((F/f'real_eval/{arm}_predictions.json').read_text());assert all(a['gt_uv_ideal']==b['gt_uv_ideal'] for a,b in zip(real,records));ss=json.loads((F/f'training/{arm}_samples.json').read_text());pred=np.array([r['predicted_uv'][0] for r in records]);inside=[];pnp=[]
 for r in records:
  gt=np.asarray(r['gt_uv_ideal']);pt=np.asarray(r['predicted_uv'][0]);inside.append(bool(np.all(pt>=gt[1:].min(0)) and np.all(pt<=gt[1:].max(0))))
  if r['pnp_valid']:pnp.append(float(np.linalg.norm(np.asarray(r['reprojected_estimated_uv'][0])-gt[0])))
 out['arms'][arm]={'synthetic_events_median_all_windows':float(np.median([r['event_count'] for r in ss])),'predicted_center_mean_native_px':pred.mean(0).tolist(),'predicted_center_inside_GT_bbox_fraction_posthoc_only':float(np.mean(inside)),'pnp_centroid_epe_mean_native_px_over_valid':float(np.mean(pnp)) if pnp else None}
(O/'posttest_domain_gap.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
