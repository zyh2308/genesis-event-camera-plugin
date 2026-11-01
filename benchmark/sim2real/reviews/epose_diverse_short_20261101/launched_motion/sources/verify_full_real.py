"""Independent real metric/source recomputation after all checkpoints are frozen."""
import os,json,hashlib
from pathlib import Path
import numpy as np,h5py
from scipy.io import loadmat
from scipy.spatial import cKDTree
P=Path('/home/zhaoyuhan/Eventbased_WAM');OLD=P/'runs/sim2real/20261101-epose-wood-short-research-v1';R=Path(os.environ['EPOSE_REPAIR_RUN']);out=R/'real_eval';summary=json.loads((out/'summary.json').read_text());verts=np.load(OLD/'cad_vertices.npy');keys=np.load(OLD/'cad_keypoints.npy');threshold=summary['success_threshold_m'];sequence=summary['protocol_frozen_before_training']['real_sequence']
raw_h5=P/'data/real/epose/figshare_28106717_v1/extracted/original_data/original_data'/sequence/'data.h5';root=P/'data/real/epose/figshare_28106717_v1/extracted/block'/sequence
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
raw_checks=[]
with h5py.File(raw_h5,'r') as h:
 ds=h['events_parallel'];origin=float(ds[3,0]);length=ds.shape[1]
 def search(v,side):
  a,b=0,length
  while a<b:
   mid=(a+b)//2;x=float(ds[3,mid])
   if x<v or (side=='right' and x==v):a=mid+1
   else:b=mid
  return a
 for n in [3,153,303,453,600]:
  ev=np.asarray(loadmat(root/f'{n:04d}_data.mat')['events']);a=search(origin+ev[0,2],'left');b=search(origin+ev[-1,2],'right');raw=np.asarray(ds[:,a:b]).T
  assert len(raw)==len(ev) and np.array_equal(raw[:,:2],ev[:,:2]) and np.array_equal(np.where(raw[:,2]>0,1,-1),ev[:,3]),(n,len(raw),len(ev))
  raw_checks.append({'index':n,'events':len(ev),'raw_xy_polarity_count_exact':True})
results={}
for arm,metrics in summary['results'].items():
 rows=json.loads((out/f'{arm}_predictions.json').read_text());assert len(rows)==200 and [r['index'] for r in rows]==list(range(3,603,3))
 assert all(r['event_last_s']<=r['time_s']+1e-7 for r in rows)
 all_epe=[];checked=0;maximum=0.
 for i,row in enumerate(rows):
  corner=float(np.linalg.norm(np.asarray(row['predicted_uv'])[1:]-np.asarray(row['gt_uv_ideal'])[1:],axis=1).mean());assert abs(corner-row['corner_epe_px'])<1e-10;all_epe.append(corner)
  if i%10:continue
  mat=loadmat(row['path']);assert sha(Path(row['path']))==row['sha256'];T=np.asarray(mat['pose']);xyz=(T@np.c_[keys,np.ones(9)].T).T[:,:3];uv=(mat['intrinsicMatrix']@xyz.T).T;uv=uv[:,:2]/uv[:,2:3];assert np.allclose(uv,row['gt_uv_ideal'],atol=1e-10)
  if not row['pnp_valid']:continue
  est=verts@np.asarray(row['estimated_R']).T+np.asarray(row['estimated_t_m']);truth=verts@T[:3,:3].T+T[:3,3];add=float(np.linalg.norm(est-truth,axis=1).mean());adds=float(cKDTree(truth).query(est,workers=1)[0].mean());maximum=max(maximum,abs(add-row['add_m']),abs(adds-row['adds_m']));checked+=1
  assert (adds<=threshold)==row['adds_success_0_1_d']
 assert maximum<1e-10
 valid=[r for r in rows if r['pnp_valid']];assert len(valid)==metrics['pnp_valid_count']
 for column,metric in [('add_m','add_mean_over_valid_pnp_m'),('adds_m','adds_mean_over_valid_pnp_m'),('translation_error_m','translation_mean_over_valid_pnp_m'),('rotation_error_raw_deg','rotation_raw_mean_over_valid_pnp_deg')]:
  if valid:assert abs(float(np.mean([r[column] for r in valid]))-metrics[metric])<1e-10
  else:assert metrics[metric] is None
 center=float(np.mean([np.linalg.norm(np.asarray(r['predicted_uv'])[0]-np.asarray(r['gt_uv_ideal'])[0]) for r in rows]));assert abs(center-metrics['centroid_epe_all_samples_px'])<1e-10
 count=sum(r['adds_success_0_1_d'] for r in rows);assert count==metrics['adds_success_0_1_d_count'] and count/200==metrics['adds_success_0_1_d_fraction_all_samples'];assert abs(np.mean(all_epe)-metrics['corner_epe_all_samples_px'])<1e-10
 assert sha(Path(metrics['checkpoint']['path']))==metrics['checkpoint']['sha256']
 results[arm]={'samples':200,'pose_metric_samples_independently_recomputed':checked,'maximum_add_adds_difference_m':maximum,'all_aggregate_metrics_recomputed':True,'checkpoint_sha256_rechecked':True,'success_count':count}
receipt={'passed':True,'real_sequence':sequence,'source_raw_event_checks':raw_checks,'results':results,'success_denominator_includes_pnp_failures':True};(out/'independent_verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
