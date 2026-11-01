"""Independent recomputation from saved estimates and original MAT poses."""
import os,json,hashlib
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat
from scipy.spatial import cKDTree
R=Path(os.environ['EPOSE_SHORT_RUN']);out=R/'real_eval';summary=json.loads((out/'summary.json').read_text());verts=np.load(R/'cad_vertices.npy');keys=np.load(R/'cad_keypoints.npy');threshold=summary['success_threshold_m'];results={}
raw_h5=Path('/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1/extracted/original_data/original_data/block/r_01_gl/data.h5')
raw_checks=[]
with h5py.File(raw_h5,'r') as h:
 times=np.asarray(h['events_parallel'][3,:]);origin=times[0]
 for n in (3,153,303,453,600):
  p=Path('/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1/extracted/block/block/r_01_gl')/f'{n:04d}_data.mat';ev=np.asarray(loadmat(p)['events']);absolute=origin+ev[:,2];a=np.searchsorted(times,absolute[0],side='left');b=np.searchsorted(times,absolute[-1],side='right');raw=np.asarray(h['events_parallel'][:,a:b]).T
  assert len(raw)==len(ev) and np.array_equal(raw[:,:2],ev[:,:2]) and np.array_equal(np.where(raw[:,2]>0,1,-1),ev[:,3])
  raw_checks.append({'index':n,'raw_event_count':len(raw),'xy_and_polarity_exact_match':True})
for arm in ['eventgenesis','v2e']:
 rows=json.loads((out/f'{arm}_predictions.json').read_text());assert len(rows)==200
 assert [r['index'] for r in rows]==list(range(3,603,3))
 assert len({r['index'] for r in rows})==200
 maximum=0.;checked=0;corner_errors=[]
 for i,row in enumerate(rows):
  assert row['path'].endswith(f"{row['index']:04d}_data.mat")
  corner=float(np.linalg.norm(np.asarray(row['predicted_uv'])[1:]-np.asarray(row['gt_uv_ideal'])[1:],axis=1).mean());assert abs(corner-row['corner_epe_px'])<1e-10;corner_errors.append(corner)
  if i%10!=0:continue
  p=Path(row['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];mat=loadmat(p);gt=np.asarray(mat['pose'],float)
  xyz=(gt@np.c_[keys,np.ones(9)].T).T[:,:3];gtuv=(np.asarray(mat['intrinsicMatrix'])@xyz.T).T;gtuv=gtuv[:,:2]/gtuv[:,2:3];assert np.allclose(gtuv,row['gt_uv_ideal'],atol=1e-10)
  if not row['pnp_valid']:continue
  assert np.allclose(gt[:3,:3],row['gt_R']) and np.allclose(gt[:3,3],row['gt_t_m'])
  estimate=verts@np.asarray(row['estimated_R']).T+np.asarray(row['estimated_t_m']);truth=verts@gt[:3,:3].T+gt[:3,3]
  adds=float(cKDTree(truth).query(estimate,workers=1)[0].mean());add=float(np.linalg.norm(estimate-truth,axis=1).mean())
  maximum=max(maximum,abs(add-row['add_m']),abs(adds-row['adds_m']));checked+=1
  assert (adds<=threshold)==row['adds_success_0_1_d']
 assert maximum<1e-10
 total=sum(bool(x['adds_success_0_1_d']) for x in rows);assert total==summary['results'][arm]['adds_success_0_1_d_count']
 assert abs(np.mean(corner_errors)-summary['results'][arm]['corner_epe_all_samples_px'])<1e-10
 assert total/200==summary['results'][arm]['adds_success_0_1_d_fraction_all_samples']
 frozen=summary['weights_frozen']['checkpoints'][arm];assert hashlib.sha256(Path(frozen['path']).read_bytes()).hexdigest()==frozen['sha256']
 results[arm]={'samples':200,'independently_recomputed_valid_pose_samples':checked,'max_abs_add_adds_difference_m':maximum,'all_sample_success_count':total,'checkpoint_hash_rechecked':True,'invalid_pnp_in_denominator':True}
(out/'independent_verification.json').write_text(json.dumps({'status':'passed','results':results,'original_hdf5_event_checks':raw_checks},indent=2));print(json.dumps(results,indent=2))
