"""Read-only synthetic 3D audit of the fixed PnP used in real inference."""
import os,sys,json,io,time,hashlib
from pathlib import Path
import numpy as np,torch,h5py,cv2
from scipy.spatial import cKDTree,ConvexHull
from scipy.spatial.distance import pdist
torch.set_num_threads(1);cv2.setNumThreads(1)
R=Path(os.environ['EPOSE_REPAIR_RUN']);early='--early' in sys.argv
keys=np.load(R/'cad_keypoints.npy');verts=np.load(R/'cad_vertices.npy');diam=float(pdist(verts[ConvexHull(verts).vertices]).max());results={}
with h5py.File(R/'bank/shared_trajectory.h5','r') as h:
 for arm in ['eventgenesis','v2e']:
  if early:
   path=R/f'training/{arm}_best_so_far.pt';blob=path.read_bytes();ck=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);validation=ck['validation'];step=ck['selected_update'];del ck
  else:
   result=json.loads((R/f'training/{arm}_result.json').read_text());validation=result['validation'];step=result['selected_update'];path=Path(result['checkpoint_path']);blob=path.read_bytes()
  checkpoint_sha=hashlib.sha256(blob).hexdigest();del blob;rows=[]
  for item in validation['rows']:
   name,window=item['sample_id'].split('/');end=int(window.split('_')[-1]);g=h['episodes'][name];K=np.asarray(g['camera_intrinsics_K_px']);T=np.linalg.inv(np.asarray(g['camera_to_world_cv'][end]))@np.asarray(g['target_pose'][end]);uv=np.asarray(item['predicted_normalized_uv'])*[346,260];row={'sample_id':item['sample_id'],'pnp_valid':False,'adds_success_0_1_d':False}
   ok,r,t=cv2.solvePnP(keys[1:].astype(float),uv[1:].astype(float),K,None,flags=cv2.SOLVEPNP_EPNP)
   if ok:
    rot=cv2.Rodrigues(r)[0];pred=keys@rot.T+t[:,0];gt=keys@T[:3,:3].T+T[:3,3]
    if np.isfinite(pred).all() and (pred[:,2]>0).all():
     cloud=verts@rot.T+t[:,0];gtcloud=verts@T[:3,:3].T+T[:3,3];adds=float(cKDTree(gtcloud).query(cloud,workers=1)[0].mean());p=(K@pred.T).T;p=p[:,:2]/p[:,2:];q=(K@gt.T).T;q=q[:,:2]/q[:,2:]
     row.update(pnp_valid=True,translation_error_m=float(np.linalg.norm(t[:,0]-T[:3,3])),adds_m=adds,adds_success_0_1_d=adds<=.1*diam,pnp_centroid_epe_native_px=float(np.linalg.norm(p[0]-q[0])),centroid_depth_ratio=float(pred[0,2]/gt[0,2]))
   rows.append(row)
  valid=[r for r in rows if r['pnp_valid']];mean=lambda key:float(np.mean([r[key] for r in valid])) if valid else None
  results[arm]={'selected_update':step,'checkpoint_sha256':checkpoint_sha,'validation_count':len(rows),'pnp_valid_count':len(valid),'translation_mean_valid_m':mean('translation_error_m'),'adds_mean_valid_m':mean('adds_m'),'pnp_centroid_epe_native_px':mean('pnp_centroid_epe_native_px'),'adds_success_count':sum(r['adds_success_0_1_d'] for r in rows),'rows':rows}
out=R/('synthetic_pose_geometry_early.json' if early else 'synthetic_pose_geometry_final.json');out.write_text(json.dumps({'scope':'synthetic-only diagnostic; no gate or checkpoint-selection change; no optimizer updates','early_snapshot':early,'diameter_m':diam,'success_threshold_m':.1*diam,'results':results,'unix':time.time()},indent=2));print(json.dumps({a:{k:v for k,v in r.items() if k!='rows'} for a,r in results.items()}),flush=True)
