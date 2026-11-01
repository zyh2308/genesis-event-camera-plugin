"""Frozen-checkpoint real pilot; event-only inference, GT used only after prediction."""
import os,sys,json,time,hashlib,gc
from pathlib import Path
import numpy as np,torch,cv2
from scipy.io import loadmat
from scipy.spatial import cKDTree,ConvexHull
from scipy.spatial.distance import pdist
R=Path(os.environ['EPOSE_REPAIR_RUN']);OLD=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-epose-wood-short-research-v1');S=OLD/'sources';sys.path.insert(0,str(S/'benchmark/sim2real/scripts'))
import run_pose_stage2_pilot as pose
P=Path('/home/zhaoyuhan/Eventbased_WAM');modelroot=P/'external/sources/downstream/SingleShotPose';sys.path.insert(0,str(modelroot))
from darknet import Darknet
sys.path.insert(0,str(R/'sources'));from pose_center_decoder import decode_classified_pose
protocol=json.loads((R/'protocol.json').read_text());frozen=json.loads((R/'weights_frozen.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for c in frozen['checkpoints'].values():assert sha(Path(c['path']))==c['sha256']
gate=json.loads((R/'synthetic_gate.json').read_text());eligible=[a for a in ['eventgenesis','v2e'] if gate['arms'][a]['passed']];assert eligible and (gate['passed'] or protocol.get('deadline_fallback_passed_arms_only',False));assert hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest()==json.loads((R/'protocol_pretraining_hash.json').read_text())['sha256']
assert json.loads((R/'independent_synthetic_verification.json').read_text())['passed']
controls=json.loads((R/'evaluation_controls_pretest.json').read_text())
for arm,c in controls['controls'].items():
 if arm in eligible:frozen['checkpoints'][arm+'_v1_control']=c
for c in frozen['checkpoints'].values():assert sha(Path(c['path']))==c['sha256']
torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
verts=np.load(OLD/'cad_vertices.npy');keys=np.load(OLD/'cad_keypoints.npy');diameter=float(pdist(verts[ConvexHull(verts).vertices]).max());threshold=.1*diameter
assert .2<diameter<.3
# Metric oracle: exact identical pose gives zero ADD and ADD-S; known 1cm translation gives 1cm ADD.
assert abs(np.linalg.norm((verts+.01*np.array([1,0,0]))-verts,axis=1).mean()-.01)<1e-12
assert cKDTree(verts).query(verts)[0].max()==0
root=P/'data/real/epose/figshare_28106717_v1/extracted/block'/protocol['real_sequence']
samples=[];dist=np.array(protocol['distortion'],float);out=R/'real_eval';out.mkdir(exist_ok=False);raw=out/'raw_undistorted';raw.mkdir()
for n in protocol['real_sample_end_indices']:
 p=root/f'{n:04d}_data.mat';m=loadmat(p);ev=np.asarray(m['events'],float);label_time=float(np.asarray(m['time']).squeeze());assert float(ev[:,2].max())<=label_time+1e-7,'future event in pose packet';K=np.asarray(m['intrinsicMatrix'],float);assert ev.ndim==2 and ev.shape[1]==4 and len(ev)>0
 und=cv2.undistortPoints(ev[:,:2].reshape(-1,1,2),K,dist,P=K).reshape(-1,2);xy=np.rint(und).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<346)&(xy[:,1]>=0)&(xy[:,1]<260)
 x,y=xy[inside].T;t=ev[inside,2];pol=(ev[inside,3]>0).astype(np.uint8)
 image=pose.encode_event_window(t,x,y,pol,float(t.min())-1e-8,float(t.max())+1e-8,346,260,8.)
 # Ground-truth UV fields are intentionally never read.
 T=np.asarray(m['pose'],float);assert T.shape==(4,4)
 samples.append({'index':n,'path':str(p),'sha256':sha(p),'image':image,'K':K,'T':T,'time_s':float(np.asarray(m['time']).squeeze()),'event_first_s':float(ev[:,2].min()),'event_last_s':float(ev[:,2].max()),'events_raw':len(ev),'events_rectified':int(inside.sum())})
 im=np.full((260,346,3),245,np.uint8);im[y,x]=(30,90,220);im[y[pol==1],x[pol==1]]=(220,50,40);cv2.imwrite(str(raw/f'{n:04d}.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
assert len(samples)==200
results={};all_predictions={};started=time.monotonic()
for arm in eligible+[a+'_v1_control' for a in eligible]:
 model=Darknet(str(modelroot/'cfg/yolo-pose.cfg')).to('cpu');checkpoint=torch.load(frozen['checkpoints'][arm]['path'],map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model'],strict=True);model.eval();rows=[]
 for item in samples:
  
  if arm.endswith('_v1_control'):norm,confidence,cell,grid=pose.predict_normalized_keypoints(model,item['image'],torch.device('cpu'))
  else:
   with torch.no_grad():norm,confidence,cell=decode_classified_pose(model(torch.from_numpy(item['image'])[None]))
  uv=norm.astype(float)*np.array([346,260]);assert np.isfinite(uv).all()
  K=item['K'];T=item['T'];gt=(T@np.c_[keys,np.ones(9)].T).T[:,:3];gtuv=(K@gt.T).T;gtuv=gtuv[:,:2]/gtuv[:,2:3]
  row={k:item[k] for k in ['index','path','sha256','time_s','event_first_s','event_last_s','events_raw','events_rectified']};row.update({'confidence':confidence,'confidence_argmax_cell_yx':list(cell),'predicted_uv':uv.tolist(),'gt_uv_ideal':gtuv.tolist(),'centroid_epe_px':float(np.linalg.norm(uv[0]-gtuv[0])),'corner_epe_px':float(np.linalg.norm(uv[1:]-gtuv[1:],axis=1).mean()),'pnp_valid':False,'adds_success_0_1_d':False,'add_success_0_1_d':False})
  try:
   ok,rvec,tvec=cv2.solvePnP(keys[1:].astype(float),uv[1:].astype(float),K,None,flags=cv2.SOLVEPNP_EPNP)
   if ok:
    rot=cv2.Rodrigues(rvec)[0];trans=tvec[:,0];estimated=keys@rot.T+trans
    if np.isfinite(rot).all() and np.isfinite(trans).all() and (estimated[:,2]>0).all():
     predcloud=verts@rot.T+trans;gtcloud=verts@T[:3,:3].T+T[:3,3]
     add=float(np.linalg.norm(predcloud-gtcloud,axis=1).mean());adds=float(cKDTree(gtcloud).query(predcloud,workers=1)[0].mean());ang=float(np.degrees(np.arccos(np.clip((np.trace(rot@T[:3,:3].T)-1)/2,-1,1))))
     reproj,_=cv2.projectPoints(keys,rvec,tvec,K,None)
     row.update({'pnp_valid':True,'estimated_R':rot.tolist(),'estimated_t_m':trans.tolist(),'gt_R':T[:3,:3].tolist(),'gt_t_m':T[:3,3].tolist(),'reprojected_estimated_uv':reproj.reshape(-1,2).tolist(),'add_m':add,'adds_m':adds,'translation_error_m':float(np.linalg.norm(trans-T[:3,3])),'rotation_error_raw_deg':ang,'adds_success_0_1_d':adds<=threshold,'add_success_0_1_d':add<=threshold})
  except cv2.error as e:row['pnp_error']=str(e)[:250]
  rows.append(row)
 valid=[x for x in rows if x['pnp_valid']]
 mean=lambda k:float(np.mean([x[k] for x in valid])) if valid else None
 results[arm]={'samples':200,'pnp_valid_count':len(valid),'pnp_failed_count':200-len(valid),'centroid_epe_all_samples_px':float(np.mean([x['centroid_epe_px'] for x in rows])),'corner_epe_all_samples_px':float(np.mean([x['corner_epe_px'] for x in rows])),'adds_mean_over_valid_pnp_m':mean('adds_m'),'add_mean_over_valid_pnp_m':mean('add_m'),'translation_mean_over_valid_pnp_m':mean('translation_error_m'),'rotation_raw_mean_over_valid_pnp_deg':mean('rotation_error_raw_deg'),'adds_success_0_1_d_count':sum(x['adds_success_0_1_d'] for x in rows),'adds_success_0_1_d_fraction_all_samples':sum(x['adds_success_0_1_d'] for x in rows)/200,'add_success_0_1_d_fraction_all_samples':sum(x['add_success_0_1_d'] for x in rows)/200,'checkpoint':frozen['checkpoints'][arm]}
 all_predictions[arm]=rows;(out/f'{arm}_predictions.json').write_text(json.dumps(rows,indent=2));print(json.dumps({'arm':arm,**results[arm]}),flush=True);del model,checkpoint;gc.collect()
summary={'eligible_new_arms':eligible,'paired_generator_comparison_available':len(eligible)==2,'scope':'fresh r_05_gl E-POSE pilot, common centroid-classification adapter; score only localization-qualified arms, no generator comparison if one arm failed; not original training or full benchmark','protocol_frozen_before_training':protocol,'weights_frozen':frozen,'cad_diameter_m':diameter,'success_threshold_m':threshold,'sample_count':200,'synthetic_gate':json.loads((R/'synthetic_gate.json').read_text()),'actual_inference_device':'g1 CPU eight threads','evaluation_controls_pretest':controls,'corner_epe_coordinates':'original346x260 ideal pixel domain, not416x416 network pixels','elapsed_s':time.monotonic()-started,'checkpoint_selection_used_real_data':False,'test_inputs':'full-frame rectified events only; CAD and K for fixed PnP; GT only for metrics','invalid_pnp_policy':'counted as failures in success fractions; mean pose errors reported explicitly over valid PnP only','results':results}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('REAL_EPOSE_COMPLETE',flush=True)







