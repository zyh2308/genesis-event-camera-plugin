"""Audit saved outputs and raw geometry on CPU; do not run candidate models."""
from pathlib import Path
import json,hashlib,datetime,statistics,collections,math,copy
import numpy as np, scipy.io as sio, h5py, cv2
from PIL import Image,ImageDraw
B=Path('/home/zhaoyuhan/Eventbased_WAM');run=B/'runs/sim2real/20261025-hqf14-e2vid-seed0-actual-r2'
outdir=B/'runs/sim2real/research-audits/20261030-supplement';outdir.mkdir(parents=True,exist_ok=True)
score=json.loads((run/'paired_score.json').read_text());m=json.loads((run/'run_manifest.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(4*1024*1024),b''):h.update(z)
 return h.hexdigest()
def err(a,b):
 d=a-b;return {'coordinate_mae_px':float(np.abs(d).mean()),'point_euclidean_mean_px':float(np.linalg.norm(d,axis=1).mean()),'point_euclidean_max_px':float(np.linalg.norm(d,axis=1).max()),'coordinate_max_px':float(np.abs(d).max())}
def pixstats(a):return {'mean':float(a.mean()),'std':float(a.std()),'min':float(a.min()),'max':float(a.max()),'fraction_eq_0':float((a==0).mean()),'fraction_eq_1':float((a==1).mean())}
seqs=sorted(score['per_arm']['eventgenesis']['per_sequence']);frames=collections.defaultdict(dict)
for row in score['per_frame']:
 if row['is_preregistered_preview']:frames[(row['arm'],row['sequence'])][row['sample_index']]=row
overview=Image.new('RGB',(720,35+len(seqs)*210),'white');draw=ImageDraw.Draw(overview);draw.text((10,10),'Fixed 50% registered frames: GT | EventGenesis | V2E (fixed [0,1])',fill='black')
visual=[];images=[];inference_hashes=[]
for seqidx,seq in enumerate(seqs):
 indices=sorted(frames[('eventgenesis',seq)])
 assert indices==sorted(frames[('v2e',seq)]) and len(indices)==5
 gt=np.load(Path(m['hqf_root'])/seq/'images.npy',mmap_mode='r')
 sheet=Image.new('RGB',(720,35+len(indices)*210),'white');d=ImageDraw.Draw(sheet);d.text((10,10),seq+' : GT | EventGenesis | V2E; fixed [0,1]',fill='black')
 for j,index in enumerate(indices):
  target=np.asarray(gt[index,...,0],dtype=np.float64)/255.;trip=[target]
  for arm in ('eventgenesis','v2e'):
   row=frames[(arm,seq)][index];p=Path(row['prediction_path']);actual=sha(p);assert actual==row['prediction_sha256']
   a=np.load(p,allow_pickle=False);assert a.shape==target.shape and np.isfinite(a).all() and a.min()>=0 and a.max()<=1
   mse=float(np.mean((a.astype(np.float64)-target)**2));psnr=-10*math.log10(mse)
   assert abs(mse-row['metrics']['mse'])<1e-8 and abs(psnr-row['metrics']['psnr_db'])<1e-5
   visual.append({'arm':arm,'sequence':seq,'sample_index':index,'prediction_sha256':actual,'mse_recomputed':mse,'psnr_recomputed':psnr,'mse_difference':mse-row['metrics']['mse'],'prediction':pixstats(a),'target':pixstats(target)})
   trip.append(a)
  for col,a in enumerate(trip):
   img=Image.fromarray(np.rint(a*255).astype('uint8')).convert('RGB');sheet.paste(img,(col*240,60+j*210))
   if j==2:overview.paste(img,(col*240,65+seqidx*210))
  d.text((10,40+j*210),'index '+str(index),fill='black')
  if j==2:draw.text((10,45+seqidx*210),seq+' index '+str(index),fill='black')
 dest=outdir/(seq+'_registered_previews.png');sheet.save(dest);images.append({'path':str(dest),'sha256':sha(dest)})
 for arm in ('eventgenesis','v2e'):
  rec=m['arms'][arm]['sequences'][seq];p=Path(rec['inference_manifest_path']);inf=json.loads(p.read_text());actual=sha(p)
  assert actual==rec['inference_manifest_sha256']
  samples=inf['sequence']['samples'];rawmins=[r['raw_model_output_min'] for r in samples];rawmaxs=[r['raw_model_output_max'] for r in samples]
  inference_hashes.append({'arm':arm,'sequence':seq,'manifest_sha256':actual,'matches':True,'frames':len(samples),'raw_min':min(rawmins),'raw_max':max(rawmaxs),'frames_requiring_low_clip':sum(x<0 for x in rawmins),'frames_requiring_high_clip':sum(x>1 for x in rawmaxs)})
dest=outdir/'hqf14_midpoint_overview.png';overview.save(dest);images.append({'path':str(dest),'sha256':sha(dest)})
# Reconstruct the pre-scoring manifest without overwriting it. Its expected
# bytes can be recovered from the runner's deterministic finalization rules.
before=copy.deepcopy(m)
for key in ('score_command','score_started_utc','score_finished_utc','score_exit_code','score_stdout_path','score_stderr_path','score_path','score_sha256'):before.pop(key,None)
before['status']='all_28_inferences_complete_unscored';before['real_scores_computed']=False
last=max(v['finished_utc'] for a in m['arms'].values() for v in a['sequences'].values());candidates=[]
for t in set([last,m['score_started_utc']]):
 before['finished_utc']=t
 for ensure in (True,False):
  blob=(json.dumps(before,indent=2,ensure_ascii=ensure,sort_keys=True)+'\n').encode();digest=hashlib.sha256(blob).hexdigest()
  if digest==score['run_manifest']['sha256']:
   p=outdir/'reconstructed_pre_score_manifest.json';p.write_bytes(blob);candidates.append({'finished_utc':t,'sha256':digest,'path':str(p),'byte_exact_matches_embedded_score_hash':True,'origin':'reconstructed; not contemporaneously saved'})
# Independently project the two actual MAT samples using consistent definitions.
root=B/'data/real/epose/figshare_28106717_v1/extracted';pose=[]
dist=np.array([-.362,.115,.0009576,.00000455,.130])
h5=root/'original_data/original_data/block/r_01_gl/data.h5'
with h5py.File(h5,'r') as h:
 timestamps=np.asarray(h['events_parallel'][3,:]);e0=timestamps[0]
 for n in (1,2):
  p=root/f'block/block/r_01_gl/{n:04d}_data.mat';mat=sio.loadmat(p);kp=np.asarray(mat['keypointprojections'],float);K=np.asarray(mat['intrinsicMatrix'],float);xyz=kp[:,2:5];stored=kp[:,:2]
  uv=(K@xyz.T).T;ideal=uv[:,:2]/uv[:,2:3]
  double=cv2.undistortPoints(ideal.reshape(-1,1,2),K,dist,P=K).reshape(-1,2)
  shared=np.column_stack([K[0,0]*xyz[:,0]/xyz[0,2]+K[0,2],K[1,1]*xyz[:,1]/xyz[0,2]+K[1,2]])
  legacy=cv2.undistortPoints(shared.reshape(-1,1,2),K,dist,P=K).reshape(-1,2)
  events=np.asarray(mat['events'],float);abswin=e0+events[:,2];a=np.searchsorted(timestamps,abswin[0],side='left');b=np.searchsorted(timestamps,abswin[-1],side='right');raw=np.asarray(h['events_parallel'][:,a:b]).T;rp=np.where(raw[:,2]>0,1.,-1.)
  pose.append({'sample':n,'mat_sha256':sha(p),'points':len(kp),'stored_uv':stored.tolist(),'camera_xyz':xyz.tolist(),'ideal_pinhole_uv':ideal.tolist(),'ideal_then_undistort_uv':double.tolist(),'legacy_shared_depth_then_undistort_uv':legacy.tolist(),'stored_vs_ideal':err(stored,ideal),'stored_vs_ideal_then_undistort':err(stored,double),'stored_vs_legacy':err(stored,legacy),'event_count_mat':len(events),'event_count_h5':len(raw),'raw_xy_match':np.array_equal(raw[:,:2],events[:,:2]),'raw_polarity_match':np.array_equal(rp,events[:,3]),'relative_time_max_abs_s':float(np.max(np.abs((raw[:,3]-e0)-events[:,2]))),'note':'ideal pinhole and undistorted coordinates are distinct hypotheses; no correction or training is performed'})
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hqf_preview_checks':visual,'inference_manifests':inference_hashes,'images':images,'pre_score_manifest_reconstruction':candidates,'epose_projection_and_event_recheck':pose,'no_new_candidate_inference':True,'no_labels_modified':True}
p=outdir/'visual_and_geometry_audit.json';p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
