"""Synthetic-only geometry, correspondence and BN causal tests. No real input."""
import argparse,ast,hashlib,json,sys,time
from pathlib import Path
import cv2,h5py,numpy as np,torch
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();R=a.campaign;P=R.parents[2]
a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device=torch.device('cuda:0');assert torch.cuda.device_count()==1
sys.path.insert(0,str(P/'external/sources/downstream/SingleShotPose'));from darknet import Darknet
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(path,data):
 with Path(path).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
protocol=json.loads((R/'recovery_20261105_mainline/pose_prepared_full_v5/protocol.json').read_text());vertices=np.load(protocol['assets']['vertices']['path']);tree=cKDTree(vertices);diam=protocol['assets']['diameter_m']
def load(method,split):
 out=[]
 for path in sorted((R/'adapters'/method/'pose'/method/split).glob('*.npz')):
  with np.load(path) as z:
   images=z['images'];uvs=z['uv_normalized'];poses=z['target_to_camera'];K=z['K'];keys=z['cad_keypoints'];ids=z['sample_ids']
   for i in range(len(images)):
    out.append(dict(image=images[i],uv=uvs[i],T=poses[i],K=K,keys=keys,id=str(ids[i]),rid=path.stem))
 return out
def pnp(uv,s):
 ok,rv,tv=cv2.solvePnP(np.asarray(s['keys'][1:],np.float64),np.asarray(uv[1:],np.float64),np.asarray(s['K'],np.float64),None,flags=cv2.SOLVEPNP_EPNP)
 if not ok:return None
 rot=cv2.Rodrigues(rv)[0];tr=tv[:,0]
 if not np.isfinite(rot).all() or not np.isfinite(tr).all() or not ((s['keys']@rot.T+tr)[:,2]>0).all():return None
 truth=vertices@s['T'][:3,:3].T+s['T'][:3,3]
 return float(tree.query((truth-tr)@rot,workers=1)[0].mean())
def decode(raw,idx=None):
 B,_,H,W=raw.shape;idx=raw[:,18].flatten(1).argmax(1) if idx is None else idx;y=idx//W;x=idx%W;b=torch.arange(B,device=device)
 v=raw[:,:18].reshape(B,9,2,H,W).permute(0,3,4,1,2)[b,y,x]
 return ((torch.cat([v[:,:1].sigmoid(),v[:,1:]],1)+torch.stack([x,y],1)[:,None])/torch.tensor([W,H],device=device)).cpu().numpy(),idx.cpu().numpy()
@torch.no_grad()
def evaluate(model,data,geometry=True):
 model.eval();rows=[]
 for k in range(0,len(data),8):
  block=data[k:k+8];raw=model(torch.from_numpy(np.stack([x['image'] for x in block])).to(device));uv=np.stack([s['uv'] for s in block]);gtcell=torch.from_numpy((uv[:,0,1]*13).astype(int)*13+(uv[:,0,0]*13).astype(int)).to(device)
  pred,cells=decode(raw);oracle,_=decode(raw,gtcell)
  for s,q,oq,c in zip(block,pred,oracle,cells):
   gt=s['uv']*[346,260];pr=q*[346,260];op=oq*[346,260];d=np.linalg.norm(pr-gt,axis=1);od=np.linalg.norm(op-gt,axis=1)
   cost=np.linalg.norm(pr[1:,None]-gt[None,1:],axis=-1);rr,cc=linear_sum_assignment(cost)
   row=dict(sample_id=s['id'],rollout=s['rid'],epe_px=float(d.mean()),centroid_px=float(d[0]),corners_px=float(d[1:].mean()),unordered_corners_diagnostic_px=float(cost[rr,cc].mean()),oracle_cell_epe_px=float(od.mean()))
   if geometry:
    row.update(adds_m=pnp(pr,s),oracle_cell_adds_m=pnp(op,s),correct_2d_pnp_adds_m=pnp(gt,s))
   rows.append(row)
 rids=sorted({s['rid'] for s in data})
 def equal(field):return float(np.mean([np.mean([x[field] for x in rows if x['rollout']==r]) for r in rids]))
 summary={field:equal(field) for field in ['epe_px','centroid_px','corners_px','unordered_corners_diagnostic_px','oracle_cell_epe_px']}
 if geometry:
  for field in ['adds_m','oracle_cell_adds_m','correct_2d_pnp_adds_m']:
   summary[field+'_success']=float(np.mean([np.mean([x[field] is not None and x[field]<.1*diam for x in rows if x['rollout']==r]) for r in rids]))
   valid=[x[field] for x in rows if x[field] is not None];summary[field+'_valid']=len(valid);summary[field+'_max']=max(valid) if valid else None
 return dict(samples=len(data),rollouts=len(rids),summary=summary,rows=rows)
def parameter_hash(model):
 h=hashlib.sha256()
 for name,value in model.named_parameters():h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
 return h.hexdigest()
all_results=[]
dump(a.out/'protocol.json',dict(scope='synthetic-only causal diagnostics',methods=['eventgenesis','v2e'],seed=0,geometry='full fixedCAD ADD-S, complete all510 validation windows, equal32rollout',unordered_matching='auxiliary diagnostic only; not pose score or declared CAD symmetry',BN='recompute running stats on1528 synthetic TRAIN samples only, frozen parameters; no target-domain adaptation',real_data_used=False))
for method in ['eventgenesis','v2e']:
 train=load(method,'train');val=load(method,'validation');assert (len(train),len(val))==(1529,510)
 # Independently reconstruct physical endpoint projections for every adapter label.
 reprojection=[]
 for split,data in [('train',train),('validation',val)]:
  byrid={rid:[x for x in data if x['rid']==rid] for rid in {x['rid'] for x in data}}
  for rid,samples in byrid.items():
   with h5py.File(R/'pose_bank/physical'/(rid+'.h5'),'r') as h:
    g=h['episodes'][rid]
    for s in samples:
     endpoint=int(s['id'].rsplit('/',1)[1])+1;T=np.linalg.inv(g['camera_to_world_cv'][endpoint])@g['target_pose'][endpoint];assert np.allclose(T,s['T'],atol=1e-10,rtol=0)
     xyz=(T@np.c_[s['keys'],np.ones(9)].T).T[:,:3];q=(s['K']@xyz.T).T;q=q[:,:2]/q[:,2:]
     reprojection.append(float(np.abs(q-s['uv']*[346,260]).max()))
 assert max(reprojection)<.0001
 path=R/'training/pose/seed0'/method/'synthetic_best.pt';original=sha(path);ck=torch.load(path,map_location='cpu',weights_only=False);model=Darknet(str(P/'external/sources/downstream/SingleShotPose/cfg/yolo-pose.cfg')).to(device);model.load_state_dict(ck['model'],strict=True);del ck
 before=evaluate(model,val);dump(a.out/(method+'_before.json'),before);print('CAUSAL_BEFORE',method,before['summary'],flush=True)
 params=parameter_hash(model);bns=[x for x in model.modules() if isinstance(x,torch.nn.modules.batchnorm._BatchNorm)];assert bns
 model.eval()
 for bn in bns:bn.reset_running_stats();bn.momentum=None;bn.train()
 rng=np.random.default_rng(20261106);indices=rng.permutation(len(train))[:1528]
 with torch.no_grad():
  for k in range(0,len(indices),8):model(torch.from_numpy(np.stack([train[i]['image'] for i in indices[k:k+8]])).to(device))
 assert parameter_hash(model)==params
 after=evaluate(model,val);assert sha(path)==original
 dump(a.out/(method+'_after_bn.json'),after);torch.save(dict(model=model.state_dict(),source_checkpoint_sha256=original,calibration_source='synthetic train only',train_calibration_windows=1528,parameters_unchanged=True),a.out/(method+'_bn_calibrated.pt'))
 result=dict(method=method,physical_label_max_error_px=max(reprojection),physical_label_windows=len(reprojection),original_checkpoint_sha256=original,parameters_unchanged=True,bn_layers=len(bns),before=before['summary'],after_bn=after['summary']);all_results.append(result);dump(a.out/(method+'_summary.json'),result);print('CAUSAL_AFTER_BN',method,after['summary'],flush=True);del model;torch.cuda.empty_cache()
dump(a.out/'completion.json',dict(status='complete',real_data_used=False,results=all_results))
