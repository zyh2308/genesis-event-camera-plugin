"""Stage10 exploratory synthetic diagnostics; never loads real data."""
import argparse,copy,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--fit-updates',type=int,default=200);a=p.parse_args()
R=a.campaign;P=R.parents[2];source=P/'external/sources/downstream/SingleShotPose'
sys.path.insert(0,str(source));from darknet import Darknet
torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available() and torch.cuda.device_count()==1
a.out.mkdir(parents=True,exist_ok=False);device=torch.device('cuda:0');pix=np.array([346.,260.])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def load(method,split):
 rows=[]
 for f in sorted((R/'adapters'/method/'pose'/method/split).glob('*.npz')):
  with np.load(f) as z:
   for image,uv,sid in zip(z['images'],z['uv_normalized'],z['sample_ids']):
    rows.append(dict(image=np.asarray(image),uv=np.asarray(uv),id=str(sid),rid=f.stem))
 assert rows;return rows
def decode(raw,indices=None):
 B,C,H,W=raw.shape;assert C==20
 idx=raw[:,18].flatten(1).argmax(1) if indices is None else indices
 yy=idx//W;xx=idx%W;b=torch.arange(B,device=raw.device)
 v=raw[:,:18].reshape(B,9,2,H,W).permute(0,3,4,1,2)[b,yy,xx]
 return ((torch.cat([v[:,:1].sigmoid(),v[:,1:]],1)+torch.stack([xx,yy],1)[:,None])/torch.tensor([W,H],device=raw.device)),idx
def loss_fn(raw,uv):
 B,_,H,W=raw.shape;b=torch.arange(B,device=raw.device);cx=(uv[:,0,0]*W).long();cy=(uv[:,0,1]*H).long()
 assert ((cx>=0)&(cx<W)&(cy>=0)&(cy<H)).all()
 v=raw[:,:18].reshape(B,9,2,H,W);v=torch.cat([v[:,:1].sigmoid(),v[:,1:]],1).permute(0,3,4,1,2)[b,cy,cx]
 wanted=uv*torch.tensor([W,H],device=raw.device)-torch.stack([cx,cy],1)[:,None]
 coord=(v-wanted).square().sum()/(2*B);ce=torch.nn.functional.cross_entropy(raw[:,18].flatten(1),cy*W+cx)
 return coord+ce,coord,ce
@torch.no_grad()
def evaluate(model,rows):
 model.eval();results=[]
 for off in range(0,len(rows),8):
  block=rows[off:off+8];raw=model(torch.from_numpy(np.stack([s['image'] for s in block])).to(device));B,_,H,W=raw.shape
  uv=torch.from_numpy(np.stack([s['uv'] for s in block])).to(device)
  truecell=(uv[:,0,1]*H).long()*W+(uv[:,0,0]*W).long()
  pred,cell=decode(raw);oracle,_=decode(raw,truecell);top=raw[:,18].flatten(1).topk(5,dim=1).indices
  for i,s in enumerate(block):
   normal=np.linalg.norm((pred[i].cpu().numpy()-s['uv'])*pix,axis=1);o=np.linalg.norm((oracle[i].cpu().numpy()-s['uv'])*pix,axis=1)
   results.append(dict(sample_id=s['id'],rollout=s['rid'],normal_epe_px=float(normal.mean()),oracle_cell_epe_px=float(o.mean()),normal_centroid_px=float(normal[0]),oracle_centroid_px=float(o[0]),normal_corner_px=float(normal[1:].mean()),oracle_corner_px=float(o[1:].mean()),cell_correct=bool(cell[i]==truecell[i]),cell_in_top5=bool((top[i]==truecell[i]).any())))
 def avg(name,subset=results):return float(np.mean([x[name] for x in subset])) if subset else None
 rids=sorted({s['rid'] for s in rows})
 return dict(samples=len(rows),rollouts=len(rids),normal_epe_equal_rollout_px=float(np.mean([avg('normal_epe_px',[x for x in results if x['rollout']==r]) for r in rids])),oracle_epe_equal_rollout_px=float(np.mean([avg('oracle_cell_epe_px',[x for x in results if x['rollout']==r]) for r in rids])),center_accuracy=avg('cell_correct'),center_top5_accuracy=avg('cell_in_top5'),normal_centroid_px=avg('normal_centroid_px'),oracle_centroid_px=avg('oracle_centroid_px'),normal_corners_px=avg('normal_corner_px'),oracle_corners_px=avg('oracle_corner_px'),epe_when_cell_correct_px=avg('normal_epe_px',[x for x in results if x['cell_correct']]),epe_when_cell_wrong_px=avg('normal_epe_px',[x for x in results if not x['cell_correct']]),rows=results)
recipe=json.loads((R/'sources/training_recipe.json').read_text())['pose'];results=[];identities=None;label_reference=None
frozen=json.loads((R/'recovery_20261103/pose_frozen_checkpoints.json').read_text())
dump(a.out/'protocol.json',dict(task='synthetic_diagnostic_only',methods=['eventgenesis','v2e'],seed=0,source_sha256=sha(__file__),original_recipe_sha256=sha(R/'sources/training_recipe.json'),fit_updates=a.fit_updates,fit_scope='fixed shared64 train windows, no augmentation, same existing loss; capacity diagnostic not formal retraining',checkpoint_selection='original frozen synthetic_best; no new real score or selection',oracle_cell='GT synthetic cell used only for diagnostic',real_data_used=False))
for method in ['eventgenesis','v2e']:
 train=load(method,'train');val=load(method,'validation');assert (len(train),len(val))==(1529,510)
 ids=[x['id'] for x in train+val]
 if identities is None:identities=ids;label_reference=np.stack([x['uv'] for x in train+val])
 else:
  assert identities==ids,'Paired sample identity mismatch'
  assert np.array_equal(label_reference,np.stack([x['uv'] for x in train+val])),'Paired labels differ'
 rids=sorted({s['rid'] for s in train});chosen=[rids[i] for i in np.linspace(0,len(rids)-1,8).astype(int)];fit=[]
 for r in chosen:
  group=[s for s in train if s['rid']==r];fit.extend(group[i] for i in np.linspace(0,len(group)-1,8).astype(int))
 assert len(fit)==64 and len({s['id'] for s in fit})==64
 path=R/'training/pose/seed0'/method/'synthetic_best.pt';before=sha(path);saved=torch.load(path,map_location='cpu',weights_only=False)
 expected=next(x for x in frozen['runs'] if x['method']==method and x['seed']==0);assert before==expected['selected_checkpoint_sha256']
 assert saved['seed']==0 and saved['generator']==method
 torch.manual_seed(0);torch.cuda.manual_seed_all(0);model=Darknet(str(source/'cfg/yolo-pose.cfg')).to(device);model.load_state_dict(saved['model'],strict=True)
 vd=evaluate(model,val);td=evaluate(model,fit)
 update=saved['completed_update'];reference=json.loads((path.parent/('synthetic_validation_%06d.json'%update)).read_text())
 assert abs(vd['normal_epe_equal_rollout_px']-reference['mean_keypoint_epe_px'])<.002,(vd['normal_epe_equal_rollout_px'],reference['mean_keypoint_epe_px'])
 item=dict(method=method,checkpoint_sha256=before,selected_update=update,before_validation=vd,before_fixed_train=td,train_rollout_ids=chosen,fit_sample_ids=[s['id'] for s in fit]);dump(a.out/(method+'_before.json'),item);print('DIAG_BEFORE',method,{k:v for k,v in vd.items() if k!='rows'},flush=True)
 del saved;optimizer=torch.optim.Adam(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay']);rng=np.random.default_rng(20261106)
 model.train()
 with (a.out/(method+'_fit_metrics.jsonl')).open('x',buffering=1) as log:
  for step in range(1,a.fit_updates+1):
   block=[fit[i] for i in rng.choice(64,8,replace=False)];x=torch.from_numpy(np.stack([s['image'] for s in block])).to(device);y=torch.from_numpy(np.stack([s['uv'] for s in block]).astype(np.float32)).to(device)
   optimizer.zero_grad(set_to_none=True);loss,coord,ce=loss_fn(model(x),y);assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip']);assert torch.isfinite(grad) and float(grad)>0;optimizer.step()
   log.write(json.dumps(dict(update=step,loss=float(loss.detach()),coord=float(coord.detach()),center_ce=float(ce.detach()),grad_norm=float(grad),sample_ids=[s['id'] for s in block]))+'\n')
 aftertrain=evaluate(model,fit);afterval=evaluate(model,val);assert sha(path)==before
 item.update(after_fixed_train=aftertrain,after_validation=afterval,original_checkpoint_unchanged=True)
 torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=a.fit_updates,method=method,seed=0,diagnostic_only=True,real_data_used=False),a.out/(method+'_diagnostic_fit.pt'))
 dump(a.out/(method+'_results.json'),item);results.append(item);print('DIAG_AFTER',method,{k:v for k,v in afterval.items() if k!='rows'},flush=True);del model,optimizer;torch.cuda.empty_cache()
dump(a.out/'completion.json',dict(status='complete',methods=2,fit_updates_per_method=a.fit_updates,real_data_used=False,original_checkpoints_unchanged=True,summary=[dict(method=r['method'],before={k:v for k,v in r['before_validation'].items() if k!='rows'},after={k:v for k,v in r['after_validation'].items() if k!='rows'}) for r in results]))
