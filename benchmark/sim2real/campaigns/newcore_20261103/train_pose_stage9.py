"""Fresh-initialized full-network paired synthetic pose training on CUDA."""
import argparse,copy,hashlib,json,math,os,random,sys,time
from pathlib import Path
import cv2,numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--generator',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--max-updates',type=int,default=0);p.add_argument('--resume',type=Path)
a=p.parse_args();R=a.campaign.resolve();sys.path.insert(0,str(R/'sources/benchmark/sim2real/scripts'))
from stage9_launch_contract import validate_launch
validate_launch(a,'pose');c=json.loads(a.config.read_text())['pose'];source=Path('/home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/SingleShotPose');sys.path.insert(0,str(source))
from darknet import Darknet
torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
device=torch.device('cuda:0');out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state_hash(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def checkpoint(path,payload):
 temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)
def savejson(path,payload):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2)+'\n');temp.replace(path)
def load(split):
 rows=[]
 for f in sorted((a.data_root/a.generator/split).glob('*.npz')):
  with np.load(f) as d:
   images=np.asarray(d['images']);uv=np.asarray(d['uv_normalized']);ids=np.asarray(d['sample_ids'])
   for i in range(len(images)):rows.append({'image':images[i],'uv':uv[i],'rid':f.stem,'id':str(ids[i])})
 return rows
train=load('train');val=load('validation');assert (len(train),len(val))==(1536,512)
model=Darknet(str(source/'cfg/yolo-pose.cfg')).to(device);model.train();initial=state_hash(model)
from stage9_launch_contract import verify_initialization
verify_initialization('pose',a.seed,initial)
optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay']);rng=np.random.default_rng(a.seed+20261103)
recipe_sha=sha(a.config);manifest_sha=sha(a.data_root/'adapter_manifest.json');start=0;best=float('inf');best_update=None;started=time.monotonic()
def decode(raw):
 B,_,H,W=raw.shape;assert raw.shape[1]==20
 idx=raw[:,18].flatten(1).argmax(1);yy=idx//W;xx=idx%W;b=torch.arange(B,device=raw.device)
 raw=raw[:,:18].reshape(B,9,2,H,W).permute(0,3,4,1,2)[b,yy,xx]
 uv=torch.cat([raw[:,:1].sigmoid(),raw[:,1:]],dim=1);return (uv+torch.stack([xx,yy],1)[:,None])/torch.tensor([W,H],device=raw.device),idx
def loss_fn(raw,uv):
 B,_,H,W=raw.shape;b=torch.arange(B,device=raw.device);cx=(uv[:,0,0]*W).long();cy=(uv[:,0,1]*H).long();assert ((cx>=0)&(cx<W)&(cy>=0)&(cy<H)).all()
 p=raw[:,:18].reshape(B,9,2,H,W);p=torch.cat([p[:,:1].sigmoid(),p[:,1:]],dim=1).permute(0,3,4,1,2)[b,cy,cx]
 wanted=uv*torch.tensor([W,H],device=raw.device)-torch.stack([cx,cy],1)[:,None]
 coord=(p-wanted).square().sum()/(2*B);center=torch.nn.functional.cross_entropy(raw[:,18].flatten(1),cy*W+cx)
 return coord+center,coord,center
@torch.no_grad()
def evaluate():
 model.eval();rows=[]
 for offset in range(0,len(val),8):
  block=val[offset:offset+8];raw=model(torch.from_numpy(np.stack([s['image'] for s in block])).to(device));uv,cell=decode(raw);pred=uv.cpu().numpy()
  for s,q,idx in zip(block,pred,cell.cpu().numpy()):
   e=np.linalg.norm((q-s['uv'])*np.array([346.,260.]),axis=1);gtcell=(s['uv'][0]*13).astype(int)
   rows.append({'sample_id':s['id'],'rollout_id':s['rid'],'keypoint_epe_px':float(e.mean()),'centroid_epe_px':float(e[0]),'corner_epe_px':float(e[1:].mean()),'center_cell_correct':int(idx)==int(gtcell[1]*13+gtcell[0])})
 byroll={rid:float(np.mean([r['keypoint_epe_px'] for r in rows if r['rollout_id']==rid])) for rid in sorted({s['rid'] for s in val})}
 model.train();return {'mean_keypoint_epe_px':float(np.mean(list(byroll.values()))),'centroid_epe_px':float(np.mean([r['centroid_epe_px'] for r in rows])),'center_cell_accuracy':float(np.mean([r['center_cell_correct'] for r in rows])),'by_rollout':byroll,'rows':rows}
def probe_prediction():
 was=model.training;model.eval()
 with torch.no_grad():q=model(torch.from_numpy(val[0]['image'][None]).to(device)).detach().cpu()
 model.train(was);return q
if a.resume:
 saved=torch.load(a.resume,map_location='cpu',weights_only=False);assert saved['recipe_sha256']==recipe_sha and saved['data_manifest_sha256']==manifest_sha and saved['seed']==a.seed and saved['generator']==a.generator
 model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);assert state_hash(model)==saved['state_sha256']
 err=float((probe_prediction()-saved['probe_prediction']).abs().max());assert err<1e-4,err
 random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']]);rng.bit_generator.state=saved['sampler_rng']
 start=saved['completed_update'];initial=saved['initial_state_sha256'];best=saved['best_metric'];best_update=saved['best_update']
 steps=[int(v['step']) for v in optimizer.state.values() if 'step' in v];assert steps and all(v==start for v in steps)
 savejson(out/'resume_verification.json',{'passed':True,'update':start,'max_forward_difference':err,'model_state_hash_match':True,'optimizer_steps':sorted(set(steps)),'rng_restored':True,'strict_continuous_trajectory_equivalence_claimed':False})
def payload(step):
 return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'completed_update':step,'initial_state_sha256':initial,'state_sha256':state_hash(model),'seed':a.seed,'generator':a.generator,'recipe_sha256':recipe_sha,'data_manifest_sha256':manifest_sha,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'sampler_rng':rng.bit_generator.state,'best_metric':best,'best_update':best_update,'probe_prediction':probe_prediction(),'input_contract':c['input'],'real_data_used':False}
savejson(out/'run_metadata.json',{'generator':a.generator,'seed':a.seed,'initial_state_sha256':initial,'fresh_random_initialization':True,'recipe_sha256':recipe_sha,'data_manifest_sha256':manifest_sha,'device':torch.cuda.get_device_name(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'effective_batch':8,'real_data_used':False,'training_input':'full event frame only; labels used only in supervised loss'})
budget=c['updates'];stop=a.max_updates or budget;assert start<stop<=budget
if start==0:
 result=evaluate();savejson(out/'initial_synthetic_validation.json',result)
with (out/'training_metrics.jsonl').open('a',buffering=1) as log:
 for step in range(start+1,stop+1):
  indices=rng.choice(len(train),8,replace=False);images=[];labels=[];aug=[];ids=[]
  for index in indices:
   s=train[int(index)];scale=float(rng.uniform(.85,1.2));dx,dy=rng.uniform(-24,24,2);q=(s['uv']-.5)*scale+.5+np.array([dx,dy])/416.
   if (q[0]>.005).all() and (q[0]<.995).all():
    A=np.array([[scale,0,(1-scale)*208+dx],[0,scale,(1-scale)*208+dy]],np.float32);image=cv2.warpAffine(s['image'].transpose(1,2,0),A,(416,416)).transpose(2,0,1).copy()
   else:image=s['image'];q=s['uv'];scale=1.;dx=dy=0.
   images.append(image);labels.append(q);aug.append([scale,float(dx),float(dy)]);ids.append(s['id'])
  x=torch.from_numpy(np.stack(images)).to(device);y=torch.from_numpy(np.asarray(labels,dtype=np.float32)).to(device)
  lr=c['learning_rate']*min(1.,step/30)*(.1+.9*.5*(1+math.cos(math.pi*step/budget)))
  for group in optimizer.param_groups:group['lr']=lr
  model.train();optimizer.zero_grad(set_to_none=True);raw=model(x);loss,coord,ce=loss_fn(raw,y);assert torch.isfinite(loss)
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip']);assert torch.isfinite(grad) and float(grad)>0
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  optimizer.step();entry={'optimizer_update':step,'loss':float(loss.detach()),'coordinate_loss':float(coord.detach()),'center_ce':float(ce.detach()),'gradient_l2_norm':float(grad),'learning_rate':lr,'rollout_sample_ids':ids,'affine':aug,'effective_batch':8,'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'elapsed_s':time.monotonic()-started,'real_data_used':False};log.write(json.dumps(entry)+'\n')
  if step%c['validate_every_updates']==0 or step==budget:
   result=evaluate();metric=result['mean_keypoint_epe_px'];assert np.isfinite(metric);savejson(out/('synthetic_validation_%06d.json'%step),result)
   if metric<best:best=metric;best_update=step;checkpoint(out/'synthetic_best.pt',payload(step))
  if step in (20,21) or step%500==0 or step==stop:
   checkpoint(out/'last.pt',payload(step));savejson(out/'live_status.json',{'status':'complete' if step==budget else 'training','completed_updates':step,'total_updates':budget,'best_update':best_update,'best_metric':None if not np.isfinite(best) else best,'initial_state_sha256':initial,'checkpoint':str(out/'last.pt'),'checkpoint_sha256':sha(out/'last.pt'),'real_data_used':False});print(json.dumps(entry),flush=True)
print('TRAINING_PHASE_COMPLETE',a.generator,a.seed,stop,flush=True)
