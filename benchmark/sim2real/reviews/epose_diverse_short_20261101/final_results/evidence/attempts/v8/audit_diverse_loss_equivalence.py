import os,sys,json,ast,time
from pathlib import Path
import numpy as np,torch,cv2
torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261101-epose-diverse-bank-v8';OLD=P/'runs/sim2real/20261101-epose-wood-short-research-v1';cfg=json.loads((R/'protocol.json').read_text());sys.path[:0]=[str(OLD/'sources/benchmark/sim2real/scripts'),str(R/'sources')];import run_pose_stage2_pilot as pose
pose.POSE_SOURCE=P/'external/sources/downstream/SingleShotPose';sys.path.insert(0,str(pose.POSE_SOURCE));from darknet import Darknet
from region_loss import RegionLoss,build_targets
from diverse_samples import collect_diverse
defs=[n for n in ast.parse((R/'sources/repair_training.py').read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name=='loss_fn'];exec(compile(ast.Module(body=defs,type_ignores=[]),'coordinate_loss','exec'),globals());rows=[]
for arm in ['eventgenesis','v2e']:
 torch.manual_seed(0);train=collect_diverse(pose,R,arm,'train');rng=np.random.default_rng(cfg['sampling_seed']);ix=rng.choice(len(train),4,replace=False);images=[];targets=[]
 for i in ix:
  s=train[i];uv=s['uv_normalized'];scale=float(rng.uniform(.85,1.20));dx,dy=rng.uniform(-24,24,2);new=(uv-.5)*scale+.5+np.array([dx,dy])/416
  if (new>.01).all() and (new<.99).all():A=np.array([[scale,0,(1-scale)*208+dx],[0,scale,(1-scale)*208+dy]],np.float32);im=cv2.warpAffine(s['image'].transpose(1,2,0),A,(416,416)).transpose(2,0,1).copy();target=pose.build_label(new*416,416,416)
  else:im=s['image'];target=s['label']
  images.append(im);targets.append(target)
 model=Darknet(str(pose.POSE_SOURCE/'cfg/yolo-pose.cfg'));model.load_state_dict(torch.load(cfg['initial_checkpoints'][arm]['path'],map_location='cpu',weights_only=False)['model']);model.train();target=torch.from_numpy(np.stack(targets))
 with torch.no_grad():raw=model(torch.from_numpy(np.stack(images)))
 a=raw.detach().clone().requires_grad_();official,detail=pose.patched_official_region_loss(a,target,RegionLoss(),0,build_targets);official=official/4;ga=torch.autograd.grad(official,a)[0]
 b=raw.detach().clone().requires_grad_();candidate,extra=loss_fn(b,target,False);gb=torch.autograd.grad(candidate,b)[0]
 row={'generator':arm,'indices':ix.tolist(),'official_coordinate_loss_mean':float(official.detach()),'vectorized_coordinate_loss_mean':float(candidate.detach()),'absolute_difference':abs(float(official.detach()-candidate.detach())),'relative_difference':abs(float(official.detach()-candidate.detach()))/max(abs(float(official.detach())),1e-12),'gradient_max_absolute_difference':float((ga-gb).abs().max()),'gradient_max_absolute_value':float(ga.abs().max()),'official_target_stats':detail,'target_centroids':target[:,1:3].tolist(),'raw_absmax':float(raw.abs().max())};rows.append(row);print(json.dumps(row),flush=True)
(R/'first_augmented_batch_loss_audit.json').write_text(json.dumps({'same_first_augmented_batch_as_failed_launch':True,'no_optimizer_updates':True,'results':rows},indent=2))
