"""Frozen-backbone output-layer short training, synthetic inputs only."""
import os,sys,json,time,hashlib,socket,copy,gzip
from pathlib import Path
import numpy as np,torch,cv2
from datetime import datetime,timezone
P=Path('/home/zhaoyuhan/Eventbased_WAM');OLD=P/'runs/sim2real/20261101-epose-wood-short-research-v1';R=Path(os.environ['EPOSE_REPAIR_RUN'])
sys.path[:0]=[str(OLD/'sources/benchmark/sim2real/scripts')]
import run_pose_stage2_pilot as pose
pose.POSE_SOURCE=P/'external/sources/downstream/SingleShotPose';sys.path.insert(0,str(pose.POSE_SOURCE))
from darknet import Darknet
from region_loss import RegionLoss,build_targets
torch.set_num_threads(8);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
keys=np.load(OLD/'cad_keypoints.npy');pose.box_keypoints=lambda s:keys.copy()
cfg=json.loads((R/'protocol.json').read_text());size=json.loads((OLD/'bank/config.json').read_text())['simulation']['target_box']['size_m']
assert cfg['real_sequence']=='block/r_05_gl'
updates=cfg['optimizer_updates_per_arm'];prior=cfg.get('prior_output_layer_updates',0);BASE=Path(cfg.get('resume_from_run',str(OLD)))
start=time.time();out=R/'training';out.mkdir(exist_ok=False)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loss_fn(output,target,confidence=True):
 B,_,H,W=output.shape;raw=output[:,:18].reshape(B,9,2,H,W)
 pred=torch.cat([raw[:,:1].sigmoid(),raw[:,1:]],dim=1)
 uv=target[:,1:19].reshape(B,9,2);cx=(uv[:,0,0]*W).long();cy=(uv[:,0,1]*H).long();b=torch.arange(B)
 offsets=pred.permute(0,3,4,1,2)[b,cy,cx];wanted=uv*torch.tensor([W,H])-torch.stack([cx,cy],1)[:,None]
 coord=(offsets-wanted).square().sum()/2/B
 yy,xx=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij');grid=torch.stack([xx,yy])[None,None]
 norm=(pred.detach()+grid)/torch.tensor([W,H])[None,None,:,None,None]
 dist=((norm-uv[:,:,:,None,None])*torch.tensor([640.,480.])[None,None,:,None,None]).square().sum(2).sqrt()
 quality=(torch.exp(2*(1-dist/80))-1).clamp(min=0)/(np.exp(2)-1+1e-5);quality=quality.mean(1)
 mask=torch.ones(B,H,W);mask[quality>.6]=0;mask[b,cy,cx]=5
 tc=torch.zeros(B,H,W);tc[b,cy,cx]=quality[b,cy,cx]
 conf=.5*((output[:,18].sigmoid()-tc).square()*mask).sum()/B
 return coord+(conf if confidence else 0),{'coordinate_loss':float(coord.detach()),'confidence_loss':float(conf.detach()),'positive_quality_mean':float(quality[b,cy,cx].mean()),'positive_confidence_mean':float(output[b,18,cy,cx].sigmoid().mean().detach())}

def transformed_samples(samples,kind):
 rng=np.random.default_rng(103 if kind=='train' else 404);result=[];records=[]
 for item in samples:
  transforms=[(1.,0.,0.)]
  count=8 if kind=='train' else 4
  while len(transforms)<count:
   scale=float(rng.uniform(.8,1.75) if kind=='train' else rng.uniform(.85,1.55));dx,dy=rng.uniform(-55,55,2)
   uv=(item['uv_normalized']-.5)*scale+.5+np.array([dx,dy])/416
   if (uv>.015).all() and (uv<.985).all():transforms.append((scale,float(dx),float(dy)))
  for j,(scale,dx,dy) in enumerate(transforms):
   A=np.array([[scale,0,(1-scale)*208+dx],[0,scale,(1-scale)*208+dy]],np.float32)
   image=cv2.warpAffine(item['image'].transpose(1,2,0),A,(416,416),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).transpose(2,0,1).copy()
   uv=(item['uv_normalized']-.5)*scale+.5+np.array([dx,dy])/416
   s=dict(item,image=image,uv_normalized=uv.astype(np.float32),label=pose.build_label(uv*416,416,416),sample_id=item['sample_id']+f'/affine_{j:02d}')
   result.append(s);records.append({'sample_id':s['sample_id'],'parent':item['sample_id'],'scale':scale,'dx':dx,'dy':dy,'A':A.tolist()})
 return result,records

def cache_features(model,head,samples,label):
 features=[];outputs=[];holder={}
 hook=head.register_forward_pre_hook(lambda m,args:holder.update(x=args[0].detach().clone()))
 with torch.no_grad():
  for i,s in enumerate(samples):
   raw=model(torch.from_numpy(s['image'])[None]);f=holder['x'];reconstruction=head(f)
   assert torch.equal(raw,reconstruction)
   features.append(f);outputs.append(raw)
   if i%32==0:print(json.dumps({'phase':'feature_cache','set':label,'done':i+1,'total':len(samples)}),flush=True)
 hook.remove();return torch.cat(features),torch.cat(outputs)

def evaluate(head,features,samples):
 with torch.no_grad():raw=head(features)
 rows=[]
 for i,s in enumerate(samples):
  norm,confidence,cell=pose.decode_normalized_keypoints(raw[i:i+1]);gt=s['uv_normalized'];cy,cx=np.floor(gt[0,::-1]*13).astype(int)
  rows.append({'sample_id':s['sample_id'],'selected_cell':list(cell),'gt_cell':[int(cy),int(cx)],'cell_correct':cell==(cy,cx),'confidence':confidence,'corner_epe_px':float(np.linalg.norm((norm[1:]-gt[1:])*416,axis=1).mean()),'centroid_epe_px':float(np.linalg.norm((norm[0]-gt[0])*416)),'predicted_normalized_uv':norm.tolist(),'gt_normalized_uv':gt.tolist()})
 return {'count':len(rows),'correct_cell_count':sum(r['cell_correct'] for r in rows),'correct_cell_fraction':float(np.mean([r['cell_correct'] for r in rows])),'mean_corner_epe_px':float(np.mean([r['corner_epe_px'] for r in rows])),'mean_centroid_epe_px':float(np.mean([r['centroid_epe_px'] for r in rows])),'rows':rows}

results=[]
for arm in ['eventgenesis','v2e']:
 begin=time.time();train0=pose.collect_samples(arm,OLD/'bank/shared_trajectory.h5',OLD/'bank/v2e_events',size,'train',4,8.)
 val=pose.collect_samples(arm,OLD/'bank/shared_trajectory.h5',OLD/'bank/v2e_events',size,'validation',4,8.)
 train,aug=transformed_samples(train0,'train');challenge,vaug=transformed_samples(val,'validation')
 if arm=='eventgenesis':
  (R/'augmentation_registry.json').write_text(json.dumps({'train':aug,'validation_challenge':vaug},indent=2))
 else:
  expected=json.loads((R/'augmentation_registry.json').read_text());assert aug==expected['train'] and vaug==expected['validation_challenge']
 base=BASE/f'training/{arm}.pt';saved=torch.load(base,map_location='cpu',weights_only=False)
 model=Darknet(str(pose.POSE_SOURCE/'cfg/yolo-pose.cfg'));model.load_state_dict(saved['model']);model.eval()
 original={k:v.clone() for k,v in model.state_dict().items()};before_hash=pose.state_sha256(original)
 head=[m for m in model.modules() if isinstance(m,torch.nn.Conv2d)][-1]
 assert head.out_channels==20 and head.kernel_size==(1,1)
 for p in model.parameters():p.requires_grad_(False)
 feat,_=cache_features(model,head,train,arm+'/train');vf,vraw=cache_features(model,head,val,arm+'/validation');cf,_=cache_features(model,head,challenge,arm+'/challenge')
 targets=torch.from_numpy(np.stack([s['label'] for s in train]))
 # Numerical/gradient oracle against the existing pinned official target builder.
 probe=head(feat[:2]).detach().requires_grad_();reference,_=pose.patched_official_region_loss(probe,targets[:2],RegionLoss(),20,build_targets)
 reference/=2;grad_ref=torch.autograd.grad(reference,probe)[0]
 test=probe.detach().clone().requires_grad_();candidate,_=loss_fn(test,targets[:2],True);grad_new=torch.autograd.grad(candidate,test)[0]
 diff=float(abs(reference.detach()-candidate.detach()));gdiff=float((grad_ref-grad_new).abs().max());assert diff<1e-4 and gdiff<1e-4,(diff,gdiff)
 # Avoid sigmoid saturation inherited from the severely underfit confidence channel.
 if prior==0:
  with torch.no_grad():head.weight[18].zero_();head.bias[18].zero_()
 for p in head.parameters():p.requires_grad_(True)
 optimizer=torch.optim.Adam(head.parameters(),lr=.001)
 if prior:optimizer.load_state_dict(saved['optimizer'])
 rng=np.random.default_rng(2026)
 for _ in range(prior):rng.choice(len(train),8,replace=False)
 records=[];t0=time.time()
 for step in range(1,updates+1):
  indices=rng.choice(len(train),8,replace=False);indices_t=torch.from_numpy(indices)
  optimizer.zero_grad(set_to_none=True);raw=head(feat[indices_t]);loss,detail=loss_fn(raw,targets[indices_t],step>cfg['confidence_warmup_updates'])
  assert torch.isfinite(loss);loss.backward();gn=float(torch.sqrt(sum(p.grad.square().sum() for p in head.parameters())))
  assert np.isfinite(gn) and gn>0
  optimizer.step();records.append({'update':step,'global_output_layer_update':prior+step,'sample_indices':indices.tolist(),'confidence_supervised':step>cfg['confidence_warmup_updates'],'loss':float(loss.detach()),'gradient_norm':gn,**detail})
  if step%1000==0:print(json.dumps({'arm':arm,'update':step,'global_update':prior+step,'elapsed_s':time.time()-t0,**detail}),flush=True)
 final_validation=evaluate(head,vf,val);final_challenge=evaluate(head,cf,challenge);final_train=evaluate(head,feat,train)
 # No backbone parameter or BN buffer may change.
 changed=[k for k,v in model.state_dict().items() if not torch.equal(v,original[k])]
 head_keys=[k for k,v in model.state_dict().items() if v.data_ptr() in {head.weight.data_ptr(),head.bias.data_ptr()}]
 assert set(changed)==set(head_keys),(changed,head_keys)
 model.eval()
 with torch.no_grad():independent=model(torch.from_numpy(val[0]['image'])[None]);cached=head(vf[:1]);assert torch.allclose(independent,cached,atol=2e-5,rtol=1e-5)
 checkpoint=out/f'{arm}.pt';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'generator':arm,'base_checkpoint':str(base),'base_checkpoint_sha256':sha(base),'seed':0,'additional_output_layer_updates':updates,'prior_output_layer_updates':prior,'prior_full_network_updates':496,'confidence_warmup_steps':cfg['confidence_warmup_updates'],'frozen_backbone':True,'input_contract':saved['input_contract'],'protocol_sha256':sha(R/'protocol.json')},checkpoint)
 restored=Darknet(str(pose.POSE_SOURCE/'cfg/yolo-pose.cfg'));restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);restored.eval()
 with torch.no_grad():actual=restored(torch.from_numpy(val[0]['image'])[None]);assert torch.equal(actual,independent)
 gate=final_validation['correct_cell_count']>=7 and final_validation['mean_corner_epe_px']<25 and final_challenge['correct_cell_fraction']>=.75 and final_challenge['mean_corner_epe_px']<35
 detail_path=out/f'{arm}_updates.jsonl.gz'
 with gzip.open(detail_path,'wt') as stream:
  for record in records:stream.write(json.dumps(record)+'\n')
 result={'generator':arm,'base_state_sha256':before_hash,'checkpoint_sha256':sha(checkpoint),'checkpoint_path':str(checkpoint),'frozen_backbone_changed_keys':changed,'official_loss_value_absolute_difference':diff,'official_loss_gradient_max_abs_difference':gdiff,'updates':updates,'prior_output_layer_updates':prior,'confidence_updates':updates-cfg['confidence_warmup_updates'],'train_augmented_windows':len(train),'training_wall_s':time.time()-t0,'total_arm_wall_s':time.time()-begin,'checkpoint_restore_exact':True,'synthetic_gate_passed':gate,'train':final_train,'validation':final_validation,'affine_challenge':final_challenge,'training_detail':{'path':str(detail_path),'sha256':sha(detail_path),'updates':len(records)}}
 results.append(result);(out/'result.json').write_text(json.dumps({'scope':'synthetic-only frozen backbone localization repair','host':socket.gethostname(),'device':'cpu','threads':8,'results':results},indent=2))
 print(json.dumps({'phase':'arm_finished','arm':arm,'gate':gate,'validation':{k:v for k,v in final_validation.items() if k!='rows'},'challenge':{k:v for k,v in final_challenge.items() if k!='rows'}}),flush=True)
 del model,restored,feat,vf,cf,optimizer

gate={'passed':all(r['synthetic_gate_passed'] for r in results),'policy':cfg['synthetic_gate'],'arms':{r['generator']:{'passed':r['synthetic_gate_passed'],'correct_cell_count':r['validation']['correct_cell_count'],'validation_count':8,'ordinary_corner_epe_network_px':r['validation']['mean_corner_epe_px'],'challenge_cell_accuracy':r['affine_challenge']['correct_cell_fraction'],'challenge_corner_epe_network_px':r['affine_challenge']['mean_corner_epe_px']} for r in results},'real_data_opened':False,'time_utc':datetime.now(timezone.utc).isoformat()}
(R/'synthetic_gate.json').write_text(json.dumps(gate,indent=2))
(R/'weights_frozen.json').write_text(json.dumps({'checkpoint_policy':f'fixed final additional{updates} output-layer updates; before fresh real pilot','checkpoints':{r['generator']:{'path':r['checkpoint_path'],'sha256':r['checkpoint_sha256']} for r in results},'time_unix':time.time()},indent=2))
print('REPAIR_TRAINING_DONE',json.dumps(gate),flush=True)
