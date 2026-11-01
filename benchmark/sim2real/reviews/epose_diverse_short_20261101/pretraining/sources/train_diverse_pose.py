"""Paired fixed-budget full-network synthetic fine tuning on richer bank."""
import os,sys,json,time,hashlib,ast,gzip,copy,argparse
from pathlib import Path
import numpy as np,torch,cv2
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=Path(os.environ['EPOSE_REPAIR_RUN']);OLD=P/'runs/sim2real/20261101-epose-wood-short-research-v1';cfg=json.loads((R/'protocol.json').read_text())
sys.path[:0]=[str(OLD/'sources/benchmark/sim2real/scripts'),str(R/'sources')]
import run_pose_stage2_pilot as pose
pose.POSE_SOURCE=P/'external/sources/downstream/SingleShotPose';sys.path.insert(0,str(pose.POSE_SOURCE));from darknet import Darknet
from diverse_samples import collect_diverse
from pose_center_decoder import decode_classified_pose
from region_loss import RegionLoss,build_targets
torch.set_num_threads(8);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
arm=sys.argv[1];assert arm in ['eventgenesis','v2e'];out=R/'training';out.mkdir(exist_ok=True)
parsed=ast.parse((R/'sources/repair_training.py').read_text(encoding='utf-8-sig'));defs=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='loss_fn'];exec(compile(ast.Module(body=defs,type_ignores=[]),'verified_coordinate_loss','exec'),globals());coordinate_loss=loss_fn
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def evaluate(model,samples):
 model.eval();rows=[]
 with torch.no_grad():
  for s in samples:
   raw=model(torch.from_numpy(s['image'])[None]);uv,confidence,cell=decode_classified_pose(raw);cy,cx=np.floor(s['uv_normalized'][0,::-1]*13).astype(int);dist=np.linalg.norm((uv-s['uv_normalized'])*416,axis=1)
   rows.append({'sample_id':s['sample_id'],'selected_cell':list(cell),'gt_cell':[int(cy),int(cx)],'cell_correct':cell==(cy,cx),'confidence':confidence,'corner_epe_px':float(dist[1:].mean()),'centroid_epe_px':float(dist[0]),'predicted_normalized_uv':uv.tolist(),'gt_normalized_uv':s['uv_normalized'].tolist()})
 return {'count':len(rows),'correct_cell_count':sum(r['cell_correct'] for r in rows),'correct_cell_fraction':float(np.mean([r['cell_correct'] for r in rows])),'mean_corner_epe_px':float(np.mean([r['corner_epe_px'] for r in rows])),'mean_centroid_epe_px':float(np.mean([r['centroid_epe_px'] for r in rows])),'rows':rows}
train=collect_diverse(pose,R,arm,'train');val=collect_diverse(pose,R,arm,'validation');assert len(train)==cfg['train_windows'] and len(val)==cfg['validation_windows']
records=[{k:s[k] for k in ['sample_id','label_time_s','event_count','uv_normalized']} for s in train+val]
for s in records:s['uv_normalized']=s['uv_normalized'].tolist()
(out/f'{arm}_samples.json').write_text(json.dumps(records,indent=2))
base=Path(cfg['initial_checkpoints'][arm]['path']);assert sha(base)==cfg['initial_checkpoints'][arm]['sha256'];saved=torch.load(base,map_location='cpu',weights_only=False);model=Darknet(str(pose.POSE_SOURCE/'cfg/yolo-pose.cfg'));model.load_state_dict(saved['model']);initial_hash=pose.state_sha256(model.state_dict());bn_before=pose.batchnorm_state_snapshot(model)
optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);rng=np.random.default_rng(cfg['sampling_seed']);best=None;history=[];updates=[];begin=time.time();live=(out/f'{arm}_updates_live.jsonl').open('w')
def atomic_checkpoint(path,payload):
 temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)
def candidate(step):
 global best
 result=evaluate(model,val);localized=result['mean_centroid_epe_px']<cfg['synthetic_gate']['max_centroid_epe_px'] and result['correct_cell_fraction']>=cfg['synthetic_gate']['min_correct_cell_fraction'];key=(not localized,result['mean_centroid_epe_px'],result['mean_corner_epe_px'])
 entry={'update':step,**{k:v for k,v in result.items() if k!='rows'}};history.append(entry)
 if best is None or key<best['key']:
  best={'key':key,'step':step,'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'result':result}
  atomic_checkpoint(out/f'{arm}_best_so_far.pt',{'model':best['model'],'optimizer':best['optimizer'],'selected_update':step,'validation':result,'protocol_sha256':sha(R/'protocol.json'),'input_contract':saved['input_contract']})
 live.flush()
 atomic_checkpoint(out/f'{arm}_last_resume.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'completed_update':step,'numpy_rng_state':rng.bit_generator.state,'torch_rng_state':torch.get_rng_state(),'history':history,'selected_update_so_far':best['step'],'protocol_sha256':sha(R/'protocol.json'),'input_contract':saved['input_contract'],'live_log':str(out/f'{arm}_updates_live.jsonl'),'manual_lr_schedule':'global step, common fixed budget from protocol.json','BN_policy':'train buffers on synthetic batches; eval only for validation and real'})
 (out/f'{arm}_checkpoint_progress.json').write_text(json.dumps({'completed_update':step,'best_update':best['step'],'resume_checkpoint':str(out/f'{arm}_last_resume.pt'),'best_checkpoint':str(out/f'{arm}_best_so_far.pt'),'saved_unix':time.time()},indent=2))
 print(json.dumps({'phase':'validation','arm':arm,**entry,'selected_update':best['step'],'wall_s':time.time()-begin}),flush=True);model.train()
candidate(0)
for step in range(1,cfg['optimizer_updates_per_arm']+1):
 indices=rng.choice(len(train),cfg['batch_size'],replace=False);images=[];targets=[];aug=[]
 for i in indices:
  s=train[i];uv=s['uv_normalized'];scale=float(rng.uniform(.85,1.20));dx,dy=rng.uniform(-24,24,2);new=(uv-.5)*scale+.5+np.array([dx,dy])/416
  if (new>.01).all() and (new<.99).all():
   A=np.array([[scale,0,(1-scale)*208+dx],[0,scale,(1-scale)*208+dy]],np.float32);image=cv2.warpAffine(s['image'].transpose(1,2,0),A,(416,416)).transpose(2,0,1).copy();label=pose.build_label(new*416,416,416)
  else:scale=1.;dx=dy=0.;image=s['image'];label=s['label']
  images.append(image);targets.append(label);aug.append([scale,float(dx),float(dy)])
 image=torch.from_numpy(np.stack(images));target=torch.from_numpy(np.stack(targets));lr=cfg['learning_rate']*min(1.,step/30)*(.1+.9*.5*(1+np.cos(np.pi*step/cfg['optimizer_updates_per_arm'])))
 for g in optimizer.param_groups:g['lr']=lr
 optimizer.zero_grad(set_to_none=True);raw=model(image);coord,detail=coordinate_loss(raw,target,False);uv=target[:,1:19].reshape(-1,9,2);classes=(uv[:,0,1]*13).long()*13+(uv[:,0,0]*13).long();ce=torch.nn.functional.cross_entropy(raw[:,18].flatten(1),classes);loss=coord+ce
 if step==1:
  probe=raw.detach().clone().requires_grad_();oracle,_=pose.patched_official_region_loss(probe,target,RegionLoss(),0,build_targets);test,_=coordinate_loss(probe,target,False);assert abs(float(oracle.detach()/cfg['batch_size']-test.detach()))<1e-4
 assert torch.isfinite(loss);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),100.));assert np.isfinite(gn) and gn>0;optimizer.step();updates.append({'update':step,'sample_indices':indices.tolist(),'affine':aug,'lr':float(lr),'loss':float(loss.detach()),'coordinate_loss':float(coord.detach()),'center_ce':float(ce.detach()),'gradient_norm':gn});live.write(json.dumps(updates[-1])+'\n')
 if step%20==0:print(json.dumps({'phase':'train','arm':arm,**{k:v for k,v in updates[-1].items() if k not in ['sample_indices','affine']},'wall_s':time.time()-begin}),flush=True)
 if step%cfg['validation_interval']==0:candidate(step)
assert best is not None
if history[-1]['update']!=cfg['optimizer_updates_per_arm']:candidate(cfg['optimizer_updates_per_arm'])
live.close()
last=out/f'{arm}_unselected_final.pt';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'executed_updates':len(updates)},last);model.load_state_dict(best['model']);model.eval();checkpoint=out/f'{arm}.pt'
torch.save({'model':best['model'],'optimizer':best['optimizer'],'selected_update':best['step'],'executed_updates':len(updates),'generator':arm,'base_checkpoint':str(base),'input_contract':saved['input_contract'],'protocol_sha256':sha(R/'protocol.json'),'BN_policy':'train on synthetic batches; eval on validation and real'},checkpoint)
restored=Darknet(str(pose.POSE_SOURCE/'cfg/yolo-pose.cfg'));restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);restored.eval()
with torch.no_grad():assert torch.equal(model(torch.from_numpy(val[0]['image'])[None]),restored(torch.from_numpy(val[0]['image'])[None]))
v=best['result'];gate=v['mean_centroid_epe_px']<cfg['synthetic_gate']['max_centroid_epe_px'] and v['correct_cell_fraction']>=cfg['synthetic_gate']['min_correct_cell_fraction']
with gzip.open(out/f'{arm}_updates.jsonl.gz','wt') as f:
 for u in updates:f.write(json.dumps(u)+'\n')
result={'generator':arm,'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'initial_state_sha256':initial_hash,'initial_checkpoint':str(base),'executed_updates':len(updates),'selected_update':best['step'],'checkpoint_restore_exact':True,'synthetic_gate_passed':gate,'validation':v,'history':history,'training_wall_s':time.time()-begin,'BN_running_buffers_changed':pose.batchnorm_state_snapshot(model)!=bn_before,'final_unselected_sha256':sha(last),'no_real_data_access':True}
(out/f'{arm}_result.json').write_text(json.dumps(result,indent=2));print('TRAINING_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['validation','history']}),flush=True)
