import os,sys,json,hashlib,gzip,ast,time
from pathlib import Path
import numpy as np,torch,cv2
torch.set_num_interop_threads(1);cv2.setNumThreads(1)
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=Path(os.environ['EPOSE_REPAIR_RUN']);OLD=P/'runs/sim2real/20261101-epose-wood-short-research-v1';cfg=json.loads((R/'protocol.json').read_text());sys.path[:0]=[str(OLD/'sources/benchmark/sim2real/scripts'),str(R/'sources')]
import run_pose_stage2_pilot as pose
pose.POSE_SOURCE=P/'external/sources/downstream/SingleShotPose';sys.path.insert(0,str(pose.POSE_SOURCE));from darknet import Darknet
from diverse_samples import collect_diverse
from verify_pinned_source import verify_pinned_source
torch.set_num_threads(8)
downstream_verified=verify_pinned_source(pose.POSE_SOURCE,'SingleShotPose',pose.POSE_SOURCE_COMMIT,OLD/'sources/benchmark/sim2real/manifests/downstream_source_bundle.json')
(R/'downstream_source_verification_after_training.json').write_text(json.dumps(downstream_verified,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'protocol.json')==json.loads((R/'protocol_pretraining_hash.json').read_text())['sha256']
source_manifest=json.loads((R/'source_snapshot_before_training.json').read_text())
for relative,digest in source_manifest['files'].items():assert sha(R/relative)==digest,relative
results=[];checkpoints={};prior_records=None;reference_samples=None;checks=[]
for arm in ['eventgenesis','v2e']:
 row=json.loads((R/f'training/{arm}_result.json').read_text());assert row['executed_updates']==cfg['optimizer_updates_per_arm'] and row['checkpoint_restore_exact'];assert sha(Path(row['checkpoint_path']))==row['checkpoint_sha256']
 with gzip.open(R/f'training/{arm}_updates.jsonl.gz','rt') as f:updates=[json.loads(l) for l in f]
 assert len(updates)==cfg['optimizer_updates_per_arm'];assert all(np.isfinite([x['loss'],x['gradient_norm']]).all() and x['gradient_norm']>0 for x in updates)
 decisions=[(x['sample_indices'],x['affine'],x['lr']) for x in updates]
 if prior_records is None:prior_records=decisions
 else:assert prior_records==decisions,'paired sampling/augmentation differs'
 train=collect_diverse(pose,R,arm,'train');val=collect_diverse(pose,R,arm,'validation');labels=[(s['sample_id'],s['label'].tolist(),s['label_time_s']) for s in train+val]
 if reference_samples is None:reference_samples=labels
 else:assert reference_samples==labels,'paired physical labels differ'
 assert all(abs(s['event_end_s']-s['label_time_s'])<1e-10 and abs(s['event_end_s']-s['event_start_s']-.01)<1e-10 for s in train+val)
 ck=torch.load(row['checkpoint_path'],map_location='cpu',weights_only=False);assert ck['protocol_sha256']==sha(R/'protocol.json');steps={int(x['step']) for x in ck['optimizer']['state'].values()};assert steps==({row['selected_update']} if row['selected_update'] else set())
 model=Darknet(str(pose.POSE_SOURCE/'cfg/yolo-pose.cfg'));model.load_state_dict(ck['model']);model.eval();independent=[]
 final_path=R/f'training/{arm}_unselected_final.pt';assert sha(final_path)==row['final_unselected_sha256'];final=torch.load(final_path,map_location='cpu',weights_only=False);executed_steps={int(v['step']) for v in final['optimizer']['state'].values()};assert executed_steps=={cfg['optimizer_updates_per_arm']}
 initial=torch.load(cfg['initial_checkpoints'][arm]['path'],map_location='cpu',weights_only=False)['model'];parameter_names=[name for name,_ in model.named_parameters()];changed=[name for name in parameter_names if not torch.equal(final['model'][name],initial[name])];assert parameter_names[0] in changed and len(changed)>20
 resume=torch.load(R/f'training/{arm}_last_resume.pt',map_location='cpu',weights_only=False);assert resume['completed_update']==cfg['optimizer_updates_per_arm'] and resume['protocol_sha256']==sha(R/'protocol.json');assert pose.state_sha256(resume['model'])==pose.state_sha256(final['model'])
 replay=np.random.default_rng(cfg['sampling_seed'])
 for update in updates:
  assert replay.choice(len(train),cfg['batch_size'],replace=False).tolist()==update['sample_indices']
  for _ in range(cfg['batch_size']):replay.uniform(.85,1.20);replay.uniform(-24,24,2)
 assert replay.bit_generator.state==resume['numpy_rng_state'];assert resume['torch_rng_state'].device.type=='cpu' and resume['torch_rng_state'].dtype==torch.uint8
 del final,initial,resume
 with torch.no_grad():
  for s,reported in zip(val,row['validation']['rows']):
   raw=model(torch.from_numpy(s['image'])[None])[0].numpy();cy,cx=np.unravel_index(raw[18].argmax(),raw[18].shape);local=raw[:18,cy,cx].reshape(9,2).copy();local[0]=1/(1+np.exp(-local[0]));uv=(local+np.array([cx,cy]))/13.;assert np.max(np.abs(uv-np.array(reported['predicted_normalized_uv'])))<2e-6
   d=np.linalg.norm((uv-s['uv_normalized'])*416,axis=1);truth=np.floor(s['uv_normalized'][0,::-1]*13).astype(int);independent.append({'center':float(d[0]),'corner':float(d[1:].mean()),'correct':bool(np.array_equal([cy,cx],truth))})
 center=float(np.mean([x['center'] for x in independent]));corner=float(np.mean([x['corner'] for x in independent]));accuracy=float(np.mean([x['correct'] for x in independent]));passed=center<cfg['synthetic_gate']['max_centroid_epe_px'] and accuracy>=cfg['synthetic_gate']['min_correct_cell_fraction']
 assert abs(center-row['validation']['mean_centroid_epe_px'])<1e-4 and abs(corner-row['validation']['mean_corner_epe_px'])<1e-4
 oracle=[]
 for s in (train+val)[::8]:
  K=np.diag([346/416,260/416,1.])@s['camera_matrix_416'];ok,r,t=cv2.solvePnP(s['keypoints_obj'][1:].astype(float),s['uv'][1:].astype(float),K,None,flags=cv2.SOLVEPNP_EPNP);assert ok;err=float(np.linalg.norm(t[:,0]-s['camera_from_object'][:3,3]));assert err<1e-4;oracle.append(err)
 checks.append({'generator':arm,'validation_count':len(val),'centroid_epe_px':center,'corner_epe_px':corner,'cell_accuracy':accuracy,'passed':passed,'oracle_max_translation_m':max(oracle),'selected_optimizer_step':list(steps),'executed_optimizer_steps':list(executed_steps),'changed_trainable_parameter_tensors':len(changed),'first_backbone_parameter_changed':True,'resume_model_and_numpy_rng_verified':True,'all_updates_finite':True,'paired_samples_and_augmentation_identical':True});results.append(row);checkpoints[arm]={'path':row['checkpoint_path'],'sha256':row['checkpoint_sha256']};del model,ck
frozen={'checkpoint_policy':cfg['checkpoint_selection'],'checkpoints':checkpoints,'time_unix':time.time()};(R/'weights_frozen.json').write_text(json.dumps(frozen,indent=2));gate={'passed':all(x['passed'] for x in checks),'arms':{x['generator']:x for x in checks},'policy':cfg['synthetic_gate'],'real_data_opened':False};(R/'synthetic_gate.json').write_text(json.dumps(gate,indent=2));(R/'training/result.json').write_text(json.dumps({'results':results,'scope':cfg['scope']},indent=2))
(R/'independent_synthetic_verification.json').write_text(json.dumps({'passed':True,'synthetic_gate_passed':gate['passed'],'checks':checks,'independent_numpy_decoder':True,'checkpoint_hashes_verified':True,'pretraining_source_manifest_verified':True,'fresh_real_not_opened':True},indent=2));print(json.dumps(gate),flush=True)
