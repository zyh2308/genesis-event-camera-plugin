"""Read-only early recovery audit; never updates the training models."""
import os,sys,json,hashlib,io,time
from pathlib import Path
import numpy as np,torch,cv2
torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=Path(os.environ['EPOSE_REPAIR_RUN']);OLD=P/'runs/sim2real/20261101-epose-wood-short-research-v1';cfg=json.loads((R/'protocol.json').read_text());sys.path[:0]=[str(OLD/'sources/benchmark/sim2real/scripts'),str(R/'sources')];import run_pose_stage2_pilot as pose
pose.POSE_SOURCE=P/'external/sources/downstream/SingleShotPose';sys.path.insert(0,str(pose.POSE_SOURCE));from darknet import Darknet
from diverse_samples import collect_diverse
from pose_center_decoder import decode_classified_pose
rows=[]
for arm in ['eventgenesis','v2e']:
 path=R/f'training/{arm}_last_resume.pt';blob=path.read_bytes();ck=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);step=ck['completed_update'];assert step>=100;model=Darknet(str(pose.POSE_SOURCE/'cfg/yolo-pose.cfg'));model.load_state_dict(ck['model']);opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);opt.load_state_dict(ck['optimizer']);assert {int(v['step']) for v in opt.state.values()}=={step};assert pose.state_sha256(model.state_dict())==pose.state_sha256(ck['model'])
 rng=np.random.default_rng(cfg['sampling_seed'])
 for _ in range(step):
  rng.choice(cfg['train_windows'],cfg['batch_size'],replace=False)
  for j in range(cfg['batch_size']):rng.uniform(.85,1.20);rng.uniform(-24,24,2)
 assert rng.bit_generator.state==ck['numpy_rng_state'];torch.set_rng_state(ck['torch_rng_state']);assert torch.equal(torch.get_rng_state(),ck['torch_rng_state']);del ck,blob,opt
 bestblob=(R/f'training/{arm}_best_so_far.pt').read_bytes();best=torch.load(io.BytesIO(bestblob),map_location='cpu',weights_only=False);model.load_state_dict(best['model']);model.eval();sample=collect_diverse(pose,R,arm,'validation')[0]
 with torch.no_grad():uv,_,_=decode_classified_pose(model(torch.from_numpy(sample['image'])[None]))
 error=float(np.max(np.abs(uv-np.array(best['validation']['rows'][0]['predicted_normalized_uv']))));assert error<1e-5
 row={'generator':arm,'rolling_completed_step':step,'all_adam_steps_restored':step,'model_state_restore_hash_exact':True,'numpy_rng_state_replayed_exact':True,'torch_rng_restore_exact':True,'best_selected_update':best['selected_update'],'best_checkpoint_payload_sha256':hashlib.sha256(bestblob).hexdigest(),'restored_forward_max_normalized_uv_difference':error,'inference_threads_audit':1,'training_threads':8,'no_optimizer_update_in_audit':True};rows.append(row);print(json.dumps(row),flush=True);del model,best,bestblob
(R/'early_checkpoint_recovery_audit.json').write_text(json.dumps({'passed':True,'read_only_audit':True,'results':rows,'completed_unix':time.time()},indent=2))
