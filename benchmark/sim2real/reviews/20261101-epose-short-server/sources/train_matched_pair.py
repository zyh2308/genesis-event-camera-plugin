import os,sys,json,argparse,time,hashlib,gc,socket,subprocess
from pathlib import Path
import numpy as np,torch
import h5py
R=Path(os.environ['EPOSE_SHORT_RUN']);S=R/'sources';sys.path[:0]=[str(S/'benchmark/sim2real/scripts')]
import run_pose_stage2_pilot as pose
from verify_pinned_source import verify_pinned_source
P=Path('/home/zhaoyuhan/Eventbased_WAM');pose.POSE_SOURCE=P/'external/sources/downstream/SingleShotPose';sys.path.insert(0,str(pose.POSE_SOURCE))
from region_loss import build_targets
verify=verify_pinned_source(pose.POSE_SOURCE,'SingleShotPose',pose.POSE_SOURCE_COMMIT,S/'benchmark/sim2real/manifests/downstream_source_bundle.json')
torch.set_num_threads(2);torch.cuda.set_device(0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
for line in apps.splitlines():
 parts=[x.strip() for x in line.split(',')]
 if parts[0]=='GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460' and int(parts[1])!=os.getpid():raise RuntimeError('reserved GPU contains another compute process; do not start training')
(R/'execution_environment.json').write_text(json.dumps({'host':socket.gethostname(),'pid':os.getpid(),'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'device_name':torch.cuda.get_device_name(0),'gpu_uuid':'GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460','physical_gpu_index':1,'logical_device':'cuda:0','cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'cpu_threads':2,'initialization':'random seed0; no ImageNet or real-pose pretrained weights'},indent=2))
keypoints=np.load(R/'cad_keypoints.npy');pose.box_keypoints=lambda size:keypoints.copy()
bank=R/'bank';cfg=json.loads((R/'short_config.json').read_text());data=json.loads((bank/'config.json').read_text());size=data['simulation']['target_box']['size_m']
expected_bases=np.array([[-.07,0,.21],[.06,.03,.23],[0,-.06,.22],[.04,.04,.25],[-.035,.025,.235],[.02,-.025,.215]])
trajectory_audit=[]
with h5py.File(bank/'shared_trajectory.h5','r') as h:
 for i,name in enumerate([f'train_{j:02d}' for j in range(4)]+[f'validation_{j:02d}' for j in range(2)]):
  g=h['episodes'][name];t=np.asarray(g['frame_timestamps_s']);T=np.asarray(g['target_pose']);assert np.array_equal(t,np.array([0,.01,.02,.03,.04]))
  end=expected_bases[i]+np.array([.025 if i%2==0 else -.025,0,.012*(1 if i%3 else -1)])
  error=max(float(np.max(np.abs(T[0,:3,3]-expected_bases[i]))),float(np.max(np.abs(T[-1,:3,3]-end))));assert error<1e-5,(name,error)
  trajectory_audit.append({'rollout_id':name,'start_end_position_max_error_m':error})
(R/'trajectory_reset_audit.json').write_text(json.dumps(trajectory_audit,indent=2))
out=R/'training';out.mkdir(exist_ok=False)
args=argparse.Namespace(seed=0,epochs=31,confidence_pretrain_epochs=15,learning_rate=.001,momentum=.9,weight_decay=.0005,output_root=out)
samples={}
for arm in ('eventgenesis','v2e'):
 samples[arm]={split:pose.collect_samples(arm,bank/'shared_trajectory.h5',bank/'v2e_events',size,split,4,8.) for split in ('train','validation')}
 assert len(samples[arm]['train'])==16 and len(samples[arm]['validation'])==8
for split in ('train','validation'):
 for a,b in zip(samples['eventgenesis'][split],samples['v2e'][split]):
  assert a['sample_id']==b['sample_id'] and np.array_equal(a['label'],b['label']) and a['label_time_s']==a['event_end_s']
# Independent exact-corner PnP oracle before training: no real data accessed.
import cv2
from scipy.spatial.transform import Rotation
oracles=[]
for sample in samples['eventgenesis']['validation']:
 ok,r,t=cv2.solvePnP(keypoints[1:].astype(float),sample['uv'][1:].astype(float),np.diag([346/416,260/416,1])@sample['camera_matrix_416'],None,flags=cv2.SOLVEPNP_EPNP)
 assert ok
 T=sample['camera_from_object'];pred=cv2.Rodrigues(r)[0];rot=float(np.degrees(np.arccos(np.clip((np.trace(pred@T[:3,:3].T)-1)/2,-1,1))));trans=float(np.linalg.norm(t[:,0]-T[:3,3]));assert trans<1e-4 and rot<.05,(trans,rot);oracles.append({'sample_id':sample['sample_id'],'translation_m':trans,'rotation_deg':rot})
(R/'synthetic_pnp_oracle.json').write_text(json.dumps(oracles,indent=2))
results=[]
for arm in ('eventgenesis','v2e'):
 print(json.dumps({'phase':'training_started','arm':arm,'train_windows':16,'validation_windows':8,'updates':496}),flush=True)
 value=pose.train_generator(arm,samples[arm]['train'],samples[arm]['validation'],args,torch.device('cuda:0'),build_targets)
 assert len(value['training_detail'])==496 and value['confidence_supervised_update_count']==256
 results.append(value);(out/'result.json').write_text(json.dumps({'scope':'exact CAD-matched short synthetic training; real test not yet accessed','host':socket.gethostname(),'source_verification':verify,'results':results},indent=2));print(json.dumps({'phase':'training_finished','arm':arm,'epe_px':value['validation_keypoint_mean_epe_px'],'wall_s':value['train_wall_seconds']}),flush=True);gc.collect();torch.cuda.empty_cache()
assert results[0]['initial_state_sha256']==results[1]['initial_state_sha256']
frozen={'checkpoint_policy':'fixed final update496 per arm; before real test','checkpoints':{v['generator']:{'path':v['checkpoint_path'],'sha256':v['checkpoint_sha256']} for v in results},'time_unix':time.time()}
(R/'weights_frozen.json').write_text(json.dumps(frozen,indent=2));print('TRAINING_COMPLETE_WEIGHTS_FROZEN',flush=True)
