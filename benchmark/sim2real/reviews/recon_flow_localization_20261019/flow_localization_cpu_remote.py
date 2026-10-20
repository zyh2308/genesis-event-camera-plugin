"""Read-only synthetic diagnostics; no network training or real data access."""
from pathlib import Path
import json,sys,hashlib,math,gc
from datetime import datetime,timezone
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(2)
P=Path('/home/zhaoyuhan/Eventbased_WAM')
R=P/'repositories/gec-sim2real-stage5-coverage-exec-20261019'
S=R/'benchmark/sim2real/scripts'
sys.path[:0]=[str(S),str(R/'benchmark/sim2real/compat')]
from stage5_flow_supervised_pilot import multiscale_supervised_epe
from steflow_align_corners import loss_warp
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out={'schema':'research_flow_localization_cpu_v1','started_utc':datetime.now(timezone.utc).isoformat(),
     'real_data_used':False,'network_training_performed':False,
     'helper_sha256':sha(S/'stage5_flow_supervised_pilot.py'),
     'warp_sha256':sha(R/'benchmark/sim2real/compat/steflow_align_corners.py')}
# Mathematical loss oracle: supplied targets at every scale are known in closed form.
h,w=32,48
gt=torch.zeros(1,2,h,w);gt[:,0]=.75;gt[:,1]=-.375
valid=torch.ones(1,1,h,w)
shapes=[(32,48),(16,24),(8,12),(4,6)]
targets=[]
for sh in shapes:
 t=torch.zeros(1,2,*sh);t[:,0]=.75*sh[1]/w;t[:,1]=-.375*sh[0]/h;targets.append(t)
oracle=float(multiscale_supervised_epe(targets,gt,valid)[0])
pred=[torch.zeros_like(t,requires_grad=True) for t in targets]
loss=multiscale_supervised_epe(pred,gt,valid)[0];loss.backward()
grad_direction=[bool((p.grad[:,0]<0).all() and (p.grad[:,1]>0).all()) for p in pred]
finite_difference=[]
for i in [0,3]:
 idx=(0,0,shapes[i][0]//2,shapes[i][1]//2)
 p0=[p.detach().clone() for p in pred];p1=[p.detach().clone() for p in pred]
 eps=.001;p0[i][idx]+=eps;p1[i][idx]-=eps
 numerical=float((multiscale_supervised_epe(p0,gt,valid)[0]-multiscale_supervised_epe(p1,gt,valid)[0])/(2*eps))
 analytical=float(pred[i].grad[idx])
 finite_difference.append({'scale':i,'numerical':numerical,'autograd':analytical,'abs_error':abs(numerical-analytical)})
opt=torch.optim.Adam(pred,lr=.02)
curve=[]
for i in range(300):
 opt.zero_grad();value=multiscale_supervised_epe(pred,gt,valid)[0];value.backward();opt.step()
 if i in [0,49,99,199,299]:
  curve.append({'update':i+1,'loss_before_update':float(value.detach()),'fixed_full_epe_after_update':float(torch.linalg.vector_norm(pred[0].detach()-gt,dim=1).mean())})
out['loss_oracle']={'known_target_loss':oracle,'expected_minimum':.001,'initial_loss':float(loss.detach()),
 'all_gradients_point_toward_target':all(grad_direction),'finite_difference':finite_difference,'free_prediction_optimization':curve,
 'note':'Only loss mathematics/gradient and direct prediction optimization are tested; no STE-FlowNet forward/backward or generator superiority claim.'}
# Temporal sampling and photometric identifiability on existing shared synthetic bank.
D=P/'data/sim2real/stage4/20261017-stage4-flowdt1-seed0-corrected-v5/steflow_30ms'
mp=D/'training_bank_manifest.json';manifest=json.loads(mp.read_text());out['data_manifest_sha256']=sha(mp)
entries={(x['generator'],x['rollout_id']):x for x in manifest['entries']}
ids=[f's{i}_{suffix}' for suffix in ['00','06'] for i in range(4)]
indices=[0,16,32,48,65]
rows=[]
for rid in ids:
 pair={}
 for arm in ['eventgenesis','v2e']:
  with np.load(entries[arm,rid]['path'],allow_pickle=False) as z:
   events=z['events'];pair[arm]={'events':np.asarray(events[indices]).copy()};del events
   if arm=='eventgenesis':
    for name in ['gray_prev','gray_next','flow_gt','flow_valid','sample_ids']:
     pair[arm][name]=np.asarray(z[name][indices]).copy()
 for k,index in enumerate(indices):
  eg=pair['eventgenesis'];v2=pair['v2e']
  g=np.asarray(eg['flow_gt'][k],dtype=np.float32);v=eg['flow_valid'][k].astype(bool)
  a0=eg['events'][k];a1=v2['events'][k]
  active=(np.any(a0!=0,axis=(0,3))|np.any(a1!=0,axis=(0,3)))&v
  magnitude=np.hypot(g[0],g[1]);moving=(magnitude>=.5)&v
  im0=torch.from_numpy(np.asarray(eg['gray_prev'][k],dtype=np.float32))[None,None]
  im1=torch.from_numpy(np.asarray(eg['gray_next'][k],dtype=np.float32))[None,None]
  flow=torch.from_numpy(g)[None]
  warped=loss_warp(im1,flow)[0,0].numpy()
  photo0=im0[0,0].numpy();photo1=im1[0,0].numpy()
  dx=np.pad(np.abs(np.diff(photo0,axis=1)),((0,0),(0,1)))
  dy=np.pad(np.abs(np.diff(photo0,axis=0)),((0,1),(0,0)))
  textured=((dx+dy)>(1/255))&v
  err0=np.abs(photo1-photo0);errgt=np.abs(warped-photo0)
  def mean_on(a,m):return float(a[m].mean(dtype=np.float64)) if m.any() else None
  rows.append({'rollout_id':rid,'split':entries['eventgenesis',rid]['split'],'sample_index':index,'sample_id':str(eg['sample_ids'][k]),
   'time_start_s':index*.03,'valid_pixels':int(v.sum()),'common_active_pixels':int(active.sum()),'gt_motion_ge_half_pixel':int(moving.sum()),
   'active_and_gt_motion_ge_half_pixel':int((active&moving).sum()),'textured_pixels_gradient_gt_1_over_255':int(textured.sum()),
   'moving_and_textured_pixels':int((moving&textured).sum()),'gray_range':[float(photo0.min()),float(photo0.max())],
   'counts':{'eventgenesis':float(a0.sum(dtype=np.float64)),'v2e':float(a1.sum(dtype=np.float64))},
   'gt_mean_magnitude_valid':mean_on(magnitude,v),'gt_mean_magnitude_active':mean_on(magnitude,active),
   'photo_mae_zero_valid':mean_on(err0,v),'photo_mae_gt_valid':mean_on(errgt,v),
   'photo_mae_zero_active':mean_on(err0,active),'photo_mae_gt_active':mean_on(errgt,active),
   'photo_mae_zero_moving':mean_on(err0,moving),'photo_mae_gt_moving':mean_on(errgt,moving)})
 del pair;gc.collect()
out['temporal_support_rows']=rows
out['sampling_contract']='8 predetermined rollouts (one train and one validation per stratum), 5 fixed physical times each; no event-count filtering; 40 windows, diagnostic only.'
out['photo_oracle_contract']='single full-resolution forward-GT warp next(x+flow)->previous(x), shared geometric validity; MAE diagnostic, not the exact four-scale formal optimization objective; no new best-checkpoint selection.'
out['finished_utc']=datetime.now(timezone.utc).isoformat()
print(json.dumps(out))
