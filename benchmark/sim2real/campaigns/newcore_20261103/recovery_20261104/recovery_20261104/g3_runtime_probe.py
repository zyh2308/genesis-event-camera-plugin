"""Small CUDA/model/loss probe, explicitly excluded from formal training/results."""
import argparse,hashlib,importlib,json,os,sys,time,types
from pathlib import Path
import h5py,numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--task',choices=['flow','reconstruction','pose'],required=True);a=p.parse_args();R=a.campaign;P=Path('/home/zhaoyuhan/Eventbased_WAM');S=R/'sources';scripts=S/'benchmark/sim2real/scripts';sys.path[:0]=[str(scripts),str(S/'benchmark/sim2real/compat')]
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0);assert torch.cuda.device_count()==1
raw=np.load(R/'pose_bank/events/eventgenesis/train_000.npy');source=R/'pose_bank/physical/train_000.h5';out=R/'recovery_20261104/g3_runtime_probe'/a.task;out.mkdir(parents=True,exist_ok=True)
def hashed(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
with h5py.File(source,'r') as h:
 g=h['episodes/train_000'];frames=np.asarray(g['v2e_display_luma_u8']);ts=np.asarray(g['frame_timestamps_s']);uv=np.vstack([g['projected_centroid_uv'][5],g['projected_corners_uv'][5]])/np.array([346.,260.])
if a.task=='flow':
 root=P/'external/sources/downstream/STE-FlowNet';sys.path.insert(0,str(root))
 from run_steflow_stage2_pilot import pure_torch_spatial_correlation,objective_dt1
 from steflow_align_corners import patch_ste
 corr=importlib.import_module('models.corr');corr.spatial_correlation_sample=pure_torch_spatial_correlation
 import models,multiscaleloss
 patch_ste(models,multiscaleloss,root)
 from prepare_steflow_stage0 import gaussian_count_representation
 from train_steflow_stage9 import apply_joint_transform
 chosen=raw[(raw[:,0]>ts[4])&(raw[:,0]<=ts[7])].copy();chosen[:,0]-=ts[4]
 event=gaussian_count_representation(chosen,260,346)
 sample={'events':event[None],'gray_prev':frames[4:5].astype(np.float32)/255,'gray_next':frames[7:8].astype(np.float32)/255,'flow_gt':np.zeros((1,2,260,346),np.float32),'flow_valid':np.ones((1,260,346),np.uint8)}
 x,gray,_,_=apply_joint_transform(sample,np.random.default_rng(0),torch.device('cuda:0'));model=models.steflow().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.0004)
 def forward_loss():
  prediction=model(x,256,use_dt4=False,flow_seq=False);loss,photo,smooth=objective_dt1(prediction,gray[:,:1],gray[:,1:],x,multiscaleloss);return loss,{'photo':float(photo.detach()),'smooth':float(smooth.detach())}
elif a.task=='reconstruction':
 root=P/'external/sources/downstream/event_cnn_minimal';sys.path[:0]=[str(root),str(root/'PerceptualSimilarity')]
 import skimage.measure
 from skimage.metrics import structural_similarity
 skimage.measure.compare_ssim=structural_similarity
 try:import IPython
 except ImportError:
  stub=types.ModuleType('IPython');stub.embed=lambda *x,**y:None;sys.modules['IPython']=stub
 from model.model import E2VIDRecurrent
 from model.loss import perceptual_loss,temporal_consistency_loss
 from e2vid_coordinate_compat import patch_temporal_sampling
 from utils import loss as warp_module
 coordinate_check=patch_temporal_sampling(warp_module,root)
 c=json.loads((S/'training_recipe.json').read_text())['reconstruction'];model=E2VIDRecurrent(c['model_args']).cuda();percept=perceptual_loss(weight=1,net='vgg',use_gpu=True)
 for q in percept.model.parameters():q.requires_grad_(False)
 optimizer=torch.optim.Adam(model.parameters(),lr=.0001,amsgrad=True)
 def forward_loss():
  model.reset_states();tc=temporal_consistency_loss(weight=2,L0=2);terms=[];tc_calls=0
  for i in range(3):
   left,right=ts[4+i],ts[5+i];q=raw[(raw[:,0]>left)&(raw[:,0]<=right)];v=np.zeros((5,112,112),np.float32)
   if len(q):
    t=(q[:,0]-q[0,0])*4/max(q[-1,0]-q[0,0],1e-9);lo=t.astype(int);fraction=t-lo;xx=q[:,1].astype(int);yy=q[:,2].astype(int);valid=(xx<112)&(yy<112)
    for bins,weight in [(lo,1-fraction),(lo+1,fraction)]:
     keep=valid&(bins>=0)&(bins<5);np.add.at(v,(bins[keep],yy[keep],xx[keep]),q[keep,3]*weight[keep])
   x=torch.from_numpy(v[None]).cuda();image=torch.from_numpy(frames[5+i:6+i,None,:112,:112].astype(np.float32)/255).cuda();prediction=model(x)['image'];terms.append(percept(prediction,image,normalize=True));term=tc(i,image,prediction,torch.zeros((1,2,112,112),device='cuda'))
   if term is not None:terms.append(term);tc_calls+=1
  return sum(terms),{'temporal_calls':tc_calls,'source':'three adjacent new-core pose windows; runtime-only cross-task diagnostic, not benchmark'}
else:
 root=P/'external/sources/downstream/SingleShotPose';sys.path.insert(0,str(root));from darknet import Darknet
 from run_pose_stage2_pilot import encode_event_window
 image=encode_event_window(raw[:,0],raw[:,1].astype(int),raw[:,2].astype(int),(raw[:,3]>0).astype(np.uint8),ts[4],ts[5],346,260,8.)
 x=torch.from_numpy(image[None]).cuda();target=torch.from_numpy(uv.astype(np.float32)[None]).cuda();model=Darknet(str(root/'cfg/yolo-pose.cfg')).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
 def forward_loss():
  raw=model(x);B,_,H,W=raw.shape;cx=(target[:,0,0]*W).long();cy=(target[:,0,1]*H).long();b=torch.arange(B,device='cuda');pred=raw[:,:18].reshape(B,9,2,H,W);pred=torch.cat([pred[:,:1].sigmoid(),pred[:,1:]],1).permute(0,3,4,1,2)[b,cy,cx];wanted=target*torch.tensor([W,H],device='cuda')-torch.stack([cx,cy],1)[:,None];coord=(pred-wanted).square().sum()/2/B;ce=torch.nn.functional.cross_entropy(raw[:,18].flatten(1),cy*W+cx);return coord+ce,{'coordinate_loss':float(coord.detach()),'center_ce':float(ce.detach())}
before=hashed(model);updates=[];model.train()
for i in range(2):
 optimizer.zero_grad(set_to_none=True);loss,detail=forward_loss();assert torch.isfinite(loss);loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads);optimizer.step();updates.append({'step':i+1,'loss':float(loss.detach()),'finite_gradient_tensors':len(grads),**detail})
after=hashed(model);assert before!=after
result={'passed':True,'task':a.task,'diagnostic_only':True,'formal_optimizer_updates':0,'diagnostic_optimizer_updates':2,'source':'new-core pose train_000, never real data','initial_state_sha256':before,'updated_state_sha256':after,'updates':updates,'max_allocated_bytes':torch.cuda.max_memory_allocated(),'visible_gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'real_data_used':False}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
