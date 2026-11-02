"""Native-image adapters for official SSM, FILM and paper RIFE weights."""
import atexit,hashlib,importlib.util,json,math,os,pickle,select,struct,subprocess,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
class Interpolator:
 def __init__(self,R,method):
  self.R=R;self.method=method;self.assets=R/'baseline_assets';self.sources=R/'baseline_sources'
  weights=json.loads((R/'sources/baseline_weights.tar.manifest.json').read_text())
  for row in weights['files']:
   p=self.assets/row['path'];assert p.stat().st_size==row['size_bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
  if method=='vid2e':
   path=self.sources/'vid2e_ssm/upsampling/utils/model.py';spec=importlib.util.spec_from_file_location('stage9_official_vid2e_model',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);self.module=module
   self.FC=module.UNet(6,4).cuda().eval();self.AT=module.UNet(20,5).cuda().eval();ckpt=torch.load(self.assets/'vid2e/SuperSloMo.ckpt',map_location='cpu',weights_only=False);self.FC.load_state_dict(ckpt['state_dictFC']);self.AT.load_state_dict(ckpt['state_dictAT']);self.mean=torch.tensor([.429,.431,.397],device='cuda')[None,:,None,None];self.warps={}
  elif method=='rife_esim':
   sys.path.insert(0,str(self.sources/'rife'));from model.RIFE import Model
   self.model=Model();weights=torch.load(self.assets/'rife/train_log/flownet.pkl',map_location='cpu',weights_only=True);converted={k.removeprefix('module.'):v for k,v in weights.items()};self.model.flownet.load_state_dict(converted,strict=True);self.model.eval()
  else:
   assert method=='film_esim'
   log=(R/('film_worker_'+str(os.getpid())+'.log')).open('wb')
   child_env=dict(os.environ,PYTHONNOUSERSITE='1');child_env.pop('PYTHONPATH',None)
   libraries=sorted((R/'runtime/film/lib/python3.11/site-packages/nvidia').glob('*/lib'));child_env['LD_LIBRARY_PATH']=':'.join(str(p) for p in libraries)
   tools=R/'runtime/esim_build_tools';child_env['PATH']=str(tools/'bin')+os.pathsep+child_env['PATH'];child_env['XLA_FLAGS']='--xla_gpu_cuda_data_dir='+str(tools)
   self.child=subprocess.Popen([str(R/'runtime/film/bin/python'),'-u',str(R/'sources/film_inference_worker.py'),str(R)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env=child_env);log.close()
   atexit.register(self.close);ready=self._receive();assert ready['ready'] and ready['interpolation_factor']==8
 def _receive(self):
  assert select.select([self.child.stdout],[],[],180)[0],'FILM worker timed out'
  header=self.child.stdout.read(8);assert len(header)==8,('FILM worker exited',self.child.poll())
  size=struct.unpack('<Q',header)[0];assert size<128*1024*1024;raw=self.child.stdout.read(size);assert len(raw)==size;return pickle.loads(raw)
 def close(self):
  if hasattr(self,'child') and self.child.poll() is None:
   self.child.stdin.close()
   try:self.child.wait(timeout=15)
   except subprocess.TimeoutExpired:self.child.terminate();self.child.wait(timeout=15)
 def _tensor(self,image,multiple):
  H,W=image.shape;raw=torch.from_numpy(np.repeat(image[None,None].astype(np.float32)/255,3,axis=1)).cuda();return F.pad(raw,(0,(-W)%multiple,0,(-H)%multiple))
 @torch.no_grad()
 def intermediates(self,left,right):
  H,W=left.shape
  if self.method=='film_esim':
   payload=pickle.dumps((left.tobytes(),right.tobytes()),protocol=5);self.child.stdin.write(struct.pack('<Q',len(payload)));self.child.stdin.write(payload);self.child.stdin.flush();frames=np.frombuffer(self._receive(),dtype=np.float32).reshape(7,H,W,3)
   for i,frame in enumerate(frames,1):yield i/8,frame
   return
  if self.method=='vid2e':
   x0=self._tensor(left,32)-self.mean;x1=self._tensor(right,32)-self.mean;flow=self.FC(torch.cat([x0,x1],1));F01,F10=flow[:,:2],flow[:,2:]
   maximum=max(float(v[:,:,:H,:W].square().sum(1).sqrt().max()) for v in [F01,F10]);n=max(1,math.ceil(maximum));assert n<=512,('excessive interpolated displacement',maximum)
   warp=self.warps.get(tuple(x0.shape[-2:]))
   if warp is None:warp=self.module.backWarp(x0.shape[-1],x0.shape[-2],torch.device('cuda'));self.warps[tuple(x0.shape[-2:])]=warp
   for j in range(1,n):
    t=j/n;temp=-t*(1-t);Ft0=temp*F01+t*t*F10;Ft1=(1-t)**2*F01+temp*F10;g0=warp(x0,Ft0);g1=warp(x1,Ft1)
    z=self.AT(torch.cat([x0,x1,F01,F10,Ft1,Ft0,g1,g0],1));f0=z[:,:2]+Ft0;f1=z[:,2:4]+Ft1;v0=z[:,4:5].sigmoid();v1=1-v0
    image=((1-t)*v0*warp(x0,f0)+t*v1*warp(x1,f1))/((1-t)*v0+t*v1)+self.mean
    yield t,image[0,:,:H,:W].permute(1,2,0).cpu().numpy()
  else:
   assert self.method=='rife_esim'
   x0=self._tensor(left,32);x1=self._tensor(right,32)
   def mid(p,q):
    flow,mask,merged,*_=self.model.flownet(torch.cat([p,q],1),[4,2,1],timestep=.5);return merged[2],flow[2]
   midpoint,_=mid(x0,x1)
   def convert(x):return x[0,:,:H,:W].permute(1,2,0).cpu().numpy()
   # The published RIFE/FILM CLIs use recursive midpoint subdivisions.
   # Main comparison fixes three subdivisions (8x), without GT adaptivity.
   levels=3
   def recurse(p,q,m,lo,hi,depth):
    t=(lo+hi)/2
    if depth>1:
     l,_=mid(p,m);yield from recurse(p,m,l,lo,t,depth-1)
    yield t,convert(m)
    if depth>1:
     r,_=mid(m,q);yield from recurse(m,q,r,t,hi,depth-1)
   yield from recurse(x0,x1,midpoint,0.,1.,levels)
