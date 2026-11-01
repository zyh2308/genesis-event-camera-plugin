"""Actual CUDA checks of the sparse endpoint reconstruction; no dense renders."""
from pathlib import Path
import sys, os, json, time, hashlib, copy, argparse
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
sys.path.insert(0,str(ROOT/'core'))
sys.path.insert(0,str(ROOT.parent/'sparse_repair_20261101'))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from types import SimpleNamespace
from check_sparse import response
from genesis_event_plugin.physics_warp import depth_aware_bidirectional_warp_projected_diagnostics as numpy_warp
from genesis_event_plugin.torch_physics import depth_aware_bidirectional_warp_projected_batch_torch as torch_warp
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator,RenderRequiredError
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig

parser=argparse.ArgumentParser()
parser.add_argument('--devices',default='cpu,cuda:0')
parser.add_argument('--output',type=Path,default=ROOT/'results')
args=parser.parse_args()
DEVICES=args.devices.split(',')
CUDA_REQUESTED=any(d.startswith('cuda') for d in DEVICES)
OUT=args.output; OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
report={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'python':sys.executable,'torch':torch.__version__,'CUDA_available':torch.cuda.is_available(),
 'devices_requested':DEVICES,
 'tests':[],'scope':'sparse endpoint-only reconstruction controls; not R2 physics'}
def save():
 (OUT/'cuda_validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
def record(name,**kw):
 row={'name':name,**kw};report['tests'].append(row);save();print(json.dumps(row),flush=True)
def arrays(t):return t.detach().cpu().numpy()
def assert_image(reference,other,tol=3e-6):
 assert np.array_equal(np.isfinite(reference),np.isfinite(other)),'finite mask mismatch'
 finite=np.isfinite(reference);err=float(np.max(np.abs(reference[finite]-other[finite]))) if finite.any() else 0.
 assert err<=tol,('radiance error',err,tol)
 return err
def canonical(e):
 e=np.asarray(e)
 return e[np.lexsort((e[:,0],e[:,3],e[:,2],e[:,1]))] if len(e) else e

def kernel_cases():
 rng=np.random.default_rng(606);B=2;H=9;W=13
 yy,xx=np.mgrid[:H,:W];uv=np.stack([xx,yy]).astype(np.float32)
 alphas=np.array([0.,1e-12,.1,.25,.5,.75,.9,1.-1e-12,1.],np.float64)
 T=len(alphas)
 for rgb in [False,True]:
  shape=(B,H,W,3) if rgb else (B,H,W)
  a=rng.uniform(.01,2.,shape).astype(np.float32);b=(a*np.exp(.7)).astype(np.float32)
  depth=np.ones((B,H,W),np.float32);seg=np.zeros((B,H,W),np.int64)
  for case in ['identity','subpixel','occlusion','missing','threshold_coverage']:
   u0=np.broadcast_to(uv,(B,T,2,H,W)).copy();u1=u0.copy()
   z0=np.ones((B,T,H,W),np.float32);z1=z0.copy();v0=np.ones_like(z0,bool);v1=v0.copy()
   if case=='subpixel':
    u0[:,:,0]+=.27;u1[:,:,0]-=.31
   if case=='occlusion':
    u0[:,:,0,:,W//2:]-=.99999;z0[:,:,:,W//2:]=.3;seg[:,:,W//2:]=1
   if case=='missing':v0[:,:,:,2:5]=False;v1[:,:,:,4:7]=False
   if case=='threshold_coverage':u0[:,:,0]+=.005;u1[:,:,0]+=.005
   refs=[];masks=[]
   for bi in range(B):
    rs=[];ms=[]
    for ti,alpha in enumerate(alphas):
     out,mask,diag=numpy_warp(a[bi],depth[bi],u0[bi,ti],z0[bi,ti],v0[bi,ti],
       b[bi],depth[bi],u1[bi,ti],z1[bi,ti],v1[bi,ti],composite='coverage_time_log_linear',
       alpha=float(alpha),surface_id0=seg[bi],surface_id1=seg[bi])
     rs.append(out);ms.append(mask)
    refs.append(rs);masks.append(ms)
   ref=np.asarray(refs);maskref=np.asarray(masks)
   for device in DEVICES:
    args=[torch.as_tensor(x,device=device) for x in [a,depth,u0,z0,v0,b,depth,u1,z1,v1]]
    out,mask,diag=torch_warp(*args,composite='coverage_time_log_linear',return_diagnostics=True,
       alpha=torch.as_tensor(alphas,dtype=torch.float64,device=device),
       surface_id0=torch.as_tensor(seg,device=device),surface_id1=torch.as_tensor(seg,device=device))
    err=assert_image(ref,arrays(out));mismatch=int(np.count_nonzero(maskref!=arrays(mask)))
    assert mismatch==0,('support mismatch',case,device,mismatch)
    assert np.array_equal(arrays(out)[:,0],a) and np.array_equal(arrays(out)[:,-1],b)
    record('kernel',case=case,rgb=rgb,device=device,max_radiance_error=err,support_mismatch=mismatch)

def scene_data(h=18,w=26):
 yy,xx=np.mgrid[:h,:w];K=np.array([[40,0,w/2],[0,40,h/2],[0,0,1.]],np.float64)
 s=np.zeros((h,w),np.int32);s[5:13,8:15]=1
 d=np.ones((h,w),np.float32)*1.4;d[s==1]=.9
 a=(.15+.2*(xx%3==0)+.4*(s==1)).astype(np.float32)
 b=(a*np.exp(.45)).astype(np.float32)
 p0=np.eye(4);p1=np.eye(4);p1[0,3]=.006;p1[1,3]=-.002
 return K,a,b,d,s,{1:p0},{1:p1}

def public_interpolation():
 K,a,b,d,s,p0,p1=scene_data()
 kwargs=dict(frame0=a,depth0=d,seg0=s,frame1=b,depth1=d,seg1=s,
  camera_to_world0=np.eye(4),camera_to_world1=np.eye(4),object_world_delta=None,t0=0.,t1=.01,
  object_world0=p0,object_world1=p1)
 config=AdaptiveSamplingConfig(min_samples=7,max_samples=7,strict=False)
 reference=PhysicsInterpolator(K,composite='coverage_time_log_linear',warp_backend='numpy',sampler_config=config)
 images,ts=reference.interpolate(**kwargs); images=np.asarray(images)
 for device in DEVICES:
  for tensor in [False,True]:
   interp=PhysicsInterpolator(K,composite='coverage_time_log_linear',warp_backend='torch',device=device,sampler_config=config)
   out,times=interp.interpolate(**kwargs,return_torch=tensor)
   out=arrays(out)[0] if tensor else np.asarray(out)
   err=assert_image(images,out)
   mismatch=int(np.count_nonzero(reference.last_visibility['intermediate_unknown']!=interp.last_visibility['intermediate_unknown']))
   assert times==ts and mismatch==0 and np.isfinite(out).all()
   record('public_interpolation',device=device,return_torch=tensor,max_radiance_error=err,support_mismatch=mismatch)
 # Strict behavior must reject unobserved detail, on CPU and GPU alike.
 for backend,device in [('numpy','cpu')]+[('torch',d) for d in DEVICES]:
  interp=PhysicsInterpolator(K,composite='coverage_time_log_linear',warp_backend=backend,device=device,
    sampler_config=config,strict_fidelity=True)
  try:interp.interpolate(**kwargs)
  except RenderRequiredError:record('strict_rejects_missing_observations',device=device,passed=True)
  else:raise AssertionError('strict accepted endpoint-only photometry')

def capture_controls():
 K,a,b,d,s,p0,p1=scene_data(h=18,w=26)
 # Run actual public capture for moving object and photometric ramp together.
 h,w=d.shape;cam=SimpleNamespace(res=(w,h),intrinsics=K)
 outputs={}
 for backend,device in [('numpy','cpu')]+[('torch',d) for d in DEVICES]:
  scene=SimpleNamespace(cur_t=0.);calls=[]
  def frame(scene,cam):
   calls.append(scene.cur_t);return dict(radiance=a if scene.cur_t==0 else b,depth=d,seg=s)
  def motion(scene,cam):return dict(camera_to_world=np.eye(4),object_to_world=p0 if scene.cur_t==0 else p1)
  plugin=GenesisPhysicsEventPlugin(noise_config=response(),device=device,frame_provider=frame,motion_state_provider=motion,
     warp_composite='coverage_time_log_linear',warp_backend=backend,strict_fidelity=False)
  plugin.attach(scene,cam);assert len(plugin.capture())==0
  scene.cur_t=.01;events=plugin.capture();events=canonical(events)
  assert calls==[0.,.01] and plugin.intermediate_state_provider is None
  key=backend+'_'+device;outputs[key]=events
  if backend=='numpy':ref=events
  assert len(events)==len(ref),('event_count',key,len(events),len(ref))
  assert np.array_equal(ref[:,1:],events[:,1:]),('event address/polarity mismatch',key)
  err=float(np.max(np.abs(ref[:,0]-events[:,0]))) if len(events) else 0.
  assert err<2e-7,(key,err)
  np.savez_compressed(OUT/(key.replace(':','_')+'_events.npz'),events=events)
  record('public_capture',backend=backend,device=device,events=len(events),event_timestamp_max_error_s=err,
    endpoint_reads=len(calls),extra_renderer_calls=0)
 # Analytic uniform log ramp: exact crossing times; unchanged static capture.
 a=np.ones((h,w),np.float32);b=np.full_like(a,np.exp(.9));s=np.zeros_like(s);p0=p1={}
 for backend,device in [('numpy','cpu')]+[('torch',d) for d in DEVICES]:
  scene=SimpleNamespace(cur_t=0.);calls=[]
  plugin=GenesisPhysicsEventPlugin(noise_config=response(),device=device,frame_provider=frame,motion_state_provider=motion,
    warp_composite='coverage_time_log_linear',warp_backend=backend,strict_fidelity=False)
  plugin.attach(scene,cam);plugin.capture();scene.cur_t=.01;ev=plugin.capture()
  expected=np.array([.2,.4,.6,.8])/.9*.01
  times=np.unique(np.round(ev[:,0],10));assert len(ev)==h*w*4 and len(times)==4
  err=float(np.max(np.abs(times-expected)));assert err<2e-8
  plugin.start_episode();plugin.capture();scene.cur_t=.02;assert len(plugin.capture())==0
  record('analytic_ramp_and_static',device=device,events=len(ev),timestamp_max_error_s=err)

if __name__=='__main__':
 started=time.perf_counter();save()
 try:
  if CUDA_REQUESTED:
   assert torch.cuda.is_available(),'actual CUDA required'
   report['GPU']=torch.cuda.get_device_name(0);report['CUDA_runtime']=torch.version.cuda
   torch.cuda.reset_peak_memory_stats()
  kernel_cases();public_interpolation();capture_controls()
  report['status']='PASS'
 except Exception as exc:
  import traceback
  report['status']='FAIL';report['failure']=repr(exc);report['traceback']=traceback.format_exc();save();raise
 finally:
  report['wall_s']=time.perf_counter()-started
  if CUDA_REQUESTED and torch.cuda.is_available():report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
  report['core_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'core/genesis_event_plugin').glob('*.py')}
  save()
