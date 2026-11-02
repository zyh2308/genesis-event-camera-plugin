"""SSM/FILM/RIFE + one identical official ESIM backend on shared native frames."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import cv2,h5py,numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--method',choices=['vid2e','film_esim','rife_esim'],required=True);p.add_argument('--kind',choices=['rich','pose'],required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();R=a.campaign.resolve();bank=R/(a.kind+'_bank');torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
sys.path[:0]=[str(R/'runtime/esim_extension'),str(R/'baseline_sources/vid2e_ssm/esim_torch'),str(R/'sources')]
import esim_torch
from esim_interpolation_models import Interpolator
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pin=json.loads((R/'sources/baseline_sources.tar.manifest.json').read_text())
for repo in pin['repositories']:
 for row in repo['files']:assert sha(R/'baseline_sources'/repo['name']/row['path'])==row['sha256']
assert json.loads((R/'runtime/esim_extension/build_receipt.json').read_text())['passed']
def oracle():
 sensor=esim_torch.EventSimulator_torch(.2,.2,500000);sensor.forward(torch.zeros((2,3),device='cuda'),torch.tensor(0,device='cuda',dtype=torch.int64));q=sensor.forward(torch.full((2,3),.9,device='cuda'),torch.tensor(10000000,device='cuda',dtype=torch.int64));assert q is not None and len(q['t'])==24
 assert set(q['p'].cpu().numpy())=={1};t=np.sort(q['t'].cpu().numpy());expected=np.repeat(np.arange(1,5)*(.2/.9)*10000000,6);assert np.max(np.abs(t-expected))<10
 sensor=esim_torch.EventSimulator_torch(.2,.2,500000);sensor.forward(torch.zeros((2,3),device='cuda'),torch.tensor(0,device='cuda',dtype=torch.int64));negative=sensor.forward(torch.full((2,3),-.9,device='cuda'),torch.tensor(10000000,device='cuda',dtype=torch.int64));assert negative is not None and len(negative['t'])==24 and (negative['p']<=0).all()
 sensor=esim_torch.EventSimulator_torch(.2,.2,500000);assert sensor.forward(torch.zeros((2,3),device='cuda'),torch.tensor(0,device='cuda',dtype=torch.int64)) is None;z=sensor.forward(torch.zeros((2,3),device='cuda'),torch.tensor(10000000,device='cuda',dtype=torch.int64));assert z is None
 return {'passed':True,'positive_ramp_events':24,'negative_ramp_events':24,'max_time_error_ns':float(np.max(np.abs(t-expected))),'static_events':0,'threshold':.2,'refractory_ns':500000}
check=oracle();model=Interpolator(R,a.method);physical_hash=sha(bank/'bank_manifest.json');out=R/'preflight/baselines'/a.method if a.preflight_only else bank/'events'/a.method;out.mkdir(parents=True,exist_ok=True);files=sorted((bank/'physical').glob('*.h5'));files=files[:1] if a.preflight_only else files;rows=[]
for f in files:
 rid=f.stem;assert not (out/(rid+'.json')).exists(),'Completed raw events are immutable'
 with h5py.File(f,'r') as h:
  g=h['episodes'][rid];images=np.asarray(g['v2e_display_luma_u8']);times=np.asarray(g['frame_timestamps_s']);split=str(g.attrs['split'])
 if a.preflight_only:images=images[:3];times=times[:3]
 sensor=esim_torch.EventSimulator_torch(.2,.2,500000);chunks=[];samples=[];torch.cuda.synchronize();start=time.monotonic()
 def feed(image,t):
  if image.ndim==3:
   image=cv2.cvtColor(np.clip(image*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
  log=np.log(image.astype(np.float32)/255+1e-5);q=sensor.forward(torch.from_numpy(np.ascontiguousarray(log)).cuda(),torch.tensor(round(t*1e9),device='cuda',dtype=torch.int64))
  if q is not None:
   v=np.stack([q[k].cpu().numpy() for k in ['t','x','y','p']],axis=1).astype(np.float64);v[:,0]*=1e-9;v[:,3]=np.where(v[:,3]>0,1.,-1.);chunks.append(v)
 feed(images[0],float(times[0]))
 for i in range(len(times)-1):
  n=0;last=0.
  for fraction,frame in model.intermediates(images[i],images[i+1]):
   assert last<fraction<1 and np.isfinite(frame).all();last=fraction;feed(frame,float(times[i]+(times[i+1]-times[i])*fraction));n+=1
  feed(images[i+1],float(times[i+1]));samples.append(n)
  if (i+1)%20==0:print(json.dumps({'method':a.method,'rollout':rid,'interval':i+1,'total':len(times)-1,'elapsed_s':time.monotonic()-start}),flush=True)
 torch.cuda.synchronize();raw=np.concatenate(chunks) if chunks else np.empty((0,4),np.float64)
 assert np.isfinite(raw).all() and (len(raw)==0 or ((np.diff(raw[:,0])>=0).all() and raw[:,0].min()>=times[0]-1e-8 and raw[:,0].max()<=times[-1]+1e-8))
 H,W=images.shape[1:];assert len(raw)==0 or ((raw[:,1]>=0).all() and (raw[:,1]<W).all() and (raw[:,2]>=0).all() and (raw[:,2]<H).all())
 path=out/(rid+'.npy');np.save(path,raw);row={'rollout_id':rid,'split':split,'event_count':len(raw),'events_sha256':sha(path),'source_physical_manifest_sha256':physical_hash,'sources_manifest_sha256':sha(R/'sources/baseline_sources.tar.manifest.json'),'weights_manifest_sha256':sha(R/'sources/baseline_weights.tar.manifest.json'),'generator_entrypoint_sha256':sha(Path(__file__)),'interpolator_adapter_sha256':sha(R/'sources/esim_interpolation_models.py'),'esim_build_receipt_sha256':sha(R/'runtime/esim_extension/build_receipt.json'),'interpolation_policy':'author SSM max-flow adaptive' if a.method=='vid2e' else 'three recursive midpoint subdivisions (8x)','intermediate_samples_per_interval':samples,'wall_s':time.monotonic()-start,'visible_gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'sensor':{'backend':'official ESIM CUDA, API-only compatibility copy','positive_threshold':.2,'negative_threshold':.2,'refractory_s':.0005,'native_sensor_noise':'none; native pipeline comparison, not matched nonideal sensor ablation'},'real_data_used':False,'preflight_only':a.preflight_only};(out/(rid+'.json')).write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
 print(json.dumps({'complete':rid,'method':a.method,'events':len(raw),'wall_s':row['wall_s']}),flush=True)
if a.preflight_only:(out/'result.json').write_text(json.dumps({'passed':True,'method':a.method,'oracle':check,'row':rows[0],'formal_training_events':False},indent=2)+'\n')
else:(out/'manifest.json').write_text(json.dumps({'schema':'stage9_official_interpolator_esim_v1','generator':a.method,'physical_manifest_sha256':physical_hash,'rollouts':rows,'real_data_used':False},indent=2)+'\n')
