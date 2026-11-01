import os,sys,json,time,subprocess
from pathlib import Path
import numpy as np,h5py,torch
R=Path(os.environ['EPOSE_REPAIR_RUN']);P=Path('/home/zhaoyuhan/Eventbased_WAM');sys.path.insert(0,str(P/'external/sources/v2e-cac6384445/v2e-pinned'))
from v2ecore.emulator import EventEmulator
torch.set_num_threads(1);root=R/'bank';out=root/'v2e_events';out.mkdir(exist_ok=True);rows=[]
if 'V2E_WORKER_INDEX' not in os.environ:
 children=[]
 for j in range(8):
  env=os.environ.copy();env.update(V2E_WORKER_INDEX=str(j),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  children.append(subprocess.Popen([sys.executable,__file__],env=env,stdout=(R/f'v2e_worker{j}.stdout').open('w'),stderr=(R/f'v2e_worker{j}.stderr').open('w')))
 codes=[p.wait() for p in children];assert codes==[0]*8,codes
 for j in range(8):rows.extend(json.loads((root/f'v2e_manifest_worker{j}.json').read_text()))
 assert len(rows)==32;(root/'v2e_manifest.json').write_text(json.dumps(sorted(rows,key=lambda r:r['rollout_id']),indent=2));print('V2E_BANK_COMPLETE',flush=True);sys.exit(0)
worker=int(os.environ['V2E_WORKER_INDEX'])
with h5py.File(root/'shared_trajectory.h5','r') as h:
 for index,name in enumerate(sorted(h['episodes'])):
  if index%8!=worker:continue
  start=time.time();g=h['episodes'][name];frames=np.asarray(g['v2e_display_luma_u8']);times=np.asarray(g['frame_timestamps_s'])
  e=EventEmulator(pos_thres=.2,neg_thres=.2,sigma_thres=.03,cutoff_hz=300.,leak_rate_hz=.01,leak_jitter_fraction=.1,noise_rate_cov_decades=.1,shot_noise_rate_hz=.001,refractory_period_s=.0005,seed=42,device='cpu',output_width=346,output_height=260)
  packets=[]
  e.generate_events(frames[0].astype(np.float32),float(times[0]))
  # Dense physical endpoints, common linear DN interpolation to the previously
  # registered 25us sensor cadence. 1ms filter updates were explicitly rejected
  # at preflight because the pinned emulator warned of large update epsilon.
  for k in range(1,len(frames)):
   a=frames[k-1].astype(np.float32);delta=frames[k].astype(np.float32)-a
   for j in range(1,41):
    im=a+(j/40)*delta;t=float(times[k-1]+j/40*(times[k]-times[k-1]));ev=e.generate_events(im,t)
    if ev is not None and len(ev):packets.append(np.asarray(ev,dtype=float))
  e.cleanup();v=np.concatenate(packets) if packets else np.zeros((0,4));order=np.argsort(v[:,0],kind='stable');v=v[order];a=np.empty(v.shape,dtype=np.uint32);a[:,0]=np.rint(v[:,0]*1e6).astype(np.uint32);a[:,1:3]=v[:,1:3].astype(np.uint32);a[:,3]=(v[:,3]>0).astype(np.uint32)
  assert len(a)>0 and a[:,0].max()<=120001 and (a[:,1]<346).all() and (a[:,2]<260).all()
  with h5py.File(out/f'{name}.h5','w') as f:f.create_dataset('events',data=a,compression='gzip');f.attrs['schema']='t_microseconds,x,y,p01'
  row={'rollout_id':name,'events':len(a),'wall_s':time.time()-start,'physical_input_cadence_hz':1000,'sensor_cadence_us':25,'interpolator':'linear DN between dense physical endpoints','slomo':False};rows.append(row);(root/f'v2e_manifest_worker{worker}.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
print('V2E_BANK_COMPLETE',flush=True)
