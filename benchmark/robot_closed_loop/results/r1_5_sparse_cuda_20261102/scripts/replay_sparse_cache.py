"""Replay immutable 100 Hz Genesis endpoint blocks through both backends."""
from pathlib import Path
import sys,json,time,hashlib,argparse
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'));sys.path.insert(0,str(ROOT/'core'))
sys.path.insert(0,str(ROOT.parent/'sparse_repair_20261101'))
import numpy as np,torch,h5py
from check_sparse import response
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin

parser=argparse.ArgumentParser()
parser.add_argument('--devices',default='cpu')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--input',type=Path)
args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2)
base=Path('/home/zhaoyuhan/Eventbased_WAM')
cache=base/'repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018a/trajectory_cache'
report={'scope':'immutable 100 Hz Genesis endpoints; original sparse warp and public capture; no dense observations',
 'results':[],'torch':torch.__version__,'devices':args.devices.split(',')}
started=time.perf_counter()
def dump():
 (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
def canon(e):return e[np.lexsort((e[:,0],e[:,3],e[:,2],e[:,1]))] if len(e) else e
def load_blocks():
 if args.input:
  with np.load(args.input,allow_pickle=False) as z:
   return [{k[len(prefix):]:z[k] for k in z.files if k.startswith(prefix)}
    for prefix in sorted({k.split('/')[0]+'/' for k in z.files})]
 blocks=[]
 for ep in [4,5]:
  path=cache/f'episode_{ep:03d}/trajectory_cache.h5'
  with h5py.File(path,'r') as h:
   ts=h['timestamps_s'][:]
   for start,end in [(.36,.40),(.55,.59)]:
    ix=np.where((ts>=start-1e-8)&(ts<=end+1e-8))[0]
    assert len(ix)==5 and np.allclose(np.diff(ts[ix]),.01,atol=1e-8)
    blocks.append(dict(episode=np.array(ep),start=np.array(start),timestamp=ts[ix],
      radiance=h['radiance'][ix],depth=h['depth'][ix],seg=h['segmentation'][ix],
      camera=h['camera_to_world_cv'][ix],poses=h['object_poses'][ix],ids=h['segmentation_pose_ids'][:],
      resolution=np.array(h.attrs['camera_resolution_wh']),vfov=np.array(h.attrs['vertical_fov_deg']),
      source_sha=np.array(hashlib.sha256(path.read_bytes()).hexdigest())))
 payload={f'block_{i:02d}/{k}':v for i,b in enumerate(blocks) for k,v in b.items()}
 np.savez_compressed(out/'sparse_endpoint_blocks.npz',**payload)
 return blocks
try:
 for bi,block in enumerate(load_blocks()):
  W,H=map(int,block['resolution']);f=H/(2*np.tan(np.deg2rad(float(block['vfov']))/2))
  cam=SimpleNamespace(res=(W,H),intrinsics=np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]]))
  ref=None;refmasks=None
  for backend,device in [('numpy','cpu')]+[('torch',d) for d in args.devices.split(',')]:
   tstart=time.perf_counter();current=[0];scene=SimpleNamespace(cur_t=float(block['timestamp'][0]));reads=[]
   def frame(s,c):
    k=current[0];reads.append(k);return dict(radiance=block['radiance'][k],depth=block['depth'][k],seg=block['seg'][k])
   def motion(s,c):
    k=current[0];return dict(camera_to_world=block['camera'][k],object_to_world={int(v):block['poses'][k,j] for j,v in enumerate(block['ids'])})
   p=GenesisPhysicsEventPlugin(noise_config=response(),device=device,frame_provider=frame,motion_state_provider=motion,
      warp_composite='coverage_time_log_linear',warp_backend=backend,disocclusion_policy='hold',strict_fidelity=False)
   p.attach(scene,cam);events=[];masks=[];samples=[]
   for k,t in enumerate(block['timestamp']):
    current[0]=k;scene.cur_t=float(t);e=p.capture()
    if len(e):events.append(e)
    if k:
     masks.append(p.interpolator.last_visibility['intermediate_unknown'].copy())
     samples.append(int(p.interpolator.last_sampling.samples))
   e=canon(np.concatenate(events) if events else np.empty((0,4)))
   assert reads==list(range(5)) and p.intermediate_state_provider is None
   name=f'block_{bi:02d}_{backend}_{device.replace(":","_")}'
   np.savez_compressed(out/(name+'_events.npz'),events=e)
   row=dict(block=bi,episode=int(block['episode']),start=float(block['start']),backend=backend,device=device,
       endpoint_reads=len(reads),events=len(e),samples=samples,wall_s=time.perf_counter()-tstart)
   if ref is None:ref=e;refmasks=masks
   row['support_mask_mismatch']=sum(int(np.count_nonzero(a!=b)) for a,b in zip(refmasks,masks))
   row['same_event_count']=len(ref)==len(e)
   row['same_event_addresses_polarities']=len(ref)==len(e) and np.array_equal(ref[:,1:],e[:,1:])
   row['max_event_time_difference_s']=float(np.max(np.abs(ref[:,0]-e[:,0]))) if len(ref)==len(e) and len(e) else None
   report['results'].append(row);dump();print(json.dumps(row),flush=True)
   assert row['support_mask_mismatch']==0,row
   assert row['same_event_addresses_polarities'],row
   assert row['max_event_time_difference_s'] is None or row['max_event_time_difference_s']<=1e-6,row
 report['status']='PASS'
except Exception as exc:
 import traceback
 report['status']='FAIL';report['failure']=repr(exc);report['traceback']=traceback.format_exc();raise
finally:
 report['wall_s']=time.perf_counter()-started;dump()
