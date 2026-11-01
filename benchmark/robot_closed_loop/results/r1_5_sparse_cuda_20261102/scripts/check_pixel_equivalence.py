"""Fixed accepted checkpoint; compare generated-input UV and inference devices."""
from pathlib import Path
import sys,json,hashlib,time,argparse,importlib.util,copy
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
import numpy as np,torch
parser=argparse.ArgumentParser();parser.add_argument('--replay',type=Path,required=True)
parser.add_argument('--model-source',type=Path,default=ROOT.parent/'sparse_repair_20261101/results/source/benchmark/robot_closed_loop/event_state_model.py')
parser.add_argument('--checkpoint',type=Path,default=ROOT.parent/'sparse_repair_20261101/results/fit/spatial_event_state.pt')
parser.add_argument('--input',type=Path,default=ROOT/'inputs/sparse_endpoint_blocks.npz')
args=parser.parse_args()
source=args.model_source
spec=importlib.util.spec_from_file_location('sparse_acceptance_state_model',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
checkpoint=args.checkpoint
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert digest=='dda51f2d5a9665295eeeeccd98682183946f8b080ceb610f82c2baeec3f8d896'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,_,_,_=module.load_checkpoint(str(checkpoint),'cpu');gpu=copy.deepcopy(model).to('cuda:0').eval()
report={'checkpoint_sha256':digest,'input_contract':module.INPUT_CONTRACT,
 'scope':'fixed checkpoint, local CUDA sparse-input/inference equivalence at 16 queries; not full R2 or formal held-out accuracy',
 'results':[],'GPU':torch.cuda.get_device_name(0)}
started=time.perf_counter()
with np.load(args.input,allow_pickle=False) as blocks:
 for bi in range(4):
  key=f'block_{bi:02d}/';ts=blocks[key+'timestamp'];W,H=map(int,blocks[key+'resolution'])
  cpu_events=np.load(args.replay/f'block_{bi:02d}_numpy_cpu_events.npz')['events']
  cuda_events=np.load(args.replay/f'block_{bi:02d}_torch_cuda_0_events.npz')['events']
  for cutoff in np.arange(float(ts[0])+.025,float(ts[-1])+1e-9,.005):
   cpu_voxel=module.five_bin_tensor(cpu_events,float(cutoff),W,H)
   cuda_voxel=module.five_bin_tensor(cuda_events,float(cutoff),W,H)
   a=torch.as_tensor(np.log1p(cpu_voxel).reshape(1,10,64,64),dtype=torch.float32)
   b=torch.as_tensor(np.log1p(cuda_voxel).reshape(1,10,64,64),dtype=torch.float32)
   with torch.inference_mode():
    logits_cpu=model(a)[0];logits_from_cuda_events=model(b)[0];logits_cuda=gpu(b.to('cuda:0'))[0]
    uv_cpu=module._softargmax_uv(logits_cpu,(W,H)).cpu().numpy()[0]
    uv_cuda_events=module._softargmax_uv(logits_from_cuda_events,(W,H)).cpu().numpy()[0]
    uv_cuda_infer=module._softargmax_uv(logits_cuda,(W,H)).cpu().numpy()[0]
   row=dict(block=bi,cutoff_s=float(cutoff),voxel_different_elements=int(np.count_nonzero(cpu_voxel!=cuda_voxel)),
    uv_CPU=uv_cpu.tolist(),uv_from_CUDA_events=uv_cuda_events.tolist(),uv_CUDA_inference=uv_cuda_infer.tolist(),
    pixel_change_from_generation=float(np.linalg.norm(uv_cpu-uv_cuda_events)),
    pixel_change_from_inference=float(np.linalg.norm(uv_cuda_events-uv_cuda_infer)))
   report['results'].append(row)
report.update(queries=len(report['results']),
 max_pixel_change_from_generation=max(r['pixel_change_from_generation'] for r in report['results']),
 max_pixel_change_from_inference=max(r['pixel_change_from_inference'] for r in report['results']),
 wall_s=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
report['status']='PASS' if report['max_pixel_change_from_generation']<.05 and report['max_pixel_change_from_inference']<.01 else 'FAIL'
(args.replay/'pixel_equivalence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
assert report['status']=='PASS'
