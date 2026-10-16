"""Independent read-only evidence checks; writes only the review output directory."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np

ROOT = Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real')
BASE = ROOT / 'benchmark/sim2real'
EVIDENCE = BASE / 'evidence/S0/20261016-stage1-pilot'
SAMPLES = EVIDENCE / 'samples/data/sim2real/stage0'
OUT = Path(__file__).resolve().parent

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()

report = {'commit':git('rev-parse','HEAD'), 'working_tree':git('status','--porcelain'),
          'core_code_diff':git('diff','--name-only','e79e656ac00f7847b98db9e12bf43b093f3afff1','HEAD','--','genesis_event_plugin'),
          'integration_diff':git('diff','--name-only','e79e656ac00f7847b98db9e12bf43b093f3afff1','HEAD','--','integrations')}
index=json.loads((EVIDENCE/'artifact_index.json').read_text())['artifacts']
bad=[]
for item in index:
    p=ROOT/item['path']
    if not p.is_file() or p.stat().st_size!=item['size_bytes'] or digest(p)!=item['sha256']:
        bad.append(item['path'])
report['artifact_index']={'files_checked':len(index),'mismatches':bad}

sources=json.loads((BASE/'manifests/downstream_source_bundle.json').read_text())['repositories']
source_results=[]
for repo in sources:
    root=Path(repo['checkout_path_wsl'])
    bad=[i['path'] for i in repo['files'] if not (root/i['path']).is_file() or digest(root/i['path'])!=i['sha256']]
    source_results.append({'repo':repo['repo'],'files':len(repo['files']),'mismatches':bad})
report['local_external_sources']=source_results

fixture=SAMPLES/'shared-trajectory-20261016-v1'
episodes=[]
with h5py.File(fixture/'shared_trajectory.h5','r') as source:
    report['shared_h5_sha256']=digest(fixture/'shared_trajectory.h5')
    for episode_id in ['train_00','validation_00']:
        ep=source['episodes/'+episode_id]
        times=ep['frame_timestamps_s'][:]
        events=ep['eventgenesis_events_txyp'][:]
        v2e_path=fixture/'v2e_output_eps005'/f'{episode_id}.h5'
        with h5py.File(v2e_path,'r') as f:
            v=f['events'][:].astype(np.float64)
        v[:,0]*=1e-6
        v[:,3]=np.where(v[:,3]>0,1,-1)
        split='train' if episode_id.startswith('train') else 'validation'
        e={'id':episode_id,'time_start_end':times[[0,-1]].tolist(),
           'radiance_shape':list(ep['radiance_linear_rgb'].shape),
           'radiance_min_max':[float(ep['radiance_linear_rgb'][:].min()),float(ep['radiance_linear_rgb'][:].max())],
           'flow_shape':list(ep['flow_forward_px_per_interval'].shape),
           'intrinsics_saved':any('intrinsic' in k.lower() for k in ep.keys()),
           'eventgenesis_count':len(events),'v2e_count':len(v),
           'events_sorted':bool(np.all(np.diff(events[:,0])>=0) and np.all(np.diff(v[:,0])>=0))}
        a=np.load(SAMPLES/'steflow_inputs_20261016-v1/eventgenesis'/f'{split}.npz')
        b=np.load(SAMPLES/'steflow_inputs_20261016-v1/v2e'/f'{split}.npz')
        e['STE_shared_arrays_equal']={k:bool(np.array_equal(a[k],b[k])) for k in ['gray_prev','gray_next','flow_gt','flow_valid','source_resolution_hw','pad_yx']}
        e['STE_event_shapes']=[list(a['events'].shape),list(b['events'].shape)]
        e['STE_counts_conserved']=[int(a['interval_event_counts'].sum())==len(events),int(b['interval_event_counts'].sum())==len(v)]
        e['STE_zero_flow_baseline_EPE_interval9']=float(np.linalg.norm(a['flow_gt'][9],axis=0)[a['flow_valid'][9].astype(bool)].mean())
        checks=[]
        for generator in ['eventgenesis','v2e']:
            with h5py.File(SAMPLES/'e2vid_loader_20261016-v1'/generator/f'{split}.h5','r') as f:
                ts=f['events/ts'][:]
                previous=0; errors=[]; image_errors=[]
                for j in range(10):
                    frame=f[f'images/image{j:09d}']; end=int(frame.attrs['event_idx'])
                    dt=float(ts[end-1]-ts[previous]) if end-previous>=3 else 0.
                    restored=f[f'flow/flow{j:09d}'][:]*dt
                    errors.append(float(np.max(np.abs(restored-ep['flow_forward_px_per_interval'][j]))))
                    image_errors.append(int(np.max(np.abs(frame[:].astype(int)-ep['v2e_display_luma_u8'][j+1].astype(int)))))
                    previous=end
                checks.append({'generator':generator,'max_restored_flow_error':max(errors),'max_image_error':max(image_errors),'event_count':len(ts)})
        e['E2VID_loader_label_checks']=checks
        episodes.append(e)
report['episodes']=episodes

# Execute only the actual upstream temporal-loss class with a counting stub.
loss_path=Path(sources[0]['checkout_path_wsl'])/'model/loss.py'
tree=ast.parse(loss_path.read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='temporal_consistency_loss')
calls=[]
def counting_loss(*args,**kwargs):
    calls.append(1)
    return 1.0
namespace={'loss':SimpleNamespace(temporal_consistency_loss=counting_loss)}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(loss_path),'exec'),namespace)
cls=namespace['temporal_consistency_loss']
activity={}
for length in [2,3]:
    calls.clear(); loss=cls(weight=2.,L0=2)
    values=[loss(i,0.,0.,0.) for i in range(length)]
    activity[str(length)]={'active_calls':len(calls),'values':values}
report['upstream_temporal_loss_activity']=activity

# Check the fallback correlation numerically and in its backward pass on CPU.
try:
    import torch
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('review_ste',BASE/'scripts/run_steflow_stage0_smoke.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    torch.manual_seed(23)
    a=torch.randn(1,3,4,5,dtype=torch.float64,requires_grad=True)
    b=torch.randn(1,3,4,5,dtype=torch.float64,requires_grad=True)
    actual=mod.pure_torch_spatial_correlation(a,b)
    pad=torch.nn.functional.pad(b,(4,4,4,4))
    reference=torch.stack([torch.stack([(a*pad[:,:,y:y+4,x:x+5]).sum(1) for x in range(9)],1) for y in range(9)],1)
    g=torch.randn_like(actual)
    ag=torch.autograd.grad((actual*g).sum(),(a,b),retain_graph=True)
    rg=torch.autograd.grad((reference*g).sum(),(a,b))
    report['correlation_CPU_reference']={'forward_max_error':float((actual-reference).abs().max().detach()),
       'gradient_max_errors':[float((x-y).abs().max()) for x,y in zip(ag,rg)],
       'native_CUDA_extension_compared':False}
except Exception as exc:
    report['correlation_CPU_reference']={'error':str(exc)}

OUT.mkdir(exist_ok=True,parents=True)
(OUT/'local_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False))
