"""Read-only g1 evidence verification; CPU only, no training or CUDA calls."""
from pathlib import Path
import hashlib, json, socket, subprocess, sys, gc
import h5py, numpy as np, torch
torch.set_num_threads(2)
ROOT=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real')
BASE=ROOT/'benchmark/sim2real'
EV=BASE/'evidence/S2/20261016-stage2-pilot'
ext=json.loads((EV/'external_artifacts.json').read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def git(*a): return subprocess.check_output(['git','-C',str(ROOT),*a],text=True).strip()
out={'host':socket.gethostname(),'GPU_used':False,'head':git('rev-parse','HEAD'),'status':git('status','--porcelain'),'core_diff':git('diff','--name-only','e79e656','HEAD','--','genesis_event_plugin')}
out['data']=[]
for row in ext['server_data_artifacts']:
    p=Path(row['server_path']); out['data'].append({'path':str(p),'size_match':p.stat().st_size==row['size_bytes'],'sha_match':digest(p)==row['sha256']})
out['checkpoints']=[]
for row in ext['server_checkpoints_not_in_git']:
    p=Path(row['server_path']); d=torch.load(p,map_location='cpu',weights_only=True); model=d.get('model',d.get('state_dict'))
    h=hashlib.sha256()
    for k,v in sorted(model.items()): h.update(k.encode()); h.update(v.contiguous().numpy().tobytes())
    out['checkpoints'].append({'path':str(p),'size_match':p.stat().st_size==row['size_bytes'],'sha_match':digest(p)==row['sha256'],
        'parameters_finite':all(bool(torch.isfinite(v).all()) for v in model.values()),'model_state_sha256':h.hexdigest(),
        'optimizer_state_count':len(d['optimizer']['state']),'optimizer_steps':sorted({float(s['step']) for s in d['optimizer']['state'].values() if 'step' in s}),
        'updates':d.get('updates'),'seed':d.get('seed'),'epochs':d.get('epochs')})
    del d,model; gc.collect()
out['exit_codes']={Path(p).name:{f.name:f.read_text().strip() for f in Path(p).glob('*exit_code.txt')} for p in ext['server_run_directories']}

data=Path(ext['server_data_artifacts'][0]['server_path']).parent.parent
out['paired_data']=[]
with h5py.File(data/'shared/shared_trajectory.h5','r') as shared:
    out['rollouts']={}
    for eid in sorted(shared['episodes']):
        e=shared['episodes'][eid]; times=e['frame_timestamps_s'][:]
        out['rollouts'][eid]={'split':str(e.attrs['split']),'frame_count':len(times),'dt_min':float(np.diff(times).min()),'dt_max':float(np.diff(times).max()),'K':e['camera_intrinsics_K_px'][:].tolist()}
        split=str(e.attrs['split'])
        with h5py.File(data/f'e2vid_inputs_v2/eventgenesis/{split}/{eid}.h5','r') as a,h5py.File(data/f'e2vid_inputs_v2/v2e/{split}/{eid}.h5','r') as b:
            image_equal=all(np.array_equal(a['images'][k][:],b['images'][k][:]) for k in a['images'])
            flow_equal=all(np.array_equal(a['flow'][k][:],b['flow'][k][:]) for k in a['flow'])
            valid_equal=all(np.array_equal(a['flow_valid'][k][:],b['flow_valid'][k][:]) for k in a['flow_valid'])
            target_equal=all(np.array_equal(a['images'][f'image{i:09d}'][:],e['v2e_display_luma_u8'][i+1]) for i in range(len(times)-1))
        with np.load(data/f'steflow_inputs_v2/eventgenesis/{split}/{eid}.npz') as a,np.load(data/f'steflow_inputs_v2/v2e/{split}/{eid}.npz') as b:
            keys=['gray_frames','flow_gt_dt1','flow_valid_dt1','flow_gt_dt4','flow_valid_dt4','endpoint_timestamps_s','pad_yx']
            equal={k:np.array_equal(a[k],b[k]) for k in keys}
            zero={}
            for dt in [1,4]:
                gt=a[f'flow_gt_dt{dt}']; mask=a[f'flow_valid_dt{dt}'].astype(bool)
                err=np.sqrt(np.sum(gt**2,axis=1)); zero[f'dt{dt}']=float(np.mean([x[m].mean() for x,m in zip(err,mask)]))
        out['paired_data'].append({'episode':eid,'e2vid_images_equal':image_equal,'e2vid_flow_equal':flow_equal,'e2vid_valid_equal':valid_equal,'images_match_shared_targets':target_equal,'STE_shared_arrays_equal':equal,'zero_flow_EPE':zero})

sources=Path('/home/zhaoyuhan/Eventbased_WAM/external/sources/downstream')
out['source_heads']={p.name:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip() for p in sources.iterdir() if (p/'.git').exists()}
out['STE_return_source']={}
for p in (sources/'STE-FlowNet/models').rglob('*.py'):
    if 'use_dt4' in p.read_text(): out['STE_return_source'][str(p)]=p.read_text()[-8000:]
print(json.dumps(out,ensure_ascii=False))
