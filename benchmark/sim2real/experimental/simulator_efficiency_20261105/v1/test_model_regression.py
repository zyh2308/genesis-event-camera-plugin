"""Fixed localization checkpoint/input regression; no labels or accuracy claim."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import h5py,numpy as np,torch
root=Path(__file__).resolve().parent
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261103-stage9-direct-v1'
repair=P/'runs/robot_r1_5_sparse_time_repair_20261101a'
source=repair/'source/benchmark/robot_closed_loop/event_state_model.py';checkpoint=repair/'fit/spatial_event_state.pt'
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert digest=='dda51f2d5a9665295eeeeccd98682183946f8b080ceb610f82c2baeec3f8d896'
spec=importlib.util.spec_from_file_location('fixed_localization_regression',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(2)
model,_,_,_=module.load_checkpoint(str(checkpoint),'cpu');model.eval()
suite=json.loads((root/'paired_suite/progress.json').read_text())['results'];results=[]
for a in [r for r in suite if r['variant']=='baseline']:
    b=next(r for r in suite if r['case']==a['case'] and r['repeat']==a['repeat'] and r['variant']=='fused_early')
    events0=np.load(root/a['path']/'trial0.npz')['events'];events1=np.load(root/b['path']/'trial0.npz')['events']
    kind='pose' if a['case'].startswith('train') else 'rich';index=10 if kind=='pose' else 100
    with h5py.File(R/(kind+'_bank/physical')/(a['case']+'.h5'),'r') as h:
        g=h['episodes'][a['case']];t0,t1=g['frame_timestamps_s'][index:index+2];H,W=g['depth_m'].shape[1:]
    for cutoff in [float((t0+t1)/2),float(t1)]:
        vox0=module.five_bin_tensor(events0,cutoff,W,H);vox1=module.five_bin_tensor(events1,cutoff,W,H)
        left=torch.tensor(np.log1p(vox0).reshape(1,10,64,64));right=torch.tensor(np.log1p(vox1).reshape(1,10,64,64))
        with torch.inference_mode():
            pred0=model(left)[0];pred1=model(right)[0]
            uv0=module._softargmax_uv(pred0,(W,H)).numpy()[0];uv1=module._softargmax_uv(pred1,(W,H)).numpy()[0]
        diff=float(np.linalg.norm(uv0-uv1))
        results.append(dict(case=a['case'],repeat=a['repeat'],cutoff_s=cutoff,voxel_different_elements=int(np.count_nonzero(vox0!=vox1)),pixel_change=diff))
report=dict(checkpoint_sha256=digest,model_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='fixed checkpoint regression on synthetic replay; 10ms events inside fixed 25ms voxel; no GT, no accuracy or Sim-to-Real claim',queries=len(results),results=results,max_pixel_change=max(r['pixel_change'] for r in results),real_data_used=False)
report['passed']=report['max_pixel_change']<.05
(root/'model_regression.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
assert report['passed']
