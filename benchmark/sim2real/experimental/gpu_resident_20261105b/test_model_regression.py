"""Fixed checkpoint regression on short windows and continuous synthetic replay."""
import hashlib,importlib.util,json
from pathlib import Path
import h5py,numpy as np,torch
root=Path(__file__).resolve().parent;P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261103-stage9-direct-v1'
repair=P/'runs/robot_r1_5_sparse_time_repair_20261101a';source=repair/'source/benchmark/robot_closed_loop/event_state_model.py';checkpoint=repair/'fit/spatial_event_state.pt'
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();assert digest=='dda51f2d5a9665295eeeeccd98682183946f8b080ceb610f82c2baeec3f8d896'
spec=importlib.util.spec_from_file_location('resident_fixed_model',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(2);model,_,_,_=module.load_checkpoint(str(checkpoint),'cpu');model.eval();results=[]
def check(events0,events1,cutoff,W,H,case,condition,repeat):
    a=module.five_bin_tensor(events0,cutoff,W,H);b=module.five_bin_tensor(events1,cutoff,W,H)
    with torch.inference_mode():
        left=model(torch.tensor(np.log1p(a).reshape(1,10,64,64)))[0];right=model(torch.tensor(np.log1p(b).reshape(1,10,64,64)))[0]
        uv0=module._softargmax_uv(left,(W,H)).numpy()[0];uv1=module._softargmax_uv(right,(W,H)).numpy()[0]
    results.append(dict(case=case,condition=condition,repeat=repeat,cutoff_s=float(cutoff),voxel_different_elements=int(np.count_nonzero(a!=b)),pixel_change=float(np.linalg.norm(uv0-uv1))))
suite=json.loads((root/'paired_suite_v2/progress.json').read_text())['results']
for row in [r for r in suite if r['condition']=='baseline']:
    case=row['case'];kind='pose' if case.startswith('train') else 'rich';index=10 if kind=='pose' else 100
    with h5py.File(R/(kind+'_bank/physical')/(case+'.h5'),'r') as h:
        g=h['episodes'][case];t0,t1=g['frame_timestamps_s'][index:index+2];H,W=g['depth_m'].shape[1:]
    events0=np.load(root/row['path']/'trial0.npz')['events']
    for condition in ['prior_fused','resident_events','resident_full']:
        other=next(r for r in suite if r['case']==case and r['repeat']==row['repeat'] and r['condition']==condition)
        events1=np.load(root/other['path']/'trial0.npz')['events']
        for cutoff in [float((t0+t1)/2),float(t1)]:check(events0,events1,cutoff,W,H,case,condition,row['repeat'])
long0=np.load(root/'long_baseline/trial0.npz')['events'];long1=np.load(root/'long_resident_exact/trial0.npz')['events']
with h5py.File(R/'rich_bank/physical/s3_00.h5','r') as h:
    g=h['episodes/s3_00'];start=float(g['frame_timestamps_s'][100]);end=float(g['frame_timestamps_s'][200]);H,W=g['depth_m'].shape[1:]
for cutoff in np.arange(start+.025,end+1e-9,.005):check(long0,long1,float(cutoff),W,H,'s3_00_long','resident_events',0)
report=dict(passed=all(r['voxel_different_elements']==0 and r['pixel_change']==0 for r in results),checkpoint_sha256=digest,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            scope='exact event voxel and fixed localization regression; no GT, training or accuracy claim',queries=len(results),results=results,max_pixel_change=max(r['pixel_change'] for r in results),real_data_used=False)
(root/'model_regression.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True);assert report['passed']
