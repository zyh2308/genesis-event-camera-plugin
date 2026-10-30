"""Read-only audit of saved CUDA evidence, with local CPU rule tracing."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
R = Path('/home/科研/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1_5-core-candidate')
O = Path(__file__).resolve().parent
D = R / 'benchmark/robot_closed_loop/results/r1_5_core_cuda_parity_20261030c'
L = R / 'benchmark/robot_closed_loop/results/runtime_logs/r1_5_core_cuda_parity_20261030c'
sys.path.insert(0, str(R))
import numpy as np
from benchmark.robot_closed_loop.diagnostics.validate_r1_5_core_candidate import load_cached_episode, event_metrics
from genesis_event_plugin.physics_warp import project_endpoint_to_alpha, depth_aware_bidirectional_warp_projected_diagnostics

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    s = json.loads((D/'summary.json').read_text())
    checks = []
    for root in (D,L):
        for line in (root/'SHA256SUMS').read_text().splitlines():
            if not line.strip(): continue
            digest,rel = line.split('  ',1)
            checks.append({'file':str(root/rel),'ok':sha(root/rel)==digest})
    for name,h in s['source_module_sha256'].items():
        checks.append({'file':str(R/'genesis_event_plugin'/name),'ok':sha(R/'genesis_event_plugin'/name)==h})
    a = np.load(D/'cuda_parity_arrays.npz')
    threshold = [np.array_equal(a[f'threshold_{i}_cpu_events'],a[f'threshold_{i}_cuda_events']) for i in range(12)]
    report = {'scope':'Saved CUDA arrays independently checked; CPU component trace only. No new CUDA, renderer, training or remote job.',
              'hash_checks':checks,'threshold_raw_arrays_exact':threshold,'episodes':[]}
    rows=[]
    cache=R/'benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059'
    if not (cache/'episode_004/same_state_source_cache.h5').is_file():
        cache=Path('/home/科研/Eventbased_WAM/repositories/genesis-event-camera-plugin/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059')
    for ep in ('004','005'):
        cpu=a[f'ep{ep}_numpy_support'].astype(bool); gpu=a[f'ep{ep}_cuda_support'].astype(bool)
        xor=cpu^gpu
        c=a[f'ep{ep}_numpy_midpoint_luma']; g=a[f'ep{ep}_cuda_midpoint_luma']
        e0=a[f'ep{ep}_numpy_events']; e1=a[f'ep{ep}_cuda_events']
        em=event_metrics(e0,e1,320)
        src,dt,dl,rgb,seg,w,h,fov,cp,fp=load_cached_episode(cache,ep)
        old=next(x for x in s['episodes'] if x['episode']==ep)
        for p,key in ((cp,'state_cache'),(fp,'direct_hdr_evidence')):
            checks.append({'file':str(p),'ok':sha(p)==old['input_sha256'][key]})
        left=int(np.flatnonzero(np.isclose(src['timestamps_s'],.5805,atol=1e-9,rtol=0))[0])
        right=left+1
        f=h/(2*np.tan(np.deg2rad(fov)/2)); K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1]])
        obj0={int(sid):src['object_poses'][left,j] for j,sid in enumerate(src['segmentation_pose_ids'])}
        obj1={int(sid):src['object_poses'][right,j] for j,sid in enumerate(src['segmentation_pose_ids'])}
        T0,T1=src['camera_to_world_cv'][left],src['camera_to_world_cv'][right]
        assert np.array_equal(T0,T1), 'Trace assumes verified fixed camera'
        p0=project_endpoint_to_alpha(src['depth'][left],src['segmentation'][left],K,T0,T0,.5,0,obj0,obj1,None)
        p1=project_endpoint_to_alpha(src['depth'][right],src['segmentation'][right],K,T1,T0,.5,1,obj0,obj1,None)
        warped,valid,diag=depth_aware_bidirectional_warp_projected_diagnostics(
            src['radiance_luma'][left],src['depth'][left],*p0,
            src['radiance_luma'][right],src['depth'][right],*p1,
            composite='same_surface_log_linear',alpha=.5,
            surface_id0=src['segmentation'][left],surface_id1=src['segmentation'][right])
        reasons={
            'coverage0':~diag['support0'],'coverage1':~diag['support1'],
            'surface_consistency0':~diag['surface_consistent0'],'surface_consistency1':~diag['surface_consistent1'],
            'surface_id':diag['surface_id0']!=diag['surface_id1'],
            'depth_match':np.abs(diag['z_buffer0']-diag['z_buffer1'])>np.maximum(.001,.001*np.maximum(np.abs(diag['z_buffer0']),np.abs(diag['z_buffer1']))) }
        delta=np.abs(np.log(np.maximum(c,1e-6))-np.log(np.maximum(g,1e-6)))
        for y,x in np.argwhere(xor):
            rows.append({'episode':ep,'x':int(x),'y':int(y),'cpu_support':bool(cpu[y,x]),'cuda_support':bool(gpu[y,x]),
                         'cpu_coverage0':float(diag['coverage0'][y,x]),'cpu_coverage1':float(diag['coverage1'][y,x]),
                         'cpu_surface_id0':int(diag['surface_id0'][y,x]),'cpu_surface_id1':int(diag['surface_id1'][y,x]),
                         'cpu_reject_reasons':','.join(k for k,m in reasons.items() if m[y,x]),
                         'cpu_luma':float(c[y,x]),'cuda_luma':float(g[y,x]),'log_delta':float(delta[y,x])})
        at=xor&~cpu
        report['episodes'].append({'episode':ep,'xor_count':int(xor.sum()),'fraction':float(xor.mean()),
            'cpu_yes_cuda_no':int((cpu&~gpu).sum()),'cpu_no_cuda_yes':int((~cpu&gpu).sum()),
            'local_cpu_mask_vs_saved_cpu_mismatches':int((valid!=cpu).sum()),
            'local_cpu_reject_reasons_on_saved_xor_cpu_rejected':{k:int((m&at).sum()) for k,m in reasons.items()},
            'xor_log_delta_max':float(delta[xor].max()),'common_support_log_delta_max':float(delta[cpu&gpu].max()),
            'events':em,'raw_events_elementwise_equal':bool(np.array_equal(e0,e1))})
    report['exit_code']=int((L/'exit_code.txt').read_text().strip())
    report['result_manifest_sha256']=sha(D/'SHA256SUMS')
    (O/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (O/'mismatch_pixels_cpu_trace.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in report.items() if k!='hash_checks'},indent=2))
    print('Hash checks',len(checks),all(x['ok'] for x in checks))

if __name__=='__main__':main()
