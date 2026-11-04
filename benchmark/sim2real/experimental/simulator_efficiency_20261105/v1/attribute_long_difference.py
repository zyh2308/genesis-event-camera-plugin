"""Keep the failed strict gate; independently establish the source and impact."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import h5py,numpy as np,torch
root=Path(__file__).resolve().parent;v2=root/'final_v2';P=Path('/home/zhaoyuhan/Eventbased_WAM')
initial=json.loads((root/'final_audit.json').read_text());oracle=json.loads((v2/'actual_input_long/results.json').read_text())
arrays={name:np.load(v2/name/'trial0.npz') for name in ['long_baseline','long_fused','long_baseline_repeat']}
def comparison(left,right):
    a=left['events'];b=right['events'];same=a.shape==b.shape
    return dict(shape_equal=same,count_baseline=len(a),count_other=len(b),event_bitwise_equal=bool(np.array_equal(a,b)),
                addresses_order_equal=bool(same and np.array_equal(a[:,1:],b[:,1:])),
                changed_timestamps=int(np.count_nonzero(a[:,0]!=b[:,0])) if same else None,
                max_time_error_s=float(np.abs(a[:,0]-b[:,0]).max(initial=0)) if same else None,
                state_errors={k:float(np.abs(left[k].astype(float)-right[k].astype(float)).max(initial=0)) for k in left.files if k!='events'})
repeat=comparison(arrays['long_baseline'],arrays['long_baseline_repeat']);candidate=comparison(arrays['long_baseline'],arrays['long_fused'])
repair=P/'runs/robot_r1_5_sparse_time_repair_20261101a';source=repair/'source/benchmark/robot_closed_loop/event_state_model.py';checkpoint=repair/'fit/spatial_event_state.pt'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='dda51f2d5a9665295eeeeccd98682183946f8b080ceb610f82c2baeec3f8d896'
spec=importlib.util.spec_from_file_location('long_input_regression',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(2);model,_,_,_=module.load_checkpoint(str(checkpoint),'cpu');model.eval()
with h5py.File(P/'runs/sim2real/20261103-stage9-direct-v1/rich_bank/physical/s3_00.h5','r') as h:
    g=h['episodes/s3_00'];H,W=g['depth_m'].shape[1:];start=float(g['frame_timestamps_s'][100]);end=float(g['frame_timestamps_s'][120])
queries=[]
for cutoff in np.arange(start+.025,end+1e-9,.005):
    voxels={key:module.five_bin_tensor(a['events'],float(cutoff),W,H) for key,a in arrays.items()};uvs={}
    with torch.inference_mode():
        for key,voxel in voxels.items():
            x=torch.tensor(np.log1p(voxel).reshape(1,10,64,64));prediction=model(x)[0];uvs[key]=module._softargmax_uv(prediction,(W,H)).numpy()[0]
    queries.append(dict(cutoff_s=float(cutoff),candidate_voxel_different=int(np.count_nonzero(voxels['long_baseline']!=voxels['long_fused'])),
                        reference_repeat_voxel_different=int(np.count_nonzero(voxels['long_baseline']!=voxels['long_baseline_repeat'])),
                        candidate_pixel_change=float(np.linalg.norm(uvs['long_baseline']-uvs['long_fused'])),reference_repeat_pixel_change=float(np.linalg.norm(uvs['long_baseline']-uvs['long_baseline_repeat']))))
checks=dict(original_strict_gate_preserved_failed=not initial['passed'],matched_input_long_bitwise=oracle['passed'] and all(r['event_bitwise_equal'] and max(r['state_errors'].values())==0 for r in oracle['cases']),
            long_addresses_order_rng_equal=candidate['addresses_order_equal'] and candidate['state_errors']['state_rng_state']==0,
            original_reference_has_repeat_variability=repeat['max_time_error_s']>0 or any(repeat['state_errors'].values()),
            long_candidate_voxels_exact=all(q['candidate_voxel_different']==0 for q in queries),long_candidate_fixed_model_exact=all(q['candidate_pixel_change']==0 for q in queries),
            all_other_initial_behavioral_checks=all(v for k,v in initial['checks'].items() if k not in ['long_trace_events_rng','completed_jobs_exit_zero']))
report=dict(passed=all(checks.values()),scope='sensor arithmetic and downstream observable equivalence; original strict end-to-end bitwise gate remains FAILED',strict_end_to_end_bitwise_passed=False,
            checks=checks,candidate=candidate,reference_repeat=repeat,same_input_oracle=oracle,queries=queries,real_data_used=False)
(root/'attribution_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['queries','same_input_oracle']},indent=2),flush=True)
assert report['passed'],checks
