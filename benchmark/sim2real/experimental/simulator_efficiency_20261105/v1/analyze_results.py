"""Summarize all timings and output differences; never hide failed variants."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
suite=json.loads((root/'paired_suite/progress.json').read_text())['results']
report=dict(protocol=dict(cases=5,repeats_per_variant=5,warmup_per_measured_trial=1,alternating_order=True,precision='original FP64 geometry/state',samples_per_10ms=377,includes='public capture, interpolation, sensor, CPU event serialization',excludes='Genesis rendering, initial attach, physical HDF5 read, result file writing'),cases=[])
for case in ['s0_00','s1_00','s2_00','s3_00','train_000']:
    rows={v:[r for r in suite if r['case']==case and r['variant']==v] for v in ['baseline','fused_early']}
    assert all(len(rs)==5 for rs in rows.values()),case
    stats={}
    for v,rs in rows.items():
        walls=[r['row']['wall_s'] for r in rs]
        stats[v]=dict(wall_s=walls,median_wall_s=float(np.median(walls)),min_wall_s=min(walls),max_wall_s=max(walls),
                      median_dvs_s=float(np.median([r['row']['phases']['dvs_wall_s'] for r in rs])),
                      median_geometry_s=float(np.median([r['row']['phases']['geometry_wall_s'] for r in rs])),
                      peak_allocated_bytes=max(r['row']['peak_allocated_bytes'] for r in rs),fallbacks=sum(r['row']['fallbacks'] for r in rs))
    comparisons=[]
    for rep in range(5):
        a=next(r for r in rows['baseline'] if r['repeat']==rep)
        b=next(r for r in rows['fused_early'] if r['repeat']==rep)
        left=np.load(root/a['path']/'trial0.npz');right=np.load(root/b['path']/'trial0.npz')
        ev0=left['events'];ev1=right['events'];same_shape=ev0.shape==ev1.shape
        event=dict(count_baseline=len(ev0),count_candidate=len(ev1),same_shape=same_shape,
                   bitwise_equal=bool(np.array_equal(ev0,ev1)),addresses_order_equal=bool(same_shape and np.array_equal(ev0[:,1:],ev1[:,1:])),
                   time_max_abs_s=float(np.abs(ev0[:,0]-ev1[:,0]).max(initial=0)) if same_shape else None)
        errors={k:float(np.abs(left[k].astype(float)-right[k].astype(float)).max(initial=0)) for k in left.files if k!='events' and left[k].shape==right[k].shape}
        comparisons.append(dict(repeat=rep,events=event,state_max_abs=errors,masks_equal=a['row']['mask_sha256']==b['row']['mask_sha256'],sampling_equal=a['row']['sampling']==b['row']['sampling'],frames_bitwise_equal=a['row']['frame_sha256']==b['row']['frame_sha256']))
    # Quantify the frozen implementation's own repeat variability too.
    ref=np.load(root/rows['baseline'][0]['path']/'trial0.npz');variability=[]
    for r in rows['baseline'][1:]:
        cur=np.load(root/r['path']/'trial0.npz')
        variability.append(dict(events_bitwise_equal=bool(np.array_equal(ref['events'],cur['events'])),
                                lp_state_max_abs=float(np.abs(ref['state_lp_log_frame']-cur['state_lp_log_frame']).max())))
    baseline=stats['baseline']['median_wall_s'];candidate=stats['fused_early']['median_wall_s']
    report['cases'].append(dict(case=case,stats=stats,speedup=baseline/candidate,time_reduction_fraction=1-candidate/baseline,comparisons=comparisons,reference_repeat_variability=variability))
report['aggregate']=dict(geometric_mean_speedup=float(np.exp(np.log([r['speedup'] for r in report['cases']]).mean())),
                         total_median_baseline_s=sum(r['stats']['baseline']['median_wall_s'] for r in report['cases']),
                         total_median_candidate_s=sum(r['stats']['fused_early']['median_wall_s'] for r in report['cases']))
report['aggregate']['weighted_speedup']=report['aggregate']['total_median_baseline_s']/report['aggregate']['total_median_candidate_s']
report['aggregate']['all_masks_and_sampling_equal']=all(c['masks_equal'] and c['sampling_equal'] for r in report['cases'] for c in r['comparisons'])
report['aggregate']['all_event_addresses_order_equal']=all(c['events']['addresses_order_equal'] for r in report['cases'] for c in r['comparisons'])
(root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'aggregate':report['aggregate'],'cases':[{k:r[k] for k in ['case','speedup','time_reduction_fraction']} for r in report['cases']]},indent=2),flush=True)
