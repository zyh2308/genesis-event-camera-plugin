"""All candidate comparisons, including original repeat variability."""
import csv,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
suite=json.loads((root/'paired_suite/progress.json').read_text())['results']
variants=sorted({r['variant'] for r in suite})
report=dict(cases=[],real_data_used=False,variants=variants)
csv_rows=[]
for case in ['s0_00','s1_00','s2_00','s3_00','train_000']:
    rows={v:[r for r in suite if r['case']==case and r['variant']==v] for v in variants}
    assert all(len(rs)==3 for rs in rows.values()),case
    stats={}
    for variant,rs in rows.items():
        walls=[r['row']['wall_s'] for r in rs]
        stats[variant]=dict(wall_s=walls,median_wall_s=float(np.median(walls)),min_wall_s=min(walls),max_wall_s=max(walls),
                            median_dvs_s=float(np.median([r['row']['phases']['dvs_wall_s'] for r in rs])),
                            median_geometry_s=float(np.median([r['row']['phases']['geometry_wall_s'] for r in rs])),
                            peak_allocated_bytes=max(r['row']['peak_allocated_bytes'] for r in rs),fallbacks=sum(r['row']['fallbacks'] for r in rs))
    comparisons={}
    for variant in variants:
        if variant=='baseline':continue
        result=[]
        for rep in range(3):
            a=next(r for r in rows['baseline'] if r['repeat']==rep)
            b=next(r for r in rows[variant] if r['repeat']==rep)
            with np.load(root/a['path']/'trial0.npz') as left,np.load(root/b['path']/'trial0.npz') as right:
                ev0=left['events'];ev1=right['events'];same_shape=ev0.shape==ev1.shape
                events=dict(count_baseline=len(ev0),count_candidate=len(ev1),same_shape=same_shape,bitwise_equal=bool(np.array_equal(ev0,ev1)),
                            addresses_order_equal=bool(same_shape and np.array_equal(ev0[:,1:],ev1[:,1:])),time_max_abs_s=float(np.abs(ev0[:,0]-ev1[:,0]).max(initial=0)) if same_shape else None)
                errors={k:float(np.abs(left[k].astype(float)-right[k].astype(float)).max(initial=0)) for k in left.files if k!='events' and left[k].shape==right[k].shape}
            result.append(dict(repeat=rep,events=events,state_max_abs=errors,masks_equal=a['row']['mask_sha256']==b['row']['mask_sha256'],sampling_equal=a['row']['sampling']==b['row']['sampling'],frames_bitwise_equal=a['row']['frame_sha256']==b['row']['frame_sha256']))
        comparisons[variant]=result
        stats[variant]['speedup']=stats['baseline']['median_wall_s']/stats[variant]['median_wall_s']
        stats[variant]['time_reduction_fraction']=1-1/stats[variant]['speedup']
    reference_repeat=[]
    with np.load(root/rows['baseline'][0]['path']/'trial0.npz') as ref:
        for r in rows['baseline'][1:]:
            with np.load(root/r['path']/'trial0.npz') as other:
                reference_repeat.append(dict(events_bitwise_equal=bool(np.array_equal(ref['events'],other['events'])),lp_state_max_abs=float(np.abs(ref['state_lp_log_frame']-other['state_lp_log_frame']).max())))
    report['cases'].append(dict(case=case,stats=stats,comparisons=comparisons,reference_repeat=reference_repeat))
    for variant in variants:
        s=stats[variant];csv_rows.append(dict(case=case,variant=variant,median_wall_s=s['median_wall_s'],min_wall_s=s['min_wall_s'],max_wall_s=s['max_wall_s'],speedup=s.get('speedup',1),dvs_s=s['median_dvs_s'],geometry_s=s['median_geometry_s'],peak_GiB=s['peak_allocated_bytes']/2**30))
report['aggregate']={}
for variant in variants:
    if variant=='baseline':continue
    base=sum(r['stats']['baseline']['median_wall_s'] for r in report['cases']);alt=sum(r['stats'][variant]['median_wall_s'] for r in report['cases'])
    checks=[c for r in report['cases'] for c in r['comparisons'][variant]]
    report['aggregate'][variant]=dict(weighted_speedup=base/alt,time_reduction_fraction=1-alt/base,geometric_mean_speedup=float(np.exp(np.log([r['stats'][variant]['speedup'] for r in report['cases']]).mean())),
                                      all_masks_and_sampling_equal=all(c['masks_equal'] and c['sampling_equal'] for c in checks),all_event_addresses_order_equal=all(c['events']['addresses_order_equal'] for c in checks),
                                      all_events_bitwise_equal=all(c['events']['bitwise_equal'] for c in checks),max_event_time_difference_s=max((c['events']['time_max_abs_s'] or 0) for c in checks),
                                      max_state_difference=max(err for c in checks for err in c['state_max_abs'].values()))
(root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
with (root/'timings.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
print(json.dumps(dict(aggregate=report['aggregate'],timings=csv_rows),indent=2),flush=True)
