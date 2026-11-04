"""Paired times, transfer calls and all raw event/state comparisons."""
import csv,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
def analyze(name):
    suite=json.loads((root/name/'progress.json').read_text())['results'];report=dict(cases=[],real_data_used=False);table=[]
    for case in ['s0_00','s1_00','s2_00','s3_00','train_000']:
        groups={c:[r for r in suite if r['case']==case and r['condition']==c] for c in ['baseline','prior_fused','resident_events','resident_full']}
        assert all(len(rs)==3 for rs in groups.values()),case
        stats={};comparisons={}
        for condition,rows in groups.items():
            times=[r['row']['wall_s'] for r in rows]
            stats[condition]=dict(wall_s=times,median_s=float(np.median(times)),min_s=min(times),max_s=max(times),peak_bytes=max(r['row']['peak_allocated_bytes'] for r in rows),
                                  geometry_s=float(np.median([r['row']['phases']['geometry_wall_s'] for r in rows])),sensor_s=float(np.median([r['row']['phases']['dvs_wall_s'] for r in rows])),
                                  transfers=[r['row']['transfers'] for r in rows],hole_pixels=[r['row'].get('hole_pixels') for r in rows],fallbacks=sum(r['row']['fallbacks'] for r in rows))
            if condition=='baseline':continue
            values=[]
            for a,b in zip(groups['baseline'],rows):
                assert a['repeat']==b['repeat']
                with np.load(root/a['path']/'trial0.npz') as left,np.load(root/b['path']/'trial0.npz') as right:
                    e0=left['events'];e1=right['events'];shape=e0.shape==e1.shape
                    values.append(dict(repeat=a['repeat'],count_baseline=len(e0),count_candidate=len(e1),events_bitwise_equal=bool(np.array_equal(e0,e1)),
                                       addresses_order_equal=bool(shape and np.array_equal(e0[:,1:],e1[:,1:])),time_max_abs_s=float(np.abs(e0[:,0]-e1[:,0]).max(initial=0)) if shape else None,
                                       state_max_abs={k:float(np.abs(left[k].astype(float)-right[k].astype(float)).max(initial=0)) for k in left.files if k.startswith('state_')},
                                       masks_equal=a['row']['mask_sha256']==b['row']['mask_sha256'],sampling_equal=a['row']['sampling']==b['row']['sampling']))
            comparisons[condition]=values
        for condition,value in stats.items():
            value['speedup_vs_baseline']=stats['baseline']['median_s']/value['median_s'];value['speedup_vs_prior']=stats['prior_fused']['median_s']/value['median_s']
            transfer=value['transfers'][0]
            table.append(dict(case=case,condition=condition,median_s=value['median_s'],min_s=value['min_s'],max_s=value['max_s'],speedup=value['speedup_vs_baseline'],speedup_vs_prior=value['speedup_vs_prior'],peak_GiB=value['peak_bytes']/2**30,
                              tensor_cpu_calls=transfer.get('tensor_cpu_calls',0),tensor_cpu_logical_bytes=transfer.get('tensor_cpu_logical_bytes',0),scalar_item_calls=transfer.get('scalar_item_calls',0),tensor_bool_calls=transfer.get('tensor_bool_calls',0)))
        report['cases'].append(dict(case=case,stats=stats,comparisons=comparisons))
    report['aggregate']={}
    for condition in ['prior_fused','resident_events','resident_full']:
        base=sum(r['stats']['baseline']['median_s'] for r in report['cases']);prior=sum(r['stats']['prior_fused']['median_s'] for r in report['cases']);alt=sum(r['stats'][condition]['median_s'] for r in report['cases'])
        checks=[c for r in report['cases'] for c in r['comparisons'][condition]]
        report['aggregate'][condition]=dict(weighted_speedup=base/alt,time_reduction_fraction=1-alt/base,speedup_vs_prior=prior/alt,
                                           all_events_bitwise_equal=all(c['events_bitwise_equal'] for c in checks),all_event_addresses_order_equal=all(c['addresses_order_equal'] for c in checks),
                                           all_masks_and_sampling_equal=all(c['masks_equal'] and c['sampling_equal'] for c in checks),max_time_error_s=max(c['time_max_abs_s'] or 0 for c in checks),max_state_error=max(error for c in checks for error in c['state_max_abs'].values()))
    (root/name/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    with (root/name/'timings.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(name,json.dumps(report['aggregate']),flush=True)
for name in ['paired_suite','paired_suite_v2']:analyze(name)
