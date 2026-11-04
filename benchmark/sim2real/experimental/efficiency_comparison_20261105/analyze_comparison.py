"""Predeclared scene-level analysis; no best-run cherry picking."""
import csv,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
name='matched_suite_static' if (ROOT/'matched_suite_static/summary.json').exists() else 'matched_suite'
report=json.loads((ROOT/name/'summary.json').read_text());rows=report['rows']
cases=['s0_00','s1_00','s2_00','s3_00','train_000'];result=[]
for case in cases:
    data={v:[x for x in rows if x['case']==case and x['variant']==v] for v in ['ours_resident','v2e_ssm_memory']}
    assert all(len(x)==3 for x in data.values())
    med={v:statistics.median(x['wall_s'] for x in values) for v,values in data.items()}
    out=dict(case=case,ours_s=med['ours_resident'],v2e_s=med['v2e_ssm_memory'],speedup=med['v2e_ssm_memory']/med['ours_resident'],simulated_s=data['ours_resident'][0]['simulated_s'])
    for v,label in [('ours_resident','ours'),('v2e_ssm_memory','v2e')]:
        values=data[v];out[label+'_min_s']=min(x['wall_s'] for x in values);out[label+'_max_s']=max(x['wall_s'] for x in values)
        out[label+'_peak_gib']=max(x['peak_bytes'] for x in values)/2**30
        out[label+'_event_counts']=[x['events']['count'] for x in values]
        out[label+'_rtf']=out['simulated_s']/med[v]
    result.append(out)
logs=np.log([x['speedup'] for x in result]);rng=np.random.default_rng(20261105)
boot=np.exp(logs[rng.integers(0,len(cases),size=(20000,len(cases)))].mean(axis=1));ci=np.quantile(boot,[.025,.975]).tolist()
gm=float(np.exp(logs.mean()));advantage=gm>=1.20 and ci[0]>1 and min(x['speedup'] for x in result)>=1/1.10
analysis=dict(cases=result,geometric_mean_speedup=gm,scene_bootstrap_95ci=ci,qualifies_practical_advantage=bool(advantage),
    bootstrap_scope='five fixed scene categories on one GPU/profile; not a dataset population or event-quality confidence interval',
    count=30,source_suite=name,decision='audit_then_qualified_freeze' if advantage else 'measure_bottleneck_then_one_optimization',
    baseline='official v2e + SuperSloMo with PNG-equivalent in-memory I/O and native API midpoint timestamps',
    events_per_second_is_not_primary=True,real_data_used=False)
(ROOT/(name+'_analysis.json')).write_text(json.dumps(analysis,indent=2)+'\n')
flat=[dict(case=x['case'],variant=x['variant'],trial=x['trial'],simulated_s=x['simulated_s'],wall_s=x['wall_s'],rtf=x['simulated_s']/x['wall_s'],peak_bytes=x['peak_bytes'],events=x['events']['count'],setup_s=x['setup_s'],serialize_s=x['serialize_s']) for x in rows]
with (ROOT/(name+'_timings.csv')).open('x',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
print(json.dumps(analysis,indent=2))
