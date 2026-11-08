"""Frozen C0/C1 predicted-factor crossing; zero inference/updates, GT scores only."""
import argparse,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists() and sha(a.predictions)==a.sha256;a.out.mkdir()
    data=[json.loads(line) for line in a.predictions.read_text().splitlines()];assert len(data)==3456
    lookup={(r['method'],r['seed'],r['arm'],r['condition'],r['sample_id']):r for r in data};assert len(lookup)==3456
    output=[];groups=defaultdict(list)
    for r in data:
        if r['arm']!='C1':continue
        old=lookup[(r['method'],r['seed'],'C0',r['condition'],r['sample_id'])]
        assert old['gt_z_m']==r['gt_z_m'] and old['K']==r['K']
        assert all(abs(math.exp(q['log_shape']+q['focal_log']-q['log_size'])-q['z_m'])<2e-6 for q in (old,r))
        variants={'C0':old['z_m'],'C1':r['z_m'],'C0_size_C1_shape':math.exp(r['log_shape']+r['focal_log']-old['log_size']),
                  'C1_size_C0_shape':math.exp(old['log_shape']+r['focal_log']-r['log_size'])}
        row={k:r[k] for k in ['method','seed','condition','rid','sample_id','gt_z_m','focal_log']}
        row.update(C0_log_size=old['log_size'],C0_log_shape=old['log_shape'],C1_log_size=r['log_size'],C1_log_shape=r['log_shape'],predicted_depth_m=variants)
        output.append(row)
        for arm,z in variants.items():groups[(r['method'],r['seed'],r['condition'],r['rid'],arm)].append((abs(z-r['gt_z_m']),z-r['gt_z_m']))
    assert len(output)==1152
    macro=[]
    for method in ['eventgenesis','v2e']:
        for condition in ['normal','half','eighth']:
            row=dict(method=method,condition=condition,arms={})
            for arm in ['C0','C1','C0_size_C1_shape','C1_size_C0_shape']:
                vals=[v for (m,s,c,rid,variant),v in groups.items() if (m,c,variant)==(method,condition,arm)]
                assert len(vals)==12 and all(len(v)==16 for v in vals)
                row['arms'][arm]=dict(depth_mae_cm=100*statistics.mean(statistics.mean(x[0] for x in v) for v in vals),depth_bias_cm=100*statistics.mean(statistics.mean(x[1] for x in v) for v in vals))
            macro.append(row)
    records=a.out/'factor_inputs_and_predictions.jsonl'
    records.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in output))
    summary=a.out/'three_seed_matched_summary.json';summary.write_text(json.dumps(macro,indent=2)+'\n')
    result=dict(schema='predicted_depth_factor_cross_v1',status='complete',diagnostic_only=True,optimizer_updates=0,new_model_forward=0,real_data_used=False,
        GT_used_only_for_error=True,uses_two_frozen_predicted_models=True,per_window_records=1152,source=dict(path=str(a.predictions),sha256=sha(a.predictions)),
        source_code_sha256=sha(__file__),artifacts=[dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size) for p in (records,summary)],
        limitation='Controlled output-factor crossing with compensating errors; not a weight-level causal attribution or accepted deployable method.')
    (a.out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print('DEPTH_FACTOR_CROSS_COMPLETE',1152)
if __name__=='__main__':main()
