"""Pure-JSON science selection; no model/data loading, forward or optimizer."""
import argparse,hashlib,json,math,operator
from pathlib import Path
METHODS=('eventgenesis','v2e');SEEDS=(0,1,2);LIGHTS=('normal','half','eighth');RIDS=('validation_000','validation_031','validation_063','validation_095')
OPS={'<':operator.lt,'<=':operator.le,'>':operator.gt,'>=':operator.ge}
def need(x,m):
    if not x:raise ValueError(m)
def _cases(rows):
    need(isinstance(rows,list) and len(rows)==6,'Exactly six method/seed native-score records required');out={}
    expected_ids={f'{rid}/{k:03}' for rid in RIDS for k in range(4,20)}
    for r in rows:
        key=(r['method'],r['seed']);need(key not in out and key in {(m,s) for m in METHODS for s in SEEDS},'Duplicate or unregistered method/seed')
        need(set(r['conditions'])==set(LIGHTS),'All three matched64 lights required')
        for light,c in r['conditions'].items():
            need(set(c['by_rollout'])==set(RIDS),'Exactly original four rollout scores required')
            windows=c['window_evidence'];ids=[w['sample_id'] for w in windows]
            need(len(ids)==64 and set(ids)==expected_ids,'Complete64 inventory without duplicate/omitted windows required')
            need(all(w['finite_prediction'] is True and w['rid']==w['sample_id'].split('/')[0] for w in windows),'Every original window must have finite prediction; no filtering')
            for field in ('center_px','success'):
                vals=[c['by_rollout'][rid][field] for rid in RIDS];summary=c['summary'][field]
                need(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in [*vals,summary]),'Finite original metric values required')
                need(math.isclose(sum(vals)/4,summary,rel_tol=1e-10,abs_tol=1e-10),'Original equal-rollout aggregation differs')
                need(all(v>=0 if field=='center_px' else 0<=v<=1 for v in [*vals,summary]),'Metric domain invalid')
        out[key]=r
    need(set(out)=={(m,s) for m in METHODS for s in SEEDS},'Incomplete paired matrix');return out
def assess_advance(control,candidate):
    old,new=_cases(control),_cases(candidate);checks=[];negative=[]
    for method in METHODS:
        for seed in SEEDS:
            for light in LIGHTS:
                a,b=old[method,seed]['conditions'][light],new[method,seed]['conditions'][light]
                need(a['window_evidence']==b['window_evidence'],'Candidate changes ordered original window/finite inventory')
                for metric in ('center_px','success'):
                    op=('<' if metric=='center_px' else '>') if method=='eventgenesis' and light=='eighth' else ('<=' if metric=='center_px' else '>=')
                    x,y=a['summary'][metric],b['summary'][metric];checks.append(dict(method=method,seed=seed,condition=light,metric='summary.'+metric,operator=op,baseline=x,candidate=y,delta=y-x,passed=OPS[op](y,x)))
                for rid in RIDS:
                    dc=b['by_rollout'][rid]['center_px']-a['by_rollout'][rid]['center_px'];dp=b['by_rollout'][rid]['success']-a['by_rollout'][rid]['success']
                    if dc>0 or dp<0:negative.append(dict(method=method,seed=seed,condition=light,rid=rid,center_delta_px=dc,success_delta=dp,center_worsened=dc>0,pose_worsened=dp<0))
    need(len(checks)==36,'Wrong mechanism check count');passed=all(c['passed'] for c in checks)
    criterion=Path(__file__).with_name('ADVANCE_CRITERION.json')
    return dict(schema='pose_readout_mechanism_advance_result_v1',status='passed' if passed else 'failed',criterion_sha256=hashlib.sha256(criterion.read_bytes()).hexdigest(),checks=checks,passed_checks=sum(c['passed'] for c in checks),failed_checks=sum(not c['passed'] for c in checks),per_rollout_negative_changes=negative,advance_to_independent_synthetic_confirmation=passed,formal_30_and_8='not_evaluated_unchanged',long_training_authorized=False,real_scoring_authorized=False)
def assess_driver_outputs(scores,case_rows):
    """Thin adapter for frozen driver scores.json + paired_cases.jsonl."""
    need(isinstance(scores,list) and len(scores)==36 and len(case_rows)==1152,'Actual full36 scores and1152 paired windows required');cases={};table={}
    finite=lambda v:all(finite(x) for x in v) if isinstance(v,list) else isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
    for c in case_rows:
        k=(c['method'],c['seed'],c['condition'],c['sample_id']);need(k not in cases,'Duplicate paired window')
        need(c['rid']==c['sample_id'].split('/')[0] and c['finite_prediction_control'] is True and c['finite_prediction_candidate'] is True,'Failed/mismatched paired prediction')
        need(all(len(c[x])==2 and finite(c[x]) for x in ('control_uv','candidate_uv')) and all(len(c[x])==3 and all(len(r)==3 for r in c[x]) and finite(c[x]) for x in ('R_control','R_candidate')) and finite(c['z_m']) and finite(c['size']) and c['z_m']>0 and c['size']>0,'Nonfinite or invalid actual prediction payload');cases[k]=c
    for r in scores:
        k=(r['method'],r['seed'],r['condition'],r['arm']);need(k not in table and r['arm'] in ('control','candidate'),'Duplicate or unknown score arm');table[k]=r
        need(r['readout']==('native_global' if r['arm']=='control' else 'predicted_peak_softfocus') and r['windows']==64,'Score role/inventory changed')
    grouped={arm:[] for arm in ('control','candidate')};used=set()
    for arm in grouped:
        for method in METHODS:
            for seed in SEEDS:
                run=dict(method=method,seed=seed,conditions={})
                for light in LIGHTS:
                    r=table[method,seed,light,arm];evidence=[]
                    for sid in r['window_ids']:
                        c=cases[method,seed,light,sid];used.add((method,seed,light,sid));evidence.append(dict(sample_id=sid,rid=c['rid'],finite_prediction=True))
                    run['conditions'][light]=dict(summary=r['score']['summary'],by_rollout=r['score']['by_rollout'],window_evidence=evidence)
                grouped[arm].append(run)
    need(len(used)==len(cases)==1152,'Unknown/omitted paired cases');return assess_advance(grouped['control'],grouped['candidate'])
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('control','candidate','scores','cases'):p.add_argument('--'+k,type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();need(not a.out.exists(),'Unique output required')
    if a.scores is not None:need(a.cases is not None and a.control is a.candidate is None,'Choose one complete input schema');r=assess_driver_outputs(json.loads(a.scores.read_text()),[json.loads(x) for x in a.cases.read_text().splitlines()])
    else:need(a.control is not None and a.candidate is not None and a.cases is None,'Choose one complete input schema');r=assess_advance(json.loads(a.control.read_text()),json.loads(a.candidate.read_text()))
    with a.out.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
