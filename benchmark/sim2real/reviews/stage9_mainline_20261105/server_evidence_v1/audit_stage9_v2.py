"""Read-only Stage9 run audit. Writes a new evidence snapshot only."""
import argparse, collections, csv, datetime, hashlib, json
from pathlib import Path

METHODS = ['eventgenesis', 'v2e', 'vid2e', 'film_esim', 'rife_esim']
BUDGET = {'reconstruction': 12000, 'flow': 8910, 'pose': 30000}

def load(p):
    return json.loads(p.read_text()) if p.is_file() else None

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def tail(p, size=64000):
    if not p.is_file(): return []
    with p.open('rb') as f:
        f.seek(max(0, p.stat().st_size-size)); data=f.read().decode('utf-8', errors='replace')
    rows=[]
    for line in data.splitlines():
        try: rows.append(json.loads(line))
        except (ValueError, TypeError): pass
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--hash-selected',action='store_true')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    observed=datetime.datetime.now(datetime.timezone.utc).isoformat()
    recipe=sha(a.campaign/'sources/training_recipe.json')
    frozen=load(a.campaign/'recovery_20261103/pose_frozen_checkpoints.json')
    cached={(x['method'],x['seed']):x for x in frozen['runs']} if frozen else {}
    rows=[];curves=[]
    for task,budget in BUDGET.items():
        for seed in range(3):
            for method in METHODS:
                p=a.campaign/'training'/task/f'seed{seed}'/method
                complete=load(p/'completion.json');live=load(p/'live_status.json') or {}
                metrics=tail(p/'training_metrics.jsonl');last=metrics[-1] if metrics else {}
                r=dict(task=task,method=method,seed=seed,budget=budget,completion_exists=bool(complete),
                       latest_update=live.get('optimizer_update',last.get('optimizer_update')),
                       live_status=live.get('status'),live_utc=live.get('updated_utc'),
                       latest_metric=last,completion_verified=False,selected_hash_verified=False)
                if complete:
                    assert complete['status']=='complete' and complete['completed_updates']==budget,(task,seed,method,'budget')
                    assert (complete['task'],complete['method'],complete['seed'])==(task,method,seed)
                    assert complete['recipe_sha256']==recipe
                    assert complete['real_data_used_in_training'] is False
                    cp=Path(complete['selected_checkpoint'])
                    assert cp==p/'synthetic_best.pt' and cp.is_file()
                    r.update(completion_verified=True,latest_update=budget,
                        initial_state_sha256=complete['initial_state_sha256'],selected_checkpoint=str(cp),
                        selected_checkpoint_sha256=complete['selected_checkpoint_sha256'],
                        completion_sha256=sha(p/'completion.json'),real_scoring_performed=complete['real_scoring_performed'])
                    if task=='pose' and (method,seed) in cached:
                        assert cached[method,seed]['selected_checkpoint_sha256']==complete['selected_checkpoint_sha256']
                        r['selected_hash_verified']=True;r['hash_evidence']='current full pose freeze receipt'
                    elif a.hash_selected:
                        assert sha(cp)==complete['selected_checkpoint_sha256']
                        r['selected_hash_verified']=True;r['hash_evidence']='independent actual-file SHA256'
                summary=load(p/'run_summary.json') or load(p/'run_metadata.json') or {}
                r['summary_keys']=list(summary)
                for name in ['synthetic_best_epe','synthetic_best_epoch','synthetic_best','best_synthetic_validation','best_validation','best_validation_epe','best_validation_lpips_vgg']:
                    if name in summary:r[name]=summary[name]
                for f in sorted(p.glob('synthetic_validation_*.json')):
                    val=load(f);point={k:v for k,v in val.items() if isinstance(v,(int,float,str,bool)) or v is None}
                    curves.append({**point, 'task':task,'seed':seed,'method':method,'source':str(f)})
                for x in metrics:
                    if any('val' in k or 'epe' in k for k in x):
                        curves.append({**{k:v for k,v in x.items() if isinstance(v,(int,float,str,bool)) or v is None},'task':task,'seed':seed,'method':method,'source':'training_metrics_tail'})
                rows.append(r)
    for task in BUDGET:
        for seed in range(3):
            init={r['initial_state_sha256'] for r in rows if r['task']==task and r['seed']==seed and r['completion_verified']}
            assert len(init)<=1,(task,seed,'different paired initialization')
    result=dict(observed_utc=observed,campaign=str(a.campaign),recipe_sha256=recipe,
                frozen_core='62da4f35ce3c5c96f04268bd1687a27657af9238',runs=rows,
                completion_counts={task:sum(r['completion_verified'] for r in rows if r['task']==task) for task in BUDGET},
                process_confirmation='Separate process receipt required; live files alone do not prove running',
                real_scores_computed_by_audit=False)
    (a.out/'training_matrix.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.out/'synthetic_curve_points.json').write_text(json.dumps(curves,indent=2)+'\n')
    with (a.out/'training_matrix.csv').open('w',newline='') as f:
        columns=['task','method','seed','completion_verified','latest_update','budget','live_status','live_utc','selected_hash_verified']
        w=csv.DictWriter(f,columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print(json.dumps(dict(observed_utc=observed,counts=result['completion_counts'],selected_hash_verified=sum(r['selected_hash_verified'] for r in rows)),indent=2),flush=True)

if __name__=='__main__':main()
