"""Independent Stage 2 read-only artifact and CPU operator checks."""
from pathlib import Path
import ast, hashlib, json, subprocess, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real')
BASE=ROOT/'benchmark/sim2real'
OUT=Path(__file__).resolve().parent
EV=BASE/'evidence/S2/20261016-stage2-pilot'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a): return subprocess.check_output(['git','-C',str(ROOT),*a],text=True).strip()
index=json.loads((EV/'artifact_index.json').read_text())['artifacts']
bad=[]
for row in index:
    p=ROOT/row['path']
    if not p.is_file() or p.stat().st_size!=row['size_bytes'] or digest(p)!=row['sha256']: bad.append(row['path'])
out={'repo':{'head':git('rev-parse','HEAD'),'status_before_review':git('status','--porcelain'),
    'core_diff':git('diff','--name-only','e79e656','HEAD','--','genesis_event_plugin'),
    'integration_diff':git('diff','--name-only','e79e656','HEAD','--','integrations')},
    'indexed_artifacts':len(index),'mismatches':bad,'results':{}}
for p in EV.rglob('result.json'):
    d=json.loads(p.read_text()); rows=d.get('paired_results',d.get('results',[]))
    summary=[]
    for row in rows:
        s={k:v for k,v in row.items() if k not in ('training_updates_detail','validation_detail','training_detail')}
        rec=row.get('training_updates_detail',row.get('training_detail',[]))
        if not rec: rec=[a for a in d.get('train_updates',[]) if a['generator']==row['generator']]
        s['observed_training_record_count']=len(rec)
        s['train_rollouts_disjoint_validation']=not bool(set(row['train_rollouts'])&set(row['validation_rollouts']))
        summary.append(s)
    out['results'][str(p.relative_to(EV))]=summary

# Execute only the checked-in correlation class, with independent unfold reference.
p=BASE/'scripts/run_steflow_stage2_pilot.py'
tree=ast.parse(p.read_text()); nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_MemoryEfficientSpatialCorrelation']
ns={'torch':torch}; exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
corr=ns['_MemoryEfficientSpatialCorrelation'].apply
torch.set_num_threads(2); torch.manual_seed(341)
checks=[]
for shape in [(2,3,6,7),(1,2,2,3),(1,1,1,1)]:
    a=torch.randn(shape,dtype=torch.float64,requires_grad=True); b=torch.randn(shape,dtype=torch.float64,requires_grad=True)
    actual=corr(a,b); B,C,H,W=shape
    expected=(a.unsqueeze(2)*F.unfold(b,kernel_size=9,padding=4).reshape(B,C,81,H,W)).sum(1).reshape(B,9,9,H,W)
    weight=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*weight).sum(),(a,b),retain_graph=True)
    gb=torch.autograd.grad((expected*weight).sum(),(a,b))
    checks.append({'shape':shape,'forward_max_error':float((actual-expected).abs().max()),'gradient_max_error':max(float((x-y).abs().max()) for x,y in zip(ga,gb))})
out['independent_CPU_correlation']=checks
(OUT/'local_verification.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print(json.dumps({'head':out['repo'],'indexed_artifacts':len(index),'mismatches':bad,'correlation':checks,'results_saved':str(OUT/'local_verification.json')},ensure_ascii=False))
