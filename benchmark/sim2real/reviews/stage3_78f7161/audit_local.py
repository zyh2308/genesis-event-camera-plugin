"""Independent read-only Stage 3 local hashes, source-data and CPU checks."""
from pathlib import Path
import ast, hashlib, json, subprocess
import numpy as np, torch
R=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real');B=R/'benchmark/sim2real'
E=B/'evidence/S3/20261016-stage3';O=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for q in iter(lambda:f.read(1048576),b''):h.update(q)
    return h.hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(R),*a],text=True).strip()
out={'head':git('rev-parse','HEAD'),'initial_status':git('status','--porcelain'),'frozen_core_diff':git('diff','--name-only','e79e656','HEAD','--','genesis_event_plugin'),'integration_diff':git('diff','--name-only','e79e656','HEAD','--','integrations')}
out['indices']=[]
for p in [E/'artifact_index.json',B/'artifact_index.json']:
    d=json.loads(p.read_text()); rows=d.get('artifacts',d.get('files',[]));bad=[]
    for row in rows:
        f=R/row['path']
        if not f.is_file() or f.stat().st_size!=row['size_bytes'] or sha(f)!=row['sha256']:bad.append(row['path'])
    out['indices'].append({'index':str(p.relative_to(R)),'count':len(rows),'mismatches':bad})
x=json.loads((E/'external_artifacts_stage3.json').read_text());out['public_source_files']=[]
for row in x['public_source_files_downloaded_wsl']:
    p=Path(row['wsl_path']);out['public_source_files'].append({'path':str(p),'size_match':p.is_file() and p.stat().st_size==row['size_bytes'],'sha_match':p.is_file() and sha(p)==row['sha256']})

# Execute checked-in metric/extraction functions with independently generated vectors.
p=B/'scripts/run_steflow_stage2_pilot.py'; tree=ast.parse(p.read_text());names=['model_output_to_flow_hw','compute_epe']
ns={'np':np,'torch':torch,'FLOW_UNIT':'px_per_interval'}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(p),'exec'),ns)
gt=np.stack([np.full((3,4),3.0),np.full((3,4),4.0)]);mask=np.ones((3,4),bool)
pred=ns['model_output_to_flow_hw'](torch.from_numpy(gt).unsqueeze(0))
metric=lambda p:ns['compute_epe'](p,gt,mask,prediction_unit='px_per_interval',gt_unit='px_per_interval')[0]
wrong=pred.copy();wrong[1]+=2
rejected=False
try:metric(pred[0])
except ValueError:rejected=True
out['independent_EPE_checks']={'output_shape':list(pred.shape),'perfect':metric(pred),'vertical_plus_2':metric(wrong),'missing_component_rejected':rejected}
(O/'local_verification.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print(json.dumps(out,ensure_ascii=False))
