"""g1 read-only Stage 3 evidence audit; no model execution or GPU allocation."""
from pathlib import Path
import hashlib,json,socket,subprocess,gc
import numpy as np,torch
torch.set_num_threads(2)
R=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real');B=R/'benchmark/sim2real';E=B/'evidence/S3/20261016-stage3'
x=json.loads((E/'external_artifacts_stage3.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for q in iter(lambda:f.read(1048576),b''):h.update(q)
    return h.hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(R),*a],text=True).strip()
out={'host':socket.gethostname(),'head':git('rev-parse','HEAD'),'status':git('status','--porcelain'),'core_diff':git('diff','--name-only','e79e656','HEAD','--','genesis_event_plugin'),'data':[],'checkpoints':[],'inference_manifests':[],'MVSEC_saved_outputs':[]}
def check(row,pathkey='server_path'):
    p=Path(row[pathkey]);return {'path':str(p),'exists':p.is_file(),'size_match':p.is_file() and p.stat().st_size==row['size_bytes'],'sha_match':p.is_file() and sha(p)==row['sha256']}
for obj in x['derived_public_data_on_g1'].values():
    for row in obj['files']:out['data'].append(check(row))
for row in x['candidate_checkpoints']:
    c=check(row);p=Path(row['server_path']);d=torch.load(p,map_location='cpu',weights_only=True);s=d.get('model',d.get('state_dict'));h=hashlib.sha256()
    for k,v in sorted(s.items()):h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
    c.update({'state_hash':h.hexdigest(),'parameters_finite':all(bool(torch.isfinite(v).all()) for v in s.values()),'updates':d.get('updates'),'epochs':d.get('epochs'),'optimizer_steps':sorted({float(v['step']) for v in d['optimizer']['state'].values() if 'step' in v})});out['checkpoints'].append(c);del d,s;gc.collect()
for row in x['hqf_event_only_inference']:
    p=Path(row['server_output_dir'])/'inference_manifest.json';d=json.loads(p.read_text());out['inference_manifests'].append({'task':'HQF','path':str(p),'size_match':p.stat().st_size==row['manifest_size_bytes'],'sha_match':sha(p)==row['manifest_sha256'],'keys':list(d),'summary':{k:v for k,v in d.items() if not isinstance(v,list)}})
for row in x['mvsec_event_only_inference_outputs']:
    p=Path(row['server_output_dir'])/'manifest.json';out['inference_manifests'].append({'task':'MVSEC','path':str(p),'size_match':p.stat().st_size==row['run_manifest']['size_bytes'],'sha_match':sha(p)==row['run_manifest']['sha256']})
    checked=[]
    for artifact in row['per_window_artifacts']:
        c=check(artifact)
        with np.load(artifact['server_path']) as f:
            c['arrays']={k:list(f[k].shape) for k in f.files};c['all_numeric_arrays_finite']=all(bool(np.isfinite(f[k]).all()) for k in f.files if np.issubdtype(f[k].dtype,np.number))
        checked.append(c)
    out['MVSEC_saved_outputs'].append({'dt':row['dt'],'generator':row['generator'],'count':len(checked),'artifacts':checked})
out['server_run_exit_codes']={p.name:{f.name:f.read_text().strip() for f in p.glob('*exit_code.txt')} for p in (R.parent.parent/'runs/sim2real').glob('20261016-stage3-*') if p.is_dir()}
out['script_hashes']={p.name:sha(p) for p in (B/'scripts').glob('*.py')}
print(json.dumps(out,ensure_ascii=False))
