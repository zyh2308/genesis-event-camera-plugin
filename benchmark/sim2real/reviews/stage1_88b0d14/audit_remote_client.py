"""Read-only g1 artifact/optimizer verification. No GPU allocation or training."""
import base64
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real')
BASE=ROOT/'benchmark/sim2real'
OUT=Path(__file__).resolve().parent
external=json.loads((BASE/'evidence/S0/20261016-stage1-pilot/external_artifacts.json').read_text())
sources=json.loads((BASE/'manifests/downstream_source_bundle.json').read_text())
payload=base64.b64encode(json.dumps({'external':external,'sources':sources}).encode()).decode()
program=r'''
import base64, hashlib, json, pathlib, subprocess, socket, sys
import torch
torch.set_num_threads(2)
data=json.loads(base64.b64decode('__PAYLOAD__'))
root=pathlib.Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real')
out={'host':socket.gethostname(),'python':sys.version.split()[0],'torch':torch.__version__,'GPU_used_by_review':False}
def git(*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
out['repo']={'head':git('rev-parse','HEAD'),'status':git('status','--porcelain'),'core_diff':git('diff','--name-only','e79e656ac00f7847b98db9e12bf43b093f3afff1','HEAD','--','genesis_event_plugin')}
source_checks=[]
for repo in data['sources']['repositories']:
    r=pathlib.Path(repo['server_path']); bad=[]
    for f in repo['files']:
        p=r/f['path']
        if not p.is_file() or digest(p)!=f['sha256']: bad.append(f['path'])
    source_checks.append({'repo':repo['repo'],'files':len(repo['files']),'mismatches':bad})
out['source_files']=source_checks
checks=[]
for item in data['external']['server_data_artifacts']:
    if 'server_path' in item:
        p=pathlib.Path(item['server_path'])
        checks.append({'role':item['role'],'exists':p.is_file(),'size_matches':p.is_file() and p.stat().st_size==item['size_bytes'],'sha_matches':p.is_file() and digest(p)==item['sha256']})
    else:
        for f in item['files']:
            p=pathlib.Path(item['server_directory'])/f['path']
            checks.append({'role':item['role']+'/'+f['path'],'exists':p.is_file(),'size_matches':p.is_file() and p.stat().st_size==f['size_bytes'],'sha_matches':p.is_file() and digest(p)==f['sha256']})
out['data_files']=checks
out['checkpoints']=[]
for item in data['external']['server_checkpoints_not_in_git']:
    p=pathlib.Path(item['path'])
    c={'task':item['task'],'generator':item['generator'],'size_bytes':p.stat().st_size,'sha_matches':digest(p)==item['sha256']}
    weights=torch.load(p,map_location='cpu',weights_only=True)
    model=weights.get('model',weights.get('state_dict'))
    h=hashlib.sha256()
    for k,v in sorted(model.items()): h.update(k.encode()); h.update(v.contiguous().numpy().tobytes())
    c.update({'seed':weights.get('seed'),'updates_metadata':weights.get('updates'),'post_update_state_sha256':h.hexdigest(),
      'optimizer_state_count':len(weights['optimizer']['state']),
      'optimizer_step_values':sorted({float(v['step']) for v in weights['optimizer']['state'].values() if 'step' in v}),
      'parameters_finite':all(bool(torch.isfinite(v).all()) for v in model.values()),
      'stored_train_loss':weights.get('train_loss',weights.get('train_objective')),
      'stored_validation_loss':weights.get('validation_loss',weights.get('synthetic_validation_objective'))})
    out['checkpoints'].append(c)
    del weights,model
out['run_exit_codes']={}
for directory in data['external']['server_run_directories']:
    d=pathlib.Path(directory)
    out['run_exit_codes'][d.name]={p.name:p.read_text().strip() for p in d.glob('*exit_code.txt')}
w=data['external']['external_fixed_loss_weight']
out['LPIPS_weight_sha_matches']=digest(w['path'])==w['sha256']
out['remote_runtime_manifests']=[]
for p in (root/'runtime/sim2real').glob('*/runtime_manifest.json'):
    out['remote_runtime_manifests'].append({'path':str(p),'content':json.loads(p.read_text())})
print(json.dumps(out,ensure_ascii=False))
'''.replace('__PAYLOAD__',payload)
encoded=base64.b64encode(program.encode()).decode()
code='import base64,sys; exec(base64.b64decode(sys.stdin.buffer.read()))'
command=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c',code])
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-p','22','zhaoyuhan@10.129.163.68',command],input=encoded,capture_output=True,text=True,timeout=180)
(OUT/'remote_audit_stdout.json').write_text(result.stdout)
(OUT/'remote_audit_stderr.txt').write_text(result.stderr)
if result.returncode:
    print(json.dumps({'exit_code':result.returncode,'stderr':result.stderr,'stdout':result.stdout[-2000:]}))
else:
    d=json.loads(result.stdout)
    print(json.dumps(d,indent=2,ensure_ascii=False))
