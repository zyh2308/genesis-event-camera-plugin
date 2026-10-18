import json, os, subprocess, hashlib, math
from pathlib import Path
from datetime import datetime, timezone

P=Path('/home/zhaoyuhan/Eventbased_WAM')
R=P/'repositories/gec-sim2real'
C=P/'runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5'
F=C/'full_training_retry1'
E=R/'benchmark/sim2real/evidence/S4/20261017-stage4-corrected-v2'
out={'observed_utc':datetime.now(timezone.utc).isoformat()}
def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=25)
    return {'exit_code':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
out['git_head']=cmd(['git','-C',str(R),'rev-parse','HEAD'])
out['git_status']=cmd(['git','-C',str(R),'status','--porcelain'])
out['frozen_core_diff']=cmd(['git','-C',str(R),'diff','--stat','e79e656ac00f7847b98db9e12bf43b093f3afff1','HEAD','--','genesis_event_plugin'])
out['tmux']=cmd(['tmux','has-session','-t','stage4-flow-corrected-20261017-v7'])
out['gpus']=cmd(['nvidia-smi','-i','2,4','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'])
out['exit_records']={p.name:p.read_text().strip() for p in sorted((C/'resume_retry1').glob('*.exit_code.txt'))}
out['arms']={}
logs={}
for arm in ['eventgenesis','v2e']:
    root=F/arm
    rows=[json.loads(x) for x in (root/'training_metrics.jsonl').read_text().splitlines() if x.strip()]
    updates=[x for x in rows if 'optimizer_update' in x]
    vals=[x for x in rows if x.get('event')=='synthetic_validation']
    logs[arm]={x['optimizer_update']:x for x in updates}
    meta=read(root/'run_metadata.json')
    pid=meta.get('pid')
    info={'metadata':meta,'live_status':read(root/'live_status.json'),
          'process':cmd(['ps','-p',str(pid),'-o','pid,etime,stat,%cpu,%mem,args']),
          'metric_count':len(updates),'first_update':updates[0] if updates else None,
          'latest_update':updates[-1] if updates else None,'validation':vals,
          'update_gaps':[[a['optimizer_update'],b['optimizer_update']] for a,b in zip(updates,updates[1:]) if b['optimizer_update']!=a['optimizer_update']+1],
          'bad_numeric_updates':[x['optimizer_update'] for x in updates if not all(math.isfinite(float(x[k])) for k in ['loss','photometric_loss','smoothness_loss','gradient_l2_norm']) or x.get('finite_gradient_tensors')!=70]}
    for name in ['stdout','stderr']:
        p=C/'resume_retry1'/f'full_resume_{arm}.{name}.log'
        info[name+'_tail']='\n'.join(p.read_text(errors='replace').splitlines()[-4:]) if p.exists() else None
    out['arms'][arm]=info
shared=sorted(set(logs['eventgenesis'])&set(logs['v2e']))
out['paired_input_comparison']={'updates_checked':len(shared),'mismatches':[i for i in shared if any(logs['eventgenesis'][i][k]!=logs['v2e'][i][k] for k in ['rollout_sample_ids','augmentation_seed_sequences'])]}

receipt=read(C/'resume_retry1/preflight_result_resume_retry1.json')
out['receipt_structure']={'keys':list(receipt),'overall_status':receipt.get('overall_status'),'checks':receipt.get('checks'),'fingerprint_sha256':receipt.get('fingerprint_sha256')}
if isinstance(out['receipt_structure']['checks'],dict):
    out['receipt_structure']['checks']={k:{kk:vv for kk,vv in v.items() if kk in ['status','evidence','evidence_path','evidence_sha256','error']} if isinstance(v,dict) else v for k,v in out['receipt_structure']['checks'].items()}
out['evidence_hashes']={}
for name in ['resume_checkpoint_oracle.json','resume_retry1_gate_verification.json','resume_full_replay_failure.json','resume_cuda_nondeterminism_diagnostic.json','paired_training_progress_190912.json']:
    p=E/name
    out['evidence_hashes'][name]={'sha256':sha(p),'bytes':p.stat().st_size}
out['negative_controls']=read(C/'resume_retry1/launcher_negative_controls/launcher_negative_controls.json')
out['strict_failure_raw']=(C/'formal_resume_equivalence.stderr.log').read_text(errors='replace')[-3000:]
p=C/'resume_equivalence/determinism_diagnostic_01/stderr.log'
out['determinism_raw']=p.read_text(errors='replace')[-3000:] if p.exists() else {'missing_claimed_path':str(p),'directory_files':[str(x) for x in p.parent.glob('*')]}
out['finished_utc']=datetime.now(timezone.utc).isoformat()
print(json.dumps(out,indent=2))
