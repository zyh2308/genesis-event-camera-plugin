from pathlib import Path
import ast,hashlib,json,math,subprocess
import numpy as np,torch
R=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real');B=R/'benchmark/sim2real';O=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ev=B/'evidence/S4/20261016-stage4-flowdt1-seed0-v4-update20'
d=json.loads((O/'remote_audit_stdout.json').read_text());checks=[]
for a in d['arms']:
    p=ev/f"{a['generator']}_training_metrics_updates_1_20.jsonl";rows=[json.loads(s) for s in p.read_text().splitlines()]
    checks.append({'generator':a['generator'],'sha256':sha(p),'records':len(rows),'updates_1_to_20':[r['optimizer_update'] for r in rows]==list(range(1,21)),'matches_live_log_prefix':rows==a['first20'],'finite_losses_and_gradients':all(all(math.isfinite(r[k]) for k in ['loss','photometric_loss','smoothness_loss','gradient_l2_norm']) for r in rows),'all_batch8_and_70_gradients':all(r['effective_batch']==8 and r['finite_gradient_tensors']==70 for r in rows)})
pairs=[json.loads(s) for s in (ev/'eventgenesis_training_metrics_updates_1_20.jsonl').read_text().splitlines()],[json.loads(s) for s in (ev/'v2e_training_metrics_updates_1_20.jsonl').read_text().splitlines()]
paired_order=all(a['rollout_sample_ids']==b['rollout_sample_ids'] for a,b in zip(*pairs))
index=json.loads((B/'artifact_index.json').read_text());rows=index.get('artifacts',[]);bad=[]
for row in rows:
    p=R/row['path']
    if not p.is_file() or p.stat().st_size!=row['size_bytes'] or sha(p)!=row['sha256']:bad.append(row['path'])

# Check the CPU gray adapter against direct official-layout grayscale values.
p=B/'scripts/run_steflow_stage2_pilot.py';t=ast.parse(p.read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_gray_batch_for_pinned_loss')
ns={'torch':torch};exec(compile(ast.Module(body=[f],type_ignores=[]),str(p),'exec'),ns)
x=torch.arange(2*5*7,dtype=torch.float32).reshape(2,1,5,7).requires_grad_();y=ns[f.name](x)
adapter={'shape':list(y.shape),'CPU':y.device.type=='cpu','values_unchanged':torch.equal(y,x.detach()[:,0]),'reference_image_gradient_disabled':not y.requires_grad}
out={'index_entries':len(rows),'index_mismatches':bad,'immutable_first20':checks,'paired_sample_order_first20':paired_order,'gray_adapter_independent_check':adapter,'implementation_diffs':subprocess.check_output(['git','diff','--name-only','e79e656','HEAD','--','genesis_event_plugin'],cwd=R,text=True).strip()}
(O/'local_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
