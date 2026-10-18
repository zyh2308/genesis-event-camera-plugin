import json, hashlib, subprocess, os, gc
from pathlib import Path
from datetime import datetime, timezone
import torch
torch.set_num_threads(2)
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'repositories/gec-sim2real'
C=P/'runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5'
F=C/'full_training_retry1';A=C/'resume_retry1'
out={'observed_utc':datetime.now(timezone.utc).isoformat(),'checkpoints':{}}
def digest_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def model_hash(s):
    h=hashlib.sha256()
    for k,v in sorted(s.items()):h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()
receipt=json.loads((A/'preflight_result_resume_retry1.json').read_text())
fp=receipt['fingerprint']
out['fingerprint']={'stored':fp['fingerprint_sha256'],
 'canonical_recomputed':hashlib.sha256(json.dumps({'code':fp['code'],'inputs':fp['inputs']},sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 'current_code_mismatches':[k for k,v in fp['code'].items() if digest_file(R/k)!=v],
 'receipt_hash':digest_file(A/'preflight_result_resume_retry1.json')}
out['gates']=[{'name':x['check'],'status':x['status']} for x in receipt['checks']]
out['boundary_source_scheduler']={}
paths={'oracle_source':C/'pilot/eventgenesis/last.pt',
       'oracle_resumed':C/'resume_equivalence/runs/process_exit_resume_from_update198/eventgenesis/last.pt',
       'eventgenesis_current':F/'eventgenesis/last.pt','v2e_current':F/'v2e/last.pt'}
for label,p in paths.items():
    with p.open('rb') as stream:
        h=hashlib.sha256()
        for b in iter(lambda:stream.read(1048576),b''):h.update(b)
        stream.seek(0);s=torch.load(stream,map_location='cpu',weights_only=False)
    adam=list(s['optimizer']['state'].values())
    steps=[int(x['step'].item()) for x in adam]
    report={k:s.get(k) for k in ['generator','seed','optimizer_update','next_epoch','next_position_samples','resume_phase','resume_allowed','initial_state_sha256','current_state_sha256','config_sha256','data_manifest_sha256','best_epoch','best_synthetic_validation_epe','real_data_used']}
    report.update({'path':str(p),'sha256':h.hexdigest(),'model_hash_verified':model_hash(s['state_dict'])==s['current_state_sha256'],
      'adam_states':len(adam),'adam_step_minmax':[min(steps),max(steps)],
      'all_adam_moments_finite':all(torch.isfinite(x[k]).all().item() for x in adam for k in ['exp_avg','exp_avg_sq']),
      'scheduler':s['scheduler'],'optimizer_lrs':[x['lr'] for x in s['optimizer']['param_groups']],
      'rng_tensor_devices':{'torch':str(s['torch_rng'].device),'cuda':[str(x.device) for x in s['cuda_rng']]}})
    out['checkpoints'][label]=report
    del s,adam;gc.collect()

diag=C/'resume_equivalence/determinism_diagnostic_01'
out['diagnostic_real_paths']={}
for name in ['sitecustomize.py','deterministic_from_update20.stderr.log','deterministic_from_update20.exit_code.txt']:
    p=diag/name;out['diagnostic_real_paths'][name]={'bytes':p.stat().st_size,'sha256':digest_file(p),'content':p.read_text()}
out['launch_gate_stdout']=(A/'final_full_gate_verify_retry1.stdout.log').read_text()
out['gate_negative_summary']=[{k:x.get(k) for k in ['case','exit_code','launch_branch_reached','blocked_before_process','run_root_created']} for x in json.loads((A/'launcher_negative_controls/launcher_negative_controls.json').read_text())['cases']]
out['pilot_epoch1']={}
for arm in ['eventgenesis','v2e']:
    rows=[json.loads(x) for x in (C/'pilot'/arm/'training_metrics.jsonl').read_text().splitlines() if x.strip()]
    out['pilot_epoch1'][arm]=[x for x in rows if x.get('event')=='synthetic_validation']
print(json.dumps(out,indent=2))
