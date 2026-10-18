import json,hashlib,subprocess,sys,gc,os,time
from pathlib import Path
from datetime import datetime,timezone
import torch
torch.set_num_threads(2)
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'repositories/gec-sim2real'
C=P/'runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5'
E=R/'benchmark/sim2real/evidence/S4/20261018-motion-diagnostic'
O=C/'research_audit_20261019';O.mkdir(exist_ok=True)
out={'observed_utc':datetime.now(timezone.utc).isoformat()}
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def cmd(a):
    p=subprocess.run(a,capture_output=True,text=True,timeout=30)
    return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
d=read(E/'motion_diagnostic.json');oracle=read(E/'v2e_resume_checkpoint_oracle.json');receipt=read(E/'run_receipt.json')
out['evidence_hashes']={p.name:sha(p) for p in E.glob('*') if p.is_file()}
out['current_git']=cmd(['git','-C',str(R),'status','--porcelain'])
out['git_head']=cmd(['git','-C',str(R),'rev-parse','HEAD'])
out['registration_history']=cmd(['git','-C',str(R),'log','--format=%H %cI %s','-6','--','benchmark/sim2real/evidence/S4/20261018-motion-diagnostic/DIAGNOSTIC_PLAN.md','benchmark/sim2real/scripts/stage4_motion_learning_diagnostic.py'])
out['source_code_hash']=sha(R/'benchmark/sim2real/scripts/stage4_motion_learning_diagnostic.py')
out['source_hash_matches_receipt']=out['source_code_hash']==receipt['diagnostic_code']['sha256']
out['v2e_oracle_checkpoints']={}
def model_hash(s):
    h=hashlib.sha256()
    for k,v in sorted(s.items()):h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()
paths={'v2e_oracle_source':Path(oracle['source_checkpoint']),'v2e_oracle_resumed':Path(oracle['resumed_checkpoint'])}
for arm in ['eventgenesis','v2e']:
    paths[arm+'_final']=C/'full_training_retry1'/arm/'last.pt'
    paths[arm+'_best']=C/'full_training_retry1'/arm/'synthetic_best.pt'
for label,p in paths.items():
    s=torch.load(p,map_location='cpu',weights_only=False)
    steps=[int(x['step'].item()) for x in s['optimizer']['state'].values()]
    out['v2e_oracle_checkpoints'][label]={**{k:s.get(k) for k in ['generator','seed','optimizer_update','next_epoch','next_position_samples','resume_phase','resume_allowed','best_epoch','best_synthetic_validation_epe','real_data_used','initial_state_sha256','config_sha256','data_manifest_sha256']},
      'path':str(p),'size':p.stat().st_size,'sha256':sha(p),'weight_hash_verified':model_hash(s['state_dict'])==s['current_state_sha256'],
      'adam_states':len(steps),'adam_step_minmax':[min(steps),max(steps)],'scheduler_last_epoch':s['scheduler']['last_epoch']}
    del s;gc.collect()
out['completion_files']={p.name:p.read_text() for p in (C/'resume_retry1').glob('*') if p.is_file() and (p.name.endswith('finished_utc.txt') or p.name=='campaign_complete.txt' or p.name.startswith('full_resume_') and p.name.endswith('exit_code.txt'))}
out['v2e_supplement_logs']={}
for p in (C/'resume_equivalence_v2e_supplement_20261018').glob('attempt_02/*'):
    if p.is_file() and (p.suffix in ['.txt','.log','.json']):
        out['v2e_supplement_logs'][p.name]={'bytes':p.stat().st_size,'sha256':sha(p),'tail':p.read_text(errors='replace')[-1500:]}
out['corrected_determinism_paths']=receipt['corrected_determinism_log_paths']
out['replay']={}
gpu=cmd(['nvidia-smi','-i','2','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'])
out['replay']['gpu_before']=gpu
values=[int(x.strip()) for x in gpu['stdout'].strip().split(',')] if gpu['exit_code']==0 else [99999,100]
if values[0]<=1024 and values[1]<=5:
    output=O/'motion_diagnostic_replay_epoch23.json'
    if output.exists():raise RuntimeError('refusing to overwrite existing replay')
    command=[sys.executable,'-u',str(R/'benchmark/sim2real/scripts/stage4_motion_learning_diagnostic.py'),
      '--model-root',d['paths']['model_root'],'--data-root',d['paths']['data_root'],
      '--shared-bank',d['paths']['shared_bank'],'--config',str(R/'benchmark/sim2real/configs/stage4_corrected_v2_flowdt1_seed0.json'),
      '--eventgenesis-checkpoint',d['paths']['checkpoints']['eventgenesis']['checkpoint_path'],
      '--v2e-checkpoint',d['paths']['checkpoints']['v2e']['checkpoint_path'],'--output',str(output),'--device','cuda:0']
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='2';env['PYTHONDONTWRITEBYTECODE']='1'
    start=time.monotonic()
    p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=240)
    (O/'replay.stdout.log').write_text(p.stdout);(O/'replay.stderr.log').write_text(p.stderr)
    out['replay'].update({'exit_code':p.returncode,'elapsed_s':time.monotonic()-start,'output':str(output),'stderr':p.stderr[-1500:]})
    if p.returncode==0:
        new=read(output);out['replay']['sha256']=sha(output)
        out['replay']['metric_comparison']={}
        for arm in ['eventgenesis','v2e']:
            out['replay']['metric_comparison'][arm]={}
            for region in ['full_valid','shared_event_active_union','gt_motion_ge_0_5','gt_motion_ge_1_0']:
                a=d['metrics']['all_strata_pixel_pooled'][arm][region];b=new['metrics']['all_strata_pixel_pooled'][arm][region]
                out['replay']['metric_comparison'][arm][region]={'old_epe':a['epe_px_per_30ms'],'new_epe':b['epe_px_per_30ms'],'abs_epe_error':abs(a['epe_px_per_30ms']-b['epe_px_per_30ms']),
                   'old_near_zero':a['near_zero_prediction_rate']['0.1'],'new_near_zero':b['near_zero_prediction_rate']['0.1'],'pixels_equal':a['pixel_count']==b['pixel_count']}
else:out['replay']['skipped']='GPU 2 occupied; no work launched'
out['finished_utc']=datetime.now(timezone.utc).isoformat()
(O/'research_verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
