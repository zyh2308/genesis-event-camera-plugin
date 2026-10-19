from pathlib import Path
import json,hashlib,subprocess,math,gc,sys
from datetime import datetime,timezone
import torch
torch.set_num_threads(2)
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'repositories/gec-sim2real'
RUN=P/'runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2'
DATA=P/'data/sim2real/stage4/e2vid_corrected_10ms_20261019_v1'
out={'observed_utc':datetime.now(timezone.utc).isoformat()}
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=25);return {'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
out['git_head']=cmd(['git','-C',str(R),'rev-parse','HEAD'])
out['git_status']=cmd(['git','-C',str(R),'status','--porcelain'])
out['core_diff']=cmd(['git','-C',str(R),'diff','e79e656ac00f7847b98db9e12bf43b093f3afff1','--stat','--','genesis_event_plugin'])
out['gpus']=cmd(['nvidia-smi','-i','2,4','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'])
out['code_hashes']={}
for name in ['train_e2vid_stage4.py','build_e2vid_stage4_10ms_data.py','stage4_final_best_motion_diagnostic.py','stage4_mvsec_dt1_metrics.py','run_pose_stage2_pilot.py']:
 p=R/'benchmark/sim2real/scripts'/name
 out['code_hashes'][str(p.relative_to(R))]=sha(p) if p.exists() else None
out['data_manifest_hash']=sha(DATA/'adapter_manifest.json')
out['data_manifest']=json.loads((DATA/'adapter_manifest.json').read_text())
out['arms']={};logs={}
for arm in ['eventgenesis','v2e']:
 root=RUN/arm
 rows=[json.loads(x) for x in (root/'training_metrics.jsonl').read_text().splitlines() if x.strip()]
 train=[r for r in rows if r.get('event')=='optimizer_update'];val=[r for r in rows if r.get('event')=='synthetic_validation']
 logs[arm]={r['optimizer_update']:r for r in train}
 info={'first':train[0],'last':train[-1],'count':len(train),'validation':val,'status':json.loads((root/'live_status.json').read_text()),
  'gaps':[[a['optimizer_update'],b['optimizer_update']] for a,b in zip(train,train[1:]) if b['optimizer_update']!=a['optimizer_update']+1],
  'bad_rows':[r['optimizer_update'] for r in train if not math.isfinite(r['gradient_l2_norm']) or r['finite_gradient_tensors']!=30 or r['temporal_consistency_valid_calls']!=76 or not all(math.isfinite(v) for v in r['loss_components'].values())]}
 for f in ['resume_oracle.json','loader_preflight.json','recurrent_state_preflight.json','run_summary.json']:
  if (root/f).exists():info[f]=json.loads((root/f).read_text())
 info['resume_log_tail']='\n'.join((root/'resume_stdout.log').read_text().splitlines()[-3:]) if (root/'resume_stdout.log').exists() else ''
 info['processes']=cmd(['pgrep','-af','train_e2vid_stage4.py.*'+arm])
 with (root/'last.pt').open('rb') as f:
  s=torch.load(f,map_location='cpu',weights_only=False)
 h=hashlib.sha256()
 for k,v in sorted(s['model'].items()):h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
 steps=[int(x['step'].item()) for x in s['optimizer']['state'].values()]
 info['checkpoint']={k:s.get(k) for k in ['schema','generator','seed','optimizer_update','next_epoch','next_batch_in_epoch','best_epoch','best_synthetic_validation_lpips_vgg','trainer_script_sha256','config_sha256','data_manifest_sha256','event','saved_utc']}
 info['checkpoint'].update({'weight_hash_verified':h.hexdigest()==s['model_sha256'],'adam_count':len(steps),'adam_steps':[min(steps),max(steps)]})
 del s;gc.collect();out['arms'][arm]=info
common=sorted(set(logs['eventgenesis'])&set(logs['v2e']))
out['paired_clips']={'count':len(common),'mismatched_updates':[i for i in common if any(logs['eventgenesis'][i][k]!=logs['v2e'][i][k] for k in ['clip_indices','clip_ids'])]}
out['upstream_source_excerpts']={}
M=P/'external/sources/downstream/event_cnn_minimal'
for relative in ['model/loss.py','trainer/trainer.py','data_loader/dataset.py','utils/flow_utils.py','utils/util.py','utils/data_augmentation.py']:
 p=M/relative
 if not p.exists():continue
 lines=p.read_text().splitlines();selected=set()
 for i,line in enumerate(lines):
  if any(t in line for t in ['grid_sample','def temporal_consistency','def __getitem__','def forward_sequence','flow_valid','torch.meshgrid','def warp','def voxel','def transform_flow']):selected.update(range(max(0,i-3),min(len(lines),i+28)))
 out['upstream_source_excerpts'][relative]={'sha256':sha(p),'lines':{str(i+1):lines[i] for i in sorted(selected)}}
out['metric_self_test']=cmd([sys.executable,str(R/'benchmark/sim2real/scripts/stage4_mvsec_dt1_metrics.py'),'--self-test']) if (R/'benchmark/sim2real/scripts/stage4_mvsec_dt1_metrics.py').exists() else None
print(json.dumps(out,indent=2))
