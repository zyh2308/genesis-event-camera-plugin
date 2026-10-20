from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, subprocess, math, sys

P=Path('/home/zhaoyuhan/Eventbased_WAM')
R=P/'repositories/gec-sim2real-stage5-exec-20261019'
RUN=P/'runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0'
out={'observed_utc':datetime.now(timezone.utc).isoformat()}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def cmd(args):
    r=subprocess.run(args,capture_output=True,text=True,timeout=20)
    return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
out['gpus']=cmd(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'])
out['gpu_processes']=cmd(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'])
out['exec_head']=cmd(['git','-C',str(R),'rev-parse','HEAD'])
out['exec_status']=cmd(['git','-C',str(R),'status','--porcelain'])
out['code_hashes']={}
for rel in ['scripts/train_e2vid_stage5.py','compat/e2vid_temporal_consistency.py','scripts/audit_e2vid_tc_geometry.py']:
    p=R/'benchmark/sim2real'/rel
    out['code_hashes'][rel]=sha(p)
out['arms']={};paired={}
for arm in ['eventgenesis','v2e']:
    root=RUN/arm
    files=sorted(x.name for x in root.iterdir())
    rows=[]
    for line in (root/'training_metrics.jsonl').read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    train=[r for r in rows if r.get('event')=='optimizer_update']
    val=[r for r in rows if r.get('event')=='synthetic_validation']
    paired[arm]={r['optimizer_update']:r for r in train}
    info={'files':files,'count':len(train),'first':train[0] if train else None,'last':train[-1] if train else None,
          'validation_count':len(val),'last_validation':val[-1] if val else None,
          'gaps':[[a['optimizer_update'],b['optimizer_update']] for a,b in zip(train,train[1:]) if b['optimizer_update']!=a['optimizer_update']+1],
          'bad_updates':[r['optimizer_update'] for r in train if not math.isfinite(r['gradient_l2_norm']) or r['finite_gradient_tensors']!=30 or r['temporal_consistency_valid_calls']!=76 or not all(math.isfinite(v) for v in r['loss_components'].values())]}
    for name in ['live_status.json','resume_oracle.json','run_summary.json','loader_preflight.json','recurrent_state_preflight.json']:
        if (root/name).exists():info[name]=json.loads((root/name).read_text())
    info['exit_files']={x.name:x.read_text() for x in root.glob('*exit*') if x.is_file() and x.stat().st_size<100}
    out['arms'][arm]=info
common=sorted(set(paired['eventgenesis'])&set(paired['v2e']))
out['pairing']={'count':len(common),'mismatches':[i for i in common if any(paired['eventgenesis'][i].get(k)!=paired['v2e'][i].get(k) for k in ['clip_indices','clip_ids'])]}
coverage=P/'data/sim2real/stage5/20261019-flow-scene-coverage-pilot-v1/shared_bank'
out['coverage']={'exists':coverage.exists(),'files':sorted(str(p.relative_to(coverage)) for p in coverage.rglob('*') if p.is_file())[:50] if coverage.exists() else []}
# Independent closed-form probe invokes the actual adapter, including subpixel warp.
import torch
torch.set_num_threads(2)
sys.path.insert(0,str(R))
from benchmark.sim2real.scripts.audit_e2vid_tc_geometry import run as geometry_run
out['tc_geometry_rerun']={'rc':0,'result':geometry_run(torch.device('cpu'))}
from benchmark.sim2real.compat.e2vid_temporal_consistency import temporal_consistency_loss_align_corners_true as tc
checks=[]
for h,w in [(112,112),(17,29)]:
    y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    a=(.25+.002*x+.003*y)[None,None].float()
    zero=torch.zeros(1,2,h,w)
    _,m=tc(a,a,a,a,zero,output_images=True)
    identity=float((m['processed0_warped_to1']-a).abs().max())
    flow=zero.clone();flow[:,0]=.25;flow[:,1]=-.375
    _,m=tc(torch.zeros_like(a),torch.zeros_like(a),a,a,flow,output_images=True)
    expected=a+.002*.25-.003*.375
    subpixel=float((m['processed0_warped_to1'][:,:,1:-1,1:-1]-expected[:,:,1:-1,1:-1]).abs().max())
    checks.append({'shape':[h,w],'identity_max_error':identity,'subpixel_max_error':subpixel,'pass':identity<2e-6 and subpixel<2e-6})
out['independent_tc_closed_form']=checks
out['observed_finished_utc']=datetime.now(timezone.utc).isoformat()
print(json.dumps(out))
