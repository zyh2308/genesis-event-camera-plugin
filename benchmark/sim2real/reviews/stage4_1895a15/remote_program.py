"""Read-only live training review. No GPU allocation or training interruption."""
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import gc,hashlib,json,subprocess,socket,time
import numpy as np,torch
torch.set_num_threads(2)
R=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real');B=R/'benchmark/sim2real'
RUN=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1048576),b''):h.update(c)
    return h.hexdigest()
def cmd(a):
    p=subprocess.run(a,text=True,capture_output=True,timeout=20);return {'exit':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
def git(*a):return cmd(['git','-C',str(R),*a])['stdout']
now=datetime.now(timezone.utc)
out={'checked_at_UTC':now.isoformat(),'checked_at_JST':now.astimezone(ZoneInfo('Asia/Tokyo')).isoformat(),'host':socket.gethostname(),'head':git('rev-parse','HEAD'),'status':git('status','--porcelain'),'core_diff':git('diff','--name-only','e79e656','HEAD','--','genesis_event_plugin'),'GPU_used_by_review':False}
out['gpu_status']=cmd(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv'])
out['gpu_processes']=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'])
out['tmux_target']=cmd(['tmux','list-panes','-t','stage4_flowdt1_seed0_shape_fix','-F','#{pane_pid} #{pane_current_command}'])
out['training_processes']=cmd(['pgrep','-af','[t]rain_steflow_stage4.py'])
out['live_status']=(B/'TRAINING_LIVE_STATUS.md').read_text()
out['arms']=[]
for gen in ['eventgenesis','v2e']:
    D=RUN/'flow_dt1_seed0'/gen;p=D/'training_metrics.jsonl';rows=[]
    for s in p.read_text().splitlines():
        try:rows.append(json.loads(s))
        except json.JSONDecodeError:pass
    first=rows[:20];last=rows[-1]
    c={'generator':gen,'files':[q.name for q in D.iterdir() if q.is_file()], 'valid_metric_records':len(rows),'first_record':rows[0],'latest_record':last,'first20':first,'metric_mtime_UTC':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()}
    for q in D.glob('*.json'):
        if q.stat().st_size<100000:c[q.name]=json.loads(q.read_text())
    for q in D.glob('*.pid'):c[q.name]=q.read_text().strip()
    for q in D.glob('*exit*txt'):c[q.name]=q.read_text().strip()
    for q in D.glob('*stderr*'):c[q.name+'_tail']=q.read_text(errors='replace')[-1500:]
    f=D/'update_000020.pt';file_hash=sha(f)
    expected={'eventgenesis':'f7eb58f5dae4b914050a38c5c68596e23a64372f68584fa5c1ba00e3c14c63ec','v2e':'ce2b5b076b9ba6df9af525a6ca478c8f3d0b20a37e7783de433986619d7cc763'}[gen]
    if file_hash!=expected:raise RuntimeError('immutable checkpoint hash mismatch before loading')
    with torch.serialization.safe_globals([np._core.multiarray._reconstruct,np.ndarray,np.dtype,np.dtypes.UInt32DType]):
        payload=torch.load(f,map_location='cpu',weights_only=True)
    state=payload.get('state_dict',payload.get('model'));h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
    c['update20_checkpoint']={'size_bytes':f.stat().st_size,'sha256':file_hash,'keys':list(payload),'model_state_sha256':h.hexdigest(),'parameters_finite':all(bool(torch.isfinite(v).all()) for v in state.values()),'optimizer_steps':sorted({float(v['step']) for v in payload['optimizer']['state'].values() if 'step' in v}),'optimizer_state_count':len(payload['optimizer']['state']), 'scheduler':payload.get('scheduler'), 'scalar_metadata':{k:v for k,v in payload.items() if isinstance(v,(str,int,float,bool))},'param_group_settings':[{k:v for k,v in g.items() if k!='params'} for g in payload['optimizer']['param_groups']]}
    del payload,state;gc.collect();out['arms'].append(c)
out['key_data_files']=[]
for p in [B/'configs/stage4_training_launch_v1_torch_cuda.json',Path('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/20261016-stage4-flowdt1-seed0-v4/steflow_30ms/training_bank_manifest.json'),RUN/'v2e_events/v2e_complete_manifest.json']:
    out['key_data_files'].append({'path':str(p),'size':p.stat().st_size,'sha256':sha(p),'content':json.loads(p.read_text())})
out['run_file_inventory']=[str(p.relative_to(RUN)) for p in RUN.iterdir()]
out['scripts_sha256']={p.name:sha(p) for p in (B/'scripts').glob('*stage4*.py')}
out['objective_adapter_sha256']=sha(B/'scripts/run_steflow_stage2_pilot.py')
print(json.dumps(out,ensure_ascii=False,default=str))
