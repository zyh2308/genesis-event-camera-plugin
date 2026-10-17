"""Independent completion, validation-curve and zero-flow audit. CPU only."""
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import gc,hashlib,json,math,subprocess
import h5py,numpy as np,torch
torch.set_num_threads(2)
R=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real');B=R/'benchmark/sim2real';RUN=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4');ROOT=RUN/'flow_dt1_seed0'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1048576),b''):h.update(c)
    return h.hexdigest()
def cmd(*args):return subprocess.run(args,text=True,capture_output=True,timeout=20).stdout.strip()
out={'checked_at_JST':datetime.now(timezone.utc).astimezone(ZoneInfo('Asia/Tokyo')).isoformat(),'head':cmd('git','-C',str(R),'rev-parse','HEAD'),'status':cmd('git','-C',str(R),'status','--porcelain'),'core_diff':cmd('git','-C',str(R),'diff','--name-only','e79e656','HEAD','--','genesis_event_plugin'),'arms':{},'checkpoints':[]}
streams={}
expected={('eventgenesis','last.pt'):'83e2788f6c0506d22f28e212b6c65faaf56d2858e541940be91d2da4edf6ab8a',('eventgenesis','synthetic_best.pt'):'10c8785439ee9b8b61497381e013b3b022e9c50eaa83509f877140d597682f85',('v2e','last.pt'):'a08a5de6bf7fd92dd4d0e7c78dcdc82b88e60bb2792125d5e2db1bbfb17ce587',('v2e','synthetic_best.pt'):'af451d42dc69236554aa5e4a34dadf0fb29554e135e8e39e7a02a30d023bb57c'}
for gen in ['eventgenesis','v2e']:
    D=ROOT/gen;p=D/'training_metrics.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()];u=[r for r in rows if 'optimizer_update' in r];v=[r for r in rows if r.get('event')=='synthetic_validation'];streams[gen]=u
    epoch_summaries=[]
    for epoch in range(1,46):
        e=[r for r in u if r['epoch']==epoch];ids=[s for r in e for s in r['rollout_sample_ids']]
        epoch_summaries.append({'epoch':epoch,'updates':len(e),'samples':len(ids),'unique_samples':len(set(ids)),'train_loss_mean':float(np.mean([r['loss'] for r in e])),'photometric_mean':float(np.mean([r['photometric_loss'] for r in e])),'smoothness_mean':float(np.mean([r['smoothness_loss'] for r in e])),'lr':sorted(set(r['learning_rate'] for r in e))})
    best=min(v,key=lambda r:r['mean_epe_px_per_30ms'])
    out['arms'][gen]={'line_count':len(rows),'update_count':len(u),'validation_count':len(v),'updates_exact_sequence':[r['optimizer_update'] for r in u]==list(range(1,8911)),'validation_epochs_exact_sequence':[r['epoch'] for r in v]==list(range(1,46)),'all_losses_gradients_finite':all(np.isfinite([r[k] for k in ['loss','photometric_loss','smoothness_loss','gradient_l2_norm']]).all() for r in u),'all_batch8':all(r['effective_batch']==8 for r in u),'all_70_gradient_tensors':all(r['finite_gradient_tensors']==70 for r in u),'train_exit':(D/'train.exit_code.txt').read_text().strip(),'log_sha256':sha(p),'live_status':json.loads((D/'live_status.json').read_text()),'best_validation':best,'last_validation':v[-1],'validation_curve':v,'epoch_summaries':epoch_summaries,'run_metadata':json.loads((D/'run_metadata.json').read_text())}
    for name in ['last.pt','synthetic_best.pt']:
        p=D/name;h=sha(p)
        if h!=expected[(gen,name)]:raise RuntimeError(f'checkpoint hash mismatch {gen}/{name}')
        with torch.serialization.safe_globals([np._core.multiarray._reconstruct,np.ndarray,np.dtype,np.dtypes.UInt32DType]):d=torch.load(p,map_location='cpu',weights_only=True)
        state=d['state_dict'];s=hashlib.sha256()
        for k,val in sorted(state.items()):s.update(k.encode());s.update(val.contiguous().numpy().tobytes())
        out['checkpoints'].append({'generator':gen,'file':name,'size_bytes':p.stat().st_size,'sha256':h,'optimizer_update':d['optimizer_update'],'epoch_index_zero_based':d['epoch'],'optimizer_steps':sorted({float(z['step']) for z in d['optimizer']['state'].values() if 'step' in z}),'scheduler':d['scheduler'],'state_hash_matches':s.hexdigest()==d['current_state_sha256'],'state_hash':s.hexdigest(),'parameters_finite':all(bool(torch.isfinite(z).all()) for z in state.values()),'best_epoch':d['best_epoch'],'best_epe':d['best_synthetic_validation_epe'],'metadata':{k:d[k] for k in ['config_sha256','data_manifest_sha256','initial_state_sha256','flow_unit','seed']}})
        del d,state;gc.collect()
out['paired_all_8910_sample_orders_match']=all(a['rollout_sample_ids']==b['rollout_sample_ids'] for a,b in zip(streams['eventgenesis'],streams['v2e'])) and len(streams['eventgenesis'])==len(streams['v2e'])==8910
out['campaign_exit']=(RUN/'training_retry3.exit_code.txt').read_text().strip()
out['training_processes']=cmd('pgrep','-af','[t]rain_steflow_stage4.py')
out['GPU']=cmd('nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv')

DATA=Path('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/20261016-stage4-flowdt1-seed0-v4/steflow_30ms')
out['zero_flow_baseline']=[]
for p in sorted((DATA/'eventgenesis/validation').glob('*.npz')):
    with np.load(p) as d:
        flow=d['flow_gt'].astype(np.float64);valid=d['flow_valid'].astype(bool);mag=np.hypot(flow[:,0],flow[:,1]);vals=[x[m] for x,m in zip(mag,valid)]
        out['zero_flow_baseline'].append({'rollout':p.stem,'stratum_index':int(p.stem[1]),'intervals':len(vals),'mean_EPE_frame_then_rollout':float(np.mean([x.mean() for x in vals])),'pooled_valid_pixels':sum(len(x) for x in vals),'zero_or_tiny_flow_fraction':float(np.mean(np.concatenate(vals)<=0.1)),'GT_magnitude_median':float(np.median(np.concatenate(vals))),'GT_magnitude_p95':float(np.percentile(np.concatenate(vals),95)),'framewise_zero_flow_EPE':[float(x.mean()) for x in vals]})
out['zero_flow_EPE_equal_rollouts']=float(np.mean([z['mean_EPE_frame_then_rollout'] for z in out['zero_flow_baseline']]))
print(json.dumps(out,ensure_ascii=False))
