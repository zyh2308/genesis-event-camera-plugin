from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import ast,gc,hashlib,json,subprocess,sys
import h5py,numpy as np,torch
torch.set_num_threads(2)
R=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real');B=R/'benchmark/sim2real'
D=Path('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/20261016-stage4-flowdt1-seed0-v4/steflow_30ms');M=json.loads((D/'training_bank_manifest.json').read_text());BANK=Path('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v1_torch_cuda_32x2s')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1048576),b''):h.update(c)
    return h.hexdigest()
out={'shared_bank':{'size_bytes':(BANK/'shared_trajectory.h5').stat().st_size,'sha_match':sha(BANK/'shared_trajectory.h5')==M['shared_bank_sha256']},'split_sha_match':sha(BANK/'registered_split.json')==M['split_sha256'],'adapter_files':[],'paired_arrays':[]}
entries={(x['generator'],x['rollout_id']):x for x in M['entries']}
for e in M['entries']:
    p=Path(e['path']);out['adapter_files'].append({'generator':e['generator'],'rollout':e['rollout_id'],'sha_match':sha(p)==e['sha256']})
with h5py.File(BANK/'shared_trajectory.h5','r') as h:
    out['rollouts']=[{'id':eid,'split':str(h[f'episodes/{eid}'].attrs['split']),'stratum':str(h[f'episodes/{eid}'].attrs['stratum']),'frames':len(h[f'episodes/{eid}/frame_timestamps_s']),'duration_s':float(h[f'episodes/{eid}/frame_timestamps_s'][-1]),'scene_seed':int(h[f'episodes/{eid}'].attrs['scene_seed'])} for eid in h['episodes']]
    for eid in sorted(h['episodes']):
        with np.load(entries[('eventgenesis',eid)]['path']) as a,np.load(entries[('v2e',eid)]['path']) as b:
            keys=['gray_prev','gray_next','flow_gt','flow_valid','event_start_s','event_end_s','sample_ids','camera_intrinsics_K_px']
            equal={k:np.array_equal(a[k],b[k]) for k in keys}
            ep=h[f'episodes/{eid}'];ix=np.arange(0,198,3)
            physical_match=bool(np.array_equal(a['flow_gt'],ep['flow_forward_30ms_px'][:]) and np.array_equal(a['flow_valid'],ep['flow_valid_30ms'][:]))
            endpoint_match=bool(np.array_equal(a['event_start_s'],ep['frame_timestamps_s'][:][ix]) and np.array_equal(a['event_end_s'],ep['frame_timestamps_s'][:][ix+3]))
            out['paired_arrays'].append({'rollout':eid,'shared_array_equality':equal,'physical_labels_match':physical_match,'endpoint_match':endpoint_match,'train_event_shape':list(a['events'].shape),'all_window_durations_30ms':bool(np.allclose(a['event_end_s']-a['event_start_s'],.03,atol=1e-7,rtol=0))})

# Read-only audit of the first complete epoch and continued training progress.
root=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/flow_dt1_seed0');streams={}
out['progress']={}
for gen in ['eventgenesis','v2e']:
    rows=[]
    for line in (root/gen/'training_metrics.jsonl').read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    upd=[r for r in rows if 'optimizer_update' in r];val=[r for r in rows if r.get('event')=='synthetic_validation'];streams[gen]={r['optimizer_update']:r for r in upd}
    epoch1=[r for r in upd if r['epoch']==1];ids=[s for r in epoch1 for s in r['rollout_sample_ids']]
    out['progress'][gen]={'latest':upd[-1],'optimizer_updates_contiguous':[r['optimizer_update'] for r in upd]==list(range(1,len(upd)+1)),'all_losses_and_gradients_finite':all(np.isfinite([r[k] for k in ['loss','photometric_loss','smoothness_loss','gradient_l2_norm']]).all() for r in upd),'epoch1_optimizer_updates':len(epoch1),'epoch1_unique_training_sample_ids':len(set(ids)),'epoch1_total_training_sample_ids':len(ids),'epoch1_has_validation_ids':any(s.split('/')[0].endswith(('_06','_07')) for s in ids),'completed_validation_epochs':[r['epoch'] for r in val],'validation_equal_weight_rollouts':[len(r['rollout_equal_weight_epe_px_per_30ms']) for r in val],'validation_metric_values_finite':all(np.isfinite(r['mean_epe_px_per_30ms']) for r in val),'synthetic_best_checkpoint_exists':(root/gen/'synthetic_best.pt').is_file()}
common=sorted(set(streams['eventgenesis'])&set(streams['v2e']))
out['paired_sample_order_all_common_updates']={'updates_checked':len(common),'all_equal':all(streams['eventgenesis'][i]['rollout_sample_ids']==streams['v2e'][i]['rollout_sample_ids'] for i in common)}
out['checked_at_JST']=datetime.now(timezone.utc).astimezone(ZoneInfo('Asia/Tokyo')).isoformat()

# Demonstrate the RNG API requirement without GPU use or changing live processes.
original=torch.get_rng_state()
try:
    torch.set_rng_state(torch.empty(original.shape,dtype=torch.uint8,device='meta'))
    out['nonCPU_RNG_state_rejected']=False
except Exception as exc:out['nonCPU_RNG_state_rejected']={'type':type(exc).__name__,'message':str(exc)}
finally:torch.set_rng_state(original)
out['source_files']=[]
bundle=json.loads((B/'manifests/downstream_source_bundle.json').read_text())
for repo in bundle['repositories']:
    if repo['repo']=='STE-FlowNet':out['source_files'].append({'repo':repo['repo'],'files':len(repo['files']),'mismatches':[f['path'] for f in repo['files'] if sha(Path(repo['server_path'])/f['path'])!=f['sha256']]})
print(json.dumps(out,ensure_ascii=False))
