"""Read only completed bounded runs; hash all referenced evidence, never train."""
import base64,datetime,gzip,hashlib,json,math,os,time
from pathlib import Path

P=Path('/home/zhaoyuhan/Eventbased_WAM')
RUNS=P/'runs/sim2real'
OUT=RUNS/'20261108-recon-photometry-detail-results-v3'
METHODS=('eventgenesis','v2e');ARMS=('control','photometric','detail')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(path.read_text())
def record(path):return dict(path=str(path),size_bytes=path.stat().st_size,sha256=sha(path))
def save(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    OUT.mkdir(exist_ok=False);roots={m:RUNS/('20261108-recon-photometry-detail-'+m+'-v3') for m in METHODS}
    deadline=time.monotonic()+2*3600
    while True:
        for root in roots.values():
            if (root/'failure.json').exists():raise RuntimeError('Own bounded study failed: '+str(root))
        if all((root/'resource_release.json').exists() for root in roots.values()):break
        if time.monotonic()>deadline:raise TimeoutError('Bounded collection wait expired')
        time.sleep(10)
    summary=dict(methods={},context=read(RUNS/'20261108-recon-photometry-context-status.json'),failure_evidence={},external_checkpoints=[],artifact_index=[],
        frozen_simulator_core='62da4f35ce3c5c96f04268bd1687a27657af9238',real_scoring_enabled=False,real_data_used=False,
        collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for m,root in roots.items():
        release=read(root/'resource_release.json');assert release['status']=='released'
        assert read(root/'completion.json')['status']=='complete'
        manifest=read(root/'sources/source_manifest.json')
        for name,digest in manifest.items():assert sha(root/'sources'/name)==digest
        data=dict(completion=read(root/'study/completion.json'),smoke=read(root/'smoke/completion.json'),release=release,
            cross_generator_smoke_pairing=read(root/'cross_generator_smoke_pairing.json'),source_manifest=manifest,
            protocol=read(root/'study/protocol.json'),training_evidence={},checkpoint_evidence={})
        for arm in ARMS:
            run=root/'study'/arm;log=run/'training_metrics.jsonl';rows=[json.loads(x) for x in log.open()]
            assert len(rows)==300 and [r['optimizer_update'] for r in rows]==list(range(4801,5101))
            assert all(r['finite_gradient_tensors']==30 and r['tc_calls']==76 and math.isfinite(r['gradient_l2']) and r['gradient_l2']>0 for r in rows)
            assert all(math.isfinite(v) for r in rows for v in r['losses'].values())
            data['training_evidence'][arm]=dict(rows=300,optimizer_updates=[r['optimizer_update'] for r in rows],
                all_losses_finite=True,all_gradients_finite=True,gradient_norm_positive=True,finite_gradient_tensors=[30],tc_calls=[76],
                finite_gradient_check_source='per-update bound-source asserts torch.isfinite on all gradient tensors; log records finite tensor count',
                log_sha256=sha(log),log_path=str(log),log_size_bytes=log.stat().st_size)
            data['checkpoint_evidence'][arm]=read(run/'update_005100.json')
            assert data['checkpoint_evidence'][arm]['independent_model_optimizer_rebuild'] and data['checkpoint_evidence'][arm]['optimizer_state_equal']
            summary['external_checkpoints'].append(record(run/'update_005100.pt'))
        summary['methods'][m]=data
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.py','.md','.txt','.log'):
                summary['artifact_index'].append(record(path))
        for version in (1,2):
            old=RUNS/('20261108-recon-photometry-detail-'+m+'-v'+str(version))
            item={name:read(old/name) for name in ('failure.json','smoke_exit.json','resource_release.json') if (old/name).exists()}
            item['smoke_tail']=(old/'logs/smoke.log').read_text()[-5000:]
            item['source_manifest']=read(old/'sources/source_manifest.json')
            summary['failure_evidence'][m+'_v'+str(version)]=item
            for path in old.rglob('*'):
                if path.is_file() and path.suffix in ('.json','.py','.md','.log'):summary['artifact_index'].append(record(path))
    labels={r['label_sequence_sha256'] for d in summary['methods'].values() for r in d['completion']['arms']}
    assert len(labels)==1,'Cross-generator training label/augmentation mismatch'
    summary['cross_generator_training_labels_equal']=True
    save(OUT/'result_summary.json',summary)
    encoded=base64.b64encode(gzip.compress((OUT/'result_summary.json').read_bytes())).decode()
    (OUT/'result_summary.b64').write_text(encoded)
    save(OUT/'completion.json',dict(status='complete',result=record(OUT/'result_summary.json'),base64_chars=len(encoded),real_data_used=False))
    print('COLLECTION_COMPLETE',len(encoded),sha(OUT/'result_summary.json'),flush=True)

if __name__=='__main__':main()
