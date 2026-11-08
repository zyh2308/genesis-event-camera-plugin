"""Read only completed bounded runs; hash all referenced evidence, never train."""
import base64,datetime,gzip,hashlib,json,math,os,time
from pathlib import Path

P=Path('/home/zhaoyuhan/Eventbased_WAM')
RUNS=P/'runs/sim2real'
OUT=RUNS/'20261108-recon-l1-quarter-results-v1'
METHODS=('eventgenesis','v2e');ARMS=('control','photometric')
PARENT=RUNS/'20261108-recon-l1-quarter-paired-gpu2-v1'
PHASES=['smoke_eventgenesis','smoke_v2e','study_eventgenesis','study_v2e']

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
    OUT.mkdir(exist_ok=False);roots={m:PARENT/m for m in METHODS}
    deadline=time.monotonic()+2*3600
    while True:
        if (PARENT/'failure.json').exists():raise RuntimeError('Own bounded paired study failed: '+str(PARENT))
        if (PARENT/'resource_release.json').exists():break
        if time.monotonic()>deadline:raise TimeoutError('Bounded collection wait expired')
        time.sleep(10)
    summary=dict(methods={},context=dict(registered_trial='synthetic_only_l1_quarter_4800_to_5100',historical_failure_is_reference_only=True,no_real_metrics_loaded=True),failure_evidence={'prior_l1_weight1_trial':dict(path=str(RUNS/'20261108-recon-photometry-detail-results-v3/result_summary.json'),sha256='760a9890b107a52914ec51c9d01fa0e2ea485d2cd2a7c0c167524a5a98c5e830',scope='Historical failed adoption evidence; reference only, not current training evidence')},external_checkpoints=[],artifact_index=[],
        frozen_simulator_core='62da4f35ce3c5c96f04268bd1687a27657af9238',real_scoring_enabled=False,real_data_used=False,
        collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    release=read(PARENT/'resource_release.json');assert release['status']=='released'
    assert release['gpu_index']==2 and release['current_job'] is None and release['completed']==PHASES
    assert release['gpu_uuid']=='GPU-a02affe0-7598-579e-ca39-f551f35ad6e9'
    assert release['run_id']==PARENT.name and release['output_path']==str(PARENT)
    parent_completion=read(PARENT/'completion.json')
    assert parent_completion['status']=='complete' and parent_completion['completed']==PHASES
    assert parent_completion['main_optimizer_updates']==1200 and parent_completion['smoke_optimizer_updates']==4
    assert parent_completion['real_data_used'] is False and parent_completion['real_scoring_enabled'] is False
    parent_protocol=read(PARENT/'protocol.json')
    assert parent_protocol['arms']==list(ARMS) and parent_protocol['auxiliary_l1_weight']==.25
    manifest=read(PARENT/'sources/source_manifest.json')
    for name,digest in manifest.items():assert sha(PARENT/'sources'/name)==digest
    registration=read(PARENT/'sources/registered_protocol.json')
    assert all(parent_protocol[key]==value for key,value in registration.items())
    assert parent_protocol['source_manifest_sha256']==sha(PARENT/'sources/source_manifest.json')
    assert parent_protocol['protocol_sha256']==sha(PARENT/'sources/PROTOCOL.md')
    assert parent_protocol['worker_sequence']==PHASES
    phase_exits={phase:read(PARENT/(phase+'_exit.json')) for phase in PHASES}
    assert all(receipt['exit_code']==0 and receipt['phase']==phase and receipt['status']=='exited' for phase,receipt in phase_exits.items())
    pairing=read(PARENT/'cross_generator_smoke_pairing.json')
    assert pairing['passed'] is True and pairing['methods']==list(METHODS)
    assert set(pairing['labels'])==set(METHODS)
    assert pairing['labels']['eventgenesis']==pairing['labels']['v2e']
    assert pairing['labels_sha256']==hashlib.sha256(json.dumps(pairing['labels']['eventgenesis']).encode()).hexdigest()
    assert pairing['completed_smoke_updates']==4 and pairing['real_data_used'] is False
    summary.update(parent_completion=parent_completion,parent_release=release,parent_protocol=parent_protocol,
        cross_generator_smoke_pairing=pairing,parent_root=str(PARENT),parent_registration=registration,parent_phase_exits=phase_exits)
    for m,root in roots.items():
        data=dict(completion=read(root/'study/completion.json'),smoke=read(root/'smoke/completion.json'),release=release,
            cross_generator_smoke_pairing=pairing,source_manifest=manifest,
            protocol=read(root/'study/protocol.json'),training_metrics={},training_evidence={},checkpoint_evidence={})
        assert data['completion']['status']=='complete' and data['completion']['method']==m
        assert data['completion']['pairing_passed'] is True and data['completion']['real_data_used'] is False
        assert data['smoke']['status']=='complete' and data['smoke']['real_data_used'] is False
        assert data['protocol']['arms']==list(ARMS) and data['protocol']['arm_count']==2
        assert data['protocol']['auxiliary_l1_weight']==.25 and data['protocol']['start_update']==4800
        assert data['protocol']['additional_updates_per_arm']==300 and data['protocol']['real_data_used'] is False
        assert data['protocol']['real_scoring_enabled'] is False
        assert data['protocol']['script_sha256']==sha(PARENT/'sources/l1_quarter_diagnostic.py')
        assert data['protocol']['protocol_sha256']==sha(PARENT/'sources/PROTOCOL.md')
        assert data['protocol']['protocol_sha256']=='7f3b400205be412b6e281d6c7e4aa9c4a005b82c477f1c574bfddf41b3b63f68'
        assert data['cross_generator_smoke_pairing']['passed'] is True
        assert pairing['labels'][m]==[r['update']['labels'] for r in data['smoke']['results']]
        completion_arms={r['arm']:r for r in data['completion']['arms']}
        assert set(completion_arms)==set(ARMS)
        for arm in ARMS:
            run=root/'study'/arm;log=run/'training_metrics.jsonl';rows=[json.loads(x) for x in log.open()]
            assert len(rows)==300 and [r['optimizer_update'] for r in rows]==list(range(4801,5101))
            assert all(r['finite_gradient_tensors']==30 and r['tc_calls']==76 and math.isfinite(r['gradient_l2']) and r['gradient_l2']>0 for r in rows)
            assert all(math.isfinite(v) for r in rows for v in r['losses'].values())
            for row in rows:
                assert all(isinstance(row[key],list) and len(row[key])==2 for key in ('labels','events'))
                for key in ('labels','events'):
                    assert all(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value) for value in row[key])
            recomputed={key:hashlib.sha256(json.dumps([r[key] for r in rows]).encode()).hexdigest() for key in ('labels','events')}
            assert recomputed['labels']==completion_arms[arm]['label_sequence_sha256']
            assert recomputed['events']==completion_arms[arm]['event_sequence_sha256']
            data['training_metrics'][arm]=rows
            data['training_evidence'][arm]=dict(rows=300,optimizer_updates=[r['optimizer_update'] for r in rows],
                all_losses_finite=True,all_gradients_finite=True,gradient_norm_positive=True,finite_gradient_tensors=[30],tc_calls=[76],
                finite_gradient_check_source='per-update bound-source asserts torch.isfinite on all gradient tensors; log records finite tensor count',
                log_sha256=sha(log),log_path=str(log),log_size_bytes=log.stat().st_size)
            data['checkpoint_evidence'][arm]=read(run/'update_005100.json')
            assert data['checkpoint_evidence'][arm]['independent_model_optimizer_rebuild'] and data['checkpoint_evidence'][arm]['optimizer_state_equal']
            receipt=data['checkpoint_evidence'][arm]
            actual_checkpoint=record(run/'update_005100.pt')
            assert receipt['path']==actual_checkpoint['path'] and receipt['sha256']==actual_checkpoint['sha256'] and receipt['size_bytes']==actual_checkpoint['size_bytes']
            assert receipt['model_hash_match'] is True and receipt['rng_roundtrip_match'] is True
            assert receipt['optimizer_counters']==[5100]
            summary['external_checkpoints'].append(actual_checkpoint)
            for name,expected in (('label_sequence_sha256',recomputed['labels']),('event_sequence_sha256',recomputed['events'])):
                data['training_evidence'][arm][name]=expected
            data['training_evidence'][arm]['sequence_digest_recomputed']=True
        summary['methods'][m]=data
    for path in PARENT.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.jsonl','.py','.md','.txt','.log'):
            summary['artifact_index'].append(record(path))
    labels={r['label_sequence_sha256'] for d in summary['methods'].values() for r in d['completion']['arms']}
    assert len(labels)==1,'Cross-generator training label/augmentation mismatch'
    summary['cross_generator_training_labels_equal']=True
    for m,data in summary['methods'].items():
        for name in ('label_sequence_sha256','event_sequence_sha256'):
            assert len({row[name] for row in data['completion']['arms']})==1
    summary['artifact_index'].append(record(Path(__file__)))
    for name in ('analyze_results.py','PROTOCOL.md'):
        path=Path(__file__).with_name(name)
        if path.is_file():
            if name=='PROTOCOL.md':assert sha(path)=='7f3b400205be412b6e281d6c7e4aa9c4a005b82c477f1c574bfddf41b3b63f68'
            summary['artifact_index'].append(record(path))
    summary['trial']=dict(arms=list(ARMS),auxiliary_l1_weight=.25,start_update=4800,end_update=5100,additional_updates_per_arm=300,primary_gate_count=9)
    save(OUT/'result_summary.json',summary)
    encoded=base64.b64encode(gzip.compress((OUT/'result_summary.json').read_bytes())).decode()
    (OUT/'result_summary.b64').write_text(encoded)
    save(OUT/'completion.json',dict(status='complete',result=record(OUT/'result_summary.json'),base64_chars=len(encoded),real_data_used=False))
    print('COLLECTION_COMPLETE',len(encoded),sha(OUT/'result_summary.json'),flush=True)

if __name__=='__main__':main()
