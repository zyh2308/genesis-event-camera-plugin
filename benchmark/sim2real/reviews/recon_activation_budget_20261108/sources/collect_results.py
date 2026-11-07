"""Collect completed bounded diagnostics without starting any computation on GPU."""
import datetime,hashlib,json,pathlib
P=pathlib.Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261103-stage9-direct-v1'
roots={m:R.parent/n for m,n in [('eventgenesis','20261108-recon-activation-budget-eventgenesis-v3'),('v2e','20261108-recon-activation-budget-v2e-v4')]}
def read(p):return json.loads(p.read_text())
def rec(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=h.hexdigest())
out=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),frozen_core='62da4f35ce3c5c96f04268bd1687a27657af9238',workers={})
arms=[]
for m,root in roots.items():
    complete=read(root/'study/completion.json');manager=read(root/'completion.json');release=read(root/'resource_release.json')
    assert complete['status']==manager['status']=='complete' and release['status']=='released'
    worker=dict(root=str(root),completion=complete,manager_completion=manager,resource_release=release,
        source_manifest=read(root/'sources/source_manifest.json'),study_protocol=read(root/'study/protocol.json'),design=read(root/'study_design.json'),
        submitted=read(root/'submitted.json'),smoke=read(root/'smoke/completion.json'),source_files={f.name:rec(f) for f in (root/'sources').glob('*') if f.is_file()},
        raw_result=rec(root/'study/completion.json'),training_logs={},checkpoint_receipts={})
    adapter=R/'adapters'/m/'reconstruction'
    worker['input_source_bindings']={str(f):rec(f) for f in [R/'sources/training_recipe.json',R/'sources/benchmark/sim2real/scripts/train_e2vid_stage9.py',adapter/'adapter_manifest.json',adapter/(m+'_train_files.txt'),adapter/(m+'_validation_files.txt')]}
    model=P/'external/sources/downstream/event_cnn_minimal'
    worker['downstream_code_bindings']={str(f):rec(f) for f in [*sorted((model/'model').glob('*.py')),model/'trainer/trainer.py',model/'data_loader/dataset.py']}
    for a in complete['arms']:
        arm=a['arm'];arms.append(a);path=root/'study'/arm/'training_metrics.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        assert [r['update'] for r in rows]==list(range(1,601)) and all(r['tc_calls']==76 and r['finite_gradient_tensors']==30 and r['gradient_l2']>0 for r in rows)
        worker['training_logs'][arm]=dict(file=rec(path),rows=len(rows),all_updates_contiguous=True,all_finite_gradient_tensor_counts=[30],all_tc_calls=[76],
            first=rows[0],last=rows[-1],loss_curve_every20=[{k:r[k] for k in ['update','losses','gradient_l2','elapsed_seconds']} for r in rows if r['update']%20==0])
        worker['checkpoint_receipts'][arm]=[read(root/'study'/arm/f'checkpoint_{n:06d}.json') for n in [20,300,600]]
    formal=R.parent/'20261107-recon-sigmoid-formal-v2/training/seed0'/m
    logs=[json.loads(s) for s in (formal/'training_metrics.jsonl').read_text().splitlines()]
    worker['formal_validation_history']=[{k:r[k] for k in ['epoch','optimizer_update','lpips_vgg_equal_rollout_mean']} for r in logs if r.get('event')=='synthetic_validation']
    selected=read(formal/'synthetic_best.json');reference=next(r for r in complete['references'] if r['name']=='sigmoid_4800')
    delta=reference['validation']['all200']['lpips_vgg']-selected['synthetic_validation_lpips_vgg']
    assert abs(delta)<1e-5,('metric mismatch',m,delta)
    worker['formal_metric_crosscheck']=dict(full200_native_vgg=selected['synthetic_validation_lpips_vgg'],reevaluated_clipped_vgg=reference['validation']['all200']['lpips_vgg'],difference=delta,passed=True)
    out['workers'][m]=worker
assert len({a['initial_state_sha256'] for a in arms})==1 and len({a['label_sequence_sha256'] for a in arms})==1
for m in roots:assert len({a['event_sequence_sha256'] for a in arms if a['method']==m})==1
out['pairing']=dict(passed=True,all_four_initialization_and_actual_label_augmentation_sequence_equal=True,events_equal_only_within_generator=True)
out['earlier_attempts']={}
smoke_updates=sum(sum(x['optimizer_updates'] for x in w['smoke']['results']) for w in out['workers'].values())
for m in roots:
    attempts=[]
    for version in [1,2,3]:
        p=R.parent/f'20261108-recon-activation-budget-{m}-v{version}'
        if p==roots[m] or not p.exists():continue
        if (p/'smoke/completion.json').exists():smoke_updates+=sum(x['optimizer_updates'] for x in read(p/'smoke/completion.json')['results'])
        attempts.append(dict(path=str(p),failure=read(p/'failure.json') if (p/'failure.json').exists() else None,manager_log=rec(p/'manager.log'),release=read(p/'resource_release.json') if (p/'resource_release.json').exists() else None))
    out['earlier_attempts'][m]=attempts
out['budget_accounting']=dict(main_four_arm_optimizer_updates=2400,all_smoke_optimizer_updates=smoke_updates,additional_optimizer_updates_for_late_reference=0,formal_long_training_started=False,real_scoring_performed=False)
target=R.parent/'20261108-recon-activation-budget-summary.json';target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(summary=rec(target),pairing=out['pairing'])),flush=True)
