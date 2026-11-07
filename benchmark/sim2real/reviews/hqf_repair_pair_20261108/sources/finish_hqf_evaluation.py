"""Bounded CPU-only completion audit and lightweight evidence export."""
import datetime,hashlib,json,os,pathlib,subprocess,sys,time,traceback
root=pathlib.Path(sys.argv[1]);project=root.parents[2]
def read(p):return json.loads(p.read_text())
def record(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=h.hexdigest())
try:
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    deadline=time.monotonic()+2*3600
    while not (root/'exit.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Evaluation did not finish within CPU watcher bound')
        time.sleep(10)
    completed=read(root/'completion.json');exit_record=read(root/'exit.json')
    assert completed['status']=='complete' and completed['exit_code']==0 and exit_record['status']=='released_complete'
    with (root/'range_audit.log').open('x') as log:
        proc=subprocess.run([sys.executable,str(root/'diagnostic_sources/audit_prediction_range.py'),'--root',str(root)],stdout=log,stderr=subprocess.STDOUT,timeout=600)
    assert proc.returncode==0,'Read-only prediction range audit failed'
    score=read(root/'results/score.json');assert len(score['per_arm'])==2 and score['real_scores_computed']
    compact={k:v for k,v in score.items() if k!='per_frame'}
    p=read(root/'protocol/hqf_repair_pair_protocol.json')
    source_manifest=read(root/'sources/source_manifest.json')
    assert all(record(root/'sources'/name)['sha256']==digest for name,digest in source_manifest.items())
    old=project/'runs/sim2real/20261103-stage9-direct-v1'
    out=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completion=completed,exit=exit_record,
        score=compact,protocol=p,authorization=read(root/'authorization.json'),runtime_preflight=read(root/'runtime_preflight/completion.json'),
        prediction_range_audit=read(root/'prediction_range_audit.json'),
        historical_selection={m:read(old/'training/reconstruction/seed0'/m/'synthetic_best.json') for m in ['eventgenesis','v2e']},
        source_manifest=source_manifest,bridge_preservation_checks=read(root/'sources/bridge_preservation_checks.json'),
        server_files={n:record(root/n) for n in ['results/score.json','results/per_sequence_scores.csv','results/run_manifest.json','results/inference_manifest_frozen.json','protocol/hqf_repair_pair_protocol.json','prediction_range_audit.json']},
        diagnostic_sources={f.name:record(f) for f in (root/'diagnostic_sources').glob('*.py')},
        earlier_preflight_failure='Attempt1 rejected a malformed SHA pin before any real inference or GPU reservation. Attempt1 files and failed preparation log retained; native sigmoid fixture had passed. Attempt2 pins both machine-verified64-character digests.',
        training_updates_during_this_evaluation=0,long_training_started=False)
    (root/'hqf_result_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    (root/'postprocess_completion.json').write_text(json.dumps(dict(status='complete',exit_code=0,summary=record(root/'hqf_result_summary.json'),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
    print(json.dumps({rid:v['equal_sequence_mean'] for rid,v in compact['per_arm'].items()}),flush=True)
except BaseException:
    (root/'postprocess_failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
    raise
