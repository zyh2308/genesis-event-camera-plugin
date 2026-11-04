"""Prepare isolated training copies and an atomic queue-only, paired-seed rollout."""
import ast,datetime,hashlib,json,os
from pathlib import Path
root=Path(__file__).resolve().parent;campaign=root.parent;scripts=campaign/'sources/benchmark/sim2real/scripts'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
operator=json.loads((root/'operator_ordered_results.json').read_text())
model=json.loads((root/'model_ordered_results.json').read_text())
assert operator['passed'] and model['passed']
# In actual contiguous feature shapes, the standalone operator must be bitwise identical.
actual=[row for row in operator['tests'] if row['shape'][0]==8]
assert len(actual)==4 and all(row[k]['bitwise_equal'] for row in actual for k in ['forward','grad1','grad2'])
trainer=(scripts/'train_steflow_stage9.py').read_text()
old='corr_module.spatial_correlation_sample = pure_torch_spatial_correlation'
assert trainer.count(old)==1
trainer=trainer.replace(old,'from ordered_adapter import spatial_correlation_sample as ordered_correlation\n    corr_module.spatial_correlation_sample = ordered_correlation')
trainer=trainer.replace('sys.path.insert(0, str(Path(__file__).resolve().parent))',f'sys.path.insert(0, {str(scripts)!r})')
trainer=trainer.replace('sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compat"))',f'sys.path.insert(0, {str(scripts.parent/"compat")!r})')
trainer=trainer.replace('"memory-bounded PyTorch implementation, math-equivalent to locked upstream op"','"reference PyTorch forward / ordered CUDA backward; seed1 paired group"')
trainer=trainer.replace('"current_state_sha256": current_hash,','"current_state_sha256": current_hash,\n        "correlation_backend": "reference_forward_ordered_cuda_backward",')
trainer=trainer.replace('saved = torch.load(args.resume, map_location="cpu", weights_only=False)','saved = torch.load(args.resume, map_location="cpu", weights_only=False)\n        if saved.get("correlation_backend") != "reference_forward_ordered_cuda_backward":\n            raise ValueError("Refusing a mixed-backend training resume")')
runner=(campaign/'sources/run_training_job.py').read_text()
old="str(S/'benchmark/sim2real/scripts/train_steflow_stage9.py')"
assert runner.count(old)==1
runner=runner.replace(old,'str(Path(__file__).resolve().parent/"train_steflow_ordered.py")')
runner=runner.replace("'real_scoring_performed':False","'real_scoring_performed':False,'correlation_backend':'reference_forward_ordered_cuda_backward'")
target=(root/'audit_model_ordered.py').read_text().replace('range(2)','range(8)').replace('batch=2,formal_batch=8','batch=8,formal_batch=8').replace('model_ordered_results.json','model_ordered_v100_results.json').replace('Diagnostic batch 2 on spare 24GB card; formal full-batch preflight remains mandatory on V100.','Target V100 full formal batch 8 correctness and throughput audit.').replace('sample_indices=[0,1]','sample_indices=list(range(8))')
for name,text in [('train_steflow_ordered.py',trainer),('run_training_job_ordered.py',runner),('audit_model_v100.py',target)]:
    ast.parse(text)
    with (root/name).open('x') as f:f.write(text)
files=['ordered_adapter.py','ordered_backward.cpp','ordered_backward.cu','build_ordered/ordered_correlation_backward.so',
       'audit_operator_ordered.py','audit_model_ordered.py','audit_model_v100.py','train_steflow_ordered.py',
       'run_training_job_ordered.py','choose_seed1_backend.py','target_v100_gate.py',
       'operator_ordered_results.json','model_ordered_results.json']
manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256={f:sha(root/f) for f in files},
              original_trainer_sha256=sha(scripts/'train_steflow_stage9.py'),original_runner_sha256=sha(campaign/'sources/run_training_job.py'),
              scope='all five pending flow seed1 jobs only; seed0/seed2 and active jobs unchanged')
with (root/'deployment_manifest.json').open('x') as f:f.write(json.dumps(manifest,indent=2)+'\n')
# Build all replacement queue files before making any visible.
pending=[]
for lane in [3,4]:
    q=campaign/('gpu'+str(lane))/'queue'
    for path in q.glob('*flow*seed1.json'):
        spec=json.loads(path.read_text());method=spec['argv'][spec['argv'].index('--method')+1]
        assert not (campaign/'training/flow/seed1'/method/'preflight20.stdout.log').exists(),method
        assert str(campaign/'sources/run_training_job.py') in spec['argv']
        pending.append((lane,path,spec,method))
assert len(pending)==5 and len({x[3] for x in pending})==5
gate_job=dict(job_id='25_correlation_v100_fullbatch_gate',phase='synthetic_operator_validation',
              argv=[str(Path('/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python')),'-u',str(root/'target_v100_gate.py')],
              output_dir=str(root/'target_gate_job'),ready_files=[str(root/'QUEUE_PATCH_COMPLETE.json')],
              source_sha256={str(root/f):sha(root/f) for f in ['target_v100_gate.py','deployment_manifest.json']},env={'PYTHONDONTWRITEBYTECODE':'1'})
backup=root/'queue_backups';backup.mkdir(exist_ok=False)
updated=[]
for lane,path,spec,method in pending:
    (backup/('gpu'+str(lane)+'_'+path.name)).write_bytes(path.read_bytes())
    spec['argv']=[str(root/'choose_seed1_backend.py') if arg==str(campaign/'sources/run_training_job.py') else arg for arg in spec['argv']]
    spec.setdefault('ready_files',[]).extend([str(root/'QUEUE_PATCH_COMPLETE.json'),str(root/'V100_BACKEND_DECISION.json')])
    spec.setdefault('source_sha256',{}).update({str(root/f):sha(root/f) for f in ['choose_seed1_backend.py','deployment_manifest.json']})
    spec['correlation_rollout_scope']='common seed1 backend decision after fullbatch target validation'
    temp=path.with_suffix('.prepared');temp.write_text(json.dumps(spec,indent=2)+'\n');updated.append((temp,path))
gate_path=campaign/'gpu3/queue/25_correlation_v100_fullbatch_gate.json'
assert not gate_path.exists()
for temp,path in updated:os.replace(temp,path)
gate_temporary=gate_path.with_suffix('.prepared');gate_temporary.write_text(json.dumps(gate_job,indent=2)+'\n')
os.replace(gate_temporary,gate_path)
receipt=dict(status='deployed_pending_target_gate',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             manifest_sha256=sha(root/'deployment_manifest.json'),modified_pending_jobs=[str(x[1]) for x in pending],
             gate_job=str(gate_path),active_jobs_modified=False,original_sources_modified=False,
             fallback='target gate failure selects original PyTorch for every seed1 method')
ready=root/'QUEUE_PATCH_COMPLETE.json';temporary=ready.with_suffix('.prepared')
with temporary.open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
os.link(temporary,ready);temporary.unlink()
print(json.dumps(receipt),flush=True)
