"""Run after the current job finishes; fall back to the original backend on failure."""
import datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
decision=root/'V100_BACKEND_DECISION.json'
assert not decision.exists(),'A published backend decision is immutable'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'deployment_manifest.json').read_text())
for relative,expected in manifest['sha256'].items():assert sha(root/relative)==expected,relative
result=dict(seed=1,all_methods=['eventgenesis','v2e','vid2e','film_esim','rife_esim'],
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            selected_backend='original_pytorch',real_data_used=False,
            deployment_manifest_sha256=sha(root/'deployment_manifest.json'))
try:
    import torch
    assert 'V100' in torch.cuda.get_device_name()
    assert torch.cuda.mem_get_info()[0] >= 26000 * 1024**2,'Insufficient isolated full-batch memory'
    assert json.loads((root/'operator_ordered_results.json').read_text())['passed']
    assert json.loads((root/'model_ordered_results.json').read_text())['passed']
    with (root/'v100_fullbatch_audit.log').open('x') as out:
        run=subprocess.run([sys.executable,'-u',str(root/'audit_model_v100.py')],stdout=out,stderr=subprocess.STDOUT,timeout=900)
    result['audit_exit_code']=run.returncode
    report_path=root/'model_ordered_v100_results.json'
    if report_path.exists():
        report=json.loads(report_path.read_text());result['full_batch_report_sha256']=sha(report_path)
        result['full_batch_report']=str(report_path)
        if run.returncode==0 and report['passed'] and report['batch']==8 and 'V100' in report['device']:
            result['selected_backend']='reference_forward_ordered_cuda_backward'
            result['whole_step_speedup']=report['whole_step_speedup']
except Exception:result['audit_error']=traceback.format_exc()
result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
temporary=decision.with_suffix('.prepared')
with temporary.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
os.link(temporary,decision)  # Atomic publication; refuses to overwrite an existing decision.
temporary.unlink()
print(json.dumps(result),flush=True)
# A failed acceleration audit must not block or mutate the existing experiment.
# The immutable common decision explicitly selects the original backend instead.
