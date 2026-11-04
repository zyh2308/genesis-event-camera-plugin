"""One immutable backend decision shared by every generator in seed 1."""
import hashlib,json,os,sys
from pathlib import Path
root=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
args=sys.argv[1:]
assert args[args.index('--task')+1]=='flow'
assert args[args.index('--seed')+1]=='1','Only the complete, untouched seed-1 group may switch'
decision=json.loads((root/'V100_BACKEND_DECISION.json').read_text())
assert decision['seed']==1 and set(decision['all_methods'])=={'eventgenesis','v2e','vid2e','film_esim','rife_esim'}
assert decision['deployment_manifest_sha256']==sha(root/'deployment_manifest.json')
manifest=json.loads((root/'deployment_manifest.json').read_text())
for relative,expected in manifest['sha256'].items():assert sha(root/relative)==expected,relative
if decision['selected_backend']=='reference_forward_ordered_cuda_backward':
    report_path=Path(decision['full_batch_report']);assert sha(report_path)==decision['full_batch_report_sha256']
    report=json.loads(report_path.read_text());assert report['passed'] and report['batch']==8 and 'V100' in report['device']
    runner=root/'run_training_job_ordered.py'
else:
    assert decision['selected_backend']=='original_pytorch'
    runner=root.parent/'sources/run_training_job.py'
print('SEED1_COMMON_BACKEND',decision['selected_backend'],flush=True)
os.execv(sys.executable,[sys.executable,'-u',str(runner)]+args)
