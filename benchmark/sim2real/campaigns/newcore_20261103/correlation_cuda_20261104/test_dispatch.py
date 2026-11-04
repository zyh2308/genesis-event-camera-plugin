"""Exercise the actual dispatcher with filesystem fixtures and intercepted exec."""
import hashlib,json,runpy,shutil,sys,tempfile
from pathlib import Path
from unittest.mock import patch
source=Path(__file__).resolve().parent/'choose_seed1_backend.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
results=[]
for case in ['native','fallback','wrong_seed','corrupt_manifest_file','corrupt_report','incomplete_group','missing_decision']:
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)/'adapter';root.mkdir();shutil.copyfile(source,root/source.name)
        original=root.parent/'sources';original.mkdir();(original/'run_training_job.py').write_text('# never executed\n')
        (root/'run_training_job_ordered.py').write_text('# never executed\n')
        manifest={'sha256':{'run_training_job_ordered.py':sha(root/'run_training_job_ordered.py')}}
        (root/'deployment_manifest.json').write_text(json.dumps(manifest))
        report=root/'fullbatch.json';report.write_text(json.dumps({'passed':True,'batch':8,'device':'Tesla V100-SXM2-32GB'}))
        decision=dict(seed=1,all_methods=['eventgenesis','v2e','vid2e','film_esim','rife_esim'],deployment_manifest_sha256=sha(root/'deployment_manifest.json'),selected_backend='original_pytorch' if case=='fallback' else 'reference_forward_ordered_cuda_backward',full_batch_report=str(report),full_batch_report_sha256=sha(report))
        if case=='corrupt_manifest_file':(root/'run_training_job_ordered.py').write_text('# changed\n')
        if case=='corrupt_report':report.write_text('{}')
        if case=='incomplete_group':decision['all_methods'].pop()
        if case!='missing_decision':(root/'V100_BACKEND_DECISION.json').write_text(json.dumps(decision))
        args=[str(root/source.name),'--task','flow','--seed','0' if case=='wrong_seed' else '1','--method','eventgenesis','--phase','full']
        with patch.object(sys,'argv',args),patch('os.execv') as execute:
            rejected=False
            try:runpy.run_path(str(root/source.name),run_name='__main__')
            except (AssertionError,FileNotFoundError):rejected=True
            if case in ['native','fallback']:
                expected=root/'run_training_job_ordered.py' if case=='native' else original/'run_training_job.py'
                assert execute.call_count==1 and execute.call_args.args[1][2]==str(expected) and not rejected,case
            else:assert rejected and execute.call_count==0,case
        results.append({'case':case,'passed':True})
out=source.parent/'dispatch_test_results.json';out.write_text(json.dumps({'passed':True,'tests':results,'gpu_used':False,'training_started':False},indent=2)+'\n')
print(out.read_text())
