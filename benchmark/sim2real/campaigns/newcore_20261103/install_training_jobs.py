"""Install guarded queues; no optimizer work until complete immutable adapters."""
import ast,hashlib,json,sys
from pathlib import Path
R=Path(sys.argv[1]).resolve();S=R/'sources';python=sys.executable
def publish(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
for path in S.glob('*.py'):ast.parse(path.read_text())
files={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(S.rglob('*.py'))}
(S/'training_source_manifest.json').write_text(json.dumps({'schema':'stage9_training_source_pin_v1','files':files},indent=2)+'\n')
jobs=[]
for gpu,task in [(3,'flow'),(4,'reconstruction'),(5,'pose')]:
 lane=R/('gpu'+str(gpu));primary=[]
 for j,method in enumerate(['eventgenesis','v2e']):
  job='30_'+str(j)+'_'+task+'_'+method+'_preflight';primary.append(job)
  manifest=R/'adapters'/method/task/('training_bank_manifest.json' if task=='flow' else 'adapter_manifest.json')
  spec={'job_id':job,'phase':'training_preflight','argv':[python,'-u',str(S/'run_training_job.py'),'--campaign',str(R),'--task',task,'--method',method,'--seed','0','--phase','preflight'],'output_dir':str(R/'job_logs'/job),'ready_files':[str(manifest)],'source_sha256':{str(S/'run_training_job.py'):files['run_training_job.py']}}
  publish(lane/'queue'/(job+'.json'),spec);jobs.append(spec)
 previous=[]
 for seed in range(3):
  current=[]
  for j,method in enumerate(['eventgenesis','v2e']):
   job=str(40+seed)+'_'+str(j)+'_'+task+'_'+method+'_seed'+str(seed);current.append(job)
   spec={'job_id':job,'phase':'formal_synthetic_training','argv':[python,'-u',str(S/'run_training_job.py'),'--campaign',str(R),'--task',task,'--method',method,'--seed',str(seed),'--phase','full'],'output_dir':str(R/'job_logs'/job),'ready_files':[str(R/'preflight/tasks'/task/'paired_gate.json')],'after':primary+previous,'source_sha256':{str(S/'run_training_job.py'):files['run_training_job.py']}}
   publish(lane/'queue'/(job+'.json'),spec);jobs.append(spec)
  previous=current
(R/'TRAINING_DEPLOYMENT.json').write_text(json.dumps({'status':'queues_installed_inputs_and_preflight_gated','primary_methods':['eventgenesis','v2e'],'seeds':[0,1,2],'tasks':['flow','reconstruction','pose'],'formal_runs':18,'jobs':jobs,'optimizer_updates_at_installation':0,'real_scoring_enabled':False,'retained_after_completion_gpu_indices':[3,4],'gpu5_release_after_campaign_finalized':True,'additional_baselines_status':'implementation and source/weight audit pending; not misrepresented as deployed'},indent=2)+'\n')
for gpu in [3,4,5]:
 for p in (R/('gpu'+str(gpu))/'queue').glob('*.json'):
  spec=json.loads(p.read_text());spec.setdefault('env',{}).update(PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_PATH=str(R/'runtime/cuda_cache'/('gpu'+str(gpu))));publish(p,spec)
print('INSTALLED',len(jobs),'JOBS',18,'FORMAL_RUNS')
