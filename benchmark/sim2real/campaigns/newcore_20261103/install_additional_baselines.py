"""Complete the registered 5-method x 3-task x 3-seed guarded job matrix."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
R=Path(sys.argv[1]).resolve();S=R/'sources';python=sys.executable;methods=['vid2e','film_esim','rife_esim'];jobs=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(gpu,spec):
 spec.setdefault('env',{}).update(PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_PATH=str(R/'runtime/cuda_cache'/('gpu'+str(gpu))))
 target=R/('gpu'+str(gpu))/'queue'/(spec['job_id']+'.json');assert not target.exists();tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(spec,indent=2)+'\n');tmp.replace(target);jobs.append(spec)
for m in methods:
 result=json.loads((R/'preflight/baselines'/m/'result.json').read_text());assert result['passed']
 assert result['row']['interpolator_adapter_sha256']==sha(S/'esim_interpolation_models.py')
 assert result['row']['generator_entrypoint_sha256']==sha(S/'replay_esim_events.py')
for j,m in enumerate(methods):
 for k,kind in enumerate(['rich','pose']):
  job=str(60+j*2+k)+'_'+m+'_'+kind+'_events';scripts=[S/'replay_esim_events.py',S/'esim_interpolation_models.py',S/'film_inference_worker.py']
  put(5,{'job_id':job,'phase':'additional_baseline_event_generation','argv':[python,'-u',str(S/'replay_esim_events.py'),'--campaign',str(R),'--kind',kind,'--method',m],'output_dir':str(R/'job_logs'/job),'ready_files':[str(R/'preflight/baselines'/m/'result.json')],'source_sha256':{str(p):sha(p) for p in scripts}})
for gpu,task in [(3,'flow'),(4,'reconstruction'),(5,'pose')]:
 for j,m in enumerate(methods):
  job='70_'+str(j)+'_'+task+'_'+m+'_preflight';manifest=R/'adapters'/m/task/('training_bank_manifest.json' if task=='flow' else 'adapter_manifest.json')
  put(gpu,{'job_id':job,'phase':'training_preflight','argv':[python,'-u',str(S/'run_training_job.py'),'--campaign',str(R),'--task',task,'--method',m,'--seed','0','--phase','preflight'],'output_dir':str(R/'job_logs'/job),'ready_files':[str(manifest)],'source_sha256':{str(S/'run_training_job.py'):sha(S/'run_training_job.py')}})
 for seed in range(3):
  for j,m in enumerate(methods):
   job=str(80+seed)+'_'+str(j)+'_'+task+'_'+m+'_seed'+str(seed)
   put(gpu,{'job_id':job,'phase':'formal_synthetic_training','argv':[python,'-u',str(S/'run_training_job.py'),'--campaign',str(R),'--task',task,'--method',m,'--seed',str(seed),'--phase','full'],'output_dir':str(R/'job_logs'/job),'ready_files':[str(R/'preflight/tasks'/task/'paired_gate.json'),str(R/'preflight/tasks'/task/(m+'.json'))],'after':['70_'+str(j)+'_'+task+'_'+m+'_preflight'],'source_sha256':{str(S/'run_training_job.py'):sha(S/'run_training_job.py')}})
background=[]
for kind in ['rich','pose']:
 for m in methods:
  out=R/'adapter_jobs'/(kind+'_'+m);out.mkdir(parents=True,exist_ok=False)
  with (out/'stdout.log').open('wb') as log:proc=subprocess.Popen([python,'-u',str(S/'build_task_adapters.py'),'--campaign',str(R),'--kind',kind,'--method',m],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','HOME':'/home/zhaoyuhan','PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True)
  background.append({'kind':kind,'method':m,'pid':proc.pid})
(R/'ADDITIONAL_BASELINE_DEPLOYMENT.json').write_text(json.dumps({'status':'installed_after_native_preflight','methods':methods,'additional_formal_runs':27,'requested_total_formal_runs':45,'jobs':jobs,'adapter_processes':background,'real_data_used':False},indent=2)+'\n')
print('ADDITIONAL_BASELINES_INSTALLED',len(jobs),'JOBS; TOTAL 45 FORMAL RUNS')
