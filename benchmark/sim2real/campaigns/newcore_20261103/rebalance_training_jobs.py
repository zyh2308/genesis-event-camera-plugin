"""Balance pending jobs before training; preserve paired GPU model per seed."""
import hashlib,json,sys
from pathlib import Path
R=Path(sys.argv[1]).resolve();assert not list((R/'training').glob('*/*/*/training_metrics.jsonl')),'Refuse schedule rewrite after model training starts'
archive=R/'queue_plan_before_rebalance_retry1';archive.mkdir(exist_ok=False);pending=[];latch=R/'QUEUE_REBALANCE_COMPLETE.json'
def flag(argv,name):return argv[argv.index(name)+1]
def gpu_for(task,seed,method):
 if task=='pose':return 5
 if task=='reconstruction':return {0:4,1:5,2:3}[seed]
 if seed==0:return 3
 if seed==1:return 4
 return 4 if method in ['film_esim','rife_esim'] else 3
snapshot=[]
for gpu in [3,4,5]:
 for path in sorted((R/('gpu'+str(gpu))/'queue').glob('*.json')):
  spec=json.loads(path.read_text())
  if spec['phase'] in ['training_preflight','formal_synthetic_training']:snapshot.append((gpu,path,spec))
formal_before=[(flag(s['argv'],'--task'),flag(s['argv'],'--method'),flag(s['argv'],'--seed')) for _,_,s in snapshot if s['phase']=='formal_synthetic_training']
assert len(formal_before)==len(set(formal_before))==45
for gpu,path,spec in snapshot:
  argv=spec['argv'];task=flag(argv,'--task');seed=int(flag(argv,'--seed'));method=flag(argv,'--method');target=gpu_for(task,seed,method)
  if spec['phase']=='training_preflight':target={'flow':3,'reconstruction':4,'pose':5}[task]
  backup=archive/('gpu'+str(gpu))/path.name;backup.parent.mkdir(parents=True,exist_ok=True);path.rename(backup)
  # Global independently checked receipts, rather than lane-local dependency names.
  spec.pop('after',None)
  if task=='reconstruction' and seed==1 and method in ['eventgenesis','v2e'] and spec['phase']=='formal_synthetic_training':
   old=spec['job_id'];spec['job_id']='66_'+('0' if method=='eventgenesis' else '1')+'_'+task+'_'+method+'_seed1';spec['output_dir']=str(R/'job_logs'/spec['job_id'])
  spec.setdefault('ready_files',[]).append(str(latch));spec.setdefault('env',{}).update(CUDA_CACHE_PATH=str(R/'runtime/cuda_cache'/('gpu'+str(target))),PYTHONDONTWRITEBYTECODE='1')
  new=R/('gpu'+str(target))/'queue'/(spec['job_id']+'.json');assert not new.exists();tmp=new.with_suffix('.tmp');tmp.write_text(json.dumps(spec,indent=2)+'\n');tmp.replace(new)
  pending.append({'task':task,'method':method,'seed':seed,'phase':spec['phase'],'gpu_index':target,'job_id':spec['job_id']})
formal=[r for r in pending if r['phase']=='formal_synthetic_training'];assert len(formal)==45 and len({(r['task'],r['method'],r['seed']) for r in formal})==45
assert all(r['gpu_index'] in [3,4] for r in formal if r['task']=='flow')
result={'passed':True,'formal_runs':45,'routing':pending,'rationale':'Flow batch8 needs V100S32GB. Balance the two identical V100S cards; move the complete reconstruction seed1 group to TITAN RTX. Same GPU model within every task/seed, with UUID recorded per run.','hardware':{'flow':'all Tesla V100S32GB; same GPU class, exact UUID may differ','reconstruction':'seed0/2 Tesla V100S32GB; seed1 all methods TITAN RTX24GB','pose':'all TITAN RTX24GB'},'planned_formal_run_count_by_gpu':{str(g):sum(r['gpu_index']==g for r in formal) for g in [3,4,5]},'retained_after_completion':[3,4],'real_scores_used':False}
tmp=latch.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(latch);print(json.dumps({k:v for k,v in result.items() if k!='routing'}),flush=True)
