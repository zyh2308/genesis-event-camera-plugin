"""Prioritize shared baseline data over long pose training; no algorithm change."""
import hashlib,json,sys
from pathlib import Path
R=Path(sys.argv[1]).resolve();queue=R/'gpu5/queue';archive=R/'data_queue_before_priority_amendment';archive.mkdir(exist_ok=False);rows=[]
paths=[]
for p in sorted(queue.glob('*.json')):
 s=json.loads(p.read_text())
 if s['phase']=='additional_baseline_event_generation':paths.append((p,s))
assert len(paths)==6,'Only alter the six still-pending owned data jobs'
for i,(path,spec) in enumerate(paths):
 assert path.resolve().is_relative_to(R);old=spec['job_id'];digest=hashlib.sha256(path.read_bytes()).hexdigest();path.rename(archive/path.name)
 spec['job_id']='35_'+str(i)+'_'+old.split('_',1)[1];spec['output_dir']=str(R/'job_logs'/spec['job_id']);new=queue/(spec['job_id']+'.json');assert not new.exists();tmp=new.with_suffix('.tmp');tmp.write_text(json.dumps(spec,indent=2)+'\n');tmp.replace(new);rows.append({'old_job':old,'new_job':spec['job_id'],'original_spec_sha256':digest})
(R/'DATA_GENERATION_PRIORITY_AMENDMENT.json').write_text(json.dumps({'passed':True,'change':'After current event replay and bounded pose preflights, generate all additional baseline data before long pose training. Avoid long training holding up other tasks or expiring adapter waiters.','model_recipe_changed':False,'physical_bank_changed':False,'training_run_count_changed':False,'training_runs':45,'retained_after_completion':[3,4],'jobs':rows},indent=2)+'\n');print('PRIORITIZED_SIX_DATA_JOBS')
