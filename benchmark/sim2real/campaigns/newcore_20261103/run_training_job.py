"""Run bounded preflight and real process-boundary resume before a full budget."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--task',choices=['flow','reconstruction','pose'],required=True);p.add_argument('--method',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--phase',choices=['preflight','full'],required=True)
a=p.parse_args();R=a.campaign.resolve();S=R/'sources';P=Path('/home/zhaoyuhan/Eventbased_WAM');cfg=S/'training_recipe.json';data=R/'adapters'/a.method/a.task;out=R/'training'/a.task/('seed'+str(a.seed))/a.method;out.mkdir(parents=True,exist_ok=True)
os.environ['STAGE9_CAMPAIGN']=str(R);os.environ['STAGE4_REPO_COMMIT']='stage9_direct_versioned_sources';os.environ['STAGE4_PHYSICAL_GPU']=os.environ['CUDA_VISIBLE_DEVICES']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(limit=None,resume=None):
 if a.task=='flow':
  cmd=[sys.executable,'-u',str(S/'benchmark/sim2real/scripts/train_steflow_stage9.py'),'--model-root',str(P/'external/sources/downstream/STE-FlowNet'),'--data-root',str(data),'--config',str(cfg),'--run-root',str(out.parent)]
 elif a.task=='reconstruction':
  cmd=[sys.executable,'-u',str(S/'benchmark/sim2real/scripts/train_e2vid_stage9.py'),'--model-root',str(P/'external/sources/downstream/event_cnn_minimal'),'--data-root',str(data),'--config',str(cfg),'--output-root',str(out)]
 else:cmd=[sys.executable,'-u',str(S/'train_pose_stage9.py'),'--campaign',str(R),'--data-root',str(data),'--config',str(cfg),'--output-root',str(out)]
 cmd+=['--generator',a.method,'--seed',str(a.seed)]
 if limit is not None:cmd+=['--max-updates',str(limit)]
 if resume is not None:cmd+=['--resume',str(resume)]
 return cmd
def run(cmd,name):
 with (out/(name+'.stdout.log')).open('xb') as log:result=subprocess.run(cmd,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,(name,result.returncode,str(out/(name+'.stdout.log')))
def get_checkpoint():
 path=out/'last.pt'
 if a.task=='reconstruction':
  # The pinned reconstruction runner uses last_resume.pt.
  options=[out/'last_resume.pt',out/'last.pt',out/'checkpoint_last.pt']
  path=next((p for p in options if p.exists()),path)
 assert path.is_file(),str(path);return path,torch.load(path,map_location='cpu',weights_only=False)
def count(d):return int(d.get('completed_update',d.get('optimizer_update',d.get('global_update',d.get('completed_updates',-1)))))
def initial(d):return d.get('initial_state_sha256',d.get('initial_model_state_sha256'))
if a.phase=='preflight':
 assert a.seed==0
 run(command(20),'preflight20');path,d=get_checkpoint();assert count(d)==20,(count(d),list(d))
 before=sha(path);init=initial(d);assert isinstance(init,str) and len(init)==64
 reference=R/'preflight/tasks'/a.task/'eventgenesis.json'
 if a.method!='eventgenesis' and reference.exists():assert init==json.loads(reference.read_text())['initial_state_sha256'],'Method-specific initial weights differ'
 optimizer=d['optimizer'];steps=[int(v['step']) for v in optimizer['state'].values() if 'step' in v];assert steps and all(v==20 for v in steps)
 run(command(21,path),'resume21');path,after=get_checkpoint();assert count(after)==21 and initial(after)==init
 steps2=[int(v['step']) for v in after['optimizer']['state'].values() if 'step' in v];assert len(steps)==len(steps2) and all(v==21 for v in steps2)
 receipt=R/'preflight/tasks'/a.task/(a.method+'.json');receipt.parent.mkdir(parents=True,exist_ok=True)
 receipt.write_text(json.dumps({'passed':True,'task':a.task,'generator':a.method,'seed':0,'completed_updates':21,'initial_state_sha256':init,'optimizer_state_tensors':len(steps),'checkpoint20_sha256':before,'checkpoint21_sha256':sha(path),'checkpoint_boundary_resume_passed':True,'recipe_sha256':sha(cfg),'real_data_used':False,'strict_full_trajectory_equivalence_claimed':False},indent=2)+'\n')
else:
 gate=R/'preflight/tasks'/a.task/'paired_gate.json';g=json.loads(gate.read_text());assert g['passed'] and g['recipe_sha256']==sha(cfg)
 if a.seed==0:
  path,d=get_checkpoint();assert count(d)==21
  run(command(resume=path),'full_training')
 else:
  run(command(20),'preflight20');path,d=get_checkpoint();assert count(d)==20
  reference=R/'training'/a.task/('seed'+str(a.seed))/'eventgenesis/completion.json'
  if reference.exists():assert initial(d)==json.loads(reference.read_text())['initial_state_sha256'],'Paired initialization differs'
  run(command(21,path),'resume21');path,d=get_checkpoint();assert count(d)==21
  run(command(resume=path),'full_training')
 path,d=get_checkpoint();budget={'flow':8910,'reconstruction':12000,'pose':30000}[a.task];assert count(d)==budget,(count(d),budget)
 selected=out/'synthetic_best.pt'
 if a.task=='reconstruction':selected=out/'synthetic_best.pt' if (out/'synthetic_best.pt').exists() else out/'best.pt'
 assert selected.is_file(),str(selected)
 (out/'completion.json').write_text(json.dumps({'status':'complete','task':a.task,'method':a.method,'seed':a.seed,'completed_updates':budget,'last_checkpoint_sha256':sha(path),'selected_checkpoint':str(selected),'selected_checkpoint_sha256':sha(selected),'initial_state_sha256':initial(d),'recipe_sha256':sha(cfg),'real_data_used_in_training':False,'real_scoring_performed':False},indent=2)+'\n')
print('TRAINING_JOB_COMPLETE',a.task,a.method,a.seed,a.phase,flush=True)
