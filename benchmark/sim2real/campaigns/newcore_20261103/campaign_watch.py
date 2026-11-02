"""Research controls and compact machine-readable live campaign status."""
import datetime,hashlib,json,os,sys,time,traceback
from pathlib import Path
R=Path(sys.argv[1]).resolve();recipe=R/'sources/training_recipe.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def records(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def pair(task):
 d=R/'preflight/tasks'/task;paths=[d/(m+'.json') for m in ['eventgenesis','v2e']]
 if not all(p.exists() for p in paths) or (d/'paired_gate.json').exists():return
 rs=[json.loads(p.read_text()) for p in paths];assert all(r['passed'] and r['recipe_sha256']==sha(recipe) for r in rs)
 assert rs[0]['initial_state_sha256']==rs[1]['initial_state_sha256'],'Paired initial model weights differ'
 key={'flow':'entries','reconstruction':'rollout_rows','pose':'rows'}[task];manname='training_bank_manifest.json' if task=='flow' else 'adapter_manifest.json';man=[json.loads((R/'adapters'/m/task/manname).read_text()) for m in ['eventgenesis','v2e']]
 assert man[0]['physical_manifest_sha256']==man[1]['physical_manifest_sha256']
 labels=[]
 for obj in man:labels.append({r['rollout_id']:r.get('label_sha256',r.get('flow_label_sha256')) for r in obj[key] if 'path' in r})
 assert labels[0]==labels[1] and all(v for v in labels[0].values()),'Paired labels differ'
 logs=[records(R/'training'/task/'seed0'/m/'training_metrics.jsonl') for m in ['eventgenesis','v2e']]
 logs=[[r for r in rows if r.get('optimizer_update',0)>0][:21] for rows in logs];assert len(logs[0])==len(logs[1])==21
 fields={'flow':['rollout_sample_ids','augmentation_seed_sequences'],'reconstruction':['clip_ids','clip_indices'],'pose':['rollout_sample_ids','affine']}[task]
 for left,right in zip(*logs):
  assert left['optimizer_update']==right['optimizer_update']
  for key in fields:assert left[key]==right[key],(task,key,left['optimizer_update'])
 save(d/'paired_gate.json',{'passed':True,'task':task,'methods':['eventgenesis','v2e'],'recipe_sha256':sha(recipe),'same_initial_weights':True,'same_physical_labels':True,'same_sample_and_augmentation_sequence':True,'boundary_resume_both_passed':True,'preflight_updates_per_method':21,'real_scores_used':False})
while True:
 try:
  for task in ['flow','reconstruction','pose']:
   try:pair(task)
   except Exception:save(R/'preflight/tasks'/task/'gate_failure.json',{'passed':False,'traceback':traceback.format_exc()})
  state={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'data_generation_and_training_deployment','tasks':{},'gpu_jobs':{},'real_scores_used_for_training':False}
  for g in [3,4,5]:
   reg=Path('/home/zhaoyuhan/Eventbased_WAM/.gpu_reservations')/(R.name+'_gpu'+str(g)+'.json');state['gpu_jobs'][str(g)]=json.loads(reg.read_text())
  for task in ['flow','reconstruction','pose']:
   state['tasks'][task]={'runs':{},'paired_gate':(R/'preflight/tasks'/task/'paired_gate.json').exists()}
   for seed in [0,1,2]:
    for m in ['eventgenesis','v2e','vid2e','film_esim','rife_esim']:
     d=R/'training'/task/('seed'+str(seed))/m;v=d/'live_status.json'
     if v.exists():state['tasks'][task]['runs'][m+'_seed'+str(seed)]=json.loads(v.read_text())
  for kind in ['rich','pose']:
   state[kind+'_data']={'physical_rollouts':len(list((R/(kind+'_bank')/'physical').glob('*.h5'))),'event_rollouts':{m:len(list((R/(kind+'_bank')/'events'/m).glob('*.npy'))) for m in ['eventgenesis','v2e','vid2e','film_esim','rife_esim']}}
  save(R/'CAMPAIGN_LIVE_STATUS.json',state)
 except Exception:save(R/'watcher_failure.json',{'traceback':traceback.format_exc()})
 time.sleep(60)
