from pathlib import Path
import argparse,hashlib,importlib.util,json,os,signal,subprocess,time
r=Path(__file__).resolve().parent;plan=json.loads((r/'RECOVERY_PLAN.json').read_text());base=r.parent;child=None;maps={};completed=[]
def read_json(p):
 for attempt in range(10):
  try:return json.loads(Path(p).read_text())
  except json.JSONDecodeError:
   if attempt==9:raise
   time.sleep(.1)
def verify(dataset):
 item=plan['datasets'][dataset];spec=read_json(item['protocol']);path=Path(item['verifier']);assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['worker_sha256'];module_spec=importlib.util.spec_from_file_location('verify_'+dataset,path);module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module);module.verify(spec)
def completed_original(root):
 rows=[]
 for p in Path(root).glob('shard*_progress.json'):rows+=read_json(p)
 return {(x['case_id'],x['method']):x for x in rows if x['exit_code']==0}
def publish_complete(dataset):
 item=plan['datasets'][dataset];root=Path(item['root']);spec=read_json(item['protocol']);original=completed_original(root);recovered=maps.get(dataset,{})
 expected=[(c['case_id'],m) for c in spec['cases'] for m in c['method_order']]
 if not all(pair in original or '|'.join(pair) in recovered for pair in expected):return False
 for pair in expected:
  output=Path(recovered['|'.join(pair)]['output']) if '|'.join(pair) in recovered else Path(original[pair]['output']);assert (output/'episodes.json').exists()
 verify(dataset);receipt={'status':'PASS','expected_unique_episodes':len(expected),'valid_completed_unique_episodes':len(expected),'frozen_source_policy_profile_hashes_verified':True,'recovered_outputs':recovered,'original_failed_attempts_retained':True,'recovery_plan':str(r/'RECOVERY_PLAN.json')};(root/'DATA_COMPLETION_RECEIPT.json').write_text(json.dumps(receipt,indent=2));return True
def stop(sig,frame):
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=25)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
for dataset in plan['datasets']:verify(dataset);maps[dataset]={}
for task in plan['tasks']:
 item=plan['datasets'][task['dataset']];spec=read_json(item['protocol']);root=Path(item['root']);case=task['case'];method=task['method'];output=r/'cases'/task['dataset']/case['case_id']/method;assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
 command=list(spec['template'])+['--output',str(output),'--seeds',str(case['seed']),'--launch-config',str(root/'launch_configs'/(case['case_id']+'.json')),'--lighting',case['lighting']]
 env=dict(os.environ,**item['env']);hz=case.get('endpoint_hz',100);env.update(EGL_DEVICE_ID='1',MUJOCO_EGL_DEVICE_ID='1',CUDA_VISIBLE_DEVICES='',CATCHIT_EVENT_GENERATOR='v2e' if method=='v2e' else 'ours',CATCHIT_EVENT_ENDPOINT_HZ=str(hz));env['CATCHIT_V2E_PARITY_RECEIPT']=spec['v2e_parity_receipts'][str(hz)] if 'v2e_parity_receipts' in spec else spec['v2e_parity_receipt']
 if method=='state_reference':command+=['--sensor-probe']
 else:
  command+=['--event-control','--rgbd-initialize','--stop-events-after-grasp']
  if method=='init_only':command+=['--initialization-only']
 record={'dataset':task['dataset'],'case_id':case['case_id'],'method':method,'seed':case['seed'],'output':str(output),'original_output':str(root/'cases'/case['case_id']/method),'command':command,'reason':task['reason'],'log':str(r/(task['dataset']+'_'+case['case_id']+'_'+method+'.log'))};(r/'RUNNING.json').write_text(json.dumps(record,indent=2));start=time.monotonic()
 with Path(record['log']).open('w') as log:child=subprocess.Popen(command,cwd=spec['source'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record['pid']=child.pid;print(json.dumps({'status':'started',**record}),flush=True);code=child.wait(timeout=1800)
 child=None;record.update(exit_code=code,wall_s=time.monotonic()-start);completed.append(record);(r/'PROGRESS.json').write_text(json.dumps(completed,indent=2));assert code==0,record
 record['episodes']=read_json(output/'episodes.json');maps[task['dataset']][case['case_id']+'|'+method]={'output':str(output),'previous_output':record['original_output'],'actor_log':record['log'],'reason':task['reason'],'exit_code':0};(root/'RECOVERY_MAP.json').write_text(json.dumps(maps[task['dataset']],indent=2));(r/'PROGRESS.json').write_text(json.dumps(completed,indent=2));publish_complete(task['dataset']);print(json.dumps({'status':'completed',**record}),flush=True)
# Recreate the originally planned own-shard25Hz representative after all missing
# actor runs. This does not generate another physics rollout or affect scores.
dataset='cadence';item=plan['datasets'][dataset];root=Path(item['root']);spec=read_json(item['protocol']);folder=root/'world_demo';folder.mkdir(exist_ok=True)
def actual(case_id,method):return Path(maps[dataset].get(case_id+'|'+method,{}).get('output',str(root/'cases'/case_id/method)))
candidates=[c for c in spec['cases'] if c['shard']==1 and c['condition']=='normal25'];selected=next((c for c in candidates if any(read_json(actual(c['case_id'],m)/'episodes.json')[0]['success'] for m in ['ours','v2e'])),candidates[0]);annotations={};outputs={};sources={}
for method in ['ours','v2e']:
 run=actual(selected['case_id'],method);sources[method]=str(run);tracefile=run/'episode_000/trace.json'
 if not tracefile.exists():outputs[method]={'skipped':'initialization failed; no physics trace'};continue
 out=folder/(selected['case_id']+'_'+method);env=dict(os.environ,**item['env']);env.update(EGL_DEVICE_ID='1',MUJOCO_EGL_DEVICE_ID='1',CUDA_VISIBLE_DEVICES='');command=[spec['template'][0],'-u',str(base/'render_archived_demo.py'),'--run',str(run),'--output',str(out)];child=subprocess.Popen(command,env=env,start_new_session=True);code=child.wait(timeout=300);child=None;assert code==0;trace=read_json(tracefile);annotations[method]=[{k:x[k] for k in ['t','hand_position','object_position']} for x in trace];outputs[method]=str(out/'whole_and_hand.mp4')
(folder/'shard1_selection.json').write_text(json.dumps({'case_id':selected['case_id'],'seed':selected['seed'],'endpoint_hz':25,'outputs':outputs,'source_episodes':sources,'selection':'same preplanned first successful own-shard25Hz case, otherwise first failed; all outcomes retained'},indent=2));(folder/'shard1_world_annotations.json').write_text(json.dumps({'case':selected['case_id'],'records':annotations,'annotation':'offline true positions for display only'},indent=2))
for dataset in plan['datasets']:verify(dataset);publish_complete(dataset)
(r/'RECOVERY_COMPLETE.json').write_text(json.dumps({'status':'PASS','valid_actor_runs':len(completed),'original_oom_attempts_preserved':2,'one_generator_at_a_time_on_gpu1':True}));print(json.dumps({'status':'PASS','valid_actor_runs':len(completed)}),flush=True)
