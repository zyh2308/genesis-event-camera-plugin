"""Deploy the final serial cost driver, then start one checkpoint supervisor."""
from pathlib import Path
import ast, base64, hashlib, json

local=Path(__file__).resolve().parent
driver=local/'run_cost_v4_study.py'
digest=hashlib.sha256(driver.read_bytes()).hexdigest()
payload=base64.b64encode(driver.read_bytes()).decode()
code='''from pathlib import Path
import base64,hashlib,json,os,socket,subprocess,time
assert socket.gethostname().split('.')[0]=='g1'
root=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/fair_frontend_blind_20261108b')
tools=root/'tools';target=tools/'run_cost_v4_study.py';checkpoint=tools/'checkpoint_blind_for_cost_v4.py'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='9b583e68a5b6598ffd93feec52ec42fdbf1560c13525606df8a95cfcadbc07ca'
assert not target.exists() and not (root/'BLIND_COST_V4_CHECKPOINT_SUPERVISOR.json').exists()
assert not (root/'blind32/SCHEDULING_COST_V4_REQUEST.json').exists()
assert not (root/'blind32/FAIR_COLLECTION_COMPLETE.json').exists()
for gpu in [0,1]:
 name='catchit_fair_blind32_g1_gpu'+str(gpu)+'_blind_20261108b'
 assert (root/'blind32'/(name+'.job.json')).exists()
with target.open('xb') as f:f.write(base64.b64decode(PAYLOAD_VALUE))
assert hashlib.sha256(target.read_bytes()).hexdigest()==DIGEST_VALUE
python='/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python'
subprocess.run([python,'-B','-c','import ast,sys;ast.parse(open(sys.argv[1]).read())',str(target)],check=True,timeout=15)
with (root/'blind_cost_v4_checkpoint_supervisor.log').open('x') as log:
 proc=subprocess.Popen([python,'-B','-u',str(checkpoint)],cwd=str(root),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
ticks=Path('/proc/'+str(proc.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
receipt={'status':'STARTED_CHECKPOINT_SUPERVISOR','pid':proc.pid,'process_start_ticks':ticks,'host':socket.gethostname(),
 'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'driver_sha256':DIGEST_VALUE,
 'helper_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'no_actor_termination':True,'protocol_and_budget_unchanged':True}
with (root/'BLIND_COST_V4_CHECKPOINT_SUPERVISOR.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''.replace('PAYLOAD_VALUE',repr(payload)).replace('DIGEST_VALUE',repr(digest))
ast.parse(code)
q=local/'remote_queue_g1';target=q/'050_deploy_cost_v4_driver_and_checkpoint.request.py'
assert not target.exists()
draft=target.with_suffix('.draft');draft.write_text(code,encoding='utf-8');draft.rename(target)
print(json.dumps({'request':str(target),'driver_sha256':digest}))
