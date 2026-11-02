"""Compute task/seed initial state hashes independently, without data or CUDA."""
import argparse,hashlib,json,os,random,subprocess,sys
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--task',choices=['flow','reconstruction','pose']);a=p.parse_args();R=a.campaign;P=Path('/home/zhaoyuhan/Eventbased_WAM');torch.set_num_threads(2)
if a.task is None:
 rows={}
 for task in ['flow','reconstruction','pose']:
  result=subprocess.run([sys.executable,'-u',__file__,'--campaign',str(R),'--task',task],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True);assert result.returncode==0,(task,result.stderr[-2000:]);rows[task]=json.loads((R/'preflight'/(task+'_initialization_reference.json')).read_text())
 (R/'preflight/initialization_reference.json').write_text(json.dumps({'passed':True,'tasks':rows,'recipe_sha256':hashlib.sha256((R/'sources/training_recipe.json').read_bytes()).hexdigest(),'device':'CPU, no data','torch_version':torch.__version__,'real_data_used':False},indent=2)+'\n');raise SystemExit(0)
root=P/'external/sources/downstream';recipe=json.loads((R/'sources/training_recipe.json').read_text());hashes={}
if a.task=='flow':sys.path.insert(0,str(root/'STE-FlowNet'));import models
elif a.task=='reconstruction':sys.path.insert(0,str(root/'event_cnn_minimal'));from model.model import E2VIDRecurrent
else:sys.path.insert(0,str(root/'SingleShotPose'));from darknet import Darknet
for seed in [0,1,2]:
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if a.task=='flow':model=models.steflow()
 elif a.task=='reconstruction':model=E2VIDRecurrent(recipe['reconstruction']['model_args'])
 else:model=Darknet(str(root/'SingleShotPose/cfg/yolo-pose.cfg'))
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 hashes[str(seed)]=h.hexdigest();del model
(R/'preflight'/(a.task+'_initialization_reference.json')).write_text(json.dumps({'task':a.task,'hashes':hashes},indent=2)+'\n');print(json.dumps(hashes),flush=True)
