"""Compare final-epoch pose weights on synthetic validation, not real data."""
import ast,argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();R=a.campaign;P=R.parents[2];T=P/'runs/sim2real/20261106-pose-synthetic-diagnostic-v1'
a.out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device=torch.device('cuda:0');assert torch.cuda.device_count()==1
cfg=json.loads((R/'recovery_20261105_mainline/pose_prepared_full_v5/protocol.json').read_text());vertices=np.load(cfg['assets']['vertices']['path']);diam=cfg['assets']['diameter_m'];tree=cKDTree(vertices)
code=ast.parse((T/'causal_tests_v2.py').read_text());functions=[n for n in code.body if isinstance(n,ast.FunctionDef) and n.name in ['sha','dump','load','pnp','decode','evaluate']];exec(compile(ast.Module(body=functions,type_ignores=[]),'audited_causal_functions','exec'),globals())
sys.path.insert(0,str(P/'external/sources/downstream/SingleShotPose'));from darknet import Darknet
summary=[]
for method in ['eventgenesis','v2e']:
 val=load(method,'validation');assert len(val)==510
 path=R/'training/pose/seed0'/method/'last.pt';digest=sha(path);ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['completed_update']==30000 and ck['generator']==method and ck['seed']==0
 model=Darknet(str(P/'external/sources/downstream/SingleShotPose/cfg/yolo-pose.cfg')).to(device);model.load_state_dict(ck['model'],strict=True);del ck
 result=evaluate(model,val);reference=json.loads((path.parent/'synthetic_validation_030000.json').read_text());assert abs(result['summary']['epe_px']-reference['mean_keypoint_epe_px'])<.002;assert sha(path)==digest
 dump(a.out/(method+'_final_validation.json'),result);row=dict(method=method,seed=0,updates=30000,last_checkpoint_sha256=digest,summary=result['summary']);summary.append(row);print('FINAL_CHECKPOINT',method,row,flush=True);del model;torch.cuda.empty_cache()
dump(a.out/'completion.json',dict(status='complete',real_data_used=False,results=summary,original_primary_protocol_unchanged=True,scope='exploratory final-epoch comparison; no retrospective full-history ADD-S selection claimed'))
