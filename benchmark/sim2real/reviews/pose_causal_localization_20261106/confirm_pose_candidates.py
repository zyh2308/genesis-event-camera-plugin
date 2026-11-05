"""Prespecified3-seed source-only confirmation; no real images or scoring."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--methods',nargs='+',default=['eventgenesis','v2e']);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);a=p.parse_args();R=a.campaign;P=R.parents[2];T=P/'runs/sim2real/20261106-pose-synthetic-diagnostic-v1'
a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.device_count()==1;device=torch.device('cuda:0')
source=T/'causal_tests_v2.py';treecode=ast.parse(source.read_text());functions=[n for n in treecode.body if isinstance(n,ast.FunctionDef) and n.name in ['sha','dump','load','pnp','decode','evaluate','parameter_hash']];exec(compile(ast.Module(body=functions,type_ignores=[]),'audited_causal_helpers','exec'),globals())
protocol=json.loads((R/'recovery_20261105_mainline/pose_prepared_full_v5/protocol.json').read_text());vertices=np.load(protocol['assets']['vertices']['path']);diam=protocol['assets']['diameter_m'];tree=cKDTree(vertices)
frozen=json.loads((R/'recovery_20261103/pose_frozen_checkpoints.json').read_text());sys.path.insert(0,str(P/'external/sources/downstream/SingleShotPose'));from darknet import Darknet
dump(a.out/'protocol.json',dict(methods=a.methods,seeds=a.seeds,cases=['original_synthetic_EPE_best','same_weights_source_BN_calibration','final_epoch30000'],primary='synthetic fullCAD ADD-S success0.1D, complete510 windows equal32rollouts',secondary='9point pixel EPE',BN_data='1528 deterministically shuffled original synthetic TRAIN windows only',real_data_used=False,original_protocol_unchanged=True,helper_source_sha256=sha(source),case_selection='no automatic selection or deployment; report all registered cases'))
rows=[]
for method in a.methods:
 assert method in ['eventgenesis','v2e','vid2e','film_esim','rife_esim']
 train=load(method,'train');val=load(method,'validation');assert (len(train),len(val))==(1529,510)
 for seed in a.seeds:
  assert seed in [0,1,2]
  for case in ['original_synthetic_EPE_best','same_weights_source_BN_calibration','final_epoch30000']:
   path=R/'training/pose'/('seed'+str(seed))/method/('last.pt' if case=='final_epoch30000' else 'synthetic_best.pt');digest=sha(path);record=next(x for x in frozen['runs'] if x['method']==method and x['seed']==seed)
   if case!='final_epoch30000':assert digest==record['selected_checkpoint_sha256']
   else:assert digest==record['last_checkpoint_sha256']
   ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['generator']==method and ck['seed']==seed
   if case=='final_epoch30000':assert ck['completed_update']==30000
   model=Darknet(str(P/'external/sources/downstream/SingleShotPose/cfg/yolo-pose.cfg')).to(device);model.load_state_dict(ck['model'],strict=True);del ck;ph=parameter_hash(model)
   if case=='same_weights_source_BN_calibration':
    model.eval();bns=[x for x in model.modules() if isinstance(x,torch.nn.modules.batchnorm._BatchNorm)];assert bns
    for bn in bns:bn.reset_running_stats();bn.momentum=None;bn.train()
    indices=np.random.default_rng(20261106).permutation(len(train))[:1528]
    with torch.no_grad():
     for off in range(0,len(indices),8):model(torch.from_numpy(np.stack([train[i]['image'] for i in indices[off:off+8]])).to(device))
    assert parameter_hash(model)==ph
   result=evaluate(model,val);assert sha(path)==digest
   tag=method+'_seed'+str(seed)+'_'+case;dump(a.out/(tag+'.json'),result)
   if case=='same_weights_source_BN_calibration':torch.save(dict(model=model.state_dict(),source_checkpoint_sha256=digest,method=method,seed=seed,source_only_BN_calibrated=True,real_data_used=False),a.out/(tag+'.pt'))
   row=dict(method=method,seed=seed,case=case,source_checkpoint_sha256=digest,summary=result['summary']);rows.append(row);print('CONFIRMED_CASE',json.dumps(row),flush=True);del model;torch.cuda.empty_cache()
dump(a.out/'completion.json',dict(status='complete',real_data_used=False,rows=rows))
