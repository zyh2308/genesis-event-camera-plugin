"""Bounded synthetic-only audit: 3 fixed windows per held-out rollout/arm."""
from pathlib import Path
import hashlib,importlib,json,os,subprocess,sys,gc
import numpy as np,torch,torch.nn.functional as F
R=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real');S=R/'benchmark/sim2real/scripts';ROOT=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/flow_dt1_seed0');DATA=Path('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/20261016-stage4-flowdt1-seed0-v4/steflow_30ms')
preflight=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip();util,used=map(int,preflight.split(','))
if util>5 or used>1024:raise RuntimeError(f'GPU0 not idle: {preflight}; audit does not stop existing processes')
sys.path.insert(0,str(S));from run_steflow_stage2_pilot import pure_torch_spatial_correlation,model_output_to_flow_hw
MODEL=Path('/home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/STE-FlowNet');sys.path.insert(0,str(MODEL));corr=importlib.import_module('models.corr');corr.spatial_correlation_sample=pure_torch_spatial_correlation;import models
torch.set_num_threads(2);torch.cuda.set_device(0);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
out={'scope':'24 fixed synthetic validation windows per model, not full evaluation and no real data','selection_indices_per_rollout':[0,33,65],'GPU':'physical0','preflight':preflight,'arms':{}}
for gen in ['eventgenesis','v2e']:
    path=ROOT/gen/'synthetic_best.pt'
    with torch.serialization.safe_globals([np._core.multiarray._reconstruct,np.ndarray,np.dtype,np.dtypes.UInt32DType]):payload=torch.load(path,map_location='cpu',weights_only=True)
    model=models.steflow().cuda();model.load_state_dict(payload['state_dict'],strict=True);model.eval();rows=[]
    for p in sorted((DATA/gen/'validation').glob('*.npz')):
        with np.load(p) as a:
            for i in [0,33,65]:
                event=a['events'][i];tensor=F.pad(torch.from_numpy(event[None].astype(np.float32)),(0,0,3,3,46,46)).cuda()
                with torch.inference_mode():pred=model_output_to_flow_hw(model(tensor,352,use_dt4=False,flow_seq=False))[:,46:306,3:349].astype(np.float64)
                gt=a['flow_gt'][i].astype(np.float64);valid=a['flow_valid'][i].astype(bool);mag=np.hypot(gt[0],gt[1]);err=np.hypot(pred[0]-gt[0],pred[1]-gt[1]);predmag=np.hypot(pred[0],pred[1]);support=event.sum(axis=(0,3))>0
                masks={'all_valid':valid,'event_supported_valid':valid&support,'GT_motion_gt_1px':valid&(mag>1),'GT_motion_le_0point1px':valid&(mag<=.1)}
                yy,xx=np.mgrid[0:260,0:346];comp=np.stack([.5-(xx+3)/352,.5-(yy+46)/352])
                rows.append({'rollout':p.stem,'index':i,'event_count':int(a['event_counts'][i]),'prediction_finite':bool(np.isfinite(pred).all()),'prediction_magnitude_mean':float(predmag[valid].mean()),'prediction_u_mean_std':[float(pred[0,valid].mean()),float(pred[0,valid].std())],'prediction_v_mean_std':[float(pred[1,valid].mean()),float(pred[1,valid].std())],'EPE_to_known_identity_sampling_compensation_field':float(np.hypot(pred[0]-comp[0],pred[1]-comp[1])[valid].mean()),'masked_metrics':{k:{'pixels':int(m.sum()),'model_EPE':float(err[m].mean()) if m.any() else None,'zero_EPE':float(mag[m].mean()) if m.any() else None} for k,m in masks.items()}})
    out['arms'][gen]={'best_epoch':payload['best_epoch'],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'windows':rows}
    del payload,model;gc.collect();torch.cuda.empty_cache()
print(json.dumps(out,ensure_ascii=False))
