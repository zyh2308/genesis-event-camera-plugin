"""CPU reproduction of R1/R2 fixes and newly identified metric/window bugs."""
from pathlib import Path
import ast, hashlib, importlib.util, json, random, sys, tempfile, types
import numpy as np, h5py, torch
torch.set_num_threads(2)
ROOT=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real')
BASE=ROOT/'benchmark/sim2real'; scripts=BASE/'scripts'
SRC=Path('/home/zhaoyuhan/Eventbased_WAM/external/sources/downstream')
DATA=Path('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage2/20261016-s2-pilot-v1')
OUT={}

# Exact checked-in flow extraction expression and metric.
tree=ast.parse((scripts/'run_steflow_stage2_pilot.py').read_text())
metric=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_epe')
extract=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prediction' for t in n.targets))
ns={'np':np,'torch':torch};exec(compile(ast.Module(body=[metric],type_ignores=[]),'<exact-metric>','exec'),ns)
gt=np.stack([np.full((3,4),3,dtype=np.float32),np.full((3,4),4,dtype=np.float32)])
ns['pred']=torch.from_numpy(gt).unsqueeze(0)
exec(compile(ast.Module(body=[extract],type_ignores=[]),'<exact-extraction>','exec'),ns)
bug=ns['prediction']; valid=np.ones((3,4),bool)
OUT['flow_EPE_reproduction']={'model_return_shape':list(ns['pred'].shape),'extracted_prediction_shape':list(bug.shape),'ground_truth_shape':list(gt.shape),
    'perfect_prediction_reported_EPE':ns['compute_epe'](bug,gt,valid)[0],
    'correct_two_component_EPE':ns['compute_epe'](ns['pred'][0].numpy(),gt,valid)[0],
    'extraction_source':ast.unparse(extract)}

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
pose=load_module('research_pose_audit',scripts/'run_pose_stage2_pilot.py')
size=json.loads((BASE/'configs/stage2_pilot.json').read_text())['simulation']['target_box']['size_m']
report=json.loads(Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s2-pose-pilot-v2/output/result.json').read_text())
posechecks=[]
for generator in ['eventgenesis','v2e']:
    logged=next(r for r in report['results'] if r['generator']==generator)
    for split in ['train','validation']:
        rows=pose.collect_samples(generator,DATA/'shared/shared_trajectory.h5',DATA/'v2e_events',size,split,2,8.0)
        with h5py.File(DATA/'shared/shared_trajectory.h5','r') as bank:
            for row in rows:
                eid=row['episode_id'];i=row['interval_index']; ep=bank[f'episodes/{eid}']; t=ep['frame_timestamps_s'][:]
                ts,x,y,p,_=pose.read_event_table(generator,eid,bank,DATA/'v2e_events')
                desired=(ts>t[i])&(ts<=t[i+1]); actual=(ts>t[i])&(ts<=t[-1])
                correct_image=pose.encode_event_window(ts,x,y,p,float(t[i]),float(t[i+1]),160,128,8.0)
                logrows=logged['training_detail'] if split=='train' else logged['validation_detail']
                rec=next(x for x in logrows if x['episode_id']==eid and x['interval_index']==i)
                posechecks.append({'generator':generator,'split':split,'episode':eid,'interval':i,'intended_duration_s':float(t[i+1]-t[i]),
                    'actual_duration_s':float(t[-1]-t[i]),'intended_event_count':int(desired.sum()),'actual_event_count':row['event_count'],
                    'archived_count_matches_buggy_window':rec['event_count']==int(actual.sum())==row['event_count'],
                    'correct_input_max_absolute_difference':float(np.abs(row['image']-correct_image).max()),'input_label_endpoint_mismatch':i+1!=len(t)-1})
OUT['pose_window_checks']=posechecks

# Reuse the exact nested E2VID adapter, without executing CUDA-only main().
sys.path[:0]=[str(SRC/'event_cnn_minimal'),str(SRC/'event_cnn_minimal/PerceptualSimilarity')]
import skimage.measure
from skimage.metrics import structural_similarity
if not hasattr(skimage.measure,'compare_ssim'): skimage.measure.compare_ssim=structural_similarity
try: import IPython
except ImportError:
    stub=types.ModuleType('IPython');stub.embed=lambda *a,**kw:None;sys.modules['IPython']=stub
from data_loader.dataset import DynamicH5Dataset
from utils.data import data_sources
from model.loss import temporal_consistency_loss
e=load_module('research_e2vid_audit',scripts/'run_e2vid_stage2_pilot.py')
tree=ast.parse((scripts/'run_e2vid_stage2_pilot.py').read_text())
cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Stage2RightEndpointH5Dataset')
ns={'DynamicH5Dataset':DynamicH5Dataset,'data_sources':data_sources,'h5py':h5py,'np':np,'torch':torch,'random':random}
exec(compile(ast.Module(body=[cls],type_ignores=[]),'<exact-adapter>','exec'),ns)
adapter=ns[cls.name]
with tempfile.TemporaryDirectory(prefix='research-stage2-audit-') as temp:
    OUT['E2VID_boundary_reexecution']=e.run_boundary_fixtures(adapter,Path(temp))
OUT['E2VID_direction_reexecution']=e.run_flow_direction_fixture(temporal_consistency_loss)
checks=[]
with h5py.File(DATA/'shared/shared_trajectory.h5','r') as bank:
    for gen in ['eventgenesis','v2e']:
        for eid in bank['episodes']:
            ep=bank[f'episodes/{eid}'];split=str(ep.attrs['split']); d=adapter(str(DATA/f'e2vid_inputs_v2/{gen}/{split}/{eid}.h5'),transforms={},num_bins=5,combined_voxel_channels=True,filter_hot_events=False)
            maxerr=0.0;mask_equal=True
            for i in range(len(ep['flow_forward_px_per_interval'])):
                s=d[i];maxerr=max(maxerr,float(np.abs(s['flow'].numpy()-ep['flow_forward_px_per_interval'][i]).max()))
                mask_equal &= np.array_equal(s['flow_valid'].numpy(),ep['flow_valid'][i].astype(bool))
            checks.append({'generator':gen,'episode':eid,'max_physical_flow_error':maxerr,'validity_matches_shared_bank':bool(mask_equal)})
            d.h5_file.close()
OUT['E2VID_loader_physical_labels']=checks
torch.manual_seed(17)
objective=temporal_consistency_loss(weight=2.0,L0=2)
image=torch.rand(1,1,12,14)
flow=torch.zeros(1,2,12,14)
predictions=[(image+0.01*torch.rand_like(image)).requires_grad_() for i in range(3)]
values=[objective(i,image,p,flow) for i,p in enumerate(predictions)]
v=values[-1];v.backward()
OUT['E2VID_temporal_gradient']={'first_two_values_none':values[0] is None and values[1] is None,'third_loss':float(v.detach()),'third_gradient_finite':bool(torch.isfinite(predictions[-1].grad).all()),'third_gradient_abs_sum':float(predictions[-1].grad.abs().sum())}
print(json.dumps(OUT,ensure_ascii=False))
