"""Compare actual synthetic-only STE-FlowNet loss/gradients/Adam step, then time it."""
import gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parent;campaign=root.parent
scripts=campaign/'sources/benchmark/sim2real/scripts'
sys.path.insert(0,str(scripts));sys.path.insert(0,str(scripts.parent/'compat'))
model_root=Path('/home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/STE-FlowNet')
sys.path.insert(0,str(model_root))
import models,models.corr as corr,multiscaleloss
from steflow_align_corners import patch_ste
from train_steflow_stage9 import apply_joint_transform
from run_steflow_stage2_pilot import pure_torch_spatial_correlation as ref,objective_dt1,set_seed,state_sha256
from native_adapter import spatial_correlation_sample as native
patch_ste(models,multiscaleloss,model_root)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
device=torch.device('cuda:0');set_seed(1)
model=models.steflow().to(device);model.train()
initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
initial_hash=state_sha256(model)
data_path=campaign/'adapters/eventgenesis/flow/eventgenesis/train/s0_00.npz'
with np.load(data_path,allow_pickle=False) as d:data={k:np.asarray(d[k]) for k in d.files}
rng=np.random.default_rng(20261104)
rows=[apply_joint_transform({k:data[k][i:i+1] for k in ['events','gray_prev','gray_next','flow_gt','flow_valid']},rng,device) for i in range(2)]
events=torch.cat([x[0] for x in rows]);gray=torch.cat([x[1] for x in rows])
def opt():return torch.optim.Adam([{'params':model.bias_parameters(),'weight_decay':0.0},{'params':model.weight_parameters(),'weight_decay':.0004}],lr=.0004,betas=(.9,.999))
def flat(x):
    if isinstance(x,torch.Tensor):return [x]
    if isinstance(x,dict):return [y for v in x.values() for y in flat(v)]
    return [y for v in x for y in flat(v)]
def one(fn):
    corr.spatial_correlation_sample=fn;model.load_state_dict(initial);model.zero_grad(set_to_none=True);optimizer=opt()
    output=model(events,256,use_dt4=False,flow_seq=False)
    loss,photo,smooth=objective_dt1(output,gray[:,0:1],gray[:,1:2],events,multiscaleloss)
    predictions=[v.detach().cpu() for v in flat(output)]
    loss.backward();grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
    assert len(grads)==70 and all(torch.isfinite(g).all() for g in grads.values())
    optimizer.step();torch.cuda.synchronize()
    updates={n:p.detach().cpu()-initial[n] for n,p in model.named_parameters()}
    return dict(predictions=predictions,gradients=grads,updates=updates,losses=[float(x.detach()) if torch.is_tensor(x) else float(x) for x in [loss,photo,smooth]],adam_steps=[float(s['step']) for s in optimizer.state.values()])
def compare(a,b):
    def stats(xs,ys):
        xa=torch.cat([x.flatten().double() for x in xs]);yb=torch.cat([y.flatten().double() for y in ys]);diff=xa-yb
        return dict(max_abs=float(diff.abs().max()),relative_l2=float(diff.norm()/xa.norm().clamp_min(1e-30)),cosine=float(torch.nn.functional.cosine_similarity(xa,yb,dim=0)),finite=bool(torch.isfinite(yb).all()))
    return dict(predictions=stats(a['predictions'],b['predictions']),
                gradients=stats(list(a['gradients'].values()),list(b['gradients'].values())),
                adam_updates=stats(list(a['updates'].values()),list(b['updates'].values())),
                loss_max_abs=max(abs(x-y) for x,y in zip(a['losses'],b['losses'])),
                per_parameter_gradient_relative_l2={k:float((a['gradients'][k]-b['gradients'][k]).norm()/a['gradients'][k].norm().clamp_min(1e-30)) for k in a['gradients']})
report=dict(device=torch.cuda.get_device_name(),batch=2,formal_batch=8,initial_state_sha256=initial_hash,
            source=str(data_path),sample_indices=[0,1],augmentation_seed=20261104,real_data_used=False,
            note='Diagnostic batch 2 on spare 24GB card; formal full-batch preflight remains mandatory on V100.',
            thresholds=dict(prediction_relative_l2=1e-4,gradient_relative_l2=.002,adam_update_relative_l2=.01,loss_max_abs=1e-4))
def save(): (root/'model_results.json').write_text(json.dumps(report,indent=2)+'\n')
a=one(ref);b=one(ref);report['reference_repeat']=compare(a,b);del b;gc.collect()
c=one(native);report['native_comparison']=compare(a,c);report['losses_reference']=a['losses'];report['losses_native']=c['losses'];report['gradient_tensors']=len(c['gradients']);report['adam_steps_correct']=all(v==1 for v in c['adam_steps']);save()
v=report['native_comparison'];report['numerical_passed']=bool(v['predictions']['relative_l2']<=1e-4 and v['gradients']['relative_l2']<=.002 and v['adam_updates']['relative_l2']<=.01 and v['loss_max_abs']<=1e-4 and report['adam_steps_correct']);save()
del a,c;gc.collect();torch.cuda.empty_cache()
timing={}
for name,fn in [('reference',ref),('native',native)]:
    corr.spatial_correlation_sample=fn;model.load_state_dict(initial);optimizer=opt()
    durations=[]
    for i in range(6):
        torch.cuda.synchronize();t=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        output=model(events,256,use_dt4=False,flow_seq=False)
        loss,photo,smooth=objective_dt1(output,gray[:,0:1],gray[:,1:2],events,multiscaleloss)
        loss.backward()
        # Preserve original finite-gradient checks/synchronizations in end-to-end timing.
        norm=0.0
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all();norm+=float(p.grad.square().sum().detach().cpu())
        optimizer.step();torch.cuda.synchronize();duration=time.perf_counter()-t
        if i:durations.append(duration)
        print('MODEL_STEP',name,i,duration,flush=True)
    timing[name]=dict(seconds=durations,median_seconds=float(np.median(durations)),peak_bytes=torch.cuda.max_memory_allocated())
report['timing']=timing;report['whole_step_speedup']=timing['reference']['median_seconds']/timing['native']['median_seconds'];report['passed']=report['numerical_passed'] and report['whole_step_speedup']>1.10;save()
print('MODEL_AUDIT',json.dumps(report),flush=True)
assert report['passed']
