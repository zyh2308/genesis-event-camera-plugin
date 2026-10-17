from pathlib import Path
import hashlib,json,sys
import numpy as np,torch
torch.set_num_threads(2)
R=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real');sys.path.insert(0,str(R/'benchmark/sim2real/scripts'))
import train_steflow_stage4 as training
import run_steflow_stage2_pilot as pilot
torch.manual_seed(17)
a=torch.randn(1,2,3,4,dtype=torch.float64,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
actual=pilot.pure_torch_spatial_correlation(a,b)
patch=torch.nn.functional.unfold(b,(9,9),padding=4).reshape(1,2,81,3,4)
expected=(a[:,:,None]*patch).sum(dim=1).reshape(1,9,9,3,4)
weights=torch.randn_like(actual)
ga=torch.autograd.grad((actual*weights).sum(),(a,b),retain_graph=True)
gb=torch.autograd.grad((expected*weights).sum(),(a,b),retain_graph=True)
out={'correlation_forward_max_error':float((actual-expected).abs().max()),'correlation_gradient_max_errors':[float((x-y).abs().max()) for x,y in zip(ga,gb)],'correlation_gradcheck_fast':torch.autograd.gradcheck(pilot.pure_torch_spatial_correlation,(a,b),fast_mode=True)}
truth=np.zeros((2,5,7),np.float32);valid=np.ones((5,7),np.uint8)
negative=[]
for name,pred,gt,mask,unit in [('missing_v',truth[0],truth,valid,'px_per_interval'),('wrong_units',truth,truth,valid,'px_per_second'),('empty_mask',truth,truth,valid*0,'px_per_interval'),('NaN',truth+np.nan,truth,valid,'px_per_interval')]:
 try:pilot.compute_epe(pred,gt,mask,prediction_unit=unit,gt_unit='px_per_interval');negative.append({'case':name,'rejected':False})
 except ValueError:negative.append({'case':name,'rejected':True})
out['metric_negative_controls']=negative
sample={'events':np.arange(2*260*346*5,dtype=np.float32).reshape(1,2,260,346,5),'gray_prev':np.ones((1,260,346),np.float32),'gray_next':np.ones((1,260,346),np.float32)*2}
event,g0,g1=training.val_input(sample,0,torch.device('cpu'))
out['validation_padding_roundtrip']={'event_shape':list(event.shape),'event_exact':bool(np.array_equal(event[:,:,46:306,3:349].numpy(),sample['events'])),'gray0_exact':bool(np.array_equal(g0[:,:,46:306,3:349].numpy()[:,0],sample['gray_prev'])),'gray1_exact':bool(np.array_equal(g1[:,:,46:306,3:349].numpy()[:,0],sample['gray_next']))}
G=Path('/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2/genesis')
out['extra_source_hashes']={str(G/n):hashlib.sha256((G/n).read_bytes()).hexdigest() for n in ['ext/pyrender/renderer.py','ext/pyrender/jit_render.py','ext/pyrender/camera.py','vis/rasterizer.py']}
P=R.parent.parent
meta=json.loads((P/'runs/sim2real/20261016-stage4-flowdt1-seed0-v4/flow_dt1_seed0/eventgenesis/run_metadata.json').read_text())
inputs=[Path(meta['config_path']),Path(meta['data_manifest']),P/'data/sim2real/stage4/training_bank_v1_torch_cuda_32x2s/bank_manifest.json',P/'data/sim2real/stage4/training_bank_v1_torch_cuda_32x2s/registered_split.json']
out['input_artifact_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
print(json.dumps(out))
