"""Check lazy metadata values, mask independence, and tensor return contract."""
import json,sys
from pathlib import Path
import numpy as np,torch
root=Path(__file__).resolve().parent;P=Path('/home/zhaoyuhan/Eventbased_WAM')
sys.path.insert(0,str(root.parent/'release_candidate_static'))
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator,RenderRequest
from release_bridge import DeferredDiagnostics,DeferredList,DeviceRenderRequest,install_geometry
torch.set_num_threads(2)
values={'float':torch.arange(7*12*16,device='cuda:0',dtype=torch.float32).reshape(7,12,16)*.001,
        'labels':torch.arange(7*12*16,device='cuda:0',dtype=torch.int64).reshape(7,12,16),
        'mask':torch.zeros((7,12,16),device='cuda:0',dtype=torch.bool)}
reference={k:v.cpu().numpy().copy() for k,v in values.items()};lazy=DeferredDiagnostics(values)
assert all(torch.is_tensor(v) for v in lazy.data.values())
assert np.array_equal(lazy['mask'],reference['mask']) and torch.is_tensor(lazy.data['float'])
exported=lazy.materialize();assert all(np.array_equal(exported[k],reference[k]) for k in reference)
same_mask=values['mask'];other=DeferredDiagnostics({'a':same_mask,'b':same_mask})
other['a'][0,0,0]=True;assert not other['b'][0,0,0]
sequence=DeferredList([values['float'],values['labels']]);assert np.array_equal(sequence[0],reference['float'])
request=DeviceRenderRequest(t=1.,alpha=.2,reason='check',mask=same_mask[0],diagnostics={'invalid_fraction':0.},available_at=1.1)
assert isinstance(request,RenderRequest);request.as_dict();assert torch.is_tensor(request._mask)
assert np.array_equal(request.mask,reference['mask'][0])
H,W=12,16;K=np.array([[30.,0,7.5],[0,30.,5.5],[0,0,1.]])
interpolator=PhysicsInterpolator(K,device='cuda:0',warp_backend='torch',composite='coverage_time_log_linear',max_photometric_interval_s=.01)
install_geometry(interpolator,'exact');frame=np.full((H,W),.2,np.float32);depth=np.ones((H,W),np.float32);seg=np.zeros((H,W),np.int64);pose=np.eye(4)
frames,times=interpolator.interpolate(frame,depth,seg,frame,depth,seg,pose,pose,None,0.,.01,return_torch=True)
assert torch.is_tensor(frames) and frames.is_cuda and frames is interpolator.last_frames_torch
reference_interpolator=PhysicsInterpolator(K,device='cuda:0',warp_backend='torch',composite='coverage_time_log_linear',max_photometric_interval_s=.01)
expected,expected_times=reference_interpolator.interpolate(frame,depth,seg,frame,depth,seg,pose,pose,None,0.,.01,return_torch=True)
assert frames.shape[-2:]==(H,W) and torch.equal(frames,expected) and times==expected_times
numpy_frames,numpy_times=interpolator.interpolate(frame,depth,seg,frame,depth,seg,pose,pose,None,0.,.01)
assert all(isinstance(f,np.ndarray) for f in numpy_frames) and numpy_times==expected_times
report=dict(numpy_default_api_preserved=True,passed=True,diagnostics_exact=True,access_is_lazy=True,mask_arrays_independent=True,render_request_compatible=True,tensor_return_attribute_same_object=True,real_data_used=False)
(root/'deferred_tests_final.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
