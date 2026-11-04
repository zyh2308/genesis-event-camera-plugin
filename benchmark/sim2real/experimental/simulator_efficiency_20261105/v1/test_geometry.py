"""Same-projection geometry regression for early exit and temporal chunks."""
import inspect,json,sys,textwrap
from pathlib import Path
import numpy as np,torch
root=Path(__file__).resolve().parent;P=Path('/home/zhaoyuhan/Eventbased_WAM')
sys.path.insert(0,json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())['core_path'])
import genesis_event_plugin.torch_physics as g
torch.set_num_threads(2)
original=g._splat_projected_coverage_batch_torch
text=textwrap.dedent(inspect.getsource(original)).replace('unresolved = next_unresolved','unresolved = next_unresolved\n            if not bool(torch.stack([item.any() for item in unresolved]).any()):\n                break')
ns=dict(g.__dict__);exec(text,ns);fast=ns['_splat_projected_coverage_batch_torch']
T,H,W=7,12,16;rng=np.random.default_rng(17);device='cuda:0'
frame=torch.tensor(rng.uniform(.01,1,(1,H,W)),dtype=torch.float32,device=device)
yy,xx=torch.meshgrid(torch.arange(H,device=device),torch.arange(W,device=device),indexing='ij')
alpha=torch.linspace(0,1,T,dtype=torch.float64,device=device)
report=dict(cases=[],float_atol=1e-6,mask_and_surface_ids='exact',real_data_used=False)
for case in ['identity','subpixel_motion','invalid_holes','occluding_surfaces','multiple_layers']:
    depth=torch.ones((1,H,W),device=device);uv=torch.stack([xx,yy]).float()[None,None].repeat(1,T,1,1,1)
    z=torch.ones((1,T,H,W),device=device);valid=torch.ones_like(z,dtype=torch.bool);sid=torch.zeros_like(depth,dtype=torch.long)
    if case!='identity':uv[:,:,0]+=.4;uv[:,:,1]+=.3
    if case=='invalid_holes':valid[:,:,:,5:7]=False
    if case=='occluding_surfaces':sid[:,:,:,] = (xx>7).long();z[:,:,:,8:]=2
    if case=='multiple_layers':uv[:,:,0]=torch.floor(uv[:,:,0]/3);z[:]=1+torch.arange(W,device=device)[None,None,None,:]*.03
    args=(frame,depth,uv,z,valid,frame*.9,depth,uv,z,valid)
    kwargs=dict(composite='coverage_time_log_linear',return_diagnostics=True,alpha=alpha,surface_id0=sid,surface_id1=sid)
    left=g.depth_aware_bidirectional_warp_projected_batch_torch(*args,**kwargs)
    g._splat_projected_coverage_batch_torch=fast
    try:right=g.depth_aware_bidirectional_warp_projected_batch_torch(*args,**kwargs)
    finally:g._splat_projected_coverage_batch_torch=original
    errors={};exact={}
    for k,a,b in [('output',left[0],right[0]),('valid',left[1],right[1])]+[(k,left[2][k],right[2][k]) for k in left[2]]:
        exact[k]=bool(torch.equal(a,b))
        if a.is_floating_point():
            assert torch.equal(torch.isnan(a),torch.isnan(b)) and torch.equal(torch.isinf(a),torch.isinf(b)),(case,k)
            finite=torch.isfinite(a)&torch.isfinite(b);err=float((a[finite]-b[finite]).abs().max()) if finite.any() else 0
            errors[k]=err;assert err<=1e-6,(case,k,err)
        else:assert torch.equal(a,b),(case,k)
    report['cases'].append(dict(case=case,passed=True,max_abs=errors,bitwise_equal=exact))
report['passed']=True;(root/'geometry_tests.json').write_text(json.dumps(report,indent=2)+'\n');print('GEOMETRY_TESTS_PASSED',flush=True)
