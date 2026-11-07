"""Procedural CPU checks; no server data or real evaluation."""
import hashlib,io,json
from pathlib import Path
import numpy as np
import torch
import pose_scale_pilot as scale
import pose_independent_depth_consistency as study

root=Path(__file__).parent
torch.set_num_threads(2);torch.manual_seed(7841)
x=torch.rand(32,3,128,128)
y=torch.tensor([.5,.5,-2.3,0.,0.,1.,1.,.5,.5]).repeat(32,1)
base=scale.CenterScale().eval();state={k:v.clone() for k,v in base.state_dict().items()}
a,b,labels=study.matched_affine(x,x,y,np.random.default_rng(71070))
assert torch.equal(a,b)
assert labels.shape==y.shape and torch.isfinite(labels).all()
assert not torch.equal(labels[:,:2],y[:,:2])
assert torch.equal(labels[:,4],y[:,4]),'Geometric resampling must not change depth'
assert torch.allclose(labels[:,3]-y[:,3],(labels[:,5]/y[:,5]).log(),atol=1e-6)
with torch.no_grad():frozen_uv=base(x)[1].clone()
trained,history=study.fit(scale,state,x,y,x*.5,y,0,True,x,2)
assert all(torch.equal(v,state[k]) for k,v in base.state_dict().items())
assert any(not torch.equal(v,state[k]) for k,v in trained.state_dict().items())
assert torch.equal(frozen_uv,base(x)[1])
stream=io.BytesIO();torch.save(trained.state_dict(),stream);stream.seek(0)
restored=scale.CenterScale().eval();restored.load_state_dict(torch.load(stream,weights_only=True),strict=True)
with torch.no_grad():
    for first,second in zip(trained(x),restored(x),strict=True):assert torch.equal(first,second)
result=dict(status='passed',device='CPU',procedural_only=True,real_data_used=False,
    checks=['matched affine image and labels','transformed intrinsics match geometric zoom',
            'affine depth invariant','finite actual optimizer update','frozen final-localizer tensors and predictions',
            'depth model parameters changed','strict checkpoint round trip'],loss=history,
    trainer_sha256=hashlib.sha256((root/'pose_independent_depth_consistency.py').read_bytes()).hexdigest())
(root/'independent_depth_local_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
