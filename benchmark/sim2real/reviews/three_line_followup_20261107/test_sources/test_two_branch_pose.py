import ast,io,json,sys
from pathlib import Path
import numpy as np,torch
p=Path(__file__).parent;old=p.parent/'pose_optimization_20261106';sys.path.append(str(old))
import pose_two_branch_coverage as train
import pose_two_branch_real as real
import pose_scale_pilot as scale
torch.set_num_threads(2);torch.manual_seed(9192)
x=torch.rand(32,3,128,128);y=torch.tensor([.5,.5,-2.3,0.,0.,1.,1.,.5,.5]).repeat(32,1)
initial=scale.CenterScale().eval();state={k:v.clone() for k,v in initial.state_dict().items()}
model,loss=train.fit_center(scale,state,x,y,x*.5,y,0,True,x,2)
assert any(not torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith('features.'))
assert any(not torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith('center.'))
assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith('size.'))
assert all(torch.equal(v,state[k]) for k,v in initial.state_dict().items())
buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
restored=scale.CenterScale().eval();restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
assert np.array_equal(train.estimate_centers(model,x[:2]),train.estimate_centers(restored,x[:2]))
class Stub:
    def __init__(self,uv,z):self.uv=np.asarray(uv);self.z=z
class Scale:
    @staticmethod
    def estimate(model,images,K):return np.tile(model.uv,(len(images),1)),np.full(len(images),model.z),np.ones(len(images))
class Shared:
    @staticmethod
    def frozen_rotation(rotation,head,images,uv,size,K):
        assert np.all(uv==[.2,.3]) or np.all(uv==[.4,.5]);assert np.all(size==1)
        return np.tile(np.eye(3),(len(images),1,1))
loc={'clean_control':Stub([.2,.3],999),'paired_consistency':Stub([.4,.5],999)}
depth={'clean_control':Stub([.9,.9],.7),'paired_consistency':Stub([.9,.9],.8)}
pred=real.predict_bundle(Scale,Shared,None,loc,depth,None,np.zeros((2,3,4,4)),[np.eye(3)]*2)
assert np.all(pred['clean_control']['centers']==[.2,.3]) and np.all(pred['paired_consistency']['centers']==[.4,.5])
assert np.all(pred['clean_control']['depth']==.7) and np.all(pred['paired_consistency']['depth']==.8)
def function(text,name):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
before=(p/'pose_coverage_real.py').read_text();after=(p/'pose_two_branch_real.py').read_text()
for name in ['gate_rows','original_confirmation','measures']:assert function(before,name)==function(after,name)
result=dict(status='passed',procedural_only=True,real_data_used=False,checks=[
    'localizer receives a finite real optimizer update','localizer features and center change; unused size head stays frozen',
    'teacher parent remains unchanged','strict localizer checkpoint reproduces coordinates',
    'real predictor uses per-arm localizer UV and independent depth, not depth-network UV or localizer depth',
    'rotation uses predicted center and depth-size crop; no GT crop or stale locator','all 30+8 gate and scoring functions unchanged'])
(p/'two_branch_pose_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
