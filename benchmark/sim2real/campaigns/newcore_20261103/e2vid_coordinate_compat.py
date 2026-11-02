"""Explicit legacy pixel-grid convention for the two E2VID temporal warps."""
import ast,hashlib
from pathlib import Path
import torch
PIN='3d94483e3bc05582560efd9fa2d50f751ecff52321ec812ae04429b36c88aeb4'
def patch_temporal_sampling(module,model_root):
 path=Path(model_root)/'utils/loss.py';source=path.read_text();assert hashlib.sha256(path.read_bytes()).hexdigest()==PIN
 function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='temporal_consistency_loss')
 class ExplicitConvention(ast.NodeTransformer):
  count=0
  def visit_Call(self,node):
   self.generic_visit(node)
   if isinstance(node.func,ast.Attribute) and node.func.attr=='grid_sample':
    assert not any(k.arg=='align_corners' for k in node.keywords);node.keywords.append(ast.keyword(arg='align_corners',value=ast.Constant(True)));self.count+=1
   return node
 transform=ExplicitConvention();function=transform.visit(function);assert transform.count==2
 tree=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]));namespace=dict(module.__dict__);exec(compile(tree,'e2vid_explicit_pixel_grid','exec'),namespace);original=module.temporal_consistency_loss;fixed=namespace['temporal_consistency_loss']
 # Independent exact pixel fixtures: identity and one-pixel right translation.
 H,W=7,9;image=torch.linspace(.2,.8,H*W).reshape(1,1,H,W);processed=(.2+.6*(torch.arange(H*W).reshape(1,1,H,W)%2)).float();zero=torch.zeros((1,2,H,W))
 before=float(original(image,image,processed,processed,zero));after=float(fixed(image,image,processed,processed,zero));assert before>1e-4 and after<1e-5,(before,after)
 next_image=torch.zeros_like(image);next_image[:,:,:,1:]=image[:,:,:,:-1];next_processed=torch.zeros_like(processed);next_processed[:,:,:,1:]=processed[:,:,:,:-1];backward=zero.clone();backward[:,0]=-1
 translation=float(fixed(image,next_image,processed,next_processed,backward));assert translation<1e-5,translation
 p0=processed.clone().requires_grad_();p1=(next_processed+.02).requires_grad_();value=fixed(image,next_image,p0,p1,backward);value.backward();norm=float(p0.grad.norm()+p1.grad.norm());assert torch.isfinite(p0.grad).all() and torch.isfinite(p1.grad).all() and norm>0
 module.temporal_consistency_loss=fixed
 return {'passed':True,'upstream_file_sha256':PIN,'samplings_changed':2,'rule':'W-1/H-1 normalization requires align_corners=True','old_zero_flow_identity_loss':before,'corrected_zero_flow_identity_loss':after,'one_pixel_translation_loss':translation,'finite_nonzero_gradient_norm':norm,'upstream_files_modified':False,'other_warps_modified':False,'flow_direction_policy':'retain declared upstream negative-forward-flow approximation identically for every arm'}
