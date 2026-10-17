"""CPU identity/translation tests of the exact pinned photometric warp."""
from pathlib import Path
import ast,json,torch,warnings
R=Path('/home/科研/Eventbased_WAM/external_sources/STE-FlowNet')
p=R/'multiscaleloss.py';src=p.read_text();tree=ast.parse(src)
f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='warp')
class CPU(ast.NodeTransformer):
    def visit_Call(self,n):
        n=self.generic_visit(n)
        if isinstance(n.func,ast.Attribute) and n.func.attr=='cuda':return n.func.value
        return n
class Consistent(ast.NodeTransformer):
    def visit_Call(self,n):
        n=self.generic_visit(n)
        if isinstance(n.func,ast.Attribute) and n.func.attr=='grid_sample':n.keywords.append(ast.keyword(arg='align_corners',value=ast.Constant(True)))
        return n
import copy
native=CPU().visit(copy.deepcopy(f));fixed=Consistent().visit(CPU().visit(copy.deepcopy(f)));fixed.name='consistent_warp'
ns={'torch':torch,'nn':torch.nn};exec(compile(ast.fix_missing_locations(ast.Module(body=[native,fixed],type_ignores=[])),str(p),'exec'),ns)
torch.manual_seed(314);image=torch.rand(1,1,32,40);zero=torch.zeros(1,2,32,40);right=zero.clone();right[:,0]=1
interior=lambda t:t[...,2:-2,2:-2]
out={'source_file':str(p),'function_line':f.lineno,'tests':{}}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for name,fn in [('pinned_current',ns['warp']),('normalization_consistent_control',ns['consistent_warp'])]:
        identity=fn(image,zero);shifted=torch.roll(image,1,dims=-1);unshift=fn(shifted,right)
        out['tests'][name]={'zero_flow_identity_interior_MAE':float(interior(identity-image).abs().mean()),'zero_flow_identity_interior_max_error':float(interior(identity-image).abs().max()),'one_pixel_translation_interior_MAE':float(interior(unshift-image).abs().mean())}
(Path(__file__).resolve().parent/'warp_coordinate_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
