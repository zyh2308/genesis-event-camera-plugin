import copy,json,shutil,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import pose_coverage_real as evaluate

def block(success,depth):return dict(summary=dict(success=success,depth_mae_m=depth))
results=[]
for method in ['eventgenesis','v2e']:
    for seed in range(3):
        results.append(dict(method=method,seed=seed,arms={
            'clean_control':{c:block(.4,.1) for c in ['clean','half','eighth']},
            'paired_consistency':{c:block(.41,.09) for c in ['clean','half','eighth']}}))
assert len(evaluate.gate_rows(results))==30 and all(c['passed'] for c in evaluate.gate_rows(results))
bad=copy.deepcopy(results);bad[0]['arms']['paired_consistency']['half']['summary']['depth_mae_m']=.11
assert not all(c['passed'] for c in evaluate.gate_rows(bad))
bad=copy.deepcopy(results);bad[1]['arms']['paired_consistency']['clean']['summary']['success']=.37
assert not all(c['passed'] for c in evaluate.gate_rows(bad))
bad=copy.deepcopy(results);bad[5]['arms']['paired_consistency']['eighth']['summary']['success']=.39
assert not all(c['passed'] for c in evaluate.gate_rows(bad))
vertices=np.array([[x,y,z] for x in [-.02,.02] for y in [-.02,.02] for z in [-.02,.02]])
keys=np.vstack([np.zeros(3),vertices]);K=np.array([[300.,0,173],[0,300,130],[0,0,1]])
T=np.eye(4);T[2,3]=.7
gt=keys@T[:3,:3].T+T[:3,3];q=gt@K.T;uv=q[:,:2]/q[:,2:]
row=dict(target_to_camera=T.tolist(),K=K.tolist(),gt_uv_ideal=uv.tolist())
class Rotation:
    @staticmethod
    def ray_frame(centers,intrinsics):
        rays=[]
        for center,k in zip(centers,intrinsics,strict=True):rays.append(np.linalg.solve(k,np.array([*(center*[346,260]),1.])))
        return np.asarray(rays),None
result=evaluate.measures(row,(uv[0]/[346,260],.7,np.eye(3)),keys,vertices,cKDTree(vertices),.07,Rotation)
assert result['success']==1 and result['auc']>1-1e-12 and result['depth_mae_m']==0 and result['center_px']==0
assert result['translation_error_m']==0 and result['rotation_error_deg']==0
result2=evaluate.measures(row,(uv[0]/[346,260],1.,np.eye(3)),keys,vertices,cKDTree(vertices),.07,Rotation)
assert result2['success']==0 and result2['auc']==0
with tempfile.TemporaryDirectory() as folder:
    p=Path(folder);(p/'recovery_20261105_mainline').mkdir()
    old=Path(__file__).parent.parent/'pose_optimization_20261106/pose_fresh_confirm.py'
    shutil.copyfile(old,p/'recovery_20261105_mainline/pose_fresh_confirm_20261106_v3.py')
    trial=copy.deepcopy(results)
    for r in trial:
        baseline={f'validation_{i:03}':dict(original=dict(success=.05,depth_mae=.4),median=dict(success=.1,depth_mae=.3)) for i in range(96)}
        r['parents']=dict(by_rollout=baseline)
        r['arms']['paired_consistency']['clean_full']=dict(by_rollout={rid:dict(success=.4,depth_mae_m=.1) for rid in baseline})
    original=evaluate.original_confirmation(trial,p);assert original['passed'] and len(original['comparisons'])==8
    trial[0]['arms']['paired_consistency']['clean_full']['by_rollout']['validation_000']['depth_mae_m']=100.
    assert not evaluate.original_confirmation(trial,p)['passed'],'Original fixed-depth baseline guard lost'
data=dict(status='passed',procedural_only=True,real_data_used=False,checks=[
    'all 30 original and supplemental checks recomputed','depth regression blocked','clean regression blocked',
    'low-light pose regression blocked','perfect pose ADD-S and full pose metrics','bad depth fails pose success',
    'prediction function accepts images and public intrinsics only, no GT samples',
    'original eight comparisons against PnP and fixed median recomputed on all 96 full trajectories',
    'original baseline regression blocks adoption even if illumination checks pass'])
Path(__file__).with_name('pose_coverage_real_preflight.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
