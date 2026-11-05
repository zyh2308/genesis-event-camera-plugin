"""Compare all physical observables from two fresh static plugin processes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--require-exact',action='store_true');a=p.parse_args()
left=json.loads((a.left/'trace.json').read_text());right=json.loads((a.right/'trace.json').read_text())
differences=[]
def check(x,y,path):
    if type(x)!=type(y):differences.append({'path':path,'kind':'type'});return
    if isinstance(x,dict):
        if x.keys()!=y.keys():differences.append({'path':path,'kind':'keys'});return
        for k in x:check(x[k],y[k],path+'/'+k)
    elif isinstance(x,list):
        if len(x)!=len(y):differences.append({'path':path,'kind':'length'});return
        for i,(u,v) in enumerate(zip(x,y)):check(u,v,path+'/'+str(i))
    elif x!=y:differences.append({'path':path,'kind':'value','left':x,'right':y})
for key in ['status','passed_local_contract','case','intervals','start_index','physical_inputs','K','object_ids','original_timestamps','profile_sha256','sensor_initial','actual_public_events','actual_event_count','actual_sensor_invocations','final_sensor_state']:
    check(left[key],right[key],'/'+key)
for i,(x,y) in enumerate(zip(left['rows'],right['rows'])):
    for key in ['index','physical_left_s','physical_right_s','events','event_count','sensor_inputs','sensor_state','geometry_diagnostics','plugin_total_steps','plugin_total_events','fast_path_active','fallbacks']:
        check(x[key],y[key],'/rows/'+str(i)+'/'+key)
check(len(left['rows']),len(right['rows']),'/row_count')
u=np.load(a.left/'events.npy');v=np.load(a.right/'events.npy')
raw_equal=u.dtype==v.dtype and u.shape==v.shape and u.tobytes()==v.tobytes()
raw={'bitwise_equal':raw_equal,'shape_left':list(u.shape),'shape_right':list(v.shape)}
if u.shape==v.shape:
    delta=u.astype(np.float64)-v.astype(np.float64)
    raw.update(changed_elements=int(np.count_nonzero(delta)),max_abs=float(np.abs(delta).max(initial=0)),max_time_delta_s=float(np.abs(delta[:,0]).max(initial=0)),xy_p_equal=bool(np.array_equal(u[:,1:],v[:,1:])))
state_equal={}
with np.load(a.left/'final_sensor_state.npz') as x,np.load(a.right/'final_sensor_state.npz') as y:
    check(sorted(x.files),sorted(y.files),'/final_npz_keys')
    for k in x.files:
        state_equal[k]=x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and x[k].tobytes()==y[k].tobytes()
result=dict(status='complete',all_physical_observables_bitwise_equal=not differences and raw_equal and all(state_equal.values()),raw_events=raw,sensor_arrays_equal=state_equal,difference_count=len(differences),differences=differences,left=str(a.left),right=str(a.right),normal_vs_deterministic=[left['deterministic_algorithms'],right['deterministic_algorithms']])
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='differences'},indent=2))
raise SystemExit(2 if a.require_exact and not result['all_physical_observables_bitwise_equal'] else 0)
