"""Describe raw normal-mode variability; never silently accept a tolerance."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parent
def diff(a,b):
    r=dict(shapes_equal=a.shape==b.shape,dtypes_equal=a.dtype==b.dtype,bitwise_equal=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes())
    if a.shape==b.shape:
        r.update(xy_p_equal=bool(np.array_equal(a[:,1:],b[:,1:])),changed_time_values=int(np.count_nonzero(a[:,0]!=b[:,0])),max_time_delta_s=float(np.abs(a[:,0]-b[:,0]).max(initial=0)))
    return r
cases=[]
for case in ['s0_00','s1_00','s2_00','s3_00','train_000']:
    old=[np.load(E/'bench'/case/'frozen_gpu'/('trial'+str(t)+'_events.npy')) for t in range(3)]
    new=[np.load(E/'bench'/case/'candidate_gpu'/('trial'+str(t)+'_events.npy')) for t in range(3)]
    r=dict(case=case,old_counts=[len(x) for x in old],candidate_counts=[len(x) for x in new],old_self=[dict(trials=[i,j],**diff(old[i],old[j])) for i,j in itertools.combinations(range(3),2)],candidate_self=[dict(trials=[i,j],**diff(new[i],new[j])) for i,j in itertools.combinations(range(3),2)],cross=[dict(old_trial=i,candidate_trial=j,**diff(old[i],new[j])) for i,j in itertools.product(range(3),range(3))])
    cases.append(r)
r=dict(status='complete',cases=cases,raw_quantization_used=False,tolerance_acceptance_used=False,deterministic_exact_gate_is_separate=True,all_cross_raw_byte_equal=all(d['bitwise_equal'] for c in cases for d in c['cross']),all_cross_event_address_polarity_equal=all(d.get('xy_p_equal',False) for c in cases for d in c['cross']))
(E/'normal_mode_event_variability.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2))
