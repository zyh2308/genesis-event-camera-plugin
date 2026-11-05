"""Read-only CPU check that both sensor tests saw identical optical samples."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
rows=[]
for kind,ours in [('step','ours_common_step_v2'),('smooth','ours_common_smooth_v2')]:
    for n in [32,63,126,189,252,377,754,1508]:
        q=f'N{n:04d}'
        left=a.run/ours/'protected'/q
        right=a.run/'v2e_sensor_full_v2'/kind/q
        with np.load(left/'actual_linear_inputs.npz') as x,np.load(right/'native_inputs.npz') as y:
            equal=np.array_equal(x['frames'],y['physical_radiance_samples'])
            max_error=float(np.max(np.abs(x['frames'].astype(np.float64)-y['physical_radiance_samples'])))
        same_times=np.array_equal(np.load(left/'times.npy'),np.load(right/'times.npy'))
        rows.append(dict(signal=kind,n=n,optical_samples_byte_equal=equal,
                         max_optical_abs_difference=max_error,absolute_times_byte_equal=same_times))
result=dict(all_passed=all(r['optical_samples_byte_equal'] and r['absolute_times_byte_equal'] for r in rows),rows=rows)
with (a.run/'pair_input_verification_v2.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
assert result['all_passed']
