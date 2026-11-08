#!/usr/bin/env python3
"""Record bounded analytic Census controls; not a learned-model experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from test_soft_census import analytic_pair, objective_at


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Use a new analytic receipt')
    rows=[]
    for name in ('identity','affine','gamma'):
        previous,following=analytic_pair(name)
        values=[]
        for displacement in (0.,.5,1.,1.5,2.):
            loss,gradient,support=objective_at(previous,following,displacement)
            values.append(dict(predicted_dx=displacement,census_pixel_sum=loss,derivative_wrt_dx=gradient,
                               loss_support_fraction=support))
        rows.append(dict(condition=name,true_dx=1.,true_dy=0.,dtype='float32',values=values,
            correct_flow_has_smaller_loss_than_half_and_one_half=(values[2]['census_pixel_sum']<values[1]['census_pixel_sum'] and values[2]['census_pixel_sum']<values[3]['census_pixel_sum']),
            neighboring_gradients_move_toward_correct_flow=values[1]['derivative_wrt_dx']<0<values[3]['derivative_wrt_dx']))
    here=Path(__file__).resolve().parent
    result=dict(schema='flow_soft_census_analytic_cpu_oracles_v1',status='passed',device='CPU',torch_version=torch.__version__,
        source_sha256={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('soft_census.py','test_soft_census.py','run_cpu_oracles.py')},
        rows=rows,shared_census_weight_calibrated=False,shared_census_weight=None,optimizer_updates=0,
        real_data_used=False,GT_masks_used=False,model_weights_loaded=False,SSH_connected=False,
        numerical_note='Ideal-identity unit assertion uses FP64; translations/lighting tests and these recorded curves use FP32. No bitwise-zero FP32 identity claim.',
        limits=['Analytic 1px textured translation only, not learned prediction quality or sim-to-real evidence.',
                'Soft Census is only approximately invariant to monotonic intensity transformations.',
                'Constant patches are unobservable and have zero useful motion gradient; Census cannot create missing information.',
                'Prediction/patch support is conservative; true physical occlusions are not resolved by this mask.',
                'No actual TRAIN gradient calibration performed yet; coefficient must be fixed identically for both generators before a separately authorized short training.'])
    if not all(r['correct_flow_has_smaller_loss_than_half_and_one_half'] and r['neighboring_gradients_move_toward_correct_flow'] for r in rows):
        raise RuntimeError('Analytic Census oracle failed')
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
