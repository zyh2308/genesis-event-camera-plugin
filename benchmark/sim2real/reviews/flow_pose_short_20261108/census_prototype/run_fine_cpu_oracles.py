#!/usr/bin/env python3
"""Retain fine displacement/gradient curves and their limitations, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
from test_soft_census import analytic_pair, objective_at


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Refusing existing fine-oracle receipt')
    curves={}
    for condition in ('identity','affine','gamma'):
        pair=analytic_pair(condition)
        curves[condition]=[dict(predicted_dx=d,loss=values[0],gradient=values[1],support_fraction=values[2])
            for d in (.9,.95,.99,1.,1.01,1.05,1.1) for values in [objective_at(*pair,d)]]
    here=Path(__file__).resolve().parent
    r=dict(schema='flow_soft_census_fine_analytic_cpu_v1',device='CPU FP32',curves=curves,
        true_displacement_px=1.,source_sha256={n:hashlib.sha256((here/n).read_bytes()).hexdigest()
            for n in ('soft_census.py','test_soft_census.py','run_fine_cpu_oracles.py')},
        note='Soft Census is approximately, not exactly, invariant. Loss at true geometry need not have exactly zero derivative. Predicted mask support changes discretely across image boundary crossings, so loss values with different support counts are not an unbiased optical-flow accuracy metric.',
        neighboring_signs_correct_at_point99_and1point01=all(v[2]['gradient']<0<v[4]['gradient'] for v in curves.values()),
        no_claim_of_exact_unbiased_subpixel_optimum=True,optimizer_updates=0,real_data_used=False,GT_masks_used=False,
        model_weights_loaded=False,shared_weight_calibrated=False)
    with a.out.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(r,indent=2,allow_nan=False))


if __name__=='__main__':main()
