"""Reuse the immutable actual-sensor tests with registered common-domain fixtures.

Only test callbacks are changed. Simulator code, profiles, arithmetic and RNG are
untouched. Run in separate processes for step/ramp because their oracles differ.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib
import json
import sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--harness-root',type=Path,required=True)
    p.add_argument('--signal',choices=['step','ramp'],required=True)
    p.add_argument('--illumination',type=float,nargs='+',default=[.12,.4,.65])
    a,rest=p.parse_known_args()
    hr=a.harness_root.resolve(strict=True)
    pins={'analytic_sensor_sweep.py':'296e10b6350a6e72da60d5f971dcc926b7734d228ceb0c44c8d5525309dbf1bc',
          'smooth_sensor_sweep.py':'85042087baa0290813a44c77972ff4275f812611ee30b55537a4674824f2c85ec'}
    for name,pin in pins.items():
        assert hashlib.sha256((hr/name).read_bytes()).hexdigest()==pin
    sys.path.insert(0,str(hr))
    module=importlib.import_module('analytic_sensor_sweep' if a.signal=='step' else 'smooth_sensor_sweep')
    runner=importlib.import_module('analytic_sensor_sweep')
    templates=[x for x in module.make_cases()['protected'] if x['illumination']==.4]
    def common_cases():
        cases=[]
        for light in a.illumination:
            for old in templates:
                c=copy.deepcopy(old)
                c['name']=c['name'].replace('I0.4',f'I{light:g}')
                c['illumination']=light
                if c['sign']:
                    initial,final=runner.levels(light,c['sign'],c['nominal_delta_log'])
                    c['initial']=initial
                    if a.signal=='step':c['changes']=[[c['phase_s'],final]]
                    else:
                        import math
                        c.update(final=final,effective_delta_log=math.log(final)-math.log(initial))
                    assert min(initial,final)*255>20 and max(initial,final)*255<255
                else:
                    import numpy as np
                    c['initial']=float(np.float32(light))
                    if a.signal=='step':c['changes']=[]
                    else:c['final']=c['initial']
                cases.append(c)
        return {'protected':cases}
    registered=common_cases()
    out=Path(rest[rest.index('--out')+1])
    registration=out.parent/(out.name+'_common_fixture_registration.json')
    registration.parent.mkdir(parents=True,exist_ok=True)
    with registration.open('x') as f:
        json.dump(dict(signal=a.signal,illumination=a.illumination,DN_per_optical_unit=255,
                       all_trajectory_DN_above20_below255=True,fixture_cases=registered,
                       original_harness_pins=pins,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,indent=2)
    module.make_cases=common_cases
    sys.argv=[str(module.__file__)]+rest
    module.main()

if __name__=='__main__':main()
