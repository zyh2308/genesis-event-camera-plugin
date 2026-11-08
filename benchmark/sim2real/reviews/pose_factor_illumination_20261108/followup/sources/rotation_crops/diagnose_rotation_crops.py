"""CPU-only frozen-export rotation/crop diagnostic. GT is never deployable input.

No forward inference, training, resampling or real evaluation. The C1 cohort
uses its saved rotation factors verbatim. An ordinary rotation geodesic remains
a label diagnostic under object symmetries; native ADD-S is the success metric.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

HELPER_SHA256 = 'ba4c13e5dc8ee3f8de3b7f5cbeb970fad4e31bdea2a49f2f15d9b612d01f9344'
R_FIELDS = {'predR':'R_pred', 'GTcenter_crop_R':'R_crop_gt_center',
            'GTsize_crop_R':'R_crop_gt_size', 'GTboth_crop_R':'R_crop_gt_center_and_size'}
TRANS = ('pred_translation', 'GTtranslation')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):h.update(block)
    return h.hexdigest()


def load_helper(directory):
    source=Path(directory)/'factor_geometry.py'
    if sha(source)!=HELPER_SHA256:raise ValueError('frozen geometry helper SHA mismatch')
    spec=importlib.util.spec_from_file_location('frozen_pose_rotation_geometry',source)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def row_key(row):
    return (str(row['method']),int(row['seed']),str(row['condition']),str(row['rid']),str(row['sample_id']))


def verify_gtboth_rotation_consistency(flat):
    """No averaging/substitution: compare exported C0/C1/slow GT-both crop R."""
    cells={}
    for row in flat:
        cells.setdefault(row_key(row),{})[row['arm']]=np.asarray(row['R_crop_gt_center_and_size'],dtype=np.float64)
    cases=[]
    for key,arms in sorted(cells.items()):
        if set(arms)!={'C0','C1','slow'}:raise ValueError('all three arms required for GT-crop consistency')
        pairs=[]
        for a,b in itertools.combinations(('C0','C1','slow'),2):
            if arms[a].shape!=(3,3) or arms[b].shape!=(3,3):raise ValueError('bad GT-crop rotation shape')
            delta=float(np.max(np.abs(arms[a]-arms[b])))
            if not np.isfinite(delta):raise ValueError('nonfinite GT-crop rotation')
            pairs.append(dict(arms=[a,b],bitwise_equal=bool(np.array_equal(arms[a],arms[b])),max_abs=delta))
        cases.append(dict(method=key[0],seed=key[1],condition=key[2],rid=key[3],sample_id=key[4],pairs=pairs))
    allpairs=[p for row in cases for p in row['pairs']]
    return dict(tolerance=1e-7,case_count=len(cases),pair_count=len(allpairs),
                all_bitwise_equal=all(p['bitwise_equal'] for p in allpairs),
                max_abs=max(p['max_abs'] for p in allpairs),
                all_within_tolerance=all(p['max_abs']<=1e-7 for p in allpairs),
                action='No result replaced or averaged; C1 own saved R always used.',cases=cases)


def _close(a,b,atol=2e-7,rtol=2e-6):
    if a is None or b is None:return a is None and b is None
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),abs_tol=atol,rel_tol=rtol)


def compare_gttranslation_macro(aggregate,factor_macro):
    target={(r['method'],int(r['seed']),r['condition']):r for r in factor_macro if r['variant']=='oracle/C1/u1z1r0'}
    actual={(r['method'],int(r['seed']),r['condition']):r for r in aggregate if r['variant']=='rotation_crop/GTtranslation/predR'}
    if len(target)!=len(actual) or set(target)!=set(actual):raise ValueError('GTtranslation/predR factor-reference inventory mismatch')
    checks=[]
    for key,row in sorted(actual.items()):
        ref=target[key]
        if set(row['by_rollout'])!=set(ref['by_rollout']):raise ValueError('factor reference rollout inventory mismatch')
        for scope in ('summary',*sorted(row['by_rollout'])):
            lhs=row['summary'] if scope=='summary' else row['by_rollout'][scope]
            rhs=ref['summary'] if scope=='summary' else ref['by_rollout'][scope]
            if set(lhs)!=set(rhs):raise ValueError('factor reference metric schema mismatch')
            for name in lhs:
                ok=lhs[name]==rhs[name] if name in ('success','sample_count','valid_adds_count') else _close(lhs[name],rhs[name])
                checks.append(dict(cell=list(key),scope=scope,metric=name,actual=lhs[name],reference=rhs[name],passed=bool(ok)))
    return dict(status='passed' if all(x['passed'] for x in checks) else 'failed',cells=len(actual),checks=checks)


def analyze(helper,flat,assets,native_scores,factor_macro,strict_inventory=True):
    records=helper.group_flat_predictions(flat)
    if strict_inventory:
        expected=set(itertools.product(('eventgenesis','v2e'),(0,1,2),('normal','half','eighth')))
        cells={}
        for record in records:cells.setdefault((record['method'],record['seed'],record['condition']),[]).append(record)
        if set(cells)!=expected or len(flat)!=3456 or len(records)!=1152:raise ValueError('full frozen 3456/1152 inventory required')
        if any(len(rows)!=64 or len({r['rid'] for r in rows})!=4 for rows in cells.values()):raise ValueError('each cell requires 64 windows / four rollouts')
    raw={(row_key(row),row['arm']):row for row in flat}
    consistency=verify_gtboth_rotation_consistency(flat)
    scored=[]
    for record in records:
        identity={k:record[k] for k in ('method','seed','condition','rid','sample_id')}
        row=raw[(row_key(record),'C1')]
        gt=helper._gt(record)
        base=record['predictions']['C1']
        for trans,rname in itertools.product(TRANS,R_FIELDS):
            p=dict(base)
            if trans=='GTtranslation':p.update(uv=gt['uv'],z_m=gt['z_m'])
            p['R']=helper._rotation(row[R_FIELDS[rname]],R_FIELDS[rname])
            scored.append(dict(identity,variant=f'rotation_crop/{trans}/{rname}',family='rotation_crop_diagnostic',
                               arm='C1',diagnostic_only=True,uses_privileged_gt=(trans=='GTtranslation' or rname!='predR'),
                               metrics=helper.score_factors(record,p,assets)))
    aggregate=helper.aggregate_rollout_equal(scored)
    # Relabel only the parity table identity, never the scientific rows/results.
    alias=[dict(row,variant='actual/C1') for row in aggregate if row['variant']=='rotation_crop/pred_translation/predR']
    native=helper.native_parity_report(alias,[r for r in native_scores if r['arm']=='C1'])
    reference=compare_gttranslation_macro(aggregate,factor_macro)
    summary=[]
    for method,condition in sorted({(r['method'],r['condition']) for r in aggregate}):
        for trans,rname in itertools.product(TRANS,R_FIELDS):
            variant=f'rotation_crop/{trans}/{rname}'
            cells=[r for r in aggregate if (r['method'],r['condition'],r['variant'])==(method,condition,variant)]
            seeds=sorted(r['seed'] for r in cells)
            if len(seeds)!=len(set(seeds)) or (strict_inventory and seeds!=[0,1,2]):raise ValueError('seed summary inventory incomplete')
            values={name:float(np.mean([r['summary'][name] for r in cells])) for name in
                    ('success','auc','rotation_error_deg','depth_mae_m','center_px','translation_error_m')}
            values['success_percent']=100*values['success']
            summary.append(dict(method=method,condition=condition,translation_condition=trans,rotation_factor=rname,
                                seeds=seeds,metrics=values,diagnostic_only=True,
                                uses_privileged_gt=(trans=='GTtranslation' or rname!='predR')))
    return dict(scored=scored,aggregate=aggregate,summary=summary,native_parity=native,
                gttranslation_reference_parity=reference,gtboth_rotation_consistency=consistency)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factor-sources',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--vertices',type=Path,required=True)
    parser.add_argument('--keypoints',type=Path,required=True)
    parser.add_argument('--diameter-m',type=float,required=True)
    parser.add_argument('--native-scores',type=Path,required=True)
    parser.add_argument('--factor-macro-json',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('fresh output required; existing results protected')
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        helper=load_helper(args.factor_sources)
        inputs={name:dict(path=str(getattr(args,name).resolve()),sha256=sha(getattr(args,name))) for name in
                ('predictions','vertices','keypoints','native_scores','factor_macro_json')}
        with args.predictions.open() as handle:flat=[json.loads(line) for line in handle if line.strip()]
        assets=helper.FactorAssets(np.load(args.vertices,allow_pickle=False),np.load(args.keypoints,allow_pickle=False),args.diameter_m)
        result=analyze(helper,flat,assets,json.loads(args.native_scores.read_text()),json.loads(args.factor_macro_json.read_text()))
        for key in ('native_parity','gttranslation_reference_parity','gtboth_rotation_consistency'):
            write_json(args.out/(key+'.json'),result[key])
        if any(result[key]['status']!='passed' for key in ('native_parity','gttranslation_reference_parity')):
            raise ValueError('native/factor numerical parity failed; evidence preserved')
        with (args.out/'per_window_diagnostic.jsonl').open('w') as handle:
            for row in result['scored']:handle.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        write_json(args.out/'macro_scores.json',result['aggregate'])
        write_json(args.out/'three_seed_summary.json',result['summary'])
        completion=dict(schema='pose_rotation_crop_cpu_diagnostic_v1',status='complete_diagnostic_only',
                        inputs=inputs,helper_sha256=HELPER_SHA256,driver_sha256=sha(Path(__file__)),
                        flat_predictions=len(flat),C1_cohort=len(result['scored'])//8,scored_variants=len(result['scored']),
                        macro_cells=len(result['aggregate']),three_seed_summary_cells=len(result['summary']),
                        native_C1_cells=result['native_parity']['checked_cells'],native_parity='passed',
                        factor_reference_cells=result['gttranslation_reference_parity']['cells'],factor_reference_parity='passed',
                        gtboth_R_all_bitwise_equal=result['gtboth_rotation_consistency']['all_bitwise_equal'],
                        gtboth_R_max_abs=result['gtboth_rotation_consistency']['max_abs'],
                        gtboth_R_within_1e_7=result['gtboth_rotation_consistency']['all_within_tolerance'],
                        optimizer_updates=0,model_forward_calls=0,real_data_used=False,
                        GT_oracles_deployable=False,scope='matched four synthetic rollouts / 64 windows each condition, not nativeclean384',
                        interpretation='If GT crop rescues dark predicted R, center/scale crop is implicated. If not, saved frozen-head low-light performance remains limited; this does not uniquely prove a training cause.')
        write_json(args.out/'completion.json',completion)
        print(json.dumps(completion))
    except BaseException:
        write_json(args.out/'failure.json',dict(status='failed',error=traceback.format_exc(),diagnostic_only=True,real_data_used=False))
        raise


if __name__=='__main__':main()
