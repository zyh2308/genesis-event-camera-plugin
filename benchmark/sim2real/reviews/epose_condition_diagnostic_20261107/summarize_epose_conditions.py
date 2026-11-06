"""Aggregate frozen E-POSE depth logs and stratified component diagnostics."""
import argparse
from itertools import permutations
import json
from pathlib import Path

import numpy as np

METHODS = ('eventgenesis', 'v2e')
SEEDS = (0, 1, 2)
FACTORS = ('depth', 'center', 'rotation')
CASES = ('full_adds_m', 'exact_z', 'exact_center', 'exact_rotation')


def full_depth(root):
    full = {}
    for m in METHODS:
        for s in SEEDS:
            tag = f'{m}_seed{s}'
            by_seq = {}
            with (root/(tag+'.jsonl')).open() as f:
                for line in f:
                    row = json.loads(line)
                    seq = row['rollout']
                    if row['predicted_depth_m']['learned'] is not None:
                        by_seq.setdefault(seq, {})[row['sample']] = (row['predicted_depth_m']['learned'], row['gt_depth_m'])
            full[tag] = by_seq
    return full


def average(values):
    return float(np.mean(values))


def score(error, threshold):
    return float(error is not None and error < threshold)


def auc(error):
    return 0.0 if error is None else max(1.0-error/.1, 0.0)


def shapley(values):
    effects = np.zeros(3)
    for order in permutations(range(3)):
        mask = 0
        for factor in order:
            next_mask = mask | (1 << factor)
            effects[factor] += (values[next_mask]-values[mask])/6
            mask = next_mask
    assert abs(effects.sum() - (values[-1]-values[0])) < 1e-12
    return effects


def merge(rows):
    methods = {}
    for m in METHODS:
        methods[m] = {key: average([r['methods'][m][key] for r in rows]) for key in rows[0]['methods'][m]}
    effects = {metric: dict(zip(FACTORS, np.mean([[r['swap'][metric][factor] for factor in FACTORS]
                                                for r in rows], axis=0).tolist(), strict=True))
               for metric in ('success', 'auc')}
    return dict(sequences=len(rows), methods=methods, swap=effects,
                real_event_count_median_per_sequence=average([r['event_count_median'] for r in rows]),
                gt_depth_mean_m=average([r['gt_depth_mean_m'] for r in rows]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--epose', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.probe/'protocol.json').read_text())
    threshold = protocol['threshold_m']
    completion = json.loads((args.probe/'completion.json').read_text())
    assert completion['status']=='complete' and completion['count']==18
    epose = json.loads((args.epose/'completion.json').read_text())['results']
    full = {(r['tag'],r['sequence']):r['summary']['learned'] for r in epose}
    depths = full_depth(args.epose)
    output = []
    maxima = {'adds_m':0.0, 'depth_m':0.0}
    for path in sorted(args.probe.glob('block__*.json')):
        seq = json.loads(path.read_text())
        name = seq['sequence']
        motion, speed, light = name.split('/')[-1].split('_')
        rows = seq['rows']
        assert len(rows)==64 and len({row['index'] for row in rows})==64
        methods = {}
        tags = [f'{m}_seed{s}' for m in METHODS for s in SEEDS]
        common = sorted(set.intersection(*(set(depths[tag][name]) for tag in tags)))
        assert common
        gt_depth = [depths[tags[0]][name][idx][1] for idx in common]
        for m in METHODS:
            tags_m = [f'{m}_seed{s}' for s in SEEDS]
            errors = [[depths[tag][name][idx][0]-depths[tag][name][idx][1] for idx in common] for tag in tags_m]
            metrics = [row['models'][tag] for row in rows for tag in tags_m]
            assert all(abs(q['log_z'] - (q['log_shape']-q['log_scale'])) < 1e-6 for q in metrics)
            maxima['adds_m'] = max(maxima['adds_m'], max(x['parity_adds_m'] for x in metrics))
            maxima['depth_m'] = max(maxima['depth_m'], max(x['parity_depth_m'] for x in metrics))
            info = dict(full_depth_mae_m=average(np.abs(errors)), full_depth_bias_m=average(errors),
                        full_success=average([full[tag,name]['success'] for tag in tags_m]),
                        full_auc=average([full[tag,name]['auc'] for tag in tags_m]),
                        sample_depth_mae_m=average([abs(x['signed_z_m']) for x in metrics]),
                        center_px=average([x['center_px'] for x in metrics]),
                        lateral_m=average([x['lateral_m'] for x in metrics]),
                        rotation_only_m=average([x['rotation_only'] for x in metrics]))
            for k in ('log_scale','log_shape','log_z'):
                info[k] = average([x[k] for x in metrics])
            for case in CASES:
                info[case+'_success'] = average([score(x[case], threshold) for x in metrics])
                info[case+'_auc'] = average([auc(x[case]) for x in metrics])
            methods[m] = info
        mixed = [row['swaps'][str(s)] for row in rows for s in SEEDS]
        swaps = dict(success=dict(zip(FACTORS, np.mean([shapley([score(e,threshold) for e in x]) for x in mixed], axis=0).tolist(), strict=True)),
                     auc=dict(zip(FACTORS, np.mean([shapley([auc(e) for e in x]) for x in mixed], axis=0).tolist(), strict=True)))
        output.append(dict(sequence=name, group=seq['group'], motion=motion, speed=speed, light=light,
                           full_common_depth_samples=len(common), sample_windows=len(rows), gt_depth_mean_m=average(gt_depth),
                           gt_depth_q10_q90_m=np.quantile(gt_depth,[.1,.9]).tolist(),
                           event_count_median=float(np.median([row['event_count'] for row in rows])),
                           methods=methods, swap=swaps))
    primary = [r for r in output if r['group']=='untouched_primary']
    conditions = {'all':merge(primary)}
    for attr, keys in [('light',('gl','ml','ll')), ('motion',('r','t')), ('speed',('01','05','10'))]:
        for key in keys:
            conditions[key] = merge([r for r in primary if r[attr]==key])
    for mo in ('r','t'):
        for li in ('gl','ml','ll'):
            conditions[mo+'_'+li] = merge([r for r in primary if r['light']==li and r['motion']==mo])
    combos = {(r['motion'],r['speed']) for r in primary if r['light']=='gl'}
    for li in ('gl','ml','ll'):
        conditions['matched_'+li] = merge([r for r in primary if r['light']==li and (r['motion'],r['speed']) in combos])
    draws = np.random.default_rng(20261107).integers(len(primary), size=(10000,len(primary)))
    intervals = {}
    for metric in ('success','auc'):
        intervals[metric] = {}
        for factor in FACTORS:
            xs = np.asarray([r['swap'][metric][factor] for r in primary])
            intervals[metric][factor] = np.quantile(xs[draws].mean(1),[.025,.975]).tolist()
    with args.out.open('x') as f:
        json.dump(dict(sequences=output, conditions=conditions, attribution_ci95=intervals, rerun_max_delta_m=maxima,
                       interpretation='positive swap contribution increases V2E minus EventGenesis metric; all orders averaged'),
                  f, indent=2, allow_nan=False)


if __name__=='__main__':
    main()
