#!/usr/bin/env python3
"""Independently check a completed temporal-weight study from its saved rows."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

SOURCE_SHA = 'ed140ca2ce789ef94dde23853d1141cca790a502aa0b56da54fd631da430c1bf'
METHODS = ('eventgenesis', 'v2e')
ARMS = ('native_control', 'temporal_weight')
STRATA = ('camera_motion_static_scene', 'independent_rigid_motion_static_camera',
          'camera_and_independent_rigid_motion', 'combined_motion_with_occlusion')
ALL, ACTIVE = 'all_physical_valid', 'common_event_active_valid'
SCENE = STRATA[2]
TAPE = ('sample_ids', 'label_augmentation_sha256', 'augmentation_seed_sequences', 'weight_sha256')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def evidence(path):
    content = path.read_bytes()
    return content, dict(path=str(path.resolve()), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def same(actual, expected):
    if actual is None or expected is None:
        return actual is None and expected is None
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def hex_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def sample_name(rid, index):
    return f'{rid}/dt1_{3 * index:03d}'


def verify_training(logs, summaries, recorded):
    replay = dict(recorded['replay'])
    replay_sha = replay.pop('replay_sha256')
    require(digest(replay) == replay_sha and recorded['actual_arms'] == list(ARMS) and
            recorded['used_replay_arm'] == 'native_control', 'Original replay binding differs')
    indices = replay['arms']['native_control']
    samples = replay['samples']
    require(len(samples) == 1584 and len(indices) == 1600 and sorted(indices[:1584]) == list(range(1584)) and
            len(set(indices[1584:])) == 16 and all(type(i) is int and 0 <= i < 1584 for i in indices), 'Wrong original replay permutation')
    rid_counts = Counter(rid for rid, _ in samples)
    require(len(rid_counts) == 24 and set(rid_counts.values()) == {66} and
            len({(rid, i) for rid, i in samples if type(i) is int and 0 <= i < 66}) == 1584, 'Wrong registered TRAIN sample inventory')
    require(all(len(rows) == 200 for rows in logs.values()), 'Require four complete 200-step logs')
    reference = logs[(METHODS[0], ARMS[0])]
    for key, rows in logs.items():
        summary = summaries[key]
        for j, row in enumerate(rows):
            require(row['additional_update'] == j + 1 and row['optimizer_update'] == 8911 + j and
                    row['finite_gradient_tensors'] == 70 and row['GT_flow_optimizer_supervision'] is False,
                    f'Wrong step or scope at {key}/{j + 1}')
            require(all(math.isfinite(row[k]) for k in ('loss', 'photometric', 'native_photometric', 'smooth', 'gradient_l2')) and
                    row['gradient_l2'] > 0, 'Nonfinite or zero gradient')
            expected_ids = [sample_name(*samples[i]) for i in indices[8*j:8*j+8]]
            epoch, batch = 45 + j // 198, j % 198
            expected_aug = [[0, epoch, batch, micro] for micro in range(8)]
            require(row['sample_ids'] == expected_ids and row['augmentation_seed_sequences'] == expected_aug and
                    replay['augmentation_seed_sequences'][8*j:8*j+8] == expected_aug, f'Wrong native sample/augmentation at {key}/{j+1}')
            require(len(row['label_augmentation_sha256']) == 8 and all(map(hex_sha, row['label_augmentation_sha256'])) and
                    hex_sha(row['weight_sha256']) and hex_sha(row['parameter_signature_sha256']), 'Missing label or weight hash')
            require(all(row[name] == reference[j][name] for name in TAPE), f'Paired replay differs at {key}/{j+1}')
            require(len(row['weight_statistics']) == 4 and row['weight_statistics'] == reference[j]['weight_statistics'] and
                    all(0.5 - 1e-6 <= s['minimum'] <= s['maximum'] <= 2 + 1e-6 and
                        0 <= s['per_window_mean_error'] <= 1e-6 for s in row['weight_statistics']), 'Weight mass/range differs')
        recomputed = {name: digest([row[name] for row in rows]) for name in TAPE}
        require(recomputed == summary['paired_sequence_sha256'], f'Arm sequence digest differs at {key}')
    require(len({row['parameter_signature_sha256'] for rows in logs.values() for row in rows}) == 1, 'Model inventory changed')
    return dict(replay_sha256=replay_sha, paired_sequence_sha256=summaries[(METHODS[0], ARMS[0])]['paired_sequence_sha256'],
                updates_per_arm=200, total_updates=800, paired_batch_size=8, total_paired_positions=1600,
                static_weight_steps=sum(all(s['minimum'] == s['maximum'] == 1 for s in r['weight_statistics']) for r in reference))


def aggregate(rows):
    names = set(rows[0]) - {'invalid_geometry_or_visibility'}
    require(all(set(row) - {'invalid_geometry_or_visibility'} == names for row in rows), 'Metric schema changed')
    result = {}
    for name in sorted(names):
        epe = [row[name]['epe_px'] for row in rows if row[name]['epe_px'] is not None]
        zero = [row[name]['zero_epe_px'] for row in rows if row[name]['zero_epe_px'] is not None]
        require(len(epe) == len(zero) and all(math.isfinite(x) and x >= 0 for x in epe + zero), 'Invalid supported EPE')
        result[name] = dict(epe_px=statistics.fmean(epe) if epe else None, zero_epe_px=statistics.fmean(zero) if zero else None,
                            supported_members=len(epe))
    return result


def verify_metrics(actual, expected):
    require(actual.keys() == expected.keys(), 'Aggregate metric inventory changed')
    for name, metrics in actual.items():
        target = expected[name]
        require(metrics['supported_members'] == target['supported_members'] and
                all(same(metrics[field], target[field]) for field in ('epe_px', 'zero_epe_px')), f'Aggregate mismatch for {name}')


def recompute_validation(val):
    require(val['window_count'] == 528 and val['real_data_used'] is False, 'Expected all 528 synthetic windows')
    rollouts = sorted(val['rollouts'], key=lambda r: r['rollout'])
    require(len(rollouts) == 8 and len({r['rollout'] for r in rollouts}) == 8 and
            Counter(r['stratum'] for r in rollouts) == Counter({s: 2 for s in STRATA}), 'Expected eight distinct rollouts, two per stratum')
    data = {}
    support_rows = []
    for row in rollouts:
        require(len(row['windows']) == 66, 'An original validation window is missing')
        support = []
        for window in row['windows']:
            require({ALL, ACTIVE} <= window.keys(), 'Missing primary support regions')
            counts = {}
            for name, m in window.items():
                require(type(m['pixels']) is int and m['pixels'] >= 0, 'Invalid pixel support count')
                counts[name] = [m['pixels'], m.get('gt_mean_motion_px'), m.get('zero_epe_px')]
                if name != 'invalid_geometry_or_visibility':
                    require((m['epe_px'] is None) == (m['pixels'] == 0) == (m['zero_epe_px'] is None), 'Unsupported row was fabricated or discarded')
            require(window[ACTIVE]['pixels'] <= window[ALL]['pixels'], 'Common support outside physical validity')
            support.append(counts)
        support_rows.append([row['rollout'], row['stratum'], support])
        regions = aggregate(row['windows'])
        verify_metrics(regions, row['regions'])
        data[row['rollout']] = dict(regions=regions, stratum=row['stratum'], windows=row['windows'])
    overall = aggregate([r['regions'] for r in data.values()])
    strata = {s: aggregate([r['regions'] for r in data.values() if r['stratum'] == s]) for s in STRATA}
    verify_metrics(overall, val['regions'])
    require(set(val['strata']) == set(STRATA), 'Incomplete strata summary')
    for s in STRATA:
        verify_metrics(strata[s], val['strata'][s])
    return dict(overall=overall, strata=strata, rollouts=data, support_sha256=digest(support_rows))


def verify_gate(arms, original):
    checks = []
    for method in METHODS:
        control, candidate = [arms[(method, arm)] for arm in ARMS]
        for region in (ALL, ACTIVE):
            a, b = control['overall'][region]['epe_px'], candidate['overall'][region]['epe_px']
            checks.append(dict(method=method, region=region, control=a, candidate=b,
                               passed=a is not None and b is not None and b < a))
        for stratum in STRATA:
            a, b = control['strata'][stratum][ALL]['epe_px'], candidate['strata'][stratum][ALL]['epe_px']
            threshold = 1.05 * a if a is not None else None
            checks.append(dict(method=method, stratum=stratum, control=a, candidate=b, threshold=threshold,
                               passed=b is not None and threshold is not None and b <= threshold))
    passed = all(c['passed'] for c in checks)
    require(len(original['checks']) == len(checks) == 12 and original['passed'] == passed, 'Original quality gate status differs')
    for ours, stored in zip(checks, original['checks'], strict=True):
        require(ours.keys() == stored.keys() and all(same(v, stored[k]) if k in ('control','candidate','threshold') else
                v == stored[k] for k, v in ours.items()), 'Original quality gate item differs')
    return dict(passed=passed, passed_items=sum(c['passed'] for c in checks), total_items=12, checks=checks)


def verify_scene(arms, report):
    keys = (('eventgenesis', 'temporal_weight'), ('eventgenesis', 'native_control'),
            ('v2e', 'temporal_weight'), ('v2e', 'native_control'))
    value_names = ('eventgenesis_temporal', 'eventgenesis_control', 'v2e_temporal', 'v2e_control')
    diff_names = ('EG_candidate_minus_EG_control', 'EG_candidate_minus_V2E_candidate')
    target = {rid for rid, row in arms[keys[0]]['rollouts'].items() if row['stratum'] == SCENE}
    require(report['stratum'] == SCENE and report['region'] == ACTIVE and report['unit'] == 'px_per_30ms' and
            report['scope'] == 'prespecified_hypothesis_on_seen_synthetic_development', 'Unregistered scene report')
    windows = {(r['rollout'], r['window_index']): r for r in report['per_window']}
    rollouts = {r['rollout']: r for r in report['per_rollout']}
    require(len(target) == 2 and len(report['per_rollout']) == 2 and set(rollouts) == target and
            len(report['per_window']) == 132 and set(windows) == set(itertools.product(target, range(66))), 'Scene report inventory differs')

    def contrasts(values):
        return [None, None] if values[0] is None else [values[0]-values[1], values[0]-values[2]]

    summary = []
    for rid in sorted(target):
        original = [arms[key]['rollouts'][rid] for key in keys]
        for index in range(66):
            record = windows[(rid, index)]
            metrics = [r['windows'][index][ACTIVE] for r in original]
            values = [m['epe_px'] for m in metrics]
            require(record['sample_id'] == sample_name(rid, index) and
                    all(record['common_support_pixels'] == m['pixels'] for m in metrics), 'Wrong common-scene sample or support')
            for name, value in zip(value_names + diff_names, values + contrasts(values), strict=True):
                require(same(record[name], value), 'Scene per-window score or paired contrast differs')
        means = [r['regions'][ACTIVE]['epe_px'] for r in original]
        require(all(m is not None for m in means) and rollouts[rid]['windows'] == 66, 'Empty paired rollout means')
        out = dict(rollout=rid, windows=66, supported_windows=original[0]['regions'][ACTIVE]['supported_members'])
        for name, value in zip(value_names + diff_names, means + contrasts(means), strict=True):
            require(same(rollouts[rid][name], value), 'Scene rollout mean/contrast differs')
            out[name] = value
        summary.append(out)
    directions = {'EG_improves_vs_own_control_in_both_rollouts': all(r[diff_names[0]] < 0 for r in summary),
                  'EG_beats_V2E_candidate_in_both_rollouts': all(r[diff_names[1]] < 0 for r in summary)}
    require(all(report[k] == v for k, v in directions.items()), 'Scene direction decision differs')
    return dict(stratum=SCENE, region=ACTIVE, unit='px_per_30ms', windows=132, paired_rollouts=summary,
                mean_EG_candidate_minus_EG_control=statistics.fmean(r[diff_names[0]] for r in summary),
                mean_EG_candidate_minus_V2E_candidate=statistics.fmean(r[diff_names[1]] for r in summary), **directions)


def verify(root):
    names = ['completion.json', 'control_replay.json', 'preregistered_scene_diagnostic.json']
    matrix = tuple(itertools.product(METHODS, ARMS))
    log_paths = {key: f'{key[0]}_{key[1]}/training_metrics.jsonl' for key in matrix}
    records = {name: evidence(root / name) for name in names + list(log_paths.values())}
    completion, replay, scene = [json.loads(records[n][0]) for n in names]
    require(completion['schema'] == 'flow_temporal_weight_paired_v1' and completion['status'] == 'complete' and
            completion['mode'] == 'study' and completion['source_manifest_sha256'] == SOURCE_SHA and
            completion['paired_weights_per_step_identical'] is True, 'Not the frozen, complete temporal-weight study')
    require(all(completion[flag] is False for flag in ('real_data_used', 'real_scores_used', 'GT_flow_optimizer_supervision',
                                                    'automatic_long_training_or_real_scoring')), 'Study scope changed')
    summaries = {(r['method'], r['arm']): r for r in completion['results']}
    require(len(completion['results']) == 4 and set(summaries) == set(matrix), 'Incomplete four-arm study')
    for row in summaries.values():
        cp = row['checkpoint']
        require(row['seed'] == 0 and row['additional_updates'] == 200 and row['finite_gradient_tensors'] == 70 and
                row['source_manifest_sha256'] == SOURCE_SHA and row['real_data_used'] is False and
                cp['additional_updates'] == 200 and cp['model_rebuild_passed'] is True and cp['optimizer_state_equal'] is True and
                hex_sha(cp['sha256']), 'Invalid arm completion or checkpoint receipt')
    for method in METHODS:
        control, candidate = [summaries[(method, arm)] for arm in ARMS]
        require(all(control[name] == candidate[name] for name in ('source_checkpoint','initial_rng','initial_lr')), 'Warm start pairing differs')
    logs = {key: [json.loads(line) for line in records[name][0].splitlines()] for key, name in log_paths.items()}
    training = verify_training(logs, summaries, replay)
    arms = {key: recompute_validation(row['validation']) for key, row in summaries.items()}
    require(len({r['support_sha256'] for r in arms.values()}) == 1, 'GT/zero-flow support inventory differs across arms')
    gate = verify_gate(arms, completion['development_quality_gate'])
    target = verify_scene(arms, scene)
    table = []
    for (method, arm), entry in arms.items():
        for split, aggregate_row in [('all', entry['overall']), *entry['strata'].items()]:
            table.append(dict(method=method, arm=arm, seed=0, stratum=split, window_count=528 if split == 'all' else 132,
                              rollout_count=8 if split == 'all' else 2, all_valid_EPE=aggregate_row[ALL]['epe_px'],
                              common_active_EPE=aggregate_row[ACTIVE]['epe_px'], zero_flow_all=aggregate_row[ALL]['zero_epe_px'],
                              zero_flow_common_active=aggregate_row[ACTIVE]['zero_epe_px']))
    for name, (_, record) in records.items():
        require(evidence(root / name)[1] == record, f'Input changed during verification: {name}')
    return dict(schema='flow_temporal_weight_independent_verification_v1', status='verified', source_manifest_sha256=SOURCE_SHA,
                scope='single_seed_seen_synthetic_development', input_artifacts={name: rec for name, (_, rec) in records.items()},
                training=training, validation_windows_per_arm=528, support_sha256=arms[matrix[0]]['support_sha256'],
                all_arm_stratum_results_px_per_30ms=table, original_development_quality_gate=gate,
                preregistered_scene=target, real_data_used=False,
                scope_note='Independent arithmetic on saved per-window scores and log fingerprints. No independent pixel-level inference, rehash of weight tensors, or reload of optimizer state.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh verification directory')
    result = verify(args.study_root)
    args.out.mkdir(parents=True)
    (args.out / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name, rows in [('all_arms_and_strata.csv', result['all_arm_stratum_results_px_per_30ms']),
                       ('scene_rollout_contrasts.csv', result['preregistered_scene']['paired_rollouts'])]:
        with (args.out / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    gate = result['original_development_quality_gate']
    print(json.dumps(dict(status=result['status'], updates=800, gate_items_passed=gate['passed_items'],
                          gate_passed=gate['passed'], scope=result['scope'])))


if __name__ == '__main__':
    main()
