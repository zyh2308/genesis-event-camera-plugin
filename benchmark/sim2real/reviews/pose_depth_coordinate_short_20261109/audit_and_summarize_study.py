"""Read a completed depth-coordinate trial and verify its pinned evidence using stdlib."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


METHODS = ('eventgenesis', 'v2e')
SEEDS = (0, 1, 2)
CONDITIONS = ('clean', 'matched_normal', 'half', 'eighth', 'clean_full')
NATIVE_METRICS = ('depth_mae_m', 'depth_bias_m', 'log_size_mae', 'log_size_bias', 'center_px', 'success', 'auc')
DRAW_FIELDS = ('update', 'normal_indices', 'auxiliary_indices', 'half_mask')
PARITY_FIELDS = ('initial_source_to_coordinate_parity', 'initial_coordinate_to_native_parity',
                 'final_coordinate_to_native_parity')
SOURCE_SHA = '60541460cb653e712c7ad7bcd32c65d62ed72b28439153fd471d799298658cb2'
FOCUS = ('validation_031', 'validation_063')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_record(record):
    path = Path(record['path'])
    require(path.is_file() and path.stat().st_size == record['size_bytes'] and sha(path) == record['sha256'],
            'Artifact bytes differ: ' + str(path))
    return path


def load(path, expected_sha=None):
    if expected_sha is not None:
        require(sha(path) == expected_sha, 'Pinned receipt differs: ' + str(path))
    return json.loads(Path(path).read_text())


def index(rows, key, expected):
    by_key = {key(row): row for row in rows}
    require(len(rows) == len(by_key) and by_key.keys() == expected, 'Incomplete or duplicate result inventory')
    return by_key


def audit_logs(candidate_path, original_path):
    digest = hashlib.sha256()
    exposure = dict(normal=0, half=0, eighth=0)
    epoch_draws = []
    with candidate_path.open() as new, original_path.open() as old:
        for step, (line, reference_line) in enumerate(zip(new, old, strict=True), 1):
            row, reference = json.loads(line), json.loads(reference_line)
            require(step <= 288 and row['update'] == reference['update'] == step, 'Noncontiguous optimizer log')
            require(all(row[key] == reference[key] for key in DRAW_FIELDS), 'Candidate sample draw differs from C1')
            require(all(math.isfinite(row[key]) for key in ('loss', 'supervised', 'clean_retention', 'gradient_l2')),
                    'Nonfinite training log')
            require(row['gradient_l2'] > 0 and row['finite_gradient_tensors'] == reference['finite_gradient_tensors'] == 24,
                    'Unexpected gradient inventory')
            normal = row['normal_indices']
            auxiliary = row['auxiliary_indices']
            half = row['half_mask']
            batch = 25 if step % 48 == 0 else 32
            require(len(normal) == len(auxiliary) == len(half) == batch, 'Wrong C1 batch inventory')
            require(all(isinstance(flag, bool) and 0 <= i < 3058 and (i < 1529) == flag
                        for i, flag in zip(auxiliary, half, strict=True)), 'Wrong paired illumination sampling')
            epoch_draws.extend(normal)
            if step % 48 == 0:
                require(sorted(epoch_draws) == list(range(1529)), 'Original epoch coverage differs')
                epoch_draws.clear()
            exposure['normal'] += batch
            exposure['half'] += sum(half)
            exposure['eighth'] += batch - sum(half)
            digest.update(json.dumps({key: row[key] for key in DRAW_FIELDS}, separators=(',', ':')).encode())
            if step == 1:
                first = {key: abs(row[key] - reference[key]) for key in ('loss', 'supervised', 'clean_retention')}
                limit = 2e-6 + 1e-6 * abs(reference['loss'])
                require(all(value <= limit for value in first.values()), 'First native scalar parity differs')
    require(exposure['normal'] == 1529 * 6 and not epoch_draws, 'Incomplete 288-update log')
    return dict(updates=288, draws_sha256=digest.hexdigest(), exposure=exposure,
                first_native_loss_absolute_errors=first, first_native_loss_tolerance=limit)


def close_metrics(observed, expected):
    require(observed.keys() == expected.keys() == set(NATIVE_METRICS), 'Native metric inventory differs')
    for name, value in observed.items():
        reference = expected[name]
        require(math.isfinite(value) and abs(value - reference) <= 1e-8 + 1e-6 * abs(reference),
                'Immutable C1 metric differs: ' + name)


def verify_formal_gates(results, gate30, gate8):
    checks = []
    for result in results:
        old = result['arms']['clean_control']
        new = result['arms']['paired_consistency']
        key = dict(method=result['method'], seed=result['seed'])
        checks.append(dict(**key, condition='clean', passed=new['clean']['summary']['success'] - old['clean']['summary']['success'] >= -.02))
        for condition in ('half', 'eighth'):
            a, b = old[condition]['summary'], new[condition]['summary']
            checks.append(dict(**key, condition=condition, passed=b['depth_mae_m'] < a['depth_mae_m']))
            checks.append(dict(**key, condition=condition + '_pose', passed=b['success'] >= a['success']))
    require(len(checks) == 30 and checks == gate30['checks'] and gate30['passed'] is all(r['passed'] for r in checks),
            'Original 30-check flags differ from independent recomputation')
    expected = {(m, b, k) for m in METHODS for b in ('original', 'median') for k in ('depth_mae', 'success')}
    index(gate8['comparisons'], lambda r: (r['method'], r['baseline'], r['metric']), expected)
    for row in gate8['comparisons']:
        metric = row['metric']
        candidate_key = 'depth_mae_m' if metric == 'depth_mae' else 'success'
        sign = -1 if metric == 'depth_mae' else 1
        group = sorted((r for r in results if r['method'] == row['method']), key=lambda r: r['seed'])
        differences = []
        for result in group:
            learned = result['arms']['paired_consistency']['clean_full']['by_rollout']
            parent = result['parents']['by_rollout']
            require(len(learned) == 96 and learned.keys() == parent.keys(), 'Original confirmation cohort differs')
            differences.append(statistics.mean(sign * (value[candidate_key] - parent[rid][row['baseline']][metric])
                                               for rid, value in learned.items()))
        require(len(row['seed_improvement']) == 3 and all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                for a, b in zip(differences, row['seed_improvement'], strict=True)), 'Original confirmation seed differences differ')
        low, high = row['ci']
        require(math.isfinite(low) and math.isfinite(high) and low <= high, 'Invalid original bootstrap interval')
        require(row['passed'] is (all(value > 0 for value in row['seed_improvement']) and low > 0),
                'Original eight-comparison flag differs')
    require(gate8['passed'] is all(r['passed'] for r in gate8['comparisons']), 'Original confirmation aggregate differs')


def comparison_row(result, condition, old, new, rid=None):
    require(old.keys() == new.keys() == set(NATIVE_METRICS), 'Incomplete native pose metric set')
    require(all(math.isfinite(v) for v in (*old.values(), *new.values())), 'Nonfinite native pose result')
    require(old['center_px'] == new['center_px'], 'Frozen independent C1 center differs')
    row = dict(method=result['method'], seed=result['seed'], condition=condition)
    if rid is not None:
        row['rid'] = rid
    for metric in NATIVE_METRICS:
        name = metric
        multiplier = 1
        if metric.startswith('depth_'):
            name = metric.removesuffix('_m') + '_cm'
            multiplier = 100
        elif metric == 'success':
            name = 'success_pct'
            multiplier = 100
        row['C1_' + name] = old[metric] * multiplier
        row['candidate_' + name] = new[metric] * multiplier
        row['candidate_minus_C1_' + name] = (new[metric] - old[metric]) * multiplier
    return row


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('completion', 'contract', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--completion-sha256', required=True)
    args = parser.parse_args()
    require(not args.out.exists() and args.out.parent.is_dir(), 'Fresh output directory required')
    study = load(args.completion, args.completion_sha256)
    contract = load(args.contract, study['contract_sha256'])
    require(study['schema'] == contract['schema'] == 'pose_depth_coordinate_short_v1'
            and study['status'] == 'complete_synthetic_only' and study['phase'] == 'study'
            and study['new_optimizer_updates'] == 1728 and study['source_manifest_sha256'] == SOURCE_SHA,
            'Wrong study, budget or locked source')
    require(study['real_data_used'] is False and study['independent32_read'] is False
            and contract['real_scoring_enabled'] is False and contract['independent32_read'] is False,
            'Unexpected data role')
    prior = Path(contract['prior_study'])
    prior_receipt = load(prior / 'priority_execution_receipt.json', contract['prior_execution_receipt_sha256'])
    prior_completion = load(prior / 'completion.json', contract['prior_study_completion_sha256'])
    forward = load(contract['forward_completion'], contract['forward_completion_sha256'])
    native_scores = next(r for r in forward['artifacts'] if Path(r['path']).name == 'native_scores.json')
    references = {(r['method'], r['seed']): r['score'] for r in load(verify_record(native_scores))
                  if r['arm'] == 'C1' and r['condition'] == 'normal'}
    keys = {(method, seed) for method in METHODS for seed in SEEDS}
    row_key = lambda r: (r['method'], r['seed'])
    original_depth = index([r for r in prior_receipt['records'] if r['arm'] == 'physical_supervised' and r['branch'] == 'depth'], row_key, keys)
    original_center = index([r for r in prior_receipt['records'] if r['arm'] == 'physical_supervised' and r['branch'] == 'center'], row_key, keys)
    original_results = index(prior_completion['results'], row_key, keys)
    admissions = index(study['full_C1_depth_tape_admissions'], row_key, keys)
    records = index(study['records'], row_key, keys)
    results = index(study['results'], row_key, keys)
    proofs, seed_rows, rollout_rows = [], [], []
    for key in sorted(keys):
        method, seed = key
        record, admission, result = records[key], admissions[key], results[key]
        original = original_depth[key]
        require(record['updates'] == 288 and record['teacher_unchanged'] and record['parameter_names_unchanged']
                and record['independent_model_adam_readback'] and record['all_labels_and_draws_match_C1']
                and record['native_validation_contract_passed'] and record['resume_supported'] is False,
                'Training or readback receipt is incomplete')
        require(admission['original_depth_tape_exact'] and record['labels_sha256'] == admission['labels_sha256'] == original['labels_sha256']
                and record['source_state_sha256'] == admission['source_state_sha256'] == original['source_state_sha256'],
                'Original C1 label or initialization receipt differs')
        require(admission['original_C1_final_state_sha256'] == original['final_state_sha256']
                and result['identical_center_state_sha256'] == original_center[key]['final_state_sha256'], 'Original C1 weights differ')
        for name in PARITY_FIELDS:
            values = record[name]
            require(len(values) == 6 and all(math.isfinite(v) and 0 <= v <= 2e-6 for v in values.values()),
                    'Depth coordinate parity receipt differs')
        artifacts = {name: record[name] for name in ('training_state', 'final_model', 'training_log')}
        paths = {name: verify_record(item) for name, item in artifacts.items()}
        old_path = verify_record(admission['original_log'])
        verify_record(admission['source_model'])
        log_proof = audit_logs(paths['training_log'], old_path)
        require(record['exposure'] == log_proof['exposure'] and record['first_native_loss_oracle']['passed']
                and record['first_native_loss_oracle']['absolute_errors'] == log_proof['first_native_loss_absolute_errors'],
                'Training log and completion receipt disagree')
        proofs.append(dict(method=method, seed=seed, labels_sha256=record['labels_sha256'], artifacts=artifacts, **log_proof))
        require(result['C1_native_summary_and_per_rollout_parity'] and result['inventories'] == record['native_validation_inventory'],
                'Original scoring admission differs')
        require(result['parents'] == original_results[key]['parents'], 'Original confirmation parents differ')
        for condition in CONDITIONS:
            a = result['arms']['physical_supervised'][condition]
            b = result['arms']['paired_consistency'][condition]
            reference = references[key] if condition == 'matched_normal' else original_results[key]['arms']['physical_supervised'][condition]
            require(a['by_rollout'].keys() == b['by_rollout'].keys() == reference['by_rollout'].keys(), 'Rollout scope differs')
            inventory = result['inventories'][condition]
            count = 96 if condition in ('clean', 'clean_full') else 4
            require(inventory['rollouts'] == len(a['by_rollout']) == count, 'Unexpected native rollout inventory')
            if condition != 'clean_full':
                require(inventory['windows'] == (384 if condition == 'clean' else 64), 'Unexpected native window inventory')
            close_metrics(a['summary'], reference['summary'])
            seed_rows.append(comparison_row(result, condition, a['summary'], b['summary']))
            for rid, values in a['by_rollout'].items():
                close_metrics(values, reference['by_rollout'][rid])
                rollout_rows.append(comparison_row(result, condition, values, b['by_rollout'][rid], rid))
    for seed in SEEDS:
        pair = [r for r in proofs if r['seed'] == seed]
        require(len({r['draws_sha256'] for r in pair}) == len({r['labels_sha256'] for r in pair}) == 1, 'Cross-generator tape differs')
    gate30, gate8 = study['original_thirty_gate'], study['original_eight_confirmation']
    verify_formal_gates(study['results'], gate30, gate8)
    means, paired = [], []
    value_keys = [key for key in seed_rows[0] if key not in ('method', 'seed', 'condition')]
    for method in METHODS:
        for condition in CONDITIONS:
            group = [r for r in seed_rows if (r['method'], r['condition']) == (method, condition)]
            means.append(dict(method=method, condition=condition, seeds=list(SEEDS),
                              **{name: statistics.mean(row[name] for row in group) for name in value_keys}))
    for seed in SEEDS:
        for condition in CONDITIONS:
            a, b = [next(r for r in seed_rows if (r['method'], r['seed'], r['condition']) == (method, seed, condition)) for method in METHODS]
            paired.append(dict(seed=seed, condition=condition,
                               **{'EG_minus_V2E_' + name: a[name] - b[name] for name in value_keys}))
    focus = [row for row in rollout_rows if row['rid'] in FOCUS and row['condition'] in CONDITIONS[1:4]]
    require(len(seed_rows) == 30 and len(focus) == 36, 'Incomplete condition or focal-rollout table')
    args.out.mkdir()
    write_csv(args.out / 'per_seed_condition.csv', seed_rows)
    write_csv(args.out / 'all_rollouts.csv', rollout_rows)
    write_csv(args.out / 'fixed_031_063.csv', focus)
    write_csv(args.out / 'paired_generators_by_seed.csv', paired)
    summary = dict(status='audited_complete_synthetic_study', completion_sha256=args.completion_sha256,
                   source_sha256=sha(__file__), verified_optimizer_updates=sum(r['updates'] for r in proofs),
                   training_and_checkpoint_proofs=proofs, original_thirty_gate=gate30, original_eight_confirmation=gate8,
                   original_eight_bootstrap_interval_source='frozen native receipt; seed deltas and flags independently checked',
                   checkpoint_readback_scope='actual file SHA rehashed; model and Adam rebuild comes from the bound runtime receipt',
                   labels_scope='candidate and original C1 SHA receipts agree; all per-step draws compared',
                   independent32_read=False, real_data_used=False, three_seed_means=means,
                   matched_generator_differences=paired, fixed_031_063=focus)
    with (args.out / 'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=summary['status'], updates=summary['verified_optimizer_updates'],
                          passed_30=sum(row['passed'] for row in gate30['checks']),
                          passed_8=sum(row['passed'] for row in gate8['comparisons']))))


if __name__ == '__main__':
    main()
