"""Summarize the fixed paired readout experiment; no models or new selection."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(name, rows):
    with (ROOT / name).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    completion = read('full/completion.json')
    audit = read('independent_verification/AUDIT.json')
    gate = read('mechanism_advance_result.json')
    assert completion['status'] == 'complete' and completion['paired_cases'] == 1152
    assert completion['all_reference_checks_passed'] and completion['all_models_unchanged']
    assert audit['status'] == 'passed' and audit['success_exact']
    assert audit['completion_sha256'] == digest(ROOT / 'full/completion.json')
    assert gate['criterion_sha256'] == digest(ROOT / 'ADVANCE_CRITERION.json')
    scores = read('full/scores.json')
    cells = {(r['method'], r['seed'], r['condition'], r['arm']): r['score'] for r in scores}
    assert len(scores) == len(cells) == 36
    pair_rows, rollout_rows, means = [], [], []
    methods = ('eventgenesis', 'v2e')
    lights = ('normal', 'half', 'eighth')
    for method in methods:
        for light in lights:
            group = []
            for seed in (0, 1, 2):
                a, b = (cells[method, seed, light, arm] for arm in ('control', 'candidate'))
                row = dict(method=method, seed=seed, condition=light)
                for metric, unit in [('center_px', 1), ('success', 100), ('depth_mae_m', 100), ('auc', 1)]:
                    suffix = {'success': 'pose_success_percent', 'depth_mae_m': 'depth_mae_cm'}.get(metric, metric)
                    row['control_' + suffix] = a['summary'][metric] * unit
                    row['candidate_' + suffix] = b['summary'][metric] * unit
                pair_rows.append(row)
                group.append(row)
                assert set(a['by_rollout']) == set(b['by_rollout'])
                for rid in sorted(a['by_rollout']):
                    rollout_rows.append(dict(method=method, seed=seed, condition=light, rid=rid,
                        control_center_px=a['by_rollout'][rid]['center_px'],
                        candidate_center_px=b['by_rollout'][rid]['center_px'],
                        control_pose_success_percent=100*a['by_rollout'][rid]['success'],
                        candidate_pose_success_percent=100*b['by_rollout'][rid]['success']))
            mean = dict(method=method, condition=light, seeds='0|1|2', windows_per_seed=64, rollouts=4)
            for key in group[0]:
                if key.startswith(('control_', 'candidate_')):
                    mean[key] = sum(r[key] for r in group) / 3
            means.append(mean)
    assert len(pair_rows) == 18 and len(rollout_rows) == 72 and len(means) == 6
    save_csv('three_seed_results.csv', means)
    save_csv('paired_seed_results.csv', pair_rows)
    save_csv('paired_rollout_results.csv', rollout_rows)
    result = dict(schema='readout_fixed_cohort_summary_v1', optimizer_updates=0,
        real_data_used=False, scope='4 seen development rollouts, 64 windows/light/seed/method',
        full_completion_sha256=digest(ROOT/'full/completion.json'),
        independent_audit_sha256=digest(ROOT/'independent_verification/AUDIT.json'),
        mechanism_criterion_sha256=gate['criterion_sha256'],
        mechanism_passed=gate['advance_to_independent_synthetic_confirmation'],
        passed_checks=gate['passed_checks'], failed_checks=gate['failed_checks'],
        independent_confirmation='not_started' if not gate['advance_to_independent_synthetic_confirmation'] else 'pending',
        formal30_and8='not_retested_unchanged', long_training_started=False,
        real_scoring_started=False, three_seed_results=means,
        script_sha256=digest(Path(__file__)))
    with (ROOT/'result_summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('mechanism_passed','passed_checks','failed_checks','independent_confirmation')}))


if __name__ == '__main__':
    main()
