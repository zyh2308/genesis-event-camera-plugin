"""Recompute the fixed new32 pose scores and registered intervals on CPU."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

REG_SHA = '8acbe3f76b841e908814b8c1d53c2c6ebabafa8e6dd78a5a69f8729d6e3540ef'
EVENT_SHA = 'ed9257f02abeeca975aa8d5563eb4704e87980c1412c87538acd8c53540d4f7c'
SCIENCE_SHA = 'cdb5a83e5c80798a700f5c001b93d7994ab2c1b272212e6437f80342f07c3c52'
ASSET_PROTOCOL_SHA = 'a08df53da85c489dd84af31ad1d9d56f1a316b8830d0941876dbf46b77aaca75'
METHODS = ('eventgenesis', 'v2e')
SEEDS = (0, 1, 2)
LIGHTS = ('normal', 'half', 'eighth')
RIDS = tuple(f'validation_{i:03d}' for i in range(32))
INDICES = tuple(range(4, 20))
CELLS = set(itertools.product(METHODS, SEEDS, LIGHTS))
METRICS = ('depth_mae_m', 'depth_bias_m', 'log_size_mae', 'log_size_bias', 'center_px', 'success', 'auc')
PIXELS = [346, 260]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(record):
    path = Path(record['path'])
    digest = record['sha256']
    require(re.fullmatch('[0-9a-f]{64}', digest), f'Invalid SHA256: {path}')
    require(path.is_file() and ('size_bytes' not in record or path.stat().st_size == record['size_bytes']), f'Input missing or size changed: {path}')
    require(sha(path) == digest, f'Input hash mismatch: {path}')
    return path


def by_key(rows, key, expected):
    indexed = {key(row): row for row in rows}
    require(len(rows) == len(indexed) and set(indexed) == expected, 'Missing, duplicate or extra matrix entries')
    return indexed


def cell_key(row):
    return row['method'], row['seed'], row['condition']


def h5_geometry(path, rid):
    fields = ('camera_intrinsics_K_px', 'cad_keypoints', 'camera_to_world_cv', 'target_pose', 'frame_timestamps_s', 'projected_centroid_uv', 'projected_corners_uv')
    with h5py.File(path, 'r') as stream:
        group = stream['episodes'][rid]
        return tuple(group[name][...] for name in fields)


def ground_truth(banks, event_artifacts, vertices):
    labels, included, excluded = {}, [], []
    for rid in RIDS:
        matched = []
        for condition in LIGHTS:
            path = Path(banks[condition]) / 'physical' / f'{rid}.h5'
            matched.append(h5_geometry(checked(event_artifacts[str(path)]), rid))
        normal = matched[0]
        for relit in matched[1:]:
            for a, b in zip(normal, relit, strict=True):
                np.testing.assert_array_equal(a, b)
        K, keys, cameras, poses, timestamps, centers, corners = normal
        require(K.shape == (3, 3) and keys.shape == (9, 3), f'Unexpected geometry shape for {rid}')
        require(timestamps.shape == (21,) and (np.diff(timestamps) > 0).all(), f'Unexpected time inventory for {rid}')
        for k in INDICES:
            sid = f'{rid}/{k:03d}'
            frame = k + 1
            transform = np.linalg.inv(cameras[frame]) @ poses[frame]
            key_camera = keys @ transform[:3, :3].T + transform[:3, 3]
            projection = key_camera @ K.T
            uv = projection[:, :2] / projection[:, 2:]
            require(np.isfinite(uv).all() and (key_camera[:, 2] > .03).all(), f'Invalid original physical geometry for {sid}')
            np.testing.assert_allclose(uv, np.vstack([centers[frame], corners[frame]]), atol=1e-7)
            if not ((uv[0] > 0).all() and (uv[0] < PIXELS).all()):
                excluded.append(sid)
                continue
            truth = vertices @ transform[:3, :3].T + transform[:3, 3]
            projected = truth @ K.T
            require((truth[:, 2] > 0).all(), f'Invalid GT CAD depth for {sid}')
            projected_uv = projected[:, :2] / projected[:, 2:] / PIXELS
            log_size = np.float32(np.log(np.ptp(projected_uv, axis=0)).mean())
            labels[sid] = (K, keys, transform, (uv[0] / PIXELS).astype(np.float32), log_size, truth)
            included.append(sid)
    require({sid.split('/')[0] for sid in included} == set(RIDS), 'A registered trajectory has no eligible windows')
    return labels, included, excluded


def measure(row, label, tree, diameter):
    K, keys, T, gt_uv, target_log_size, truth = label
    uv = np.asarray(row['pred_uv'], dtype=np.float32)
    R = np.asarray(row['R_pred'], dtype=np.float64)
    z, size = np.float32(row['z_m']), np.float32(row['size'])
    require(uv.shape == (2,) and R.shape == (3, 3), 'Unexpected prediction geometry shape')
    require(np.isfinite(np.r_[uv, R.ravel(), z, size]).all() and z > 0 and size > 0, 'Invalid predicted center, rotation, depth or size')
    np.testing.assert_array_equal(row['K'], K)
    np.testing.assert_array_equal(row['anchor'], keys[0])
    np.testing.assert_array_equal(np.asarray(row['gt_uv'], dtype=np.float32), gt_uv)
    np.testing.assert_allclose(row['T_gt'], T, rtol=0, atol=1e-12)
    require(np.float32(row['gt_log_size']) == target_log_size, 'GT log size differs from projected CAD')
    ray = np.linalg.solve(K, np.r_[uv * PIXELS, 1.])
    translation = ray * z - R @ keys[0]
    valid = bool(((keys @ R.T + translation)[:, 2] > 0).all())
    adds = float(tree.query((truth - translation) @ R, workers=1)[0].mean()) if valid else None
    depth_bias = z - (T[:3, :3] @ keys[0] + T[:3, 3])[2]
    log_size_bias = float(np.log(size) - target_log_size)
    return (abs(depth_bias), depth_bias, abs(log_size_bias), log_size_bias,
            np.linalg.norm((uv - gt_uv) * PIXELS), float(valid and adds < .1 * diameter),
            max(1 - adds / .1, 0) if valid else 0.), valid


def recompute(predictions, labels, ids, vertices, diameter):
    tree = cKDTree(vertices)
    groups = defaultdict(lambda: defaultdict(list))
    seen = defaultdict(list)
    invalid = defaultdict(int)
    recorded_gt = {}
    with predictions.open() as stream:
        for line in stream:
            row = json.loads(line)
            key, sid = cell_key(row), row['sample_id']
            require(key in CELLS and sid in labels and sid.split('/')[0] == row['rid'], 'Unexpected cell, trajectory or sample ID')
            identity = tuple(row[name] for name in ('K', 'T_gt', 'anchor', 'gt_uv', 'gt_log_size'))
            require(recorded_gt.setdefault(sid, identity) == identity, f'GT identity differs across paired cells for {sid}')
            values, valid = measure(row, labels[sid], tree, diameter)
            groups[key][row['rid']].append(values)
            seen[key].append(sid)
            invalid[key] += int(not valid)
    require(set(seen) == CELLS and all(sids == ids for sids in seen.values()), 'Prediction matrix order or eligibility differs')
    scores = {}
    for key, by_rollout in groups.items():
        require(set(by_rollout) == set(RIDS), 'Missing prediction trajectory')
        rows = {rid: dict(zip(METRICS, np.mean(by_rollout[rid], axis=0).tolist())) for rid in RIDS}
        scores[key] = dict(by_rollout=rows, summary={m: float(np.mean([r[m] for r in rows.values()])) for m in METRICS})
    return scores, invalid


def compare_scores(scores, declared, ids):
    errors = dict.fromkeys(METRICS, 0.)
    for key, score in scores.items():
        row = declared[key]
        require(row['arm'] == 'C1_RAW' and row['windows'] == len(ids) and row['window_ids'] == ids, 'Score metadata/eligibility differs')
        reference = row['score']
        require(set(reference['by_rollout']) == set(RIDS), 'Producer rollout inventory differs')
        scopes = [(score['summary'], reference['summary'])]
        scopes.extend((score['by_rollout'][rid], reference['by_rollout'][rid]) for rid in RIDS)
        for current, old in scopes:
            require(set(current) == set(old) == set(METRICS), 'Producer score metric fields differ')
            for metric in METRICS:
                a, b = current[metric], old[metric]
                agrees = a == b if metric == 'success' else np.isclose(a, b, rtol=1e-6, atol=1e-8)
                require(np.isfinite(a) and np.isfinite(b) and agrees, f'Score parity failed for {key} {metric}: {a} versus {b}')
                errors[metric] = max(errors[metric], abs(a - b))
    return errors


def condition_table(scores, ids, registered):
    seed = registered['statistics']['bootstrap_seed']
    draws = registered['statistics']['bootstrap_draws']
    indices = np.random.default_rng(seed).integers(32, size=(draws, 32))
    rows, checks = [], []
    for light in LIGHTS:
        summaries = {}
        for method in METHODS:
            cells = [scores[method, seed, light]['summary'] for seed in SEEDS]
            success = [100 * cell['success'] for cell in cells]
            summaries[method] = {metric: float(np.mean([cell[metric] for cell in cells])) for metric in METRICS}
            summaries[method].update(success_pct=float(np.mean(success)), seed_success_pct=success, seed_success_sd_pp=float(np.std(success, ddof=1)))
        delta = np.array([[scores[METHODS[0], s, light]['by_rollout'][rid]['success'] - scores[METHODS[1], s, light]['by_rollout'][rid]['success'] for rid in RIDS] for s in SEEDS])
        seed_deltas = delta.mean(axis=1)
        paired = delta.mean(axis=0)
        interval_pp = None
        passed = None
        if light in registered['primary_conditions']:
            low, high = np.quantile(paired[indices].mean(axis=1), [.0125, .9875])
            passed = bool((seed_deltas > 0).all() and low > 0)
            interval_pp = [100 * float(low), 100 * float(high)]
            checks.append(dict(condition=light, metric='ADD-S_success', difference='eventgenesis_minus_v2e', seed_mean_delta=seed_deltas.tolist(),
                               mean_delta=float(paired.mean()), ci_level=.975, paired_trajectory_ci=[float(low), float(high)], passed=passed))
        rows.append(dict(condition=light, eligible_windows_per_seed=len(ids), methods=summaries, paired_seed_delta_pp=(100 * seed_deltas).tolist(),
                         eventgenesis_minus_v2e_pp=100 * float(paired.mean()), paired_ci_97_5_pp=interval_pp, registered_gate_passed=passed))
    return rows, checks


def audit(root, completion_sha, registration):
    completion_path = checked(dict(path=str(root / 'completion.json'), sha256=completion_sha))
    contract = read(checked(dict(path=str(registration), sha256=REG_SHA)))
    completion = read(completion_path)
    require(completion['schema'] == 'pose_confirm32_frozen_evaluation_v1' and completion['status'] == 'complete', 'Frozen evaluation incomplete')
    require(completion['registration_sha256'] == REG_SHA and completion['optimizer_updates'] == 0 and completion['real_data_used'] is False and completion['GT_forward_inputs'] is False, 'Evaluation scope differs')
    require(root.resolve() == (Path(contract['root']) / 'evaluation').resolve() and completion['score_rows'] == 18, 'Evaluation root or matrix size differs')
    science = contract['scientific_registration']
    require(science['sha256'] == SCIENCE_SHA, 'Scientific registration pin differs')
    registered = read(checked(science))
    require(registered['primary_conditions'] == ['normal', 'half'] and registered['mandatory_negative_condition'] == 'eighth' and registered['window_indices'] == list(INDICES), 'Registered study conditions differ')
    require(registered['statistics']['bootstrap_draws'] == 10000 and registered['statistics']['bootstrap_seed'] == 20261109137 and registered['model_arm'] == 'RAW_C1_physical_supervised', 'Registered model or statistics differ')
    records = completion['artifacts']
    names = {'predictions.jsonl', 'scores.json', 'eligibility.json', 'baseline_parity.json', 'conditional_advantage.json'}
    files = by_key(records, lambda r: Path(r['path']).name, names)
    files = {name: checked(record) for name, record in files.items()}
    require(all(path.parent.resolve() == root.resolve() for path in files.values()), 'Evaluation artifact escaped its root')
    event_record = completion['event_completion']
    require(event_record['sha256'] == EVENT_SHA and Path(event_record['path']) == Path(contract['root']) / 'event_completion.json', 'Event receipt pin differs')
    event = read(checked(event_record))
    require(event['registration_sha256'] == REG_SHA and event['status'] == 'complete' and event['real_data_used'] is False, 'Event receipt scope differs')
    artifacts = {r['path']: r for r in event['artifacts']}
    require(len(artifacts) == len(event['artifacts']), 'Duplicate event artifact record')
    source_records = completion['source_records']
    cad_protocol = {r['path']: r for r in source_records if r['sha256'] == ASSET_PROTOCOL_SHA}
    require(len(cad_protocol) == 1, 'Frozen CAD protocol record missing or ambiguous')
    assets = read(checked(next(iter(cad_protocol.values()))))['assets']
    vertices = np.load(checked(assets['vertices']), allow_pickle=False)
    diameter = float(assets['diameter_m'])
    require(vertices.shape == (8194, 3) and np.isfinite(vertices).all() and diameter > 0, 'Frozen CAD geometry differs')
    models = by_key(completion['models'], lambda r: (r['method'], r['seed']), set(itertools.product(METHODS, SEEDS)))
    require(all(r['unchanged'] is True and len(r['states_sha256']) == 3 and all(re.fullmatch('[0-9a-f]{64}', h) for h in r['states_sha256']) for r in models.values()), 'Frozen model state receipts differ')
    checkpoint_names = {f'{m}_seed{s}_{suffix}.pt' for m, s in models for suffix in ('physical_supervised', 'physical_supervised_localizer', 'rotation')}
    checkpoints = {r['path']: r for r in source_records if Path(r['path']).name in checkpoint_names}
    require(len(checkpoints) == 18 and {Path(p).name for p in checkpoints} == checkpoint_names, 'Frozen model file records differ')
    for record in checkpoints.values():
        checked(record)
    parities = by_key(read(files['baseline_parity.json']), cell_key, CELLS)
    require(completion['all_baseline_parities_passed'] is True and all(r['passed'] is True and np.isfinite(r['max_abs']) for r in parities.values()), 'Original 18-cell baseline parity receipt differs')
    labels, ids, excluded = ground_truth(event['banks'], artifacts, vertices)
    eligibility = by_key(read(files['eligibility.json']), lambda r: (r['method'], r['condition']), set(itertools.product(METHODS, LIGHTS)))
    require(all(r['eligible_ids'] == ids and r['excluded_by_original_geometry_rule'] == excluded for r in eligibility.values()), 'Stored eligibility differs from original geometry rule')
    require(completion['cases'] == 18 * len(ids), 'Evaluation prediction count differs')
    scores, invalid = recompute(files['predictions.jsonl'], labels, ids, vertices, diameter)
    declared = by_key(read(files['scores.json']), cell_key, CELLS)
    errors = compare_scores(scores, declared, ids)
    table, checks = condition_table(scores, ids, registered)
    decision = read(files['conditional_advantage.json'])
    require(decision == completion['conditional_advantage'], 'Decision artifact and completion disagree')
    require(decision['bootstrap_seed'] == 20261109137 and decision['draws'] == 10000 and decision['mandatory_negative_condition'] == 'eighth', 'Producer bootstrap contract differs')
    require(decision['checks'] == checks and decision['passed'] == all(r['passed'] for r in checks), 'Independent bootstrap/gate disagrees with producer')
    for record in records:
        checked(record)
    checked(event_record)
    checked(dict(path=str(completion_path), sha256=completion_sha))
    return dict(schema='pose_new32_independent_cpu_audit_v1', status='passed', evaluation_completion_sha256=completion_sha,
                preparation_contract_sha256=REG_SHA, event_completion_sha256=EVENT_SHA, verified_output_artifacts=len(records),
                verified_physical_H5_files=96, verified_frozen_checkpoint_files=18, pose_predictions_recomputed=18*len(ids),
                score_cells_recomputed=18, scalar_score_checks=18*33*len(METRICS), metric_max_abs_errors=errors,
                baseline_parity_cells_verified_from_receipt=18, GT_identity_exact_across_cells=True, GT_H5_pose_atol=1e-12,
                invalid_pose_counts=[dict(method=m, seed=s, condition=k, count=invalid[m,s,k]) for m,s,k in sorted(CELLS)],
                excluded_windows=excluded, physical_rollouts=32, CAD_diameter_m=diameter, condition_table=table,
                registered_checks=checks, both_primary_conditions_passed=all(r['passed'] for r in checks),
                bootstrap_seed=20261109137, bootstrap_draws=10000, model_forward_calls=0, optimizer_updates=0, real_data_used=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-root', type=Path, required=True)
    parser.add_argument('--completion-sha256', required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(args.out.parent.is_dir() and not args.out.exists(), 'Use a fresh audit output path')
    result = audit(args.eval_root, args.completion_sha256, args.registration)
    with args.out.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], pose_predictions=result['pose_predictions_recomputed'],
                          both_primary_conditions_passed=result['both_primary_conditions_passed'], condition_table=result['condition_table']), allow_nan=False))
