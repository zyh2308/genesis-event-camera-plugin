"""Run a paired synthetic depth-coordinate smoke or short study on the owned GPU4."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch

from depth_coordinates import fit_depth, require, state_sha

METHODS = ('eventgenesis', 'v2e')
SEEDS = (0, 1, 2)
CANDIDATE = 'focal_normalized_depth_coordinate'
GPU4 = 'GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3'
CORE = '62da4f35ce3c5c96f04268bd1687a27657af9238'
BASE_SHA = 'd2368c9f246a61efc4382238741f1e7a62cf891aa8d6d653136832ed64ea3b62'
RAW_CACHE_CONTRACT_SHA = 'd2860889e582223bf248d7b8e8e6fcc8f80aa71041a371243fdae06010e312ff'
RAW_CACHE_SOURCES_SHA = '4819053bf08966a153055b81eab86c0f70f519109afcb104cf71059cf23800c5'
POLICY_SHA = 'e1883b68715ecf067a9f4dccbd7a61af5f5dc26b557e061859f4066e7de8d38c'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def record(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def pinned(path, digest):
    require(sha(path) == digest, 'Pinned file changed: ' + str(path))
    return read(path)


def verify_sources(directory, expected):
    directory = Path(directory)
    manifest = directory / 'source_manifest.json'
    files = pinned(manifest, expected)
    result = [record(manifest)]
    for name, digest in files.items():
        require(Path(name).name == name, 'Unexpected source path')
        item = record(directory / name)
        require(item['sha256'] == digest, 'Frozen source changed: ' + name)
        result.append(item)
    return result


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_contract(c):
    require(c['schema'] == 'pose_depth_coordinate_short_v1' and c['candidate'] == CANDIDATE,
            'Wrong registered experiment')
    require(c['frozen_core'] == CORE and c['base_contract_sha256'] == BASE_SHA, 'Frozen scientific source differs')
    require(c['methods'] == list(METHODS) and c['seeds'] == list(SEEDS), 'Complete paired matrix required')
    require(c['new_depth_optimizer_updates'] == 1728 and c['smoke_optimizer_updates'] == 2
            and c['updates_per_run'] == 288 and c['epochs'] == 6 and c['batch_size'] == 32, 'Budget changed')
    require(c['learning_rate'] == .00015 and c['normal_teacher_weight'] == 2
            and c['half_probability'] == .75 and c['auxiliary_teacher_weight'] == 0, 'Original C1 settings differ')
    require(c['real_scoring_enabled'] is False and c['independent32_read'] is False, 'Data role changed')
    require(c['raw_cache_source_manifest_sha256'] == RAW_CACHE_SOURCES_SHA, 'Unexpected RAW cache source')
    require(c['phase_timeout_seconds'] == {'smoke': 1800, 'study': 3600}, 'Runtime ceiling changed')
    gpu = c['gpu']
    require(gpu['index'] == 4 and gpu['uuid'] == GPU4 and gpu['parent_pid'] == 335186
            and int(gpu['parent_start_ticks']) == 53273702, 'Wrong owned GPU4 queue')
    require(gpu['parent_cmdline'][-4:] == [c['campaign'], '4', GPU4, '6'], 'Wrong queue arguments')


def gpu_admission(c, base, E):
    gpu = c['gpu']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU4, 'Exact assigned GPU UUID required')
    process = Path('/proc') / str(gpu['parent_pid'])
    stat = (process / 'stat').read_text()
    fields = stat[stat.rfind(')') + 1:].split()
    require(fields[0] not in ('Z', 'X') and int(fields[19]) == int(gpu['parent_start_ticks']), 'Queue PID identity changed')
    argv = [value.decode() for value in (process / 'cmdline').read_bytes().split(b'\0') if value]
    require(argv == gpu['parent_cmdline'] and gpu['parent_script_path'] in argv, 'Queue command changed')
    require(sha(gpu['parent_script_path']) == gpu['parent_script_sha256'], 'Queue script changed')
    E.check_gpu(base)
    require(torch.cuda.get_device_name(0) == gpu['name'], 'Assigned CUDA device model differs')
    return dict(passed=True, gpu_index=4, gpu_uuid=GPU4, parent_pid=gpu['parent_pid'],
                parent_start_ticks=int(gpu['parent_start_ticks']), worker_pid=os.getpid())


def load_cache(c, receipt, method, ctx, P, E, verify_lineage):
    item = next(r for r in c['input_caches'] if Path(r['path']).name == f'{method}_RAW_inputs_masks128.pt')
    require(item in receipt['input_caches'], 'Cache does not belong to the immutable input receipt')
    E.remember(ctx.source_records, item['path'], item['sha256'])
    cached = torch.load(item['path'], map_location='cpu', weights_only=False)
    require(cached['contract_sha256'] == RAW_CACHE_CONTRACT_SHA
            and cached['source_manifest_sha256'] == RAW_CACHE_SOURCES_SHA, 'RAW cache provenance differs')
    data = cached['data']
    if verify_lineage:
        P.verify_input_lineage(data['input_lineage'])
        by_path = {r['path']: r for r in ctx.source_records}
        for source in data['input_lineage']:
            if source['path'] in by_path:
                require(by_path[source['path']] == source, 'Conflicting RAW input source')
            else:
                ctx.source_records.append(source)
    return data


def training_tensors(data, device):
    result = tuple(data[key].to(device) for key in ('normal', 'y', 'aux', 'ay', 'clean'))
    require(tuple(len(value) for value in result) == (1529, 1529, 3058, 3058, 3058),
            'Original 1,529/3,058 RAW window inventory differs')
    return result


def admit_depth_tape(ctx, E, P, c, records, data, method, seed):
    run = next(r for r in ctx.completion['results'] if (r['method'], r['seed']) == (method, seed))
    source = E.remember(ctx.source_records, run['localization_path'], run['localization_sha256'])
    model_state = torch.load(source['path'], map_location='cpu', weights_only=True)
    expected = records[(method, seed, 'physical_supervised', 'depth')]
    require(expected['updates'] == 288 and expected['source_state_sha256'] == state_sha(model_state),
            'Original C1 depth initialization differs')
    require(expected['teacher_unchanged'] and expected['independent_model_optimizer_rebuild']
            and expected['optimizer_state_equal'], 'Original C1 proof is incomplete')
    reused = next(r for r in ctx.receipt['records']
                  if (r['method'], r['seed'], r['arm'], r['branch']) == (method, seed, 'physical_supervised', 'depth'))
    for key in ('source_state_sha256', 'final_state_sha256', 'labels_sha256', 'updates'):
        require(reused[key] == expected[key], 'Frozen C1 depth differs from the original control')
    path = Path(c['prior_study']) / 'training_evidence' / f'{method}_seed{seed}_physical_supervised_depth.jsonl'
    require(str(path) == reused['original_log']['path'], 'Original depth log path differs')
    source_log = E.remember(ctx.source_records, path, reused['original_log']['sha256'])
    tape = SimpleNamespace(shared=ctx.shared, native=ctx.native)
    labels, draws = P.replay(tape, *data, seed)
    require(len(draws) == 288 and labels == expected['labels_sha256']
            and draws == P.trajectory(P.read_log(path)), 'Full original C1 depth tape differs')
    return dict(method=method, seed=seed, original_depth_tape_exact=True, source_model=source,
                original_log=source_log, labels_sha256=labels, source_state_sha256=state_sha(model_state),
                original_C1_final_state_sha256=expected['final_state_sha256'],
                original_C1_center_state_sha256=records[(method, seed, 'physical_supervised', 'center')]['final_state_sha256'])


def clean_full(ctx, method, clean384):
    bank = ctx.D / 'pose_bank'
    paths = sorted((bank / 'physical').glob('*.h5'))
    require(len(paths) == 96, 'Original clean confirmation trajectory inventory differs')
    pinned_paths = {r['path'] for r in ctx.source_records}
    rows, tensors = [], []
    for path in paths:
        event = bank / 'events' / method / (path.stem + '.npy')
        require(str(path) in pinned_paths and str(event) in pinned_paths, 'Clean confirmation input lacks original cache lineage')
        episode = ctx.fresh.adapter(bank, [path], method, ctx.encode)
        tensors.append(ctx.shared.tensors(episode, ctx.device).cpu())
        rows.extend({key: value for key, value in row.items() if key != 'image'} for row in episode)
    groups = {}
    for row in rows:
        groups.setdefault(row['rid'], []).append(row)
    require(len(groups) == 96, 'Original clean confirmation lost trajectories')
    expected = [group[i]['id'] for group in groups.values() for i in np.linspace(0, len(group)-1, 4, dtype=int)]
    require(expected == [row['id'] for row in clean384['rows']], 'Clean384 selection differs from the native evaluator')
    return dict(rows=rows, raw=torch.cat(tensors))


def score_pair(ctx, E, method, seed, candidate, views, prior_run, admission, references):
    center, _, _ = E.load_branch(ctx, method, seed, 'physical_supervised', 'center')
    control, _, _ = E.load_branch(ctx, method, seed, 'physical_supervised', 'depth')
    require(state_sha(control.state_dict()) == admission['original_C1_final_state_sha256'],
            'Expected original C1 depth control is absent')
    require(state_sha(center.state_dict()) == admission['original_C1_center_state_sha256'],
            'Expected original C1 center control is absent')
    rotation_path = ctx.A / f'jobs/117_pose_rotation_paired_20261106/results/{method}_seed{seed}_rotation.pt'
    E.remember(ctx.source_records, rotation_path, prior_run['parents']['rotation_sha256'])
    head = ctx.rotation.RotationHead().to(ctx.device).eval()
    head.load_state_dict(torch.load(rotation_path, map_location='cpu', weights_only=True)['model'], strict=True)
    states = [(model, state_sha(model.state_dict())) for model in (center, control, candidate, head)]
    tree = cKDTree(ctx.vertices)
    arms = dict(clean_control=prior_run['arms']['clean_control'], physical_supervised={}, paired_consistency={})
    inventories = {}
    for condition, view in views.items():
        rows = view['rows']
        inputs = view['raw'].to(ctx.device)
        intrinsics = [row['K'] for row in rows]
        uv = ctx.native.estimate_centers(center, inputs)
        target_scale = ctx.shared.labels(ctx.scale, rows, ctx.vertices, ctx.device)[:, 2].cpu().numpy()
        inventories[condition] = dict(windows=len(rows), rollouts=len({row['rid'] for row in rows}))
        for arm, model in [('physical_supervised', control), ('paired_consistency', candidate)]:
            _, depth, size = ctx.scale.estimate(model, inputs, intrinsics)
            rotations = ctx.shared.frozen_rotation(ctx.rotation, head, inputs, uv, size, intrinsics)
            score = ctx.shared.score(ctx.rotation, rows, uv, depth, size, target_scale, rotations,
                                     ctx.vertices, tree, ctx.threshold)
            require(all(np.isfinite(value) for value in score['summary'].values()), 'Nonfinite native pose metrics')
            arms[arm][condition] = score
        expected = (references[(method, seed, 'normal')] if condition == 'matched_normal'
                    else prior_run['arms']['physical_supervised'][condition])
        actual = arms['physical_supervised'][condition]
        E.compare_summary(actual['summary'], expected['summary'])
        require(actual['by_rollout'].keys() == expected['by_rollout'].keys(), 'C1 control rollout inventory differs')
        for rid, metrics in actual['by_rollout'].items():
            E.compare_summary(metrics, expected['by_rollout'][rid])
        require(arms['paired_consistency'][condition]['summary']['center_px'] == actual['summary']['center_px'],
                'Independent frozen C1 center changed')
    require(all(state_sha(model.state_dict()) == state for model, state in states), 'Inference changed a frozen model')
    return dict(method=method, seed=seed, arms=arms, parents=prior_run['parents'], inventories=inventories,
                identical_center_state_sha256=states[0][1], rotation_state_sha256=states[3][1],
                C1_native_summary_and_per_rollout_parity=True)


def verify_smoke(path, digest, contract_sha, sources_sha):
    result = pinned(path, digest)
    require(result['schema'] == 'pose_depth_coordinate_short_v1' and result['phase'] == 'smoke'
            and result['status'] == 'complete_synthetic_only' and result['new_optimizer_updates'] == 2,
            'Actual paired two-update smoke is required')
    require(result['contract_sha256'] == contract_sha and result['source_manifest_sha256'] == sources_sha
            and result['real_data_used'] is False, 'Smoke source or data role differs')
    require(len(result['records']) == 2 and {(r['method'], r['seed']) for r in result['records']} == {(m, 0) for m in METHODS},
            'Paired smoke record inventory differs')
    require(all(r['updates'] == 1 and r['independent_model_adam_readback']
                and r['first_native_loss_oracle']['passed'] and r['native_validation_contract_passed']
                for r in result['records']), 'Smoke proof is incomplete')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--phase', choices=('smoke', 'study'), required=True)
    parser.add_argument('--smoke-receipt', type=Path)
    parser.add_argument('--smoke-receipt-sha256')
    args = parser.parse_args()
    require(args.out.is_absolute() and args.out.parent.is_dir() and not args.out.exists(), 'Fresh absolute output required')
    args.out.mkdir()
    start = time.monotonic()
    try:
        c = pinned(args.contract, args.contract_sha256)
        validate_contract(c)
        smoke = args.phase == 'smoke'
        if not smoke:
            verify_smoke(args.smoke_receipt, args.smoke_receipt_sha256, args.contract_sha256, args.source_manifest_sha256)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        cv2.setNumThreads(1)
        own = Path(__file__).resolve().parent
        sources = verify_sources(own, args.source_manifest_sha256)
        require(sha(own / 'pose_priority_policy.py') == POLICY_SHA, 'Original C1 policy copy differs')
        base = pinned(c['base_contract_path'], BASE_SHA)
        diagnostic = Path(c['diagnostic_source_dir'])
        sources.extend(verify_sources(diagnostic, c['diagnostic_source_manifest_sha256']))
        E = load_module(diagnostic / 'export_pose_factors.py', 'depth_coordinate_native_context')
        gpu_start = gpu_admission(c, base, E)
        ctx = E.load_context(base, check_gpu=True)
        sources.extend(verify_sources(c['raw_cache_sources'], RAW_CACHE_SOURCES_SHA))
        P = load_module(Path(c['raw_cache_sources']) / 'legacy_center_bridge.py', 'depth_coordinate_raw_tape')
        E.remember(ctx.source_records, args.contract, args.contract_sha256)
        E.remember(ctx.source_records, c['base_contract_path'], BASE_SHA)
        if not smoke:
            E.remember(ctx.source_records, args.smoke_receipt, args.smoke_receipt_sha256)
        cache = c['raw_cache_receipt']
        raw_receipt = pinned(cache['path'], cache['sha256'])
        E.remember(ctx.source_records, cache['path'], cache['sha256'])
        require(raw_receipt['status'] == 'complete_synthetic_only' and raw_receipt['phase'] == 'smoke'
                and raw_receipt['contract_sha256'] == RAW_CACHE_CONTRACT_SHA
                and raw_receipt['source_manifest_sha256'] == RAW_CACHE_SOURCES_SHA
                and raw_receipt['real_scores_computed'] is False, 'Input-only cache receipt differs')
        prior_root = Path(c['prior_study'])
        prior_complete = pinned(prior_root / 'completion.json', c['prior_study_completion_sha256'])
        prior_execution = pinned(prior_root / 'priority_execution_receipt.json', c['prior_execution_receipt_sha256'])
        for name, key in [('completion.json', 'prior_study_completion_sha256'),
                          ('priority_execution_receipt.json', 'prior_execution_receipt_sha256')]:
            E.remember(ctx.source_records, prior_root / name, c[key])
        require(prior_complete['status'] == 'complete' and prior_complete['real_data_used'] is False,
                'Original C0/C1 metrics receipt differs')
        prior_runs = {(r['method'], r['seed']): r for r in prior_complete['results']}
        prior_records = {(r['method'], r['seed'], r['arm'], r['branch']): r for r in prior_execution['records']}
        require(len(prior_runs) == 6 and len(prior_records) == 36, 'Original control inventory differs')
        forward = pinned(c['forward_completion'], c['forward_completion_sha256'])
        E.remember(ctx.source_records, c['forward_completion'], c['forward_completion_sha256'])
        score_record = next(r for r in forward['artifacts'] if Path(r['path']).name == 'native_scores.json')
        reference_scores = pinned(score_record['path'], score_record['sha256'])
        E.remember(ctx.source_records, score_record['path'], score_record['sha256'])
        references = {(r['method'], r['seed'], r['condition']): r['score'] for r in reference_scores if r['arm'] == 'C1'}
        require(len(references) == 18 and forward['optimizer_updates'] == 0 and forward['real_data_used'] is False,
                'Frozen C1 matched reference scores differ')
        admissions = {}
        for method in METHODS:
            data = load_cache(c, raw_receipt, method, ctx, P, E, verify_lineage=True)
            tensors = training_tensors(data, ctx.device)
            for seed in SEEDS:
                admissions[(method, seed)] = admit_depth_tape(ctx, E, P, c, prior_records, tensors, method, seed)
            del data, tensors
            gc.collect()
            torch.cuda.empty_cache()
        require(all(len({admissions[(method, seed)]['labels_sha256'] for method in METHODS}) == 1 for seed in SEEDS),
                'Shared labels differ across generators')
        write(args.out / 'pre_update_admission.json', dict(status='passed', optimizer_updates=0,
              records=list(admissions.values()), execution_gpu=gpu_start))
        records, results = [], []
        for method in METHODS:
            data = load_cache(c, raw_receipt, method, ctx, P, E, verify_lineage=False)
            tensors = training_tensors(data, ctx.device)
            views = dict(data['views'])
            require(views.keys() == {'clean', 'matched_normal', 'half', 'eighth'}, 'Original validation views differ')
            require(len(views['clean']['rows']) == 384, 'Original clean384 inventory differs')
            expected_relit = {f'validation_{rid:03d}/{frame:03d}' for rid in (0, 31, 63, 95) for frame in range(4, 20)}
            for condition in ('matched_normal', 'half', 'eighth'):
                ids = [row['id'] for row in views[condition]['rows']]
                require(len(ids) == 64 and set(ids) == expected_relit, 'Original fixed64 validation inventory differs')
            views['clean_full'] = clean_full(ctx, method, views['clean'])
            for seed in ((0,) if smoke else SEEDS):
                require(time.monotonic() - start < c['phase_timeout_seconds'][args.phase], 'Phase runtime ceiling exceeded')
                admission = admissions[(method, seed)]
                source = torch.load(admission['source_model']['path'], map_location='cpu', weights_only=True)
                require(state_sha(source) == admission['source_state_sha256'], 'Original model changed after admission')
                old_log = P.read_log(admission['original_log']['path'])
                model, fitted = fit_depth(ctx.native, ctx.scale, source, tensors, method, seed, smoke, args.out, old_log)
                if smoke:
                    tape = SimpleNamespace(shared=ctx.shared, native=ctx.native)
                    expected = P.replay(tape, *tensors, seed, smoke=True)[0]
                else:
                    expected = admission['labels_sha256']
                require(fitted['labels_sha256'] == expected, 'Actual labels differ from original C1 depth tape')
                fitted['all_labels_and_draws_match_C1'] = True
                for key in ('training_state_path', 'final_model_path', 'training_log_path'):
                    fitted[key.removesuffix('_path')] = record(fitted[key])
                result = score_pair(ctx, E, method, seed, model, views, prior_runs[(method, seed)], admission, references)
                fitted['native_validation_contract_passed'] = True
                fitted['native_validation_inventory'] = result['inventories']
                records.append(fitted)
                if not smoke:
                    write(args.out / f'{method}_seed{seed}_native_scores.json', result)
                    results.append(result)
                del model, source
            del views, data, tensors
            gc.collect()
            torch.cuda.empty_cache()
        updates = sum(r['updates'] for r in records)
        require(updates == (2 if smoke else 1728), 'Wrong actual update count')
        gate = confirmation = None
        if not smoke:
            coverage_path = Path(base['prior_source_dir']) / 'pose_coverage_real.py'
            require(any(r['path'] == str(coverage_path) and r['sha256'] == sha(coverage_path)
                        for r in ctx.source_records), 'Original gate module is not source-pinned')
            coverage = load_module(coverage_path, 'frozen_pose_confirmation_functions')
            checks = coverage.gate_rows(results)
            confirmation = coverage.original_confirmation(results, ctx.R)
            require(len(checks) == 30 and len(confirmation['comparisons']) == 8, 'Original 30+8 gate inventory differs')
            gate = dict(checks=checks, passed=all(row['passed'] for row in checks))
        all_sources = sources + ctx.source_records
        require(all(sha(r['path']) == r['sha256'] for r in all_sources), 'Frozen input, source, or control bytes changed')
        gpu_end = gpu_admission(c, base, E)
        require(time.monotonic() - start < c['phase_timeout_seconds'][args.phase], 'Phase runtime ceiling exceeded')
        write(args.out / 'completion.json', dict(schema='pose_depth_coordinate_short_v1', phase=args.phase,
              status='complete_synthetic_only', candidate=CANDIDATE, contract_sha256=args.contract_sha256,
              source_manifest_sha256=args.source_manifest_sha256, new_optimizer_updates=updates,
              real_data_used=False, independent32_read=False, source_records=all_sources,
              execution_gpu_start=gpu_start, execution_gpu_end=gpu_end, records=records,
              full_C1_depth_tape_admissions=list(admissions.values()), results=results,
              original_thirty_gate=gate, original_eight_confirmation=confirmation,
              canonical_arm_mapping=dict(clean_control='original_C0', physical_supervised='original_RAW_C1',
                                         paired_consistency=CANDIDATE),
              elapsed_seconds=time.monotonic()-start))
    except BaseException:
        write(args.out / 'failure.json', dict(status='failed', traceback=traceback.format_exc(),
              elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()
