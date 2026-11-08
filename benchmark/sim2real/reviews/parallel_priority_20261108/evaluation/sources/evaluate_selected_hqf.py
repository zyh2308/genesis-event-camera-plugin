#!/usr/bin/env python3
"""Frozen all15 / selected HQF3 development evaluation; no scoring on prepare/import."""
import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
METHODS = ('eventgenesis', 'v2e', 'vid2e', 'film_esim', 'rife_esim')
SEEDS = (0, 1, 2)
SEQUENCES = ('boxes', 'reflective_materials', 'still_life')
SCOPE = 'posthoc_selected_hqf3_development'
CORE = '62da4f35ce3c5c96f04268bd1687a27657af9238'
RECIPE_SHA = 'cf04fa50cd833a21b2a1cae7127acff8678317b9535e0adc99fc7c0eb5d0013a'
TRAINER_SHA = 'e7d9db6632930c8319ddaee174414d1984ae1b7a380b26c026a795b110b77b28'
HELPER_SHA = '4aec0d0ef8ca88c7c0eb1fe1ae7cc3905aff87431b68253d69729965a7bc8f04'
SCHEMA = 'recon_selected_hqf3_development_protocol_v1'
METRICS = ('lpips_alex', 'ssim', 'psnr_db', 'mse')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def record(path):
    p = Path(path).resolve(strict=True)
    return dict(path=str(p), size_bytes=p.stat().st_size, sha256=sha(p))


def check(r):
    p = Path(r['path']).resolve(strict=True)
    need(p.is_file() and p.stat().st_size == r['size_bytes'] and sha(p) == r['sha256'],
         'Bound file changed: ' + str(p))
    return p


def load_helper():
    path = HERE / 'hqf_repair_pair_bridge.py'
    need(sha(path) == HELPER_SHA, 'Original HQF bridge bytes changed')
    spec = importlib.util.spec_from_file_location('hqf_selected3_original_bridge', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def matrix_contract(plan, matrix, completion):
    """Pure admission contract, also exercised using non-model CPU fixtures."""
    need(plan['schema'] == 'recon_selected_full_plan_v1', 'Unrecognized training plan')
    need(plan['methods'] == list(METHODS) and plan['seeds'] == list(SEEDS), 'Changed plan matrix')
    need(plan['epochs'] == 200 and plan['training_budget_updates'] == 12000, 'Changed full budget')
    need(plan['final_activation'] == 'sigmoid' and plan['loss'] ==
         dict(lpips_vgg=1., temporal=2., auxiliary_l1=0.), 'Changed model/loss')
    need(plan['frozen_core'] == CORE and plan['recipe_sha256'] == RECIPE_SHA and
         plan['trainer_sha256'] == TRAINER_SHA, 'Changed frozen core/recipe/trainer')
    need(plan['real_sequences'] == list(SEQUENCES) and plan['real_result_scope'] == SCOPE,
         'Changed posthoc development subset')
    need(plan['real_scoring_enabled'] is False and
         plan['no_real_feedback_to_training_or_selection'] is True and
         plan['original_hqf14_results_preserved'] is True, 'Training isolation not established')
    need(matrix['schema'] == 'recon_selected_frozen_checkpoints_v1' and
         matrix['status'] == 'frozen_all15_before_real_scoring', 'All15 must be frozen before real use')
    need(matrix['methods'] == list(METHODS) and matrix['seeds'] == list(SEEDS) and
         matrix['real_sequences'] == list(SEQUENCES) and matrix['result_scope'] == SCOPE,
         'Frozen inventory/scope changed')
    need(matrix['recipe']['sha256'] == RECIPE_SHA, 'Frozen recipe differs')
    need(completion['status'] == 'full15_training_complete_selected_eval_pending' and
         completion['training_runs'] == 15 and completion['evaluated_sequence_runs'] == 0 and
         completion['real_scoring_enabled'] is False and completion['real_result_scope'] == SCOPE and
         completion['full14_historical_results_unchanged'] is True, 'Full15 completion receipt missing')
    rows = matrix['runs']
    expected = {(m, s) for m in METHODS for s in SEEDS}
    need(len(rows) == 15 and {(r['method'], r['seed']) for r in rows} == expected,
         'Require every method and all three seeds exactly once')
    for r in rows:
        need(r['run_id'] == r['method'] + '_seed' + str(r['seed']), 'Run identity differs')
        need(r['completed_updates'] == 12000, 'Any incomplete training rejects the whole table')
        need(len(r['initial_state_sha256']) == 64, 'Initial state digest missing')
    for s in SEEDS:
        need(len({r['initial_state_sha256'] for r in rows if r['seed'] == s}) == 1,
             'Generators do not share the initial state within seed')
    return rows


def selected_manifest(parent, parent_record):
    need(parent['status'] == 'ready_unscored' and parent['expected_sequence_count'] == 14 and
         parent['ready_unscored_sequence_count'] == 14 and
         parent['real_scoring_enabled'] is False and parent['real_gt_metrics_computed'] is False,
         'Original HQF14 data manifest is not pre-score/complete')
    names = [r['sequence'] for r in parent['sequences']]
    need(len(names) == 14 and len(set(names)) == 14 and set(SEQUENCES) <= set(names),
         'Original HQF14 sequence inventory differs')
    by = {r['sequence']: r for r in parent['sequences']}
    return dict(schema='hqf_selected3_manifest_v1', status='ready_unscored',
                expected_sequence_count=3, ready_unscored_sequence_count=3,
                real_scoring_enabled=False, real_gt_metrics_computed=False,
                result_scope=SCOPE, source_hqf14_manifest=parent_record,
                derivation='Exact original manifest rows; no paths, events, frames or timestamps changed',
                sequences=[copy.deepcopy(by[s]) for s in SEQUENCES])


def verify_selected_manifest(p, full_files=True):
    parent = read(check(p['original_data_manifest']))
    actual = read(check(p['data_manifest']))
    expected = selected_manifest(parent, p['original_data_manifest'])
    need(actual == expected, 'Selected data rows must equal unchanged HQF14 rows exactly')
    need([r['sequence'] for r in parent['sequences']] == p['original_sequences'],
         'Original data inventory changed')
    helper = load_helper()
    for row in actual['sequences']:
        seq = row['sequence']; mm = row['evreal_memmap']
        need(row['status'] == 'ready_unscored' and mm['exists'] is True and not mm['missing_files'],
             'Selected sequence is not ready')
        need(Path(mm['path']).resolve() == (Path(p['hqf_root']) / seq).resolve(), 'Selected data path changed')
        need(set(mm['files']) == set(helper.FILES), 'Selected file inventory differs')
        for filename, rec in mm['files'].items():
            if full_files:
                check(dict(path=str(Path(mm['path']) / filename), **rec))
            recorded_hash = mm.get('schema_checks', {}).get('hashes', {}).get(filename)
            if recorded_hash is not None:
                need(recorded_hash == rec['sha256'], 'Array hash metadata disagrees')
    return actual, {r['sequence']: r for r in actual['sequences']}


def validate_checkpoint_metadata(cp, row, recipe, plan, selection, reference_init, adapter_sha):
    need(cp['schema'] == 'e2vid_stage4_resume_v1' and
         (cp['generator'], cp['seed']) == (row['method'], row['seed']), 'Checkpoint identity/schema changed')
    u = cp['optimizer_update']
    need(isinstance(u, int) and 300 <= u <= 12000 and u % 300 == 0 and
         u == selection['optimizer_update'] and selection['epoch'] * 60 == u,
         'Selected checkpoint is not its declared synthetic-validation boundary')
    need(cp['next_epoch'] * 60 + cp['next_batch_in_epoch'] == u and
         len(cp['optimizer_step_counts']) == 30 and set(cp['optimizer_step_counts']) == {u},
         'Checkpoint optimizer/cursor differs')
    need(cp['config_sha256'] == RECIPE_SHA and cp['trainer_script_sha256'] == TRAINER_SHA and
         cp['data_manifest_sha256'] == adapter_sha, 'Checkpoint recipe/source/data fingerprint differs')
    need(cp['model_args'] == recipe['reconstruction']['model_args'] and
         cp['model_args']['final_activation'] == 'sigmoid', 'Checkpoint architecture differs')
    need(cp['real_data_used'] is False and cp.get('real_scoring_enabled', False) is False,
         'Real data was used by training')
    need(cp['initial_state_sha256'] == row['initial_state_sha256'] == reference_init,
         'Checkpoint is not from the paired original initialization')
    need(cp['reconstruction_repair'] == recipe['reconstruction_repair'], 'Repair provenance changed')
    tr, va = cp['train_rollouts'], cp['validation_rollouts']
    need(len(tr) == len(set(tr)) == 24 and len(va) == len(set(va)) == 8 and not set(tr) & set(va),
         'Whole-rollout 24/8 split changed')
    return (tr, va)


def collect_training(root, inspect_models=True):
    root = Path(root).resolve(strict=True)
    plan_path = root / 'sources/registered_plan.json'
    plan, matrix, completion = read(plan_path), read(root / 'frozen_checkpoint_matrix.json'), read(root / 'completion.json')
    rows = matrix_contract(plan, matrix, completion)
    source_manifest = root / 'sources/source_manifest.json'
    need(sha(source_manifest) == matrix['source_manifest_sha256'], 'Frozen source-manifest digest differs')
    source_map = read(source_manifest)
    for name, digest in source_map.items():
        need(Path(name).name == name and sha(root / 'sources' / name) == digest, 'Training source changed: ' + name)
    recipe_path = check(matrix['recipe']); recipe = read(recipe_path)
    need(recipe_path == root / 'sources/training_recipe.json' and sha(recipe_path) == RECIPE_SHA and
         sha(root / 'sources/train_e2vid_repaired.py') == TRAINER_SHA, 'Unexpected training recipe/source path')
    need(str(root) == plan['root'] and sha(plan_path) == source_map['registered_plan.json'], 'Plan/root binding differs')
    need(recipe['frozen_simulator_core'] == CORE and recipe['real_scoring_enabled'] is False and
         recipe['reconstruction']['epochs'] == 200 and
         recipe['reconstruction']['expected_optimizer_updates'] == 12000 and
         recipe['reconstruction']['model_args']['final_activation'] == 'sigmoid', 'Recipe changed')
    cpu = root / 'cpu_admission.json'; admission = read(cpu)
    need(admission['passed'] is True and admission['source_manifest_sha256'] == sha(source_manifest), 'CPU admission not bound')
    initial_record = admission['initial_state_reference']; initial = read(check(initial_record))['tasks']['reconstruction']['hashes']
    gate_path = root / 'ALL15_SYNTHETIC_GATE.json'; gate = read(gate_path)
    need(gate['passed'] is True and gate['source_manifest_sha256'] == sha(source_manifest), 'All15 synthetic gate missing')
    lineage = []
    split = None
    for row in rows:
        run = root / 'training' / ('seed' + str(row['seed'])) / row['method']
        cp_path = check(row['checkpoint']); selection_path = check(row['selection_record'])
        need(cp_path == run / 'synthetic_best.pt' and selection_path == run / 'synthetic_best.json',
             'Checkpoint selection path changed')
        selection = read(selection_path); done = read(run / 'completion.json'); summary = read(run / 'run_summary.json')
        need(selection['selection_source'] == 'synthetic validation only' and
             selection['sha256'] == row['checkpoint']['sha256'] and selection['size_bytes'] == row['checkpoint']['size_bytes'] and
             Path(selection['checkpoint']).resolve() == cp_path, 'Synthetic selection record differs')
        need(done['status'] == 'complete' and done['method'] == row['method'] and done['seed'] == row['seed'] and
             done['optimizer_update'] == 12000 and done['real_data_used'] is False and
             done['real_scoring_enabled'] is False and done['source_manifest_sha256'] == sha(source_manifest) and
             done['selected_checkpoint'] == row['checkpoint'] and done['selection_record'] == row['selection_record'],
             'Per-run completion not bound')
        need(done['summary'] == summary and summary['status'] == 'complete' and summary['optimizer_update'] == 12000 and
             summary['optimizer_budget'] == 12000 and summary['completed_epochs'] == 200 and
             summary['config_sha256'] == RECIPE_SHA and summary['trainer_script_sha256'] == TRAINER_SHA and
             summary['initial_model_sha256'] == row['initial_state_sha256'] == initial[str(row['seed'])] and
             summary['real_scoring_enabled'] is False, 'Training did not finish its common full budget')
        last_path = run / 'last.pt'
        need(Path(summary['final_checkpoint']).resolve() == last_path and
             sha(last_path) == summary['final_checkpoint_sha256'], 'Final full-budget checkpoint missing/changed')
        stability_path = run / 'synthetic_output_stability_guard.json'; stability = read(stability_path)
        need(stability['passed'] is True and stability['optimizer_update'] == 12000 and
             stability['real_data_used'] is False and len(stability['rollouts']) == 8 and
             stability['contrast_ratio_mean'] > .1 and stability['degenerate_fraction_mean'] < .05 and
             all(r['minimum'] >= 0 and r['maximum'] <= 1 for r in stability['rollouts']),
             'Final synthetic output-stability guard missing/failed')
        boundary_path = run / 'boundary_preflight.json'; boundary = read(boundary_path)
        need(boundary['passed'] is True and boundary['oracle']['status'] == 'pass', 'Independent resume boundary check failed')
        adapter_path = Path(plan['campaign']) / 'adapters' / row['method'] / 'reconstruction/adapter_manifest.json'
        adapter_rec = record(adapter_path)
        need(summary['data_manifest_sha256'] == adapter_rec['sha256'], 'Adapter lineage differs')
        log_path = run / 'training_metrics.jsonl'
        updates, validations = [], []
        with log_path.open(encoding='utf-8') as f:
            for line in f:
                e = json.loads(line)
                if e.get('event') == 'optimizer_update':
                    updates.append(e['optimizer_update'])
                elif e.get('event') == 'synthetic_validation':
                    validations.append(e)
        need(updates == list(range(1, 12001)), 'Training log lacks the complete update inventory')
        need([v['optimizer_update'] for v in validations] == list(range(300, 12001, 300)), 'Synthetic validation inventory differs')
        need(all(math.isfinite(v['lpips_vgg_equal_rollout_mean']) and len(v['rollouts']) == 8 for v in validations),
             'Synthetic selection metrics are incomplete/nonfinite')
        chosen = min(validations, key=lambda v: (v['lpips_vgg_equal_rollout_mean'], v['optimizer_update']))
        need(selection['optimizer_update'] == chosen['optimizer_update'] and
             selection['synthetic_validation_lpips_vgg'] == chosen['lpips_vgg_equal_rollout_mean'],
             'Selected checkpoint is not the synthetic-only minimum with earlier tie rule')
        if inspect_models:
            import torch
            cp = torch.load(cp_path, map_location='cpu', weights_only=False)
            this_split = validate_checkpoint_metadata(cp, row, recipe, plan, selection, initial[str(row['seed'])], adapter_rec['sha256'])
            if split is None:
                split = this_split
            need(this_split == split, 'Generators/seeds do not use the same physical rollout split')
            h = hashlib.sha256()
            for name, value in sorted(cp['model'].items()):
                need(value.dtype == torch.float32 and torch.isfinite(value).all().item(), 'Model contains non-FP32/nonfinite tensors')
                h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
            need(h.hexdigest() == cp['model_sha256'], 'Model tensor digest differs')
            del cp; gc.collect()
        lineage.append(dict(run_id=row['run_id'], completion=record(run / 'completion.json'),
                            summary=record(run / 'run_summary.json'), training_log=record(log_path),
                            adapter_manifest=adapter_rec, final_checkpoint=record(last_path),
                            final_synthetic_stability=record(stability_path), resume_boundary=record(boundary_path)))
    return dict(plan=record(plan_path), frozen_matrix=record(root / 'frozen_checkpoint_matrix.json'),
                completion=record(root / 'completion.json'), source_manifest=record(source_manifest),
                cpu_admission=record(cpu), synthetic_gate=record(gate_path),
                initialization_reference=initial_record, runs=lineage), plan, matrix, recipe


def build_protocol(training_root, out):
    helper = load_helper()
    lineage, plan, matrix, recipe = collect_training(training_root)
    old = plan['hqf']
    need(old['real_scores_computed'] is False and old['real_scoring_enabled'] is False, 'Inherited protocol must be unscored')
    for key in ('inherited_protocol', 'data_manifest', 'downstream_source_bundle'):
        check(old[key])
    for value in old['evaluation_sources'].values():
        check(value)
    need(old['training_recipe']['sha256'] == RECIPE_SHA, 'Frozen inference contract uses a different model recipe')
    helper.verify_downstream_sources(old)
    parent = read(check(old['data_manifest']))
    selected = selected_manifest(parent, old['data_manifest'])
    data_path = Path(out) / 'hqf_selected3_manifest.json'
    p = dict(schema=SCHEMA, status='prepared_unscored', created_utc=helper.utc(), task='reconstruction',
             methods=list(METHODS), seeds=list(SEEDS), sequences=list(SEQUENCES), expected_runs=15,
             expected_sequence_runs=45, training_budget_updates=12000,
             result_scope=SCOPE, real_test_exposure='Posthoc HQF3 display/development subset selected after earlier HQF results; not HQF14 or a blind test',
             real_scores_computed=False, real_scoring_enabled=False, authorization_required=True,
             no_real_feedback_to_training_or_selection=True, full_hqf14_historical_results_preserved=True,
             training_root=str(Path(training_root).resolve()), training_lineage=lineage,
             frozen_checkpoint_matrix=matrix, runs=matrix['runs'], training_recipe=matrix['recipe'],
             original_sequences=old['sequences'], original_data_manifest=old['data_manifest'],
             inherited_protocol=old['inherited_protocol'], hqf_root=old['hqf_root'], repository=old['repository'],
             model_root=old['model_root'], evreal_root=old['evreal_root'],
             downstream_source_bundle=old['downstream_source_bundle'], source_verification=old['source_verification'],
             evaluation_sources=old['evaluation_sources'], inherited_contract=old['inherited_contract'],
             pre_registered_preview_indices={s: old['pre_registered_preview_indices'][s] for s in SEQUENCES},
             selection='All15 first finish 12000 updates; independently freeze each minimum synthetic equal-rollout LPIPS-VGG checkpoint, ties earlier update; never real selection',
             photometry='Unchanged frozen validate_pair/classical_metrics/LPIPS-Alex; no output-selected gain, histogram or crop',
             aggregation='Arithmetic frame mean within sequence; equal three-sequence mean per seed; arithmetic mean and sample SD(ddof=1) over all three seeds; no window/frame pooling across sequences',
             legacy_raw_scorer_note='The unchanged score_pair has hardcoded HQF14 labels/count text; raw artifact is retained, canonical selected3_summary.json and tables contain the actual selected3 inventory.',
             bridge_source=record(__file__), original_bridge_source=record(HERE / 'hqf_repair_pair_bridge.py'))
    # No output is published until all actual selected file hashes and source guards pass.
    need(not Path(out).exists(), 'Use a fresh protocol directory')
    Path(out).mkdir(parents=True)
    dump_new(data_path, selected)
    p['data_manifest'] = record(data_path)
    verify_selected_manifest(p)
    return p


def verify_protocol(path, inspect_models=True, full_data=True):
    p = read(path)
    need(p['schema'] == SCHEMA and p['status'] == 'prepared_unscored', 'Wrong protocol schema/state')
    need(p['methods'] == list(METHODS) and p['seeds'] == list(SEEDS) and p['sequences'] == list(SEQUENCES) and
         p['expected_runs'] == 15 and p['expected_sequence_runs'] == 45 and p['training_budget_updates'] == 12000,
         'Changed evaluation matrix')
    need(p['real_scores_computed'] is False and p['real_scoring_enabled'] is False and
         p['no_real_feedback_to_training_or_selection'] is True and p['result_scope'] == SCOPE and
         p['full_hqf14_historical_results_preserved'] is True, 'Isolation/scope differs')
    need(Path(str(path) + '.sha256').read_text().split()[0] == sha(path), 'Protocol digest sidecar differs')
    for name in ('bridge_source', 'original_bridge_source', 'inherited_protocol', 'training_recipe', 'downstream_source_bundle'):
        check(p[name])
    need(Path(p['bridge_source']['path']).resolve() == Path(__file__).resolve(), 'Use the registered bridge source')
    for value in p['evaluation_sources'].values():
        check(value)
    helper = load_helper()
    need(helper.verify_downstream_sources(p) == p['source_verification'], 'Pinned downstream source verification changed')
    lineage, plan, matrix, _ = collect_training(p['training_root'], inspect_models=inspect_models)
    need(lineage == p['training_lineage'] and matrix == p['frozen_checkpoint_matrix'] and matrix['runs'] == p['runs'],
         'Frozen training bindings changed')
    need(plan['hqf']['data_manifest'] == p['original_data_manifest'] and
         plan['hqf']['inherited_contract'] == p['inherited_contract'] and
         plan['hqf']['evaluation_sources'] == p['evaluation_sources'], 'Inherited evaluation contract changed')
    verify_selected_manifest(p, full_files=full_data)
    return p


def authorize(protocol_path, auth_path):
    a = read(auth_path)
    need(a['schema'] == 'recon_selected_hqf3_development_authorization_v1' and a['authorized'] is True and
         a['task'] == 'reconstruction' and a['result_scope'] == SCOPE and a['protocol_sha256'] == sha(protocol_path) and
         a['source_user_instruction'] and a['research_verification_note'] and
         a['no_feedback_to_training_or_selection'] is True and a['all15_frozen_before_any_real_prediction'] is True,
         'Missing explicit hash-bound selected3 authorization')
    return verify_protocol(protocol_path)


def metric_values(sequence):
    m = sequence['metrics']; psnr = m['psnr']
    values = dict(lpips_alex=m['lpips_alex_mean'], ssim=m['ssim_mean'], mse=m['mse_mean'],
                  psnr_db=float('inf') if psnr['mean_is_positive_infinity'] else psnr['mean_db'])
    need(all(math.isfinite(v) for k, v in values.items() if k != 'psnr_db') and
         (math.isfinite(values['psnr_db']) or values['psnr_db'] == float('inf')), 'Invalid metric value')
    return values


def aggregate(score):
    expected = {m + '_seed' + str(s) for m in METHODS for s in SEEDS}
    need(set(score['per_arm']) == expected, 'Cannot aggregate a partial method/seed matrix')
    per_sequence, per_seed, table = [], [], []
    counts = {}
    for method in METHODS:
        seed_values = {k: [] for k in METRICS}
        for seed in SEEDS:
            rid = method + '_seed' + str(seed); arm = score['per_arm'][rid]
            seqs = arm['per_sequence']
            need(set(seqs) == set(SEQUENCES) and arm['sequence_count'] == 3, 'Cannot pool or omit sequences')
            values = []
            for seq in SEQUENCES:
                row = seqs[seq]
                n = row['scored_frame_count']
                need(isinstance(n, int) and n >= 1 and row['num_frames_including_anchor'] == n + 1 and
                     row['initial_anchor_excluded'] is True, 'Wrong endpoint denominator')
                if seq in counts:
                    need(counts[seq] == n, 'Methods/seeds scored different frame inventories')
                counts[seq] = n
                v = metric_values(row); values.append(v)
                serial = dict(v)
                serial['psnr_is_positive_infinity'] = math.isinf(v['psnr_db'])
                if serial['psnr_is_positive_infinity']:
                    serial['psnr_db'] = None
                per_sequence.append(dict(method=method, seed=seed, sequence=seq, scored_frames=n, **serial))
            macro = {k: statistics.mean(v[k] for v in values) for k in METRICS}
            for k in METRICS:
                seed_values[k].append(macro[k])
            serial_macro = dict(macro)
            serial_macro['psnr_is_positive_infinity'] = math.isinf(macro['psnr_db'])
            if serial_macro['psnr_is_positive_infinity']:
                serial_macro['psnr_db'] = None
            per_seed.append(dict(method=method, seed=seed, sequences=3, scored_frames=sum(counts.values()), **serial_macro))
        entry = dict(method=method, seeds=3, sequences_per_seed=3, scope=SCOPE)
        for k in METRICS:
            v = seed_values[k]
            # Positive-infinite PSNR means exact matches, not missing data. Its SD is undefined.
            entry[k + '_mean'] = None if any(math.isinf(x) for x in v) else statistics.mean(v)
            entry[k + '_sample_sd'] = None if any(math.isinf(x) for x in v) else statistics.stdev(v)
            if k == 'psnr_db':
                entry['psnr_mean_is_positive_infinity'] = any(math.isinf(x) for x in v)
        table.append(entry)
    return dict(schema='recon_selected_hqf3_summary_v1', result_scope=SCOPE, methods=list(METHODS), seeds=list(SEEDS),
                sequences=list(SEQUENCES), frame_counts=counts, training_budget_updates=12000,
                aggregation='Frame mean -> equal 3 sequence means -> 3 seed mean and sample SD(ddof=1)',
                blind_final_test=False, original_hqf14_results_overwritten=False,
                per_sequence=per_sequence, per_seed=per_seed, method_table=table)


def write_tables(out, summary):
    for name in ('per_sequence', 'per_seed', 'method_table'):
        rows = summary[name]
        with (out / (name + '.csv')).open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    table = summary['method_table']
    lines = ['# HQF selected3 — posthoc development result', '',
             'All five methods × three seeds completed the same 12,000-update budget. Each checkpoint was selected only with synthetic validation.', '',
             'The three previously inspected sequences were chosen after earlier HQF results. This table is development/display evidence, not HQF14 or a blind final test.', '',
             '| Method | LPIPS-Alex ↓ | SSIM ↑ | PSNR (dB) ↑ | MSE ↓ |',
             '|---|---:|---:|---:|---:|']
    for r in table:
        cells = []
        for key in METRICS:
            mean, sd = r[key + '_mean'], r[key + '_sample_sd']
            cells.append('∞ (SD undefined)' if mean is None else f'{mean:.6g} ± {sd:.3g}')
        lines.append('| ' + r['method'] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Within each seed: every inferable frame except the initial recurrent anchor, then equal weight per sequence. The table reports all three seed means and sample SD (ddof=1).',
              'The unchanged legacy scoring function’s raw artifact contains historical HQF14 labels; the canonical summary above binds the actual three-sequence inventory. No numerical metric function was changed.', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def score_existing(p, protocol_path, out, device):
    helper = load_helper()
    need(verify_protocol(protocol_path, inspect_models=False, full_data=False) == p,
         'Protocol/training bindings changed before scoring')
    data, by = verify_selected_manifest(p, full_files=True)
    data['__path__'] = p['data_manifest']['path']; data['__sha256__'] = p['data_manifest']['sha256']
    scorer = helper.load_scorer(p); scorer.DEFAULT_PROTOCOL = Path(protocol_path).resolve()
    frozen_path = out / 'inference_manifest_frozen.json'; frozen = read(frozen_path)
    need(frozen['status'] == 'all_45_inferences_complete_unscored' and
         frozen['protocol_sha256'] == sha(protocol_path) and frozen['real_scores_computed'] is False,
         'Every all15 × selected3 inference must finish and freeze before scoring')
    inferences = {}
    for row in p['runs']:
        rid = row['run_id']; inferences[rid] = {}
        for seq in SEQUENCES:
            path = check(frozen['arms'][rid]['sequences'][seq]['manifest'])
            inferences[rid][seq] = helper.validate_inference_contract(p, row, seq, path)
    import torch
    metric, source, weights, backbone = scorer.load_lpips_alex(Path(p['model_root']), torch.device(device))
    for key, path in (('metric_source', source), ('metric_weights', weights), ('backbone', backbone)):
        need(record(path) == frozen['metric_preflight'][key], 'Metric assets changed after CPU preflight')
    protocol = dict(schema=p['schema'], sequences=list(SEQUENCES),
                    photometry_and_metrics=p['inherited_contract']['photometry_and_metrics'],
                    protocol_freeze=dict(data_manifest=p['data_manifest'], pre_registered_preview_indices=p['pre_registered_preview_indices']))
    raw_path = out / 'legacy_score_raw_unchanged.json'
    raw = scorer.score_pair(protocol=protocol, protocol_sha256=sha(protocol_path), data_manifest=data, data_rows=by,
                            hqf_root=Path(p['hqf_root']), inference_manifests=inferences, run_manifest_path=frozen_path,
                            output_path=raw_path, metric_model=metric, device=torch.device(device),
                            metric_provenance=dict(metric_source=record(source), metric_weights=record(weights), backbone=record(backbone),
                                                   implementation='Unchanged pinned score_pair and metric functions; three selected sequence rows, all15 actual identities'))
    summary = aggregate(raw)
    expected_frames = sum(summary['frame_counts'].values()) * 15
    need(len(raw['per_frame']) == expected_frames, 'Raw score does not cover the complete frame matrix')
    summary.update(real_scores_computed=True, protocol=record(protocol_path), raw_score=record(raw_path),
                   frozen_inferences=record(frozen_path), scored_frame_rows=expected_frames,
                   legacy_raw_metadata_note=p['legacy_raw_scorer_note'])
    dump_new(out / 'selected3_summary.json', summary)
    write_tables(out, summary)
    return summary


def run_evaluation(protocol_path, auth_path, out, device):
    p = authorize(protocol_path, auth_path); helper = load_helper(); out = Path(out).resolve()
    need(not out.exists(), 'Refusing to overwrite an evaluation output directory')
    if str(device).startswith('cuda'):
        need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8', ':16:8'), 'Set CUDA deterministic inference environment before launch')
    preflight = helper.preflight_metric_assets(p)
    out.mkdir(parents=True); (out / 'logs').mkdir()
    state = dict(schema='recon_selected_hqf3_all15_inference_run_v1', status='running', created_utc=helper.utc(),
                 protocol_sha256=sha(protocol_path), authorization=record(auth_path), result_scope=SCOPE,
                 real_scores_computed=False, frozen_simulator_core=CORE, metric_preflight=preflight,
                 run_identity={r['run_id']: dict(method=r['method'], seed=r['seed']) for r in p['runs']}, arms={})
    def save_state():
        path = out / 'run_manifest.json'; partial = out / 'run_manifest.json.partial'
        partial.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n'); os.replace(partial, path)
    save_state()
    try:
        for row in p['runs']:
            rid = row['run_id']; cp = check(row['checkpoint'])
            arm = dict(checkpoint_path=str(cp), checkpoint_sha256=row['checkpoint']['sha256'], sequences={}, inference_manifest_sha256={})
            state['arms'][rid] = arm; save_state()
            for seq in SEQUENCES:
                target = out / 'inference' / rid / seq
                argv = [sys.executable, '-B', '-u', p['evaluation_sources']['infer_hqf_e2vid_candidate.py']['path'],
                        '--evreal-root', p['evreal_root'], '--model-root', p['model_root'], '--hqf-root', p['hqf_root'],
                        '--checkpoint', str(cp), '--sequence', seq, '--model-config', p['training_recipe']['path'],
                        '--output-root', str(target), '--device', device, '--save-every', '1', '--save-float-arrays']
                log = out / 'logs' / (rid + '__' + seq + '.log')
                with log.open('x') as f:
                    result = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, cwd=p['repository'], check=False)
                need(result.returncode == 0, f'Inference failed {rid}/{seq}; retained {log}')
                path = target / 'inference_manifest.json'
                inf = helper.validate_inference_contract(p, row, seq, path)
                arm['sequences'][seq] = dict(exit_code=0, command=argv, manifest=record(path),
                    effective_training_core=CORE, legacy_runner_reported_core=inf['provenance']['frozen_simulator_core_commit'])
                arm['inference_manifest_sha256'][seq] = sha(path); save_state()
        state['status'] = 'all_45_inferences_complete_unscored'; save_state()
        dump_new(out / 'inference_manifest_frozen.json', state)
        summary = score_existing(p, protocol_path, out, device)
        state.update(status='complete', real_scores_computed=True, finished_utc=helper.utc(),
                     summary=record(out / 'selected3_summary.json'), scored_frame_rows=summary['scored_frame_rows'])
        save_state()
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_utc=helper.utc()); save_state(); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--training-root', type=Path, required=True); q.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('verify'); q.add_argument('--protocol', type=Path, required=True)
    q = sub.add_parser('run')
    for name in ('protocol', 'authorization', 'out'):
        q.add_argument('--' + name, type=Path, required=True)
    q.add_argument('--device', default='cuda:0'); q.add_argument('--enable-real-scoring', action='store_true', required=True)
    a = parser.parse_args()
    if a.command == 'prepare':
        p = build_protocol(a.training_root, a.out); path = a.out / 'selected3_protocol.json'; dump_new(path, p)
        path.with_suffix('.json.sha256').write_text(sha(path) + '  ' + path.name + '\n')
        dump_new(a.out / 'authorization_template_DO_NOT_RUN.json', dict(schema='recon_selected_hqf3_development_authorization_v1',
            authorized=False, task='reconstruction', result_scope=SCOPE, protocol_sha256=sha(path),
            source_user_instruction='', research_verification_note='', no_feedback_to_training_or_selection=True,
            all15_frozen_before_any_real_prediction=True))
        print(json.dumps(dict(status='prepared_unscored', runs=15, sequence_runs=45, protocol=record(path), real_scores_computed=False)))
    elif a.command == 'verify':
        verify_protocol(a.protocol); print(json.dumps(dict(status='verified_unscored', runs=15, sequences=3, real_scores_computed=False)))
    else:
        run_evaluation(a.protocol, a.authorization, a.out, a.device)


if __name__ == '__main__':
    main()
