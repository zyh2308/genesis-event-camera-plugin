#!/usr/bin/env python3
"""Synthetic-only STE paired study of bounded temporal contrast pixel weights."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch

import temporal_photo as photo_weight

RUNS = Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real')
ROOT = RUNS / '20261109-flow-temporal-weight-paired-v1'
PRIOR = RUNS / '20261108-flow-census-paired-short-v1'
PRIOR_MANIFEST = '45343592cc7eff1c86f64bafe0d91362f96371f84071daf19ae32cff9060ff04'
METHODS = ('eventgenesis', 'v2e')
ARMS = ('native_control', 'temporal_weight')
SCENE = 'camera_and_independent_rigid_motion'
REGION = 'common_event_active_valid'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
        return result.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verified_manifest(root, expected=None):
    manifest = root / 'sources/source_manifest.json'
    digest = sha(manifest)
    check(expected is None or digest == expected, 'Prior source manifest changed')
    files = read(manifest)
    check(files and all(Path(name).name == name and sha(manifest.parent / name) == pin
                        for name, pin in files.items()), 'Source manifest member changed')
    return digest


def update(C, trainer, loss, model, optimizer, train, replay, arm, step, smoke):
    events, gray, metadata = C.batch(trainer, train, replay, step)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(events, 256, use_dt4=False, flow_seq=False)
    previous, following = [gray[:, channel].detach().cpu().contiguous() for channel in (0, 1)]
    levels = photo_weight.prepare(previous, following, outputs)
    weights = [level[2] for level in levels]
    level_weights = (1., 1., 1., 1.)
    if arm == 'native_control' or smoke:
        native_total, native_photo, native_smooth = trainer.objective_dt1(
            outputs, gray[:, 0:1], gray[:, 1:2], events, loss)
    if arm == 'temporal_weight' or smoke:
        equivalent, weighted = photo_weight.photometric_terms(outputs, levels, loss.warp, level_weights)
    if smoke:
        check(bool(torch.equal(equivalent, native_photo)), 'Native photometric reconstruction differs on actual batch')
    if arm == 'native_control':
        objective, photometric, smooth = native_total, native_photo, native_smooth
    else:
        native_photo = equivalent
        smooth = native_smooth if smoke else loss.smooth_loss(outputs)
        photometric, objective = weighted, weighted + 10 * smooth
    check(all(bool(torch.isfinite(value)) for value in (objective, photometric, native_photo, smooth)), 'Nonfinite losses')
    objective.backward()
    _, signature, named = C.parameter_signature(model)
    parameters = [p for _, p in named]
    norm = C.gradient_norm([p.grad for p in parameters], parameters)
    optimizer.step()
    check(C.B.counts(optimizer) == [8911 + step] and len(optimizer.state) == 70, 'Adam counter did not advance')
    weight_record = []
    for weight in weights:
        low, high = float(weight.min()), float(weight.max())
        mean_error = float((weight.mean(dim=(-2, -1)) - 1).abs().max())
        check(0.5 - 1e-6 <= low <= high <= 2 + 1e-6 and mean_error <= 1e-6, 'Pixel mass or weight range changed')
        weight_record.append(dict(minimum=low, maximum=high, per_window_mean_error=mean_error))
    return dict(additional_update=step + 1, optimizer_update=8911 + step, **metadata,
                weight_sha256=C.B.digest_tensors(weights), weight_statistics=weight_record,
                loss=float(objective.detach()), photometric=float(photometric.detach()),
                native_photometric=float(native_photo.detach()), smooth=float(smooth.detach()),
                gradient_l2=norm, finite_gradient_tensors=70, parameter_signature_sha256=signature,
                native_photometric_forward_parity=smoke, GT_flow_optimizer_supervision=False)


def scene_rows(results, cache):
    indexed = {(r['method'], r['arm']): r['validation'] for r in results}
    full = {key: {r['rollout']: r for r in val['rollouts'] if r['stratum'] == SCENE}
            for key, val in indexed.items()}
    keys = (('eventgenesis', 'temporal_weight'), ('eventgenesis', 'native_control'),
            ('v2e', 'temporal_weight'), ('v2e', 'native_control'))
    rids = set(full[keys[0]])
    check(len(rids) == 2 and all(set(group) == rids for group in full.values()), 'Primary scene rollout set differs')
    rows, summaries = [], []
    for rid in sorted(rids):
        cases = [full[key][rid]['windows'] for key in keys]
        ids = cache['eventgenesis']['validation'][rid]['sample_ids']
        check(all(len(case) == len(ids) == 66 for case in cases), 'Wrong matched primary scene windows')
        for index, (sample_id, *window_values) in enumerate(zip(ids, *cases, strict=True)):
            values = [w[REGION]['epe_px'] for w in window_values]
            supports = [w[REGION]['pixels'] for w in window_values]
            check(len(set(supports)) == 1 and len({value is None for value in values}) == 1, 'Primary scene support differs')
            diffs = [None, None] if values[0] is None else [values[0] - values[1], values[0] - values[2]]
            rows.append(dict(rollout=rid, window_index=index, sample_id=str(sample_id), common_support_pixels=supports[0],
                             eventgenesis_temporal=values[0], eventgenesis_control=values[1], v2e_temporal=values[2],
                             v2e_control=values[3], EG_candidate_minus_EG_control=diffs[0],
                             EG_candidate_minus_V2E_candidate=diffs[1]))
        means = [full[key][rid]['regions'][REGION]['epe_px'] for key in keys]
        check(all(x is not None and math.isfinite(x) for x in means), 'No finite primary scene aggregate')
        summaries.append(dict(rollout=rid, windows=66, eventgenesis_temporal=means[0], eventgenesis_control=means[1],
                              v2e_temporal=means[2], v2e_control=means[3], EG_candidate_minus_EG_control=means[0]-means[1],
                              EG_candidate_minus_V2E_candidate=means[0]-means[2]))
    return dict(scope='prespecified_hypothesis_on_seen_synthetic_development', stratum=SCENE, region=REGION,
                unit='px_per_30ms', per_window=rows, per_rollout=summaries,
                EG_improves_vs_own_control_in_both_rollouts=all(r['EG_candidate_minus_EG_control'] < 0 for r in summaries),
                EG_beats_V2E_candidate_in_both_rollouts=all(r['EG_candidate_minus_V2E_candidate'] < 0 for r in summaries))


def study(C, contract, out, smoke, source_digest):
    trainer, models, loss, verified = C.setup(contract)
    cache, look, domains = C.load_cache(contract, ('train',) if smoke else ('train', 'validation'))
    replay = C.K.paired_replay({rid: row['stratum'] for rid, row in look['train']['eventgenesis'].items()})
    write(out / 'control_replay.json', dict(replay=replay, actual_arms=list(ARMS), used_replay_arm='native_control'))
    results, initial_hashes, learning_rates = [], [], []
    deadline = time.monotonic() + (900 if smoke else 7200)
    update_count = 1 if smoke else 200
    for method in METHODS:
        checkpoint, source = C.checkpoint(contract, method)
        initial_hashes.append(checkpoint['initial_state_sha256'])
        for arm in ARMS:
            run = out / f'{method}_{arm}'
            run.mkdir()
            model, optimizer, scheduler, lr = C.B.restore(trainer, models, checkpoint)
            model._diagnostic_method = method
            learning_rates.append(lr)
            initial_rng = C.rng_fingerprint()
            rows = []
            with (run / 'training_metrics.jsonl').open('x') as log:
                for step in range(update_count):
                    check(time.monotonic() < deadline, 'Study wall-clock deadline exceeded')
                    row = update(C, trainer, loss, model, optimizer, cache[method]['train'], replay, arm, step, smoke)
                    rows.append(row)
                    log.write(json.dumps(row, allow_nan=False) + '\n')
                    log.flush()
                    if step + 1 == 198:
                        scheduler.step()
                        check([g['lr'] for g in optimizer.param_groups] == lr, 'Original scheduler boundary changed')
                    if (step + 1) % 20 == 0:
                        print('FLOW_TEMPORAL_WEIGHT_UPDATE', method, arm, step + 1, flush=True)
            saved = C.B.save_checkpoint(run / 'last.pt', trainer, models, model, optimizer, scheduler, source, update_count)
            loaded = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
            check(loaded['scheduler'] == scheduler.state_dict() and set(loaded['rng_state']) == {'python','numpy','torch','cuda'}, 'Scheduler/RNG save differs')
            del loaded
            validation = None if smoke else C.B.evaluate(trainer, model, cache, look['validation'][method])
            paired = {name: C.hash_json([row[name] for row in rows]) for name in
                      ('sample_ids', 'label_augmentation_sha256', 'augmentation_seed_sequences', 'weight_sha256')}
            result = dict(method=method, seed=0, arm=arm, additional_updates=update_count,
                          source_checkpoint=source, source_manifest_sha256=source_digest,
                          initial_lr=lr, initial_rng=initial_rng, paired_sequence_sha256=paired,
                          checkpoint=saved, validation=validation, finite_gradient_tensors=70,
                          native_photometric_forward_parity=smoke, real_data_used=False)
            write(run / 'completion.json', result)
            results.append(result)
            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
        check(sha(Path(source['path'])) == source['sha256'], 'Formal checkpoint changed')
        del checkpoint
    check(len(set(initial_hashes)) == 1 and all(lr == learning_rates[0] for lr in learning_rates), 'Original initialization/LR differs')
    check(all(r['paired_sequence_sha256'] == results[0]['paired_sequence_sha256'] for r in results), 'Paired label/sample/augmentation/pixel-weight hash differs')
    check(all(results[i]['initial_rng'] == results[i+1]['initial_rng'] for i in (0, 2)), 'Within-method restored RNG differs')
    result = dict(schema='flow_temporal_weight_paired_v1', status='complete', source_manifest_sha256=source_digest,
                  prior_manifest_sha256=PRIOR_MANIFEST, model_source_verification=verified,
                  mode='smoke' if smoke else 'study', results=results, raw_endpoint_domains=domains,
                  paired_weights_per_step_identical=True, GT_flow_optimizer_supervision=False,
                  real_data_used=False, real_scores_used=False, automatic_long_training_or_real_scoring=False)
    if not smoke:
        # Only the legacy contract adapter maps the new candidate to the old
        # expected slot. Saved arm identities remain canonical and distinct.
        gate_rows = [dict(row, arm='census' if row['arm'] == 'temporal_weight' else row['arm']) for row in results]
        result['development_quality_gate'] = C.K.quality(gate_rows)
        write(out / 'preregistered_scene_diagnostic.json', scene_rows(results, cache))
    check(C.input_sources(contract)[1] == verified, 'Native model source changed during study')
    check(verified_manifest(PRIOR, PRIOR_MANIFEST) == PRIOR_MANIFEST and verified_manifest(ROOT) == source_digest,
          'Study source manifest changed')
    write(out / 'completion.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('cpu', 'smoke', 'study'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    check(args.out.resolve() == ROOT / args.mode and not args.out.exists(), 'Use a fresh canonical output directory')
    source_digest = verified_manifest(ROOT)
    verified_manifest(PRIOR, PRIOR_MANIFEST)
    sys.path.insert(0, str(PRIOR / 'sources'))
    import run_census_paired_short as C
    contract = C.K.validate(read(PRIOR / 'sources/contract.json'))
    args.out.mkdir()
    try:
        if args.mode == 'cpu':
            check(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU admission requires hidden CUDA')
            C.cpu_admission(contract, PRIOR, args.out)
            write(args.out / 'admission_binding.json', dict(status='passed', source_manifest_sha256=source_digest,
                  prior_source_manifest_sha256=PRIOR_MANIFEST, cpu_completion_sha256=sha(args.out / 'completion.json')))
        else:
            cpu = read(ROOT / 'cpu/admission_binding.json')
            check(cpu['status'] == 'passed' and cpu['source_manifest_sha256'] == source_digest and
                  cpu['cpu_completion_sha256'] == sha(ROOT / 'cpu/completion.json'), 'CPU admission missing or incompatible')
            if args.mode == 'study':
                smoke = read(ROOT / 'smoke/completion.json')
                check(smoke['status'] == 'complete' and smoke['mode'] == 'smoke' and
                      smoke['source_manifest_sha256'] == source_digest and len(smoke['results']) == 4 and
                      smoke['paired_weights_per_step_identical'] and
                      all(r['additional_updates'] == 1 and r['native_photometric_forward_parity'] and
                          r['checkpoint']['model_rebuild_passed'] and r['checkpoint']['optimizer_state_equal']
                          for r in smoke['results']), 'All four actual smoke updates/recoveries required')
            study(C, contract, args.out, args.mode == 'smoke', source_digest)
    except BaseException:
        write(args.out / 'failure.json', dict(status='failed', mode=args.mode, error=traceback.format_exc(), real_data_used=False))
        raise


if __name__ == '__main__':
    main()
