"""Aggregate actual-entry timing evidence; fail closed on incomplete validation.

Pure Python standard library. No Torch import, GPU work, artifact rewriting,
training, or result selection. All output is derived from the stored trials.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


CASES = ('s0_00', 's1_00', 's2_00', 's3_00', 'train_000')
METHODS = ('frozen_gpu', 'candidate_gpu', 'v2e_auto')
EXPECTED_TRIALS = 3
PHYSICAL_KEYS = ('case', 'intervals', 'index', 'shape_hw', 'simulated_s',
                 'timestamps_sha256', 'physical_sha256', 'display_sha256',
                 'intrinsics_sha256', 'segmentation_ids_sha256')
METRICS = ('genwall_s', 'initialized_pipeline_s', 'setup_s',
           'peak_worker_allocated_bytes', 'peak_reported_bytes')
STAT_KEYS = ('median', 'min', 'max', 'q25', 'q75', 'iqr')
MISSING = object()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def quantile(values, p):
    values = sorted(values)
    position = (len(values)-1)*p
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (position-lo)*(values[hi]-values[lo])


def stats(values):
    if not values:
        return None
    q25, q75 = quantile(values, .25), quantile(values, .75)
    return dict(n=len(values), median=statistics.median(values),
                min=min(values), max=max(values), q25=q25, q75=q75,
                iqr=q75-q25, quantile_definition='linear, position=(n-1)*p')


def geometric_mean(values):
    return math.exp(math.fsum(math.log(x) for x in values)/len(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--profile', type=Path, required=True,
                        help='Immutable profile used by all benchmark workers')
    parser.add_argument('--expected-gpu', required=True,
                        help='Actual physical GPU2 UUID, not logical cuda:0 or index2')
    parser.add_argument('--expected-cublas', default='<unset>',
                        help='Exact recorded value; use <unset> if intentionally unset')
    parser.add_argument('--metadata', type=Path,
                        help='Optional separately recorded benchmark environment receipt')
    args = parser.parse_args()
    root = args.root.resolve()
    profile_path = args.profile.resolve()
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    profile_sha = sha(profile_path)
    expected_noise = profile['noise_config_kwargs']
    external = json.loads(args.metadata.read_text(encoding='utf-8')) if args.metadata else {}
    issues, jobs, input_groups = [], [], {}
    runtime_reference, source_by_method = None, {}

    def issue(where, text):
        issues.append(dict(where=where, problem=text))

    if not args.expected_gpu.startswith('GPU-'):
        issue('expected GPU', 'Provide the physical GPU2 UUID beginning GPU-, independently verified with nvidia-smi')

    expected_cublas = None if args.expected_cublas == '<unset>' else args.expected_cublas
    requested = [(case, method) for case in CASES for method in METHODS]
    requested.append(('s2_00', 'v2e_auto_min63'))
    for case, method in requested:
        where = case+'/'+method
        folder = root/'bench'/case/method
        timings_path = folder/'timings.json'
        inputs_path = folder/'physical_inputs.json'
        try:
            document = json.loads(timings_path.read_text(encoding='utf-8'))
            physical = json.loads(inputs_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            issue(where, 'Missing or unreadable evidence: '+str(exc))
            continue
        before_issues = len(issues)
        job = dict(case=case, method=method, directory=str(folder),
                   timings_sha256=sha(timings_path), physical_inputs_sha256=sha(inputs_path))
        rows = document.get('rows', [])
        source_hashes = document.get('core_source_sha256')
        if not isinstance(source_hashes, dict) or not source_hashes or 'gpu_geometry.py' not in source_hashes:
            issue(where, 'Immutable GPU core source hash receipt is missing')
        elif method in source_by_method and source_hashes != source_by_method[method]:
            issue(where, 'GPU source hashes changed between cases for the same method')
        else:
            source_by_method[method] = source_hashes
        if document.get('status') != 'complete':
            issue(where, 'timings status is not complete')
        if len(rows) != EXPECTED_TRIALS or sorted(r.get('trial', -1) for r in rows) != [0, 1, 2]:
            issue(where, 'Exactly trials0,1,2 are required; found '+str([r.get('trial') for r in rows]))
        inputs = {key: physical.get(key, '<missing>') for key in PHYSICAL_KEYS}
        if any(key not in physical for key in PHYSICAL_KEYS):
            issue(where, 'Physical-input description omits a required field')
        if physical.get('case') != case or physical.get('intervals') != 20:
            issue(where, 'Expected registered case and20 intervals')
        hash_values = [physical.get(key) for key in ('timestamps_sha256', 'intrinsics_sha256', 'segmentation_ids_sha256')]
        hash_values += list(physical.get('physical_sha256', {}).values())
        hash_values += list(physical.get('display_sha256', []))
        if not hash_values or not all(isinstance(h, str) and len(h) == 64
                                     and all(c in '0123456789abcdef' for c in h) for h in hash_values):
            issue(where, 'Malformed physical/display input SHA256')
        signature = canonical(inputs)
        if case in input_groups and signature != input_groups[case]['signature']:
            issue(where, 'Physical/display hashes or registered physical window differ between methods')
        else:
            input_groups[case] = dict(signature=signature, physical_inputs=inputs)

        sidecar_path = folder/'environment.json'
        sidecar = json.loads(sidecar_path.read_text(encoding='utf-8')) if sidecar_path.exists() else {}
        external_job = external.get('jobs', {}).get(where, {})

        def evidence(row, *keys):
            # Values must be in recorded output, never inferred from source intent.
            for container in (row, document.get('metadata', {}), document, sidecar,
                              external_job, external.get('metadata', {}), external):
                for key in keys:
                    if key in container:
                        return container[key]
            return MISSING

        metric_values = {key: [] for key in METRICS}
        v2e_information, trial_evidence = [], []
        for row in rows:
            trial = row.get('trial')
            at = where+'/trial'+str(trial)
            if row.get('case') != case or row.get('method') != method:
                issue(at, 'Row case or method does not match its evidence directory')
            if row.get('gpu_uuid') != args.expected_gpu:
                issue(at, 'Recorded CUDA_VISIBLE_DEVICES does not equal the expected physical GPU2 UUID')
            if row.get('deterministic_algorithms') is not False:
                issue(at, 'Normal-mode timing requires deterministic_algorithms=false; exact gates are separate')
            if row.get('full_stage9_noise') is not True or row.get('same_temporal_density_claimed') is not False:
                issue(at, 'Full noise declaration or no-equal-temporal-density declaration missing')
            actual_profile = evidence(row, 'profile_sha256', 'sensor_profile_sha256')
            actual_noise = evidence(row, 'noise_config_kwargs', 'noise_config')
            actual_cublas = evidence(row, 'cublas_workspace_config', 'CUBLAS_WORKSPACE_CONFIG')
            actual_sampler = evidence(row, 'sampler_kwargs')
            actual_plugin = evidence(row, 'plugin_kwargs')
            actual_runtime = evidence(row, 'runtime')
            if actual_profile != profile_sha:
                issue(at, 'Recorded profile SHA256 missing or different from immutable expected profile')
            if actual_noise is MISSING or canonical(actual_noise) != canonical(expected_noise):
                issue(at, 'Recorded complete noise config missing or different from expected profile')
            if actual_cublas is MISSING or actual_cublas != expected_cublas:
                issue(at, 'Recorded CUBLAS_WORKSPACE_CONFIG missing or different')
            if actual_sampler is MISSING or canonical(actual_sampler) != canonical(profile['sampler_kwargs']):
                issue(at, 'Recorded full sampler config missing or different from profile')
            if actual_plugin is MISSING or canonical(actual_plugin) != canonical(profile['plugin_kwargs']):
                issue(at, 'Recorded full plugin config missing or different from profile')
            runtime_expected = dict(num_threads=2, interop_threads=1, matmul_tf32=False, cudnn_tf32=False)
            if not isinstance(actual_runtime, dict) or any(type(actual_runtime.get(k)) is not type(v)
                    or actual_runtime.get(k) != v for k, v in runtime_expected.items()):
                issue(at, 'Runtime receipt must show2 Torch threads,1 interop thread and both TF32 flags false')
            elif not actual_runtime.get('torch') or not actual_runtime.get('cuda'):
                issue(at, 'Torch/CUDA runtime versions missing')
            elif runtime_reference is None:
                runtime_reference = actual_runtime
            elif canonical(actual_runtime) != canonical(runtime_reference):
                issue(at, 'Torch/CUDA/thread/precision runtime differs across methods or trials')
            if row.get('simulated_s') != physical.get('simulated_s'):
                issue(at, 'Trial duration differs from registered physical input window')
            wall, setup, initialized = row.get('wall_s'), row.get('setup_s'), row.get('initialized_pipeline_s')
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       and math.isfinite(x) and x >= 0 for x in (wall, setup, initialized)):
                issue(at, 'Timing fields must be finite nonnegative seconds')
            elif wall <= 0 or initialized <= 0 or not math.isclose(initialized, wall+setup, rel_tol=1e-12, abs_tol=1e-12):
                issue(at, 'Initialized pipeline must equal setup+generation and have positive duration')
            else:
                metric_values['genwall_s'].append(wall)
                metric_values['setup_s'].append(setup)
                metric_values['initialized_pipeline_s'].append(initialized)
            reported_peak = row.get('peak_bytes')
            worker_peak = row.get('worker_peak_bytes', reported_peak)
            for key, value in (('peak_reported_bytes', reported_peak),
                               ('peak_worker_allocated_bytes', worker_peak)):
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    issue(at, 'Missing/invalid '+key)
                else:
                    metric_values[key].append(value)
            events_path = folder/('trial'+str(trial)+'_events.npy')
            if not events_path.is_file():
                issue(at, 'Raw events array missing')
            if method in ('frozen_gpu', 'candidate_gpu'):
                steps = row.get('substeps')
                if row.get('fallbacks') != 0 or not isinstance(steps, list) or len(steps) != 20 or set(steps) != {377}:
                    issue(at, 'EventGenesis must have zero fallback and exactly377 substeps at every one of20 intervals')
            else:
                if row.get('minimum_factor') != (63 if method.endswith('min63') else 2):
                    issue(at, 'V2E native-auto minimum factor differs from registration')
                if row.get('substeps') is not None:
                    issue(at, 'V2E per-batch automatic sampling must not be misreported as a constant substep list')
                info = {key: row.get(key) for key in ('input_frames', 'first_frame_s', 'last_frame_s', 'max_dt_s', 'factor', 'minimum_factor')}
                if not isinstance(info['input_frames'], int) or info['input_frames'] <= 0:
                    issue(at, 'V2E frame count missing or invalid')
                time_values = [info[k] for k in ('first_frame_s', 'last_frame_s', 'max_dt_s')]
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in time_values):
                    issue(at, 'V2E supported timestamp range/maxdt missing or invalid')
                elif not (0 <= info['first_frame_s'] < info['last_frame_s'] <= physical['simulated_s']+1e-12 and info['max_dt_s'] > 0):
                    issue(at, 'V2E timestamps are outside the registered physical duration')
                info['trial'] = trial
                v2e_information.append(info)
            trial_evidence.append(dict(trial=trial, gpu_uuid=row.get('gpu_uuid'),
                gpu_name=row.get('gpu_name'), profile_sha256=None if actual_profile is MISSING else actual_profile,
                cublas_workspace_config=None if actual_cublas is MISSING else actual_cublas,
                noise_config_exact=actual_noise is not MISSING and canonical(actual_noise) == canonical(expected_noise),
                runtime=None if actual_runtime is MISSING else actual_runtime,
                raw_events=str(events_path), timing_boundary=row.get('timing_boundary'),
                primary_timing_boundary=row.get('primary_timing_boundary')))
        job.update(metrics={key: stats(values) for key, values in metric_values.items()},
                   trials=trial_evidence, v2e_temporal_information=v2e_information,
                   complete_three_trials=len(rows) == EXPECTED_TRIALS,
                   validation_passed=len(issues) == before_issues,
                   core_source_sha256=document.get('core_source_sha256'))
        jobs.append(job)

    lookup = {(j['case'], j['method']): j for j in jobs}
    comparisons = []
    for case in CASES:
        group = [lookup.get((case, m)) for m in METHODS]
        if not all(group):
            continue
        comparison = dict(case=case, validation_passed=all(j['validation_passed'] for j in group))
        for metric in ('initialized_pipeline_s', 'genwall_s'):
            medians = [j['metrics'][metric]['median'] if j['metrics'][metric] else None for j in group]
            if all(v is not None and v > 0 for v in medians):
                old, new, v2e = medians
                comparison[metric] = dict(old_over_new_speedup=old/new,
                    new_time_reduction_percent=100*(1-new/old),
                    candidate_over_v2e_slowdown=new/v2e,
                    frozen_over_v2e_slowdown=old/v2e)
        comparisons.append(comparison)
    totals = {}
    for metric in ('initialized_pipeline_s', 'genwall_s'):
        usable = [c[metric] for c in comparisons if metric in c]
        totals[metric] = dict(cases=len(usable),
            geometric_mean_old_over_new_speedup=geometric_mean([c['old_over_new_speedup'] for c in usable]) if len(usable) == len(CASES) else None,
            geometric_mean_candidate_over_v2e_slowdown=geometric_mean([c['candidate_over_v2e_slowdown'] for c in usable]) if len(usable) == len(CASES) else None,
            all_five_cases_validated=len(comparisons) == len(CASES) and all(c['validation_passed'] for c in comparisons))
    summary = dict(schema='eventgenesis_sampler_efficiency_summary_v1',
        status='complete' if not issues else 'failed_validation',
        primary_metric='initialized_pipeline_s',
        definitions=dict(genwall_s='Stored wall_s: cached endpoint input to sorted CPU raw events, setup excluded',
            initialized_pipeline_s='Stored setup_s+wall_s: includes per-clip initialization and each sensor first frame; pretrained model/private compiled kernel already loaded',
            peak_worker_allocated_bytes='Whole worker Torch allocated-memory peak, including persistent model allocations; not device total usage',
            peak_reported_bytes='Original per-method peak_bytes; ours subtracts unrelated pre-call allocation, V2E reports whole worker peak',
            old_over_new_speedup='median frozen time / median candidate time; >1 means candidate is faster',
            candidate_over_v2e_slowdown='median candidate time / median V2E native-auto time; >1 means candidate is slower',
            aggregation='Geometric mean of per-case ratios across all five fixed cases, never a ratio of arbitrary average clip times'),
        limitations=['No equal-event-quality, temporal-density, or accuracy claim is made for V2E.',
            'Three repeats quantify timing spread; IQR is descriptive and is not a confidence interval.',
            'Physical Genesis rendering, input disk loading and event serialization are excluded.',
            'V2E minimum63 is a separate numerical-step control, not a default or quality-matched configuration.',
            'Normal-mode timing cannot replace the separate deterministic full-output quality gates.'],
        expected=dict(gpu_uuid=args.expected_gpu, profile=str(profile_path), profile_sha256=profile_sha,
                      noise_config_kwargs=expected_noise, cublas_workspace_config=expected_cublas,
                      runtime=runtime_reference,
                      trials_per_job=EXPECTED_TRIALS, main_cases=list(CASES), main_methods=list(METHODS)),
        metadata_receipt=str(args.metadata) if args.metadata else None,
        source_script_sha256=sha(Path(__file__)), validation=dict(passed=not issues, issues=issues,
            jobs_found=len(jobs), jobs_expected=len(requested), physical_inputs_match_by_case=len(jobs) == len(requested) and not any('Physical/display hashes' in x['problem'] or 'Physical-input' in x['problem'] for x in issues)),
        physical_inputs={k: v['physical_inputs'] for k,v in input_groups.items()},
        jobs=jobs, per_case_comparisons=comparisons, aggregate_ratios=totals)
    json_path = root/'benchmark_summary.json'
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    csv_path = root/'benchmark_summary.csv'
    columns = ['case', 'method', 'validation_passed']+[m+'_'+s for m in METRICS for s in STAT_KEYS]
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for job in jobs:
            row = {k: job[k] for k in ('case', 'method', 'validation_passed')}
            for metric in METRICS:
                if job['metrics'][metric]:
                    row.update({metric+'_'+stat: job['metrics'][metric][stat] for stat in STAT_KEYS})
            writer.writerow(row)
    ratio_path = root/'benchmark_case_ratios.csv'
    ratio_keys = ('old_over_new_speedup', 'new_time_reduction_percent',
                  'candidate_over_v2e_slowdown', 'frozen_over_v2e_slowdown')
    with ratio_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case', 'metric', 'validation_passed']+list(ratio_keys))
        writer.writeheader()
        for c in comparisons:
            for metric in ('initialized_pipeline_s', 'genwall_s'):
                if metric in c:
                    writer.writerow(dict(case=c['case'], metric=metric,
                                         validation_passed=c['validation_passed'], **c[metric]))
    print(json.dumps(dict(status=summary['status'], jobs=len(jobs),
        validation_issues=len(issues), primary=totals['initialized_pipeline_s'],
        json=str(json_path), csv=str(csv_path), ratios_csv=str(ratio_path)), indent=2))
    return 0 if not issues else 2


if __name__ == '__main__':
    raise SystemExit(main())
