"""Alternate exact-v1 and isolated-candidate process order across three trials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--intervals', type=int, default=20)
    parser.add_argument('--cases', nargs='+', default=['s0_00', 's1_00', 's2_00', 's3_00', 'train_000'])
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    source = args.repo/'benchmark/sim2real/reviews/efficiency_continuation_20261105/source/measure_plugin.py'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    reference = Path('/home/zhaoyuhan/Eventbased_WAM/runs/simulator_sampler_optimization_20261105e/candidate_release')
    candidate = args.run/'candidate_release'
    mode = 'deterministic' if args.deterministic else 'normal'
    output = args.run/('benchmark_'+mode+'_'+str(args.intervals))
    output.mkdir(parents=True, exist_ok=False)
    rows, order = [], []
    for case_index, case in enumerate(args.cases):
        for trial in range(3):
            packages = [('reference', reference), ('candidate', candidate)]
            if (trial+case_index)%2:
                packages.reverse()
            for label, core in packages:
                destination = output/(case+'_'+label+'_'+str(trial))
                env = dict(os.environ)
                env['EVENTGENESIS_GPU_BUILD_DIR'] = str(args.run.parent/('build_reference' if label=='reference' else 'build_zbuf_candidate'))
                argv = [sys.executable, '-u', str(source), '--worker', '--core', str(core), '--out', str(destination), '--case', case, '--intervals', str(args.intervals), '--trial-index', str(trial)]
                if args.deterministic:
                    argv.append('--deterministic')
                with (output/(destination.name+'.log')).open('x') as log:
                    subprocess.run(argv, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
                row = json.loads((destination/'result.json').read_text())
                row['label'] = label
                rows.append(row)
                order.append({'case': case, 'trial': trial, 'label': label})
                (output/'progress.json').write_text(json.dumps({'rows': rows, 'order': order}, indent=2)+'\n')
                print('MEASURE', case, trial, label, row['initialized_pipeline_s'], flush=True)
    summaries = []
    for case in args.cases:
        group = {label: [x for x in rows if x['case']==case and x['label']==label] for label in ('reference', 'candidate')}
        expected = [json.loads((output/(case+'_'+label+'_'+str(t))/'physical_inputs.json').read_text()) for t in range(3) for label in ('reference','candidate')]
        if not all(x==expected[0] for x in expected):
            raise RuntimeError('Input content differs between paired measurements')
        med = {label: statistics.median(x['initialized_pipeline_s'] for x in group[label]) for label in group}
        summaries.append({'case': case, 'reference_median_s': med['reference'], 'candidate_median_s': med['candidate'], 'speedup': med['reference']/med['candidate'], 'time_reduction_fraction': 1-med['candidate']/med['reference'], 'events': {label:[x['events']['count'] for x in group[label]] for label in group}, 'allocated_peak_gib': {label:max(x['pipeline_peak_allocated_bytes'] for x in group[label])/2**30 for label in group}, 'reserved_peak_gib': {label:max(x['pipeline_peak_reserved_bytes'] for x in group[label])/2**30 for label in group}})
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_sha:
        raise RuntimeError('Measurement source changed during paired runs')
    report = {'status':'complete', 'mode':mode, 'intervals':args.intervals, 'trials_per_package_case':3, 'cases':summaries, 'rows':rows, 'order':order, 'measurement_source_sha256':source_sha, 'geometric_mean_speedup':statistics.geometric_mean(x['speedup'] for x in summaries), 'sum_of_medians_speedup':sum(x['reference_median_s'] for x in summaries)/sum(x['candidate_median_s'] for x in summaries), 'timing_boundary':'warm cached CPU inputs to sorted CPU FP64 raw events, including fresh plugin setup; no rendering, disk input/output, process startup or JIT', 'strict_default_cuda_qualification':'inherited_failed_not_overridden_by_timing'}
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PAIRED_COMPLETE', mode, args.intervals, json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
