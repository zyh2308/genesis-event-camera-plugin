"""Fresh-process P0 gates against the exact archived index-reuse v1 package."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--mode', choices=['p0', 'shorts', 'loader'], required=True)
    parser.add_argument('--delta', type=Path)
    parser.add_argument('--coverage', action='store_true')
    args = parser.parse_args()
    root = args.run.resolve()
    review = args.repo / 'benchmark/sim2real/reviews'
    old = review / 'sampler_cost_20261105/source'
    own = review / 'efficiency_continuation_20261105/source'
    base = Path('/home/zhaoyuhan/Eventbased_WAM/runs/simulator_sampler_optimization_20261105e/candidate_release')
    candidate = root / 'candidate_release'
    def run(script, *argv, allowed=(0,)):
        command = [sys.executable, str(script), *map(str, argv)]
        print('RUN', json.dumps(command), flush=True)
        result = subprocess.run(command)
        if result.returncode not in allowed:
            raise subprocess.CalledProcessError(result.returncode, command)
        return result.returncode
    if args.mode == 'p0':
        extra = ['--delta', args.delta] if args.delta else []
        run(own / 'prepare_candidate.py', '--base-release', base, '--out', candidate, *extra)
        checker = 'check_coverage.py' if args.coverage else 'check_projection_storage.py'
        run(own / checker, '--reference', base/'genesis_event_plugin/gpu_geometry.py', '--candidate', candidate/'genesis_event_plugin/gpu_geometry.py', '--out', root/('coverage_check.json' if args.coverage else 'projection_check.json'))
        cases = ['s2_00']
    elif args.mode == 'shorts':
        cases = ['s0_00', 's1_00', 's3_00', 'train_000']
    else:
        cases = ['s3_00']
    for index, case in enumerate(cases):
        intervals = 20 if args.mode == 'loader' else 2
        folder = root / ('loader_inputs' if args.mode == 'loader' else 'short_gates')
        order = [('reference', base), ('candidate', candidate)]
        if index % 2:
            order.reverse()
        for label, core in order:
            run(old/'full_plugin_gate.py', '--core', core, '--out', folder/label/case, '--case', case, '--intervals', intervals)
        run(old/'compare_plugin_gates_v2.py', '--left', folder/'reference'/case, '--right', folder/'candidate'/case, '--out', root/'comparisons'/(args.mode+'_'+case+'.json'), '--require-exact')
    if args.mode == 'loader':
        import numpy as np
        events_dir = root/'formal_events'
        events_dir.mkdir(exist_ok=False)
        for label in ('reference', 'candidate'):
            data = np.load(root/'loader_inputs'/label/'s3_00/events.npy')
            origin = json.loads((root/'loader_inputs'/label/'s3_00/trace.json').read_text())['physical_origin_s']
            data[:, 0] += origin
            np.save(events_dir/(label+'.npy'), data, allow_pickle=False)
        manifest = json.loads((review/'sampler_cost_20261105/evidence/formal_cases.json').read_text())
        case = manifest['cases'][0]
        case.update(id='projection_storage_exact_200ms', reference=str(events_dir/'reference.npy'), candidate=str(events_dir/'candidate.npy'), start_s=origin, end_s=origin+.2)
        manifest['cases'] = [case]
        manifest_path = root/'formal_cases.json'
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        project = base.parents[2]
        campaign = project/'runs/sim2real/20261103-stage9-direct-v1'
        scripts = campaign/'sources/benchmark/sim2real/scripts'
        run(old/'upstream_check_formal_loaders_v2.py', '--cases-json', manifest_path, '--scripts-root', scripts, '--flow-trainer-source', scripts/'train_steflow_stage9.py', '--e2vid-trainer-source', scripts/'train_e2vid_stage9.py', '--e2vid-model-root', project/'external/sources/downstream/event_cnn_minimal', '--output-root', root/'formal_loader_gate')
    print('CHECK_STAGE_COMPLETE', args.mode, flush=True)


if __name__ == '__main__':
    main()
