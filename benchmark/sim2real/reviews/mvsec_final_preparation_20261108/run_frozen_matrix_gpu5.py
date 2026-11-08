"""One leased GPU5, fifteen immutable MVSEC evaluations; no training or tuning.

Default: local/server preflight only. --execute requires existing reviewed frozen
protocol, manifest, authorization and their SHA256 values. This runner never
creates scoring authorization. --self-test exercises pure offline contracts only.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

PROJECT = Path('/home/zhaoyuhan/Eventbased_WAM')
RUNS = PROJECT / 'runs/sim2real'
ROOT = RUNS / '20261108-mvsec-frozen-matrix-gpu5-v1'
PREPARATION = RUNS / '20261108-mvsec-final-preparation-v1'
UUID = 'GPU-06ee62ba-c171-40d4-b4ae-ea63391dc885'
METHODS = ('eventgenesis', 'v2e', 'vid2e', 'film_esim', 'rife_esim')
SEEDS = (0, 1, 2)
SEQUENCES = ('indoor_flying1', 'indoor_flying2', 'indoor_flying3', 'outdoor_day1')
COUNTS = dict(zip(SEQUENCES, (2165, 2624, 2911, 800)))
PHASES = tuple((f'seed{seed}_{method}', seed, method)
               for seed in SEEDS for method in METHODS)
BRIDGE_SHA = '54b39d630bcf90096133abad9aa890200f0cd9a7b9223bb06afef210f9498d28'
PROBE_SHA = '12a166df66f902e571d079ae2ace119a08276dda63d52568cdf7553affdaf7d7'
MODEL_COMMIT = 'eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6'
CORE = '62da4f35ce3c5c96f04268bd1687a27657af9238'
TOTAL_SECONDS = 18 * 3600
CHILD_SECONDS = 2 * 3600


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def canonical_file(path):
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and path.resolve(strict=True) == path, 'Expected canonical regular file: ' + str(path))


def file_record(path):
    canonical_file(path)
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha(path))


def checked_hash(path, digest):
    canonical_file(path)
    require(isinstance(digest, str) and len(digest) == 64
            and all(c in '0123456789abcdef' for c in digest), 'Invalid explicit SHA256')
    require(sha(path) == digest, 'Immutable file SHA changed: ' + str(path))


def frozen_contract(protocol, manifest, authorization, protocol_sha, manifest_sha):
    """The original hash-bound gate, with complete fixed matrix/scope checks."""
    for name, value, schema in (
        ('protocol', protocol, 'stage9_mvsec_dt1_protocol_v2'),
        ('manifest', manifest, 'stage9_mvsec_matrix_manifest_v2'),
    ):
        require(value.get('schema') == schema and value.get('status') == 'frozen_by_research_review'
                and value.get('real_scores_authorized') is True, name + ' is not frozen and authorized')
        require(value.get('required_sequences') == list(SEQUENCES)
                and value.get('evaluation_sequences') == list(SEQUENCES)
                and value.get('missing_required_sequences') == []
                and value.get('result_scope') == 'required_four_sequence_final', name + ' incomplete scope')
    require(protocol.get('methods') == list(METHODS) and protocol.get('seeds') == list(SEEDS)
            and protocol.get('dt') == 1 and protocol.get('training_budget_updates') == 8910
            and protocol.get('model_commit') == MODEL_COMMIT
            and protocol.get('no_real_data_model_selection') is True, 'Frozen model/matrix changed')
    require(manifest.get('complete_checkpoint_matrix') is True
            and manifest.get('missing_training_runs') == []
            and manifest.get('total_registered_windows') == 8500
            and manifest.get('registered_window_counts') == COUNTS, 'Incomplete fixed inputs or weights')
    expected = {f'{method}/seed{seed}' for _, seed, method in PHASES}
    require(set(manifest.get('checkpoint_binding', {})) == expected, 'Need exactly fifteen frozen weights')
    require(authorization.get('authorized_real_test') is True
            and authorization.get('authorization_id') not in (None, '')
            and authorization.get('approved_by') not in (None, '')
            and authorization.get('evaluation_scope') == 'required_four_sequence_final'
            and authorization.get('protocol_sha256') == protocol_sha
            and authorization.get('manifest_sha256') == manifest_sha, 'Hash-bound research authorization absent')


def preserves_candidate(candidate, frozen):
    for key, value in candidate.items():
        if key not in ('status', 'real_scores_authorized'):
            require(frozen.get(key) == value, 'Freeze changed prepared field: ' + key)


def numerical_contract(report, manifest, authorization, report_sha, prepared_candidate_sha):
    """Require fixed synthetic forward parity, never use any real score here."""
    require(isinstance(report, dict) and report.get('status') == 'passed',
            'Missing or failed synthetic numerical preflight')
    require(report.get('schema') == 'mvsec_synthetic_forward_equivalence_v1'
            and report.get('GPU_used') is True and report.get('gpu_uuid') == UUID
            and report.get('cpu_self_test_only') is not True
            and report.get('real_data_read') is False and report.get('real_scores_computed') is False
            and report.get('training_started') is False and report.get('scoring_authorization_created') is False,
            'CPU-only test or non-synthetic report cannot authorize full-model scoring')
    require(report.get('manifest_sha256') == prepared_candidate_sha,
            'Numerical probe must use this completed preparation receipt candidate manifest')
    require(report.get('thresholds', {}).get('max_abs_px') == .001
            and report.get('thresholds', {}).get('rms_px') == .0001,
            'Numerical thresholds must not be weakened')
    require(authorization.get('numerical_preflight_sha256') == report_sha,
            'Research authorization does not bind numerical preflight SHA')
    bindings = manifest.get('checkpoint_binding', {})
    checkpoints = report.get('checkpoints')
    require(len(bindings) == 15 and isinstance(checkpoints, dict)
            and set(checkpoints) == set(bindings), 'Numerical preflight must cover exactly fifteen frozen weights')
    for key, binding in bindings.items():
        row = checkpoints[key]
        require(isinstance(row, dict) and row.get('passed') is True
                and row.get('checkpoint_sha256') == binding['selected_checkpoint_sha256'],
                'Numerical preflight checkpoint mismatch/failure: ' + key)
        cases = row.get('cases')
        require(isinstance(cases, list) and len(cases) == 2,
                'Numerical preflight needs exactly two synthetic cases: ' + key)
        for case in cases:
            require(isinstance(case, dict) and case.get('passed') is True,
                    'Synthetic forward case failed: ' + key)
            for metric, threshold in [('max_abs_px', .001), ('rms_px', .0001)]:
                value = case.get(metric)
                require(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and 0 <= value <= threshold,
                        'Synthetic forward parity exceeds fixed threshold: ' + key + '/' + metric)


def numerical_provenance(args, report, manifest, verified_model):
    require(report.get('flow_unit') == 'px_per_interval' and report.get('dtype') == 'float32'
            and report.get('tf32') is False and report.get('eval_mode') is True
            and report.get('inference_mode') is True, 'Numerical model mode/unit contract changed')
    prepared = read(PREPARATION / 'cpu_preparation/PREPARATION_COMPLETE_NO_SCORING.json')
    require(report.get('manifest', {}).get('path') == prepared['manifest']
            and report['manifest'].get('sha256') == prepared['manifest_sha256']
            and report['manifest'].get('size_bytes') == Path(prepared['manifest']).stat().st_size,
            'Probe did not use the exact preparation-receipt candidate file')
    oracle = report.get('fp64_oracle', {})
    require(oracle.get('passed') is True and oracle.get('status') == 'passed'
            and oracle.get('dtype') == 'float64' and oracle.get('max_abs_threshold') == 1e-12,
            'FP64 independent mathematical oracle failed or missing')
    for name in ('reference_max_abs', 'adapter_max_abs', 'reference_adapter_max_abs'):
        value = oracle.get(name)
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and 0 <= value <= 1e-12, 'FP64 oracle error exceeds fixed bound')
    fixtures = report.get('fixtures')
    case_ids = ['sparse_gaussian_counts', 'structured_gaussian_counts']
    require(isinstance(fixtures, list) and len(fixtures) == 2
            and [r.get('case_id') for r in fixtures] == case_ids, 'Unexpected synthetic fixtures')
    for fixture in fixtures:
        require(fixture.get('base_seed') == 20261108 and fixture.get('native_shape') == [2, 260, 346, 5]
                and fixture.get('input_shape') == [1, 2, 352, 352, 5], 'Synthetic fixture seed/shape changed')
        value = fixture.get('input_sha256')
        require(isinstance(value, str) and len(value) == 64
                and all(c in '0123456789abcdef' for c in value), 'Missing synthetic input fingerprint')
    source = report.get('model_source', {})
    require(source.get('repository') == 'STE-FlowNet' and source.get('commit') == MODEL_COMMIT
            and source.get('verification') == 'manifest_file_sha256_and_tree_sha256'
            and source.get('files_verified') == 14
            and source.get('bytes_verified') == verified_model['bytes_verified']
            and source.get('tree_sha256') == verified_model['tree_sha256'], 'Probe model-source tree differs')
    for key, row in report['checkpoints'].items():
        require(row.get('model_commit') == MODEL_COMMIT and row.get('model_source_verification') == source,
                'Per-checkpoint model-source verification changed: ' + key)
        require([c.get('case_id') for c in row['cases']] == case_ids
                and [c.get('input_sha256') for c in row['cases']] == [f['input_sha256'] for f in fixtures],
                'Different synthetic inputs used by a checkpoint: ' + key)
    sources = report.get('sources', {})
    require(all(name in sources for name in ('reference', 'adapter', 'compat', 'probe_script')),
            'Probe source evidence incomplete')
    reference = Path(manifest['campaign']) / 'sources/benchmark/sim2real/scripts/run_steflow_stage2_pilot.py'
    for role, expected in [('reference', reference),
                           ('adapter', Path(manifest['source_scripts']['adapter']['path'])),
                           ('compat', Path(manifest['source_scripts']['model_compat']['path']))]:
        item = sources[role]
        require(item.get('path') == str(expected), 'Wrong probe source location: ' + role)
        checked_hash(expected, item.get('sha256'))
    require(sources['adapter']['sha256'] == manifest['source_scripts']['adapter']['sha256']
            and sources['compat']['sha256'] == manifest['source_scripts']['model_compat']['sha256'],
            'Numerical and scoring source hashes differ')
    script = sources['probe_script']
    require(script.get('sha256') == PROBE_SHA, 'Unreviewed numerical probe implementation')
    checked_hash(Path(script['path']), PROBE_SHA)


def model_tree(model_root, bundle_path):
    require(model_root.is_dir() and model_root.resolve(strict=True) == model_root,
            'Noncanonical model source root')
    rows = [row for row in read(bundle_path)['repositories'] if row['repo'] == 'STE-FlowNet']
    require(len(rows) == 1 and rows[0]['commit'] == MODEL_COMMIT, 'Model source pin changed')
    record = rows[0]
    require(record['included_file_count'] == 14 and len(record['files']) == 14, 'Need original fourteen-file tree')
    values = []
    for item in record['files']:
        relative = Path(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source relative path')
        path = model_root / relative
        checked_hash(path, item['sha256'])
        require(path.stat().st_size == item['size_bytes'], 'Model source size changed')
        values.append([item['path'], item['size_bytes'], item['sha256']])
    tree = hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    require(sum(row[1] for row in values) == record['included_size_bytes']
            and tree == record['included_tree_sha256'], 'Model source tree/count/bytes changed')
    return dict(commit=MODEL_COMMIT, files=14, bytes_verified=sum(row[1] for row in values),
                tree_sha256=tree, model_root=str(model_root))


def command(args, method, seed):
    return [sys.executable, '-B', '-u', str(args.sources / 'scripts/stage9_mvsec_bridge_v2.py'),
            'evaluate', '--protocol', str(args.protocol), '--manifest', str(args.manifest),
            '--authorization', str(args.authorization), '--method', method, '--seed', str(seed),
            '--model-root', str(args.model_root), '--device', 'cuda:0',
            '--out', str(args.root / f'seed{seed}' / method)]


def verify_immutable(args):
    checked_hash(args.protocol, args.protocol_sha256)
    checked_hash(args.manifest, args.manifest_sha256)
    checked_hash(args.authorization, args.authorization_sha256)
    checked_hash(args.numerical_preflight, args.numerical_preflight_sha256)
    checked_hash(args.sources / 'source_manifest.json', args.source_manifest_sha256)
    source_manifest = read(args.sources / 'source_manifest.json')
    require(source_manifest['schema'] == 'mvsec_fixed_cpu_preparation_source_snapshot_v1'
            and source_manifest['frozen_core'] == CORE, 'Wrong snapshot/core')
    for row in source_manifest['files']:
        relative = Path(row['relative_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe snapshot path')
        path = args.sources / relative
        checked_hash(path, row['sha256'])
        require(path.stat().st_size == row['size_bytes'], 'Snapshot source size changed')
    checked_hash(args.sources / 'scripts/stage9_mvsec_bridge_v2.py', BRIDGE_SHA)


def preflight(args):
    require(args.root == ROOT and args.preparation_root == PREPARATION,
            'Output/preparation root names must remain fixed')
    require(args.root.parent.resolve(strict=True) == RUNS and not args.root.exists()
            and not args.root.is_symlink(), 'Output already exists or is noncanonical; preserve it')
    require(args.sources == PREPARATION / 'source_snapshot'
            and args.sources.resolve(strict=True) == args.sources, 'Use the completed fixed source snapshot')
    for path in (args.protocol, args.manifest, args.authorization, args.numerical_preflight):
        require(path.is_relative_to(PROJECT), 'Reviewed contract must remain in the project')
    verify_immutable(args)
    ready = read(PREPARATION / 'completion.json')
    require(ready['status'] == 'complete_unscored' and ready['real_scores_computed'] is False,
            'CPU preparation is incomplete or scored')
    ready = read(PREPARATION / 'cpu_preparation/PREPARATION_COMPLETE_NO_SCORING.json')
    require(ready['status'] == 'complete_unscored' and ready['registered_windows'] == 8500
            and ready['indoor_windows'] == 7700 and ready['outdoor_windows'] == 800
            and ready['checkpoint_count'] == 15 and ready['real_scores_computed'] is False
            and ready['source_snapshot_manifest_sha256'] == args.source_manifest_sha256,
            'Completed preparation count/source contract changed')
    protocol, manifest, auth = read(args.protocol), read(args.manifest), read(args.authorization)
    frozen_contract(protocol, manifest, auth, args.protocol_sha256, args.manifest_sha256)
    report = read(args.numerical_preflight)
    numerical_contract(report, manifest, auth,
                       args.numerical_preflight_sha256, ready['manifest_sha256'])
    verified_model = model_tree(args.model_root, args.sources / 'manifests/downstream_source_bundle.json')
    numerical_provenance(args, report, manifest, verified_model)
    for name, frozen in [('protocol', protocol), ('manifest', manifest)]:
        candidate = Path(ready[name])
        checked_hash(candidate, ready[name + '_sha256'])
        preserves_candidate(read(candidate), frozen)
    registration = Path(manifest['registration'])
    checked_hash(registration, manifest['registration_sha256'])
    require(protocol['registration_sha256'] == manifest['registration_sha256'] == ready['registration_sha256']
            and str(registration) == ready['registration'], 'Window registration lineage differs')
    windows = read(registration)['windows']
    require(len(windows) == 8500 and collections.Counter(row['sequence'] for row in windows) == COUNTS
            and len({(r['sequence'], r['aps_start_index']) for r in windows}) == 8500,
            'Window population changed')
    del windows
    checked_hash(Path(manifest['recipe']), manifest['recipe_sha256'])
    require(read(Path(manifest['recipe']))['frozen_simulator_core'] == CORE, 'Recipe core differs')
    require(protocol['source_scripts'] == manifest['source_scripts'], 'Protocol/source paths differ')
    for role, relative in [('adapter', 'scripts/run_mvsec_candidate_inference.py'),
                           ('metrics', 'scripts/stage4_mvsec_dt1_metrics.py'),
                           ('model_compat', 'compat/steflow_align_corners.py')]:
        item = manifest['source_scripts'][role]
        require(Path(item['path']) == args.sources / relative, 'Unexpected bound source location')
        checked_hash(Path(item['path']), item['sha256'])
    checkpoints = []
    for _, seed, method in PHASES:
        binding = manifest['checkpoint_binding'][f'{method}/seed{seed}']
        require(binding['seed'] == seed and binding['method'] == method
                and binding['completed_updates'] == 8910
                and binding['recipe_sha256'] == manifest['recipe_sha256'], 'Weight identity/budget changed')
        expected = Path(manifest['campaign']) / 'training/flow' / f'seed{seed}' / method
        require(Path(binding['run']) == expected and Path(binding['selected_checkpoint']) == expected / 'synthetic_best.pt',
                'Refuse substituted last/preflight/other checkpoint')
        for key, hash_key in [('selected_checkpoint', 'selected_checkpoint_sha256'),
                              ('completion', 'completion_sha256'), ('data_manifest', 'data_manifest_sha256')]:
            checked_hash(Path(binding[key]), binding[hash_key])
        require(Path(binding['selected_checkpoint']).stat().st_size == binding['selected_checkpoint_size_bytes'],
                'Selected weight size changed')
        checkpoints.append(dict(method=method, seed=seed, path=binding['selected_checkpoint'],
                                sha256=binding['selected_checkpoint_sha256']))
    return dict(status='preflight_passed_no_scoring', source_files=read(args.sources / 'source_manifest.json')['files'],
                contracts=[file_record(p) for p in (args.protocol, args.manifest, args.authorization, args.numerical_preflight)],
                preparation=ready, model=verified_model,
                checkpoints=checkpoints, registered_windows=8500, GPU_used=False, scores_computed=False)


def stop_own_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        require(os.getpgid(child.pid) == child.pid, 'Own child process-group identity changed')
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=30)
    except ProcessLookupError:
        pass


def validate_result(args, method, seed):
    root = args.root / f'seed{seed}' / method
    result = read(root / 'result.json')
    require(result['status'] == 'authorized_real_score_full_scope' and result['method'] == method
            and result['seed'] == seed and result['window_count'] == 8500
            and result['complete_required_scope'] is True
            and result['evaluation_sequences'] == list(SEQUENCES)
            and result['missing_required_sequences'] == []
            and result['four_sequence_aggregate_published'] is False
            and result['no_real_score_feedback'] is True, 'Incomplete child result')
    for name, digest in [('protocol', args.protocol_sha256), ('manifest', args.manifest_sha256),
                          ('authorization', args.authorization_sha256)]:
        require(result[name + '_sha256'] == digest, 'Child contract SHA differs')
    binding = read(args.manifest)['checkpoint_binding'][f'{method}/seed{seed}']
    require(result['checkpoint_sha256'] == binding['selected_checkpoint_sha256'], 'Child weight SHA differs')
    samples, counts = set(), collections.Counter()
    with (root / 'window_metrics.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row['method'] == method and row['seed'] == seed
                    and row['checkpoint_sha256'] == result['checkpoint_sha256'], 'Mixed result identities')
            require(row['sample_id'] not in samples, 'Duplicate scored window')
            samples.add(row['sample_id']); counts[row['sequence']] += 1
    require(len(samples) == 8500 and counts == COUNTS, 'Missing child window rows')
    return [file_record(root / name) for name in ('result.json', 'window_metrics.jsonl', 'sequence_metrics.csv')]


def execute(args, evidence):
    import fcntl
    require(socket.gethostname() == 'g1', 'Only the assigned server g1 is allowed')
    deadline = args.entered_monotonic + TOTAL_SECONDS
    require(time.monotonic() < deadline, 'Overall18-hour deadline exhausted before GPU acquisition')
    args.root.mkdir(exist_ok=False)
    state = child = gpu_lock = None
    completed, artifacts = [], []
    registry = PROJECT / '.gpu_reservations'
    record = registry / (ROOT.name + '_gpu5.json')
    def interrupted(signum, _frame):
        raise SystemExit('Own manager received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        require(registry.is_dir() and registry.resolve(strict=True) == registry, 'Reservation registry missing')
        gpu = subprocess.check_output(['nvidia-smi', '-i', '5', '--query-gpu=uuid,name', '--format=csv,noheader'],
                                      text=True, timeout=15).strip().split(',', 1)
        require(gpu[0].strip() == UUID and gpu[1].strip() == 'NVIDIA TITAN RTX', 'GPU5 UUID/model mismatch')
        gpu_lock = (registry / ('.gpu-' + UUID + '.lock')).open('a+')
        with (registry / '.gpu_reservations.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                                           text=True, timeout=15)
            require(UUID not in apps, 'GPU5 has another active compute process; preserve and exit')
            for path in registry.glob('*.json'):
                existing = read(path)
                if existing.get('gpu_uuid') == UUID:
                    require(str(existing.get('status', '')).startswith(('released', 'complete')),
                            'GPU5 has an existing live reservation')
            require(not record.exists(), 'Preserve existing own lease record')
            state = dict(owner='sim-to-real research chat', host='g1', gpu_index=5, gpu_uuid=UUID,
                         gpu_model='NVIDIA TITAN RTX', run_id=ROOT.name, output_path=str(ROOT),
                         pid=os.getpid(), process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                         status='reserved', current_job=None, completed=[], started_utc=now(),
                         purpose='authorized frozen MVSEC fifteen-cell final evaluation; no training/tuning')
            atomic(record, state)
        (args.root / 'logs').mkdir()
        write_new(args.root / 'preflight.json', evidence)
        for seed in SEEDS:
            (args.root / f'seed{seed}').mkdir()
        parameters = dict(protocol=file_record(args.protocol), manifest=file_record(args.manifest),
                          authorization=file_record(args.authorization), source_manifest=file_record(args.sources / 'source_manifest.json'),
                          numerical_preflight=file_record(args.numerical_preflight),
                          manager=file_record(Path(__file__).resolve()), model_root=str(args.model_root),
                          gpu_uuid=UUID, overall_timeout_seconds=TOTAL_SECONDS, per_child_timeout_seconds=CHILD_SECONDS,
                          phases=[dict(phase=label, seed=seed, method=method, argv=command(args, method, seed))
                                  for label, seed, method in PHASES], automatic_retries=0,
                          changes_to_weights_training_or_protocol=False)
        write_new(args.root / 'source_parameter_protocol_manifest.json', parameters)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', NVIDIA_TF32_OVERRIDE='0', PYTHONDONTWRITEBYTECODE='1')
        for label, seed, method in PHASES:
            require(time.monotonic() < deadline, 'Overall18-hour deadline exhausted')
            verify_immutable(args)
            phase_deadline = min(deadline, time.monotonic() + CHILD_SECONDS)
            receipt = dict(phase=label, seed=seed, method=method, started_utc=now(), status='running',
                           argv=command(args, method, seed), timeout_seconds=min(CHILD_SECONDS, deadline - time.monotonic()))
            with (args.root / 'logs' / (label + '.log')).open('x') as log:
                child = subprocess.Popen(receipt['argv'], env=env, cwd=str(args.sources / 'scripts'),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                receipt['pid'] = child.pid
                while True:
                    state.update(status='running', current_job=receipt, heartbeat_utc=now(), completed=completed[:])
                    atomic(record, state); atomic(args.root / 'live.json', state)
                    code = child.poll()
                    if code is not None:
                        break
                    if time.monotonic() >= phase_deadline:
                        stop_own_child(child)
                        receipt.update(status='timed_out', exit_code=child.poll(), finished_utc=now())
                        write_new(args.root / (label + '_exit.json'), receipt)
                        raise TimeoutError('Own child/overall timeout: ' + label)
                    time.sleep(min(5., max(.01, phase_deadline - time.monotonic())))
                receipt.update(status='exited', exit_code=code, finished_utc=now())
            write_new(args.root / (label + '_exit.json'), receipt)
            require(code == 0, 'Own child failed; no retry/tuning: ' + label)
            child = None
            verify_immutable(args)
            artifacts.extend(validate_result(args, method, seed))
            completed.append(label)
        require(completed == [p[0] for p in PHASES], 'Matrix phases incomplete')
        write_new(args.root / 'artifact_index.json', artifacts)
        write_new(args.root / 'completion.json', dict(status='complete_frozen_real_matrix', completed=completed,
                  cells=15, windows_per_cell=8500, finished_utc=now(), protocol_sha256=args.protocol_sha256,
                  manifest_sha256=args.manifest_sha256, authorization_sha256=args.authorization_sha256,
                  numerical_preflight_sha256=args.numerical_preflight_sha256,
                  no_real_feedback_to_training=True, paper_aggregation_generated=False))
    except BaseException:
        write_new(args.root / 'failure.json', dict(status='failed', completed=completed,
                  error=traceback.format_exc(), utc=now(), automatic_retries=0))
        raise
    finally:
        cleanup_error = None
        try:
            stop_own_child(child)
        except BaseException:
            cleanup_error = traceback.format_exc()
        child_stopped = child is None or child.poll() is not None
        if state is not None:
            # Never claim a GPU is released while our child might still run.
            state.update(status='released' if child_stopped else 'cleanup_failed',
                         current_job=None if child_stopped else state.get('current_job'),
                         completed=completed[:], released_utc=now() if child_stopped else None,
                         cleanup_error=cleanup_error)
            atomic(record, state); write_new(args.root / 'resource_release.json', state)
        if gpu_lock is not None:
            fcntl.flock(gpu_lock, fcntl.LOCK_UN); gpu_lock.close()
        if cleanup_error is not None:
            raise RuntimeError('Own process cleanup failed; preserve receipt and investigate: ' + cleanup_error)


def selftest():
    """No GPU, SSH, subprocess, datasets or scores; reject closed/misbound gates."""
    require(len(PHASES) == 15 and len({p[0] for p in PHASES}) == 15, 'Phase matrix duplicates')
    require(PHASES[0] == ('seed0_eventgenesis', 0, 'eventgenesis')
            and PHASES[-1] == ('seed2_rife_esim', 2, 'rife_esim'), 'Wrong fixed order')
    p = dict(schema='stage9_mvsec_dt1_protocol_v2', status='frozen_by_research_review', real_scores_authorized=True,
             required_sequences=list(SEQUENCES), evaluation_sequences=list(SEQUENCES), missing_required_sequences=[],
             result_scope='required_four_sequence_final', methods=list(METHODS), seeds=list(SEEDS), dt=1,
             training_budget_updates=8910, model_commit=MODEL_COMMIT, no_real_data_model_selection=True)
    m = dict(schema='stage9_mvsec_matrix_manifest_v2', status='frozen_by_research_review', real_scores_authorized=True,
             required_sequences=list(SEQUENCES), evaluation_sequences=list(SEQUENCES), missing_required_sequences=[],
             result_scope='required_four_sequence_final', complete_checkpoint_matrix=True, missing_training_runs=[],
             total_registered_windows=8500, registered_window_counts=COUNTS,
             checkpoint_binding={f'{method}/seed{seed}': {} for _, seed, method in PHASES})
    a = dict(authorized_real_test=True, authorization_id='OFFLINE_FIXTURE', approved_by='OFFLINE_FIXTURE',
             evaluation_scope='required_four_sequence_final', protocol_sha256='a' * 64, manifest_sha256='b' * 64)
    frozen_contract(p, m, a, 'a' * 64, 'b' * 64)
    checked = ['fifteen_fixed_order_cells', 'authorized_complete_fixture_only']
    for label, target, key, value in [('closed_protocol', p, 'real_scores_authorized', False),
                                     ('wrong_auth_hash', a, 'manifest_sha256', '0' * 64),
                                     ('missing_outdoor', m, 'missing_required_sequences', ['outdoor_day1']),
                                     ('incomplete_weights', m, 'complete_checkpoint_matrix', False)]:
        before = target[key]; target[key] = value
        try:
            frozen_contract(p, m, a, 'a' * 64, 'b' * 64)
        except ValueError:
            checked.append(label + '_rejected')
        else:
            raise AssertionError('Unsafe offline fixture accepted: ' + label)
        finally:
            target[key] = before
    try:
        preserves_candidate({'primary_mask': 'fixed'}, {'primary_mask': 'changed'})
    except ValueError:
        checked.append('changed_candidate_science_rejected')
    else:
        raise AssertionError('Freeze silently changed scientific field')
    numerical_manifest = dict(checkpoint_binding={key: dict(selected_checkpoint_sha256=hashlib.sha256(key.encode()).hexdigest())
                                               for key in m['checkpoint_binding']})
    numerical = dict(schema='mvsec_synthetic_forward_equivalence_v1', status='passed', manifest_sha256='d' * 64,
                     GPU_used=True, gpu_uuid=UUID, real_data_read=False, real_scores_computed=False,
                     training_started=False, scoring_authorization_created=False,
                     thresholds=dict(max_abs_px=.001, rms_px=.0001), checkpoints={
        key: dict(checkpoint_sha256=binding['selected_checkpoint_sha256'], passed=True,
                  cases=[dict(max_abs_px=0., rms_px=0., passed=True) for _ in range(2)])
        for key, binding in numerical_manifest['checkpoint_binding'].items()})
    numerical_auth = dict(numerical_preflight_sha256='c' * 64)
    numerical_contract(numerical, numerical_manifest, numerical_auth, 'c' * 64, 'd' * 64)
    for label, bad in [('missing_numerical_preflight', None),
                       ('failed_numerical_preflight', {**numerical, 'status': 'failed'}),
                       ('cpu_self_test_as_gpu_probe', {**numerical, 'GPU_used': False, 'cpu_self_test_only': True}),
                       ('wrong_prepared_candidate', {**numerical, 'manifest_sha256': '0' * 64})]:
        try:
            numerical_contract(bad, numerical_manifest, numerical_auth, 'c' * 64, 'd' * 64)
        except ValueError:
            checked.append(label + '_rejected')
        else:
            raise AssertionError('Invalid numerical evidence accepted: ' + label)
    first = next(iter(numerical['checkpoints']))
    original = numerical['checkpoints'][first]['checkpoint_sha256']
    numerical['checkpoints'][first]['checkpoint_sha256'] = '0' * 64
    try:
        numerical_contract(numerical, numerical_manifest, numerical_auth, 'c' * 64, 'd' * 64)
    except ValueError:
        checked.append('wrong_numerical_checkpoint_rejected')
    else:
        raise AssertionError('Wrong numerical checkpoint binding accepted')
    finally:
        numerical['checkpoints'][first]['checkpoint_sha256'] = original
    print(json.dumps(dict(status='passed_offline_static_checks', checks=checked,
                         GPU_used=False, SSH_used=False, real_data_read=False, scores_computed=False)))


def main():
    entered = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--preparation-root', type=Path, default=PREPARATION)
    parser.add_argument('--sources', type=Path, default=PREPARATION / 'source_snapshot')
    parser.add_argument('--source-manifest-sha256')
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--numerical-preflight', type=Path)
    parser.add_argument('--numerical-preflight-sha256')
    for name in ('protocol', 'manifest', 'authorization'):
        parser.add_argument('--' + name, type=Path)
        parser.add_argument('--' + name + '-sha256')
    args = parser.parse_args()
    args.entered_monotonic = entered
    if args.self_test:
        require(not args.execute, 'Offline self-test cannot execute scoring')
        selftest(); return
    required = ('source_manifest_sha256', 'model_root', 'protocol', 'protocol_sha256',
                'manifest', 'manifest_sha256', 'authorization', 'authorization_sha256',
                'numerical_preflight', 'numerical_preflight_sha256')
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error('Existing hash-bound reviewed inputs required: ' + ', '.join(missing))
    evidence = preflight(args)
    if not args.execute:
        print(json.dumps(evidence, ensure_ascii=False, allow_nan=False)); return
    execute(args, evidence)


if __name__ == '__main__':
    main()
