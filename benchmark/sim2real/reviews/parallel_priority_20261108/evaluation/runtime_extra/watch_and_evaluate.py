"""CPU dependency watcher, then one leased g3 HQF selected3 evaluation.

No GPU query, lease or CUDA context before all15 freeze and CPU preparation.
No training, source edits, checkpoint selection, automatic retry, or automation.
--start explicitly enables the future dependency pipeline under the existing
selected3 user instruction. The evaluator itself remains an unmodified child.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

PROJECT = Path('/home/zhaoyuhan/Eventbased_WAM')
RUNS = PROJECT / 'runs/sim2real'
TRAINING = RUNS / '20261108-recon-selected-full-v1'
METHODS = ['eventgenesis', 'v2e', 'vid2e', 'film_esim', 'rife_esim']
SEEDS = [0, 1, 2]
SEQUENCES = ['boxes', 'reflective_materials', 'still_life']
SCOPE = 'posthoc_selected_hqf3_development'
EVALUATOR_SHA = '9df71e420eb6ce3536201c68ea2f5bb69f8c59f30cc2aea12c91aa92fe5a50c4'
EVAL_MANIFEST_SHA = 'd92e20ba05622e632bd53a69388b41b665af7478484a0ac1e862716cbd0a670d'
WAIT_SECONDS = 36 * 3600
EVALUATION_SECONDS = 8 * 3600


def need(ok, message):
    if not ok: raise ValueError(message)


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): digest.update(block)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def checked(path, digest):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path,
         'Expected canonical immutable input: ' + str(path))
    need(sha(path) == digest, 'Immutable source/input changed: ' + str(path))


def record(path):
    path = Path(path)
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha(path))


def source_check(args):
    checked(Path(__file__).resolve(), args.watcher_sha256)
    manifest_path = args.sources / 'source_manifest.json'
    checked(manifest_path, EVAL_MANIFEST_SHA)
    manifest = read(manifest_path)
    need(manifest['schema'] == 'recon_selected_evaluation_sources_v1' and manifest['source_locked'] is True,
         'Evaluation sources are not locked')
    for name, row in manifest['files'].items():
        need(Path(name).name == name, 'Unsafe evaluation source filename')
        path = args.sources / name; checked(path, row['sha256'])
        need(path.stat().st_size == row['size_bytes'], 'Evaluation source size differs')
    checked(args.sources / 'evaluate_selected_hqf.py', EVALUATOR_SHA)
    training_manifest = TRAINING / 'sources/source_manifest.json'
    checked(training_manifest, args.training_source_manifest_sha256)
    for name, digest in read(training_manifest).items():
        need(Path(name).name == name, 'Unsafe training source filename')
        checked(TRAINING / 'sources' / name, digest)


def plan_contract(plan):
    need(plan['schema'] == 'recon_selected_full_plan_v1' and plan['root'] == str(TRAINING), 'Wrong training plan')
    need(plan['methods'] == METHODS and plan['seeds'] == SEEDS and plan['real_sequences'] == SEQUENCES,
         'Changed method/seed/selected-sequence scope')
    need(plan['training_budget_updates'] == 12000 and plan['epochs'] == 200
         and plan['final_activation'] == 'sigmoid' and plan['loss'] == dict(lpips_vgg=1., temporal=2., auxiliary_l1=0.),
         'Unapproved training/selection recipe')
    need(plan['real_result_scope'] == SCOPE and plan['real_scoring_enabled'] is False
         and plan['no_real_feedback_to_training_or_selection'] is True
         and plan['original_hqf14_results_preserved'] is True and bool(plan['source_user_instruction']),
         'Missing standing instruction or isolation')
    lanes = plan['lanes']
    need(len(lanes) == 3 and {r['gpu_index'] for r in lanes} == {2, 5, 6}
         and {r['seed'] for r in lanes} == {0, 1, 2} and len({r['gpu_uuid'] for r in lanes}) == 3,
         'Use exactly the existing three planned cards')
    need(all(r['host'] == 'g3' and r['gpu_model'] == 'NVIDIA GeForce RTX 4090'
             and r['gpu_uuid'].startswith('GPU-') for r in lanes), 'Wrong planned host/device')
    return sorted(lanes, key=lambda r: r['gpu_index'])


def frozen_contract(matrix, completion):
    need(matrix['schema'] == 'recon_selected_frozen_checkpoints_v1'
         and matrix['status'] == 'frozen_all15_before_real_scoring'
         and matrix['methods'] == METHODS and matrix['seeds'] == SEEDS
         and matrix['real_sequences'] == SEQUENCES and matrix['result_scope'] == SCOPE, 'Wrong frozen matrix')
    rows = matrix['runs']
    need(len(rows) == 15 and {(r['method'], r['seed']) for r in rows} == {(m, s) for m in METHODS for s in SEEDS},
         'Full fifteen-weight freeze is required')
    need(all(r['completed_updates'] == 12000 for r in rows), 'Training is incomplete')
    need(completion['status'] == 'full15_training_complete_selected_eval_pending'
         and completion['training_runs'] == 15 and completion['evaluated_sequence_runs'] == 0
         and completion['real_scoring_enabled'] is False and completion['real_result_scope'] == SCOPE
         and completion['full14_historical_results_unchanged'] is True, 'Full training completion required')


def derive_authorization(template, protocol_sha, plan, note):
    plan_contract(plan)
    need(template['schema'] == 'recon_selected_hqf3_development_authorization_v1'
         and template['authorized'] is False and template['protocol_sha256'] == protocol_sha
         and template['task'] == 'reconstruction' and template['result_scope'] == SCOPE
         and template['no_feedback_to_training_or_selection'] is True
         and template['all15_frozen_before_any_real_prediction'] is True and bool(note.strip()),
         'Wrong disabled authorization template/verification note')
    return dict(template, authorized=True, source_user_instruction=plan['source_user_instruction'],
                research_verification_note=note,
                authorization_derivation='Existing user instruction plus completed programmatic research guards; no new human signature',
                generated_utc=now())


def group_members(group):
    result = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit(): continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z': result.append(int(directory.name))
        except (OSError, ValueError, IndexError): pass
    return result


def stop_own_group(child):
    if child is None: return
    group = child.pid
    if group_members(group):
        try: os.killpg(group, signal.SIGTERM)
        except ProcessLookupError: pass
        until = time.monotonic() + 45
        while group_members(group) and time.monotonic() < until: time.sleep(.25)
        if group_members(group):
            try: os.killpg(group, signal.SIGKILL)
            except ProcessLookupError: pass
            until = time.monotonic() + 15
            while group_members(group) and time.monotonic() < until: time.sleep(.25)
    try: child.wait(timeout=1)
    except subprocess.TimeoutExpired: pass
    need(not group_members(group), 'Own evaluator process group did not stop')


def run_phase(args, label, argv, deadline, env, progress):
    child = None; directory = args.root / 'controller'
    receipt = dict(phase=label, argv=argv, started_utc=now(), status='running')
    try:
        with (directory / (label + '.log')).open('x') as log:
            child = subprocess.Popen(argv, cwd=str(args.sources), env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt['pid'] = child.pid
            while child.poll() is None:
                need(time.monotonic() < deadline, 'Own bounded phase exceeded deadline: ' + label)
                progress(label, receipt)
                time.sleep(min(5., max(.01, deadline - time.monotonic())))
        receipt.update(status='exited', exit_code=child.returncode, finished_utc=now())
        write_new(directory / (label + '_exit.json'), receipt)
        need(child.returncode == 0 and not group_members(child.pid), 'Own child failed or left descendants: ' + label)
    finally:
        try:
            stop_own_group(child)
        except BaseException:
            write_new(directory / (label + '_cleanup_failure.json'),
                      dict(phase=label, group=child.pid if child is not None else None,
                           error=traceback.format_exc(), utc=now()))
            raise
        finally:
            exit_path = directory / (label + '_exit.json')
            if not exit_path.exists():
                receipt.update(status='failed_or_stopped', exit_code=child.poll() if child is not None else None,
                               finished_utc=now())
                write_new(exit_path, receipt)


def training_failure():
    paths = [TRAINING / 'GLOBAL_FAILURE.json']
    paths += [TRAINING / 'lanes' / f'seed{s}' / name for s in SEEDS
              for name in ('parent_failure.json', 'lane_failure.json')]
    need(not any(p.exists() for p in paths), 'Training failure preserved; do not prepare or score')
    gate = TRAINING / 'ALL15_SYNTHETIC_GATE.json'
    if gate.exists(): need(read(gate)['passed'] is True, 'All15 synthetic gate failed')


def acquire_one(args, lanes, deadline, progress):
    """No CUDA imports; only acquire an idle planned UUID after preparation."""
    import fcntl
    registry = PROJECT / '.gpu_reservations'
    need(registry.is_dir() and registry.resolve() == registry, 'Existing reservation registry required')
    while time.monotonic() < deadline:
        training_failure()
        progress('waiting_for_free_planned_gpu', None)
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free', '--format=csv,noheader,nounits'],
                                            text=True, timeout=15)
        observed = {int(r[0].strip()): r for r in csv.reader(inventory.splitlines())}
        with (registry / '.gpu_reservations.lock').open('a+') as global_lock:
            try: fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(.25); continue
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                                           text=True, timeout=15)
            old = [read(p) for p in registry.glob('*.json')]
            for lane in lanes:
                idx, uuid = lane['gpu_index'], lane['gpu_uuid']; row = observed.get(idx)
                if row is None: continue
                need(row[1].strip() == uuid and row[2].strip() == lane['gpu_model'],
                     'Planned GPU identity/model changed')
                if int(row[3].strip()) < 21000 or uuid in apps: continue
                if any(r.get('gpu_uuid') == uuid and not str(r.get('status', '')).startswith(('released', 'complete')) for r in old): continue
                lock = (registry / ('.gpu-' + uuid + '.lock')).open('a+')
                try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close(); continue
                path = registry / (args.root.name + '_g3_gpu' + str(idx) + '.json')
                if path.exists():
                    fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
                    raise FileExistsError('Preserve previous own reservation: ' + str(path))
                state = dict(owner='sim-to-real research chat', host='g3', gpu_index=idx, gpu_uuid=uuid,
                             gpu_model=lane['gpu_model'], run_id=args.root.name, output_path=str(args.root),
                             pid=os.getpid(), process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                             status='reserved', current_job=None, started_utc=now(),
                             purpose='all15-frozen selected HQF3 development evaluation; no training or real feedback')
                atomic(path, state)
                return lock, path, state
        time.sleep(min(15., max(.01, deadline - time.monotonic())))
    raise TimeoutError('No safe planned GPU became available within evaluation bound')


def verify_final(root, protocol_sha, authorization_sha):
    run = read(root / 'run_manifest.json'); summary = read(root / 'selected3_summary.json')
    need(run['status'] == 'complete' and run['real_scores_computed'] is True
         and run['protocol_sha256'] == protocol_sha and run['scored_frame_rows'] > 0, 'Evaluation did not complete')
    need(run['authorization']['sha256'] == authorization_sha, 'Evaluation authorization binding changed')
    need(summary['schema'] == 'recon_selected_hqf3_summary_v1' and summary['result_scope'] == SCOPE
         and summary['methods'] == METHODS and summary['seeds'] == SEEDS and summary['sequences'] == SEQUENCES
         and summary['real_scores_computed'] is True and summary['blind_final_test'] is False
         and summary['original_hqf14_results_overwritten'] is False, 'Wrong complete development result scope')
    need(len(summary['method_table']) == 5 and len(summary['per_seed']) == 15 and len(summary['per_sequence']) == 45,
         'Partial table cannot be published')
    for item in [run['summary'], run['authorization'], summary['protocol'], summary['raw_score'], summary['frozen_inferences']]:
        checked(Path(item['path']), item['sha256']); need(Path(item['path']).stat().st_size == item['size_bytes'], 'Result size changed')
    return [record(root / name) for name in ('run_manifest.json', 'selected3_summary.json', 'method_table.csv',
                                           'per_seed.csv', 'per_sequence.csv', 'RESULTS.md')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--training-source-manifest-sha256', required=True)
    parser.add_argument('--watcher-sha256', required=True)
    parser.add_argument('--research-verification-note', required=True)
    parser.add_argument('--start', action='store_true', required=True)
    args = parser.parse_args(); args.sources = args.sources or args.root / 'sources'
    need(socket.gethostname() == 'g3', 'Run only on the authorized g3 host')
    need(args.root.parent == RUNS and args.root.is_dir() and args.root.resolve() == args.root
         and args.root != TRAINING and args.sources.resolve() == args.sources, 'Use a new canonical evaluation root')
    need(not (args.root / 'protocol').exists() and not (args.root / 'evaluation').exists(), 'Preserve existing outputs')
    source_check(args)
    plan = read(TRAINING / 'sources/registered_plan.json'); lanes = plan_contract(plan)
    control = args.root / 'controller'; control.mkdir(exist_ok=False)
    write_new(control / 'launch.json', dict(pid=os.getpid(), started_utc=now(), training_root=str(TRAINING),
              watcher=record(Path(__file__).resolve()), evaluator=record(args.sources / 'evaluate_selected_hqf.py'),
              source_manifest=record(args.sources / 'source_manifest.json'),
              wait_seconds=WAIT_SECONDS, evaluation_seconds=EVALUATION_SECONDS, GPU_reserved=False,
              source_user_instruction=plan['source_user_instruction'], result_scope=SCOPE))
    lease = lease_path = state = None
    def interrupted(signum, _frame): raise SystemExit('Own watcher interrupted ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    def progress(phase, current):
        status = dict(phase=phase, current_job=current, heartbeat_utc=now(), pid=os.getpid(),
                      GPU_reserved=state is not None, real_scores_completed=False)
        atomic(control / 'live.json', status)
        if state is not None:
            state.update(status='running', current_job=current, heartbeat_utc=status['heartbeat_utc'])
            atomic(lease_path, state)
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', NVIDIA_TF32_OVERRIDE='0',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
    evaluator = args.sources / 'evaluate_selected_hqf.py'
    try:
        until = time.monotonic() + WAIT_SECONDS
        while not (TRAINING / 'frozen_checkpoint_matrix.json').exists() or not (TRAINING / 'completion.json').exists():
            training_failure(); need(time.monotonic() < until, 'All15 freeze did not finish within36h')
            progress('waiting_for_all15_training_freeze_no_gpu', None); time.sleep(min(15., max(.01, until-time.monotonic())))
        training_failure(); source_check(args)
        frozen_contract(read(TRAINING / 'frozen_checkpoint_matrix.json'), read(TRAINING / 'completion.json'))
        deadline = time.monotonic() + EVALUATION_SECONDS
        write_new(control / 'training_freeze.json', dict(matrix=record(TRAINING / 'frozen_checkpoint_matrix.json'),
                  completion=record(TRAINING / 'completion.json'), GPU_reserved=False))
        run_phase(args, 'prepare', [sys.executable, '-B', '-u', str(evaluator), 'prepare', '--training-root', str(TRAINING),
                  '--out', str(args.root / 'protocol')], min(deadline, time.monotonic()+3600), cpu_env, progress)
        protocol = args.root / 'protocol/selected3_protocol.json'
        run_phase(args, 'verify', [sys.executable, '-B', '-u', str(evaluator), 'verify', '--protocol', str(protocol)],
                  min(deadline, time.monotonic()+3600), cpu_env, progress)
        code = ('import json,sys; from pathlib import Path; import evaluate_selected_hqf as e; '
                'p=e.verify_protocol(Path(sys.argv[1])); r=e.load_helper().preflight_metric_assets(p); '
                'f=Path(sys.argv[2]).open("x"); json.dump(r,f,indent=2); f.close()')
        assets = control / 'cpu_metric_assets.json'
        run_phase(args, 'metric_assets', [sys.executable, '-B', '-c', code, str(protocol), str(assets)],
                  min(deadline, time.monotonic()+3600), cpu_env, progress)
        need(read(assets)['status'] == 'cpu_metric_initialization_passed_no_real_images', 'Metric CPU admission failed')
        source_check(args)
        auth = derive_authorization(read(args.root / 'protocol/authorization_template_DO_NOT_RUN.json'),
                                    sha(protocol), plan, args.research_verification_note)
        authorization = args.root / 'authorization.json'; write_new(authorization, auth)
        snapshot = dict(protocol=record(protocol), authorization=record(authorization), metric_assets=record(assets))
        write_new(control / 'prepared_authorized_before_gpu.json', dict(**snapshot, GPU_reserved=False))
        memory = {line.split(':', 1)[0]: int(line.split()[1]) * 1024
                  for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        resources = dict(cpu_count=os.cpu_count(), load_average=list(os.getloadavg()),
                         available_memory_bytes=memory['MemAvailable'], free_disk_bytes=shutil.disk_usage(args.root).free)
        write_new(control / 'cpu_disk_resources_before_gpu.json', resources)
        need(resources['free_disk_bytes'] >= 20 * 1024**3 and resources['available_memory_bytes'] >= 8 * 1024**3,
             'Insufficient memory/disk for complete bounded evaluation')
        lease, lease_path, state = acquire_one(args, lanes, deadline, progress)
        gpu_env = dict(cpu_env, CUDA_VISIBLE_DEVICES=state['gpu_uuid'])
        source_check(args)
        checked(protocol, snapshot['protocol']['sha256']); checked(authorization, snapshot['authorization']['sha256'])
        run_phase(args, 'evaluate_all45', [sys.executable, '-B', '-u', str(evaluator), 'run', '--protocol', str(protocol),
                  '--authorization', str(authorization), '--out', str(args.root / 'evaluation'),
                  '--device', 'cuda:0', '--enable-real-scoring'], deadline, gpu_env, progress)
        source_check(args)
        artifacts = verify_final(args.root / 'evaluation', snapshot['protocol']['sha256'], snapshot['authorization']['sha256'])
        write_new(control / 'artifact_index.json', artifacts)
        write_new(control / 'completion.json', dict(status='complete_selected3_development', finished_utc=now(),
                  methods=5, seeds=3, sequence_runs=45, result_scope=SCOPE, selected_gpu_uuid=state['gpu_uuid'],
                  protocol_sha256=snapshot['protocol']['sha256'], authorization_sha256=snapshot['authorization']['sha256'],
                  no_feedback_to_training_or_selection=True, original_hqf14_unchanged=True))
    except BaseException:
        write_new(control / 'failure.json', dict(status='failed', utc=now(), error=traceback.format_exc(), automatic_retries=0))
        raise
    finally:
        if state is not None:
            # run_phase finally has stopped every own child group before returning.
            cleanup = [read(p) for p in control.glob('*_cleanup_failure.json')]
            live_groups = [r['group'] for r in cleanup if r.get('group') is not None and group_members(r['group'])]
            state.update(status='cleanup_failed' if live_groups else 'released',
                         current_job=state.get('current_job') if live_groups else None,
                         released_utc=None if live_groups else now(), cleanup_failures=cleanup)
            atomic(lease_path, state); write_new(control / 'resource_release.json', state)
        if lease is not None:
            import fcntl
            fcntl.flock(lease, fcntl.LOCK_UN); lease.close()


if __name__ == '__main__': main()
