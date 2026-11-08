#!/usr/bin/env python3
"""One sealed read-only synthetic-flow command on g1 GPU0/1; CPU-only lease parent."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

P = Path('/home/zhaoyuhan/Eventbased_WAM')
RUNS = P / 'runs/sim2real'
GPUS = {0: 'GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23',
        1: 'GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460'}
MAX_SECONDS = 1200


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        need(key not in out, 'Duplicate JSON key: ' + key)
        out[key] = value
    return out


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def save(path, value):
    path = Path(path)
    partial = path.with_name(path.name + '.' + str(os.getpid()) + '.partial')
    with partial.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
    os.replace(partial, path)


def within(path, root, exists=False):
    p = Path(path)
    need(p.is_absolute(), 'Use absolute paths')
    resolved = p.resolve(strict=exists)
    need(resolved == p and resolved.is_relative_to(root), 'Path escapes root or uses a symlink: ' + str(p))
    return resolved


def check(rec, root=None):
    p = Path(rec['path'])
    if root is not None:
        p = within(p, root, exists=True)
    need(p.is_file() and p.stat().st_size == rec['size_bytes'] and sha(p) == rec['sha256'],
         'Bound file changed: ' + str(p))
    return p


def sealed_input(stream, expected_sha):
    raw = stream.read(1024**2 + 1)
    need(len(raw) <= 1024**2, 'Sealed command exceeds 1MiB')
    need(hashlib.sha256(raw).hexdigest() == expected_sha, 'stdin command digest differs')
    contract = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_object)
    need(isinstance(contract, dict), 'Sealed command must be one JSON object')
    return contract


def verify_contract(c, root, gpu_index, parent_source):
    need(c['schema'] == 'flow_full_synthetic_cell_command_v1', 'Wrong sealed command schema')
    need(c['task'] == 'flow_full_synthetic_readonly_singlecell' and gpu_index == 1 and c['task_root'] == str(root) and
         c['gpu_index'] == gpu_index and c['gpu_uuid'] == GPUS[gpu_index], 'Task/GPU identity differs')
    need(c['optimizer_updates'] == 0 and c['real_data_used'] is False and
         c['real_scores_computed'] is False and c['model_weights_read_only'] is True,
         'Only synthetic read-only evaluation is permitted')
    need(c['timeout_seconds'] == MAX_SECONDS and c['min_free_mib'] == 22000, 'Changed time/resource bound')
    need(c['parent_source']['path'] == str(parent_source) and check(c['parent_source'], root) == parent_source,
         'Parent source changed')
    manifest_path = check(c['source_manifest'], root)
    need(manifest_path.parent == root / 'sources', 'Sources must be in the task source root')
    manifest = read(manifest_path)
    files = manifest.get('files', manifest)
    need(isinstance(files, dict) and files, 'Empty source manifest')
    verified = set()
    for name, digest in files.items():
        need(isinstance(name, str) and Path(name).name == name, 'Unsafe source manifest filename')
        path = within(root / 'sources' / name, root / 'sources', exists=True)
        expected = digest if isinstance(digest, str) else digest['sha256']
        need(sha(path) == expected, 'Source bytes changed: ' + name)
        if isinstance(digest, dict):
            need(path.stat().st_size == digest['size_bytes'], 'Source size changed: ' + name)
        verified.add(path)
    entry = check(c['entrypoint'], root)
    need(entry in verified and entry.suffix == '.py', 'Entrypoint is not a source-manifest Python file')
    worker_contract = check(c['worker_contract'], root)
    need(worker_contract in verified, 'Scientific worker contract is not source-manifest bound')
    cwd = within(c['cwd'], root, exists=True); out = within(c['out'], root)
    need(cwd == root / 'sources' and out != root and not out.exists(), 'Wrong cwd or existing worker output')
    need(out.parent == root and out.name == 'rife_seed1', 'Output must be its registered shard')
    need(out.name == 'rife_seed1' and gpu_index == 1, 'GPU/shard assignment differs')
    args = c['argv']
    need(isinstance(args, list) and all(isinstance(v, str) and '\x00' not in v for v in args), 'argv must be sealed strings')
    need(args[:4] == [str(Path(sys.executable).resolve()), '-B', '-u', str(entry)], 'Only the same Python and pinned script are allowed')
    expected_seeds = ['1']
    need(args[4:] == ['--contract', str(worker_contract), '--seeds', *expected_seeds, '--methods', 'rife_esim', '--out', str(out)],
         'Only the exact registered read-only shard CLI is permitted')
    ready_path = check(c['ready'], root); ready = read(ready_path)
    need(ready['schema'] == 'flow_full_synthetic_cell_ready_v1' and ready['passed'] is True and
         ready['task_root'] == str(root) and ready['source_manifest_sha256'] == c['source_manifest']['sha256'] and
         ready['worker_contract_sha256'] == c['worker_contract']['sha256'] and
         ready['read_only_model'] is True and ready['optimizer_updates'] == 0 and
         ready['real_scores_computed'] is False, 'Root source/CPU-ready receipt not sealed')
    return dict(ready=ready, out=out, cwd=cwd, argv=args, verified_source_files=len(verified))


def procinfo(pid):
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]),
                    session=int(fields[3]), start_ticks=int(fields[19]), uid=root.stat().st_uid)
    except (OSError, ValueError, IndexError):
        return None


def group_members(pgid):
    result = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            info = procinfo(p.name)
            if info and info['pgid'] == pgid and info['state'] != 'Z':
                result.append(info)
    return result


def child_boundary(child):
    owner = procinfo(os.getpid()); info = procinfo(child.pid)
    need(info and info['ppid'] == owner['pid'] and info['pgid'] == child.pid and info['session'] == child.pid,
         'New worker did not create its own process/session group')
    return dict(owner_pid=owner['pid'], owner_start_ticks=owner['start_ticks'],
                child_pid=child.pid, pgid=child.pid, child_start_ticks=info['start_ticks'], uid=owner['uid'])


def checked_group(boundary):
    owner = procinfo(os.getpid())
    need(owner and boundary['owner_pid'] == os.getpid() and boundary['owner_start_ticks'] == owner['start_ticks'],
         'Cannot signal a process group not created by this parent process')
    rows = group_members(boundary['pgid'])
    need(all(r['uid'] == boundary['uid'] and r['session'] == boundary['pgid'] and
             r['start_ticks'] >= boundary['child_start_ticks'] for r in rows), 'Own process-group boundary changed')
    return rows


def stop_own(child, boundary):
    if child is None:
        return
    need(boundary is not None and child.pid == boundary['child_pid'], 'Missing own-child boundary')
    for signum, seconds in ((signal.SIGTERM, 8), (signal.SIGKILL, 5)):
        if not checked_group(boundary):
            break
        try:
            os.killpg(boundary['pgid'], signum)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + seconds
        while checked_group(boundary) and time.monotonic() < deadline:
            time.sleep(.1)
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    need(not checked_group(boundary), 'Own worker process group did not stop; do not claim released')


def registry_admissible(rows, uuid):
    for name, old in rows:
        if old.get('gpu_uuid') == uuid:
            status = old.get('status', '')
            need(isinstance(status, str) and status.startswith('released'),
                 'GPU has an unreleased/unrecognized registry state: ' + name)
            need(old.get('current_job') is None, 'Released registry record still has an active job: ' + name)


def hardware_admissible(gpu_row, app_rows, uuid):
    fields = [v.strip() for v in gpu_row.strip().split(',')]
    need(len(fields) == 3 and fields[0] == uuid and fields[1] == 'NVIDIA GeForce RTX 3090' and
         int(fields[2]) >= 22000, 'GPU UUID/model/free memory does not match')
    for line in app_rows.splitlines():
        if line.strip():
            parts = [v.strip() for v in line.split(',')]
            need(len(parts) == 2 and parts[0] != uuid, 'Existing compute process; no takeover')
    return fields


def interrupted(signum, _frame):
    raise SystemExit('Own read-only evaluation parent interrupted: ' + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--gpu-index', type=int, choices=(1,), required=True)
    parser.add_argument('--contract-sha256', required=True)
    a = parser.parse_args()
    need(socket.gethostname() == 'g1', 'This runtime is restricted to g1')
    root = within(a.task_root, RUNS, exists=True)
    need(root.parent == RUNS and root.name.startswith('20261108-flow-full-synthetic-readonly-'), 'Unregistered task-root name')
    c = sealed_input(sys.stdin.buffer, a.contract_sha256)
    v = verify_contract(c, root, a.gpu_index, Path(__file__).resolve())
    registry = P / '.gpu_reservations'; need(registry.is_dir(), 'Existing global reservation registry missing')
    lane = root / ('parent_gpu' + str(a.gpu_index)); lane.mkdir(exist_ok=False)
    uuid = GPUS[a.gpu_index]; lease_path = registry / (root.name + '_g1_gpu' + str(a.gpu_index) + '_readonly_eval.json')
    child = boundary = state = lock = None; successful = False
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        lock = (registry / ('.gpu-' + uuid + '.lock')).open('a+')
        with (registry / '.gpu_reservations.lock').open('a+') as global_lock:
            fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            need(not lease_path.exists(), 'Existing task record; never overwrite/reclaim')
            rows = [(str(p), read(p)) for p in registry.glob('*.json')]
            registry_admissible(rows, uuid)
            gpu = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu_index), '--query-gpu=uuid,name,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15)
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
            fields = hardware_admissible(gpu, apps, uuid)
            me = procinfo(os.getpid())
            state = dict(owner='sim-to-real research chat', host='g1', gpu_index=a.gpu_index, gpu_uuid=uuid,
                         gpu_model=fields[1], pid=os.getpid(), process_start_ticks=me['start_ticks'],
                         task_root=str(root), status='starting_actual_worker', current_job=None,
                         started_utc=now(), sealed_contract_sha256=a.contract_sha256,
                         source_manifest_sha256=c['source_manifest']['sha256'],
                         CUDA_context_created_by_parent=False, dummy_process_used=False,
                         optimizer_updates=0, real_data_used=False, hard_timeout_seconds=MAX_SECONDS)
            save(lease_path, state)
        # Revalidate every bound byte after admission, immediately before spawning.
        verify_contract(c, root, a.gpu_index, Path(__file__).resolve())
        save(lane / 'sealed_command_receipt.json', c)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   NVIDIA_TF32_OVERRIDE='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (lane / 'worker.log').open('x') as log:
            child = subprocess.Popen(v['argv'], cwd=v['cwd'], env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            boundary = child_boundary(child)
            state.update(status='running_readonly_synthetic_eval', current_job=dict(argv=v['argv'], cwd=str(v['cwd']),
                         out=str(v['out']), boundary=boundary, started_utc=now()))
            deadline = time.monotonic() + MAX_SECONDS
            while child.poll() is None:
                need(time.monotonic() < deadline, 'Own worker exceeded20 minute bound')
                state['heartbeat_utc'] = now(); save(lease_path, state); save(lane / 'live.json', state)
                time.sleep(2)
        save(lane / 'worker_exit.json', dict(argv=v['argv'], exit_code=child.returncode, finished_utc=now(), boundary=boundary))
        need(child.returncode == 0, 'Read-only worker failed; no automatic rerun')
        need(not checked_group(boundary), 'Worker leader exited but its children remain')
        successful = True
        save(lane / 'parent_completion.json', dict(status='complete', exit_code=0, task='flow_full_synthetic_readonly',
             finished_utc=now(), worker_out=str(v['out']), optimizer_updates=0, real_scores_computed=False))
    except BaseException:
        save(lane / 'parent_failure.json', dict(status='failed', error=traceback.format_exc(), utc=now()))
        raise
    finally:
        cleanup_error = None
        try:
            stop_own(child, boundary)
        except BaseException:
            cleanup_error = traceback.format_exc()
        try:
            stopped = child is None or (boundary is not None and not checked_group(boundary))
        except BaseException:
            stopped = False
            cleanup_error = (cleanup_error or '') + traceback.format_exc()
        released = dict(status='released' if stopped else 'cleanup_failed', lease_acquired=state is not None,
                        stopped_all_own_group_members=stopped, successful=successful, cleanup_error=cleanup_error,
                        gpu_uuid=uuid, released_utc=now() if stopped else None, optimizer_updates=0,
                        CUDA_context_created_by_parent=False, dummy_process_used=False)
        if state is not None:
            state.update(released, current_job=None if stopped else state['current_job'])
            save(lease_path, state)
        save(lane / 'resource_release.json', released)
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
        if cleanup_error:
            raise RuntimeError(cleanup_error)


if __name__ == '__main__':
    main()
