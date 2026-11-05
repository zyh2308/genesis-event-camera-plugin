"""Bounded g1 GPU2 queue; follows the project's global + UUID flock protocol."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/home/zhaoyuhan/Eventbased_WAM')
UUID = 'GPU-a02affe0-7598-579e-ca39-f551f35ad6e9'
RUN_ID = 'simulator_efficiency_continuation_20261105h'
REG = PROJECT / '.gpu_reservations'
RECORD = REG / (RUN_ID + '_gpu2.json')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command(argv):
    return subprocess.check_output(argv, text=True)


def preflight():
    result = {'utc': now()}
    for key, argv in {
        'gpu': ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'],
        'apps': ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader'],
        'cpu': ['uptime'], 'memory': ['free', '-b'],
        'disk': ['df', '-B1', str(ROOT)], 'mount': ['findmnt', '-T', str(ROOT)],
    }.items():
        result[key] = command(argv)
    assert UUID not in result['apps'], 'GPU2 has a compute owner; do not start'
    matches = [line for line in result['gpu'].splitlines() if UUID in line]
    assert len(matches) == 1
    fields = [field.strip() for field in matches[0].split(',')]
    assert fields[0] == '2' and 'V100S' in fields[2]
    assert int(fields[3].split()[0]) < 100 and int(fields[5].split()[0]) == 0
    return result


def save(state):
    tmp = RECORD.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(RECORD)


def main():
    if sys.flags.optimize:
        raise RuntimeError('Optimized Python would disable required project assertions')
    def interrupted(signum, _frame):
        raise SystemExit('lease runner signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    assert command(['hostname']).strip() == 'g1'
    assert command(['id', '-un']).strip() == 'zhaoyuhan'
    assert ROOT == PROJECT / 'runs' / RUN_ID
    global_lock = (REG / '.gpu_reservations.lock').open('a+')
    gpu_lock = (REG / ('.gpu-' + UUID + '.lock')).open('a+')
    state = None
    acquired = False
    try:
        fcntl.flock(global_lock, fcntl.LOCK_EX)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        before = preflight()
        checked = []
        for path in sorted(REG.glob('*.json')):
            data = json.loads(path.read_text())
            if UUID in json.dumps(data):
                checked.append({'path': str(path), 'status': data.get('status')})
                assert str(data.get('status', '')).startswith(('released', 'complete')), str(path)
        state = {'status': 'reserved_efficiency_continuation', 'gpu_uuid': UUID,
                 'gpu_index': 2, 'host': 'g1', 'run_id': RUN_ID,
                 'pid': os.getpid(), 'process_start_ticks': Path('/proc/self/stat').read_text().split()[21],
                 'started_utc': now(), 'output_path': str(ROOT), 'cpu_threads': 2,
                 'owner': 'user-authorized plugin efficiency branch; no training or production data writes'}
        save(state)
        (ROOT / 'resource_preflight.json').write_text(json.dumps({'before_reservation': before, 'checked_reservations': checked, 'after_reservation': preflight(), 'lease': state}, indent=2) + '\n')
    finally:
        fcntl.flock(global_lock, fcntl.LOCK_UN)
    if not acquired:
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=UUID, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', MAX_JOBS='2',
                      TORCH_CUDA_ARCH_LIST='7.0',
                      EVENTGENESIS_GPU_BUILD_DIR=str(ROOT / 'private_build'))
    receipt = PROJECT / 'runs/sim2real/20261103-stage9-direct-v1/runtime/esim_build_tools_install.json'
    install = json.loads(receipt.read_text())
    assert install['exit_code'] == 0 and install['shared_env_modified'] is False
    tools = Path(install['prefix'])
    env = {'CUDA_HOME': str(tools), 'CC': str(tools / 'bin/x86_64-conda-linux-gnu-cc'),
           'CXX': str(tools / 'bin/x86_64-conda-linux-gnu-c++')}
    assert all(Path(value).exists() for value in env.values())
    os.environ.update(env)
    os.environ['PATH'] = str(tools / 'bin') + os.pathsep + os.environ['PATH']
    (ROOT / 'build_environment.json').write_text(json.dumps({'install_receipt': str(receipt), 'install_receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(), 'environment': {key: os.environ[key] for key in (*env, 'EVENTGENESIS_GPU_BUILD_DIR', 'TORCH_CUDA_ARCH_LIST', 'MAX_JOBS')}, 'compiler_math_unchanged': True}, indent=2) + '\n')
    queue = ROOT / 'queue'
    queue.mkdir(exist_ok=True)
    deadline = time.monotonic() + 7200
    try:
        while time.monotonic() < deadline and not (ROOT / 'FINISH').exists():
            state['heartbeat_utc'] = now()
            save(state)
            pending = sorted(queue.glob('*.json'))
            if not pending:
                time.sleep(3)
                continue
            path = pending[0]
            job = json.loads(path.read_text())
            path.rename(path.with_suffix('.running'))
            assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest() == h for f, h in job.get('sha256', {}).items())
            preflight()
            state.update(status='running_efficiency_continuation', job=job['id'])
            save(state)
            started = time.monotonic()
            with (ROOT / (job['id'] + '.log')).open('x') as log:
                child = subprocess.Popen(job['argv'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    state.update(child_pid=child.pid, child_started_utc=now())
                    save(state)
                    try:
                        code = child.wait(timeout=job.get('timeout_s', 1800))
                    except subprocess.TimeoutExpired:
                        code = 124
                finally:
                    # Every exit path reaps our directly created process group
                    # before the lease can be marked released, including save errors.
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                    assert child.poll() is not None
            done = {'exit_code': code, 'wall_s': time.monotonic() - started, 'job': job, 'finished_utc': now()}
            (ROOT / (job['id'] + '.done.json')).write_text(json.dumps(done, indent=2) + '\n')
            state.update(status='reserved_efficiency_continuation', last_job=job['id'], last_exit_code=code)
            state.pop('child_pid', None)
            save(state)
    except BaseException:
        state['error'] = traceback.format_exc()
        raise
    finally:
        # A child is reaped before control reaches here. Do not release based on SSH state.
        state.update(status='released_with_error' if 'error' in state else 'released_complete', finished_utc=now())
        save(state)
        (ROOT / 'resource_release.json').write_text(json.dumps(state, indent=2) + '\n')
        fcntl.flock(gpu_lock, fcntl.LOCK_UN)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        # Initialization can fail after registration but before entering the queue.
        if RECORD.exists():
            failed = json.loads(RECORD.read_text())
            if failed.get('pid') == os.getpid() and not str(failed.get('status', '')).startswith('released'):
                failed.update(status='released_initialization_failed', finished_utc=now(), error=traceback.format_exc())
                save(failed)
                (ROOT / 'resource_release.json').write_text(json.dumps(failed, indent=2) + '\n')
        raise
