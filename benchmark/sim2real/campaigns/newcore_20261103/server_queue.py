"""Single GPU job queue with a persistent, low-load CUDA standby context."""
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

campaign = Path(sys.argv[1]).resolve()
index = int(sys.argv[2])
uuid = sys.argv[3]
lock_fd = int(sys.argv[4])
fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
lane = campaign / ('gpu' + str(index))
lane.mkdir(exist_ok=True)
queue = lane / 'queue'
queue.mkdir(exist_ok=True)
registry = Path('/home/zhaoyuhan/Eventbased_WAM/.gpu_reservations') / (campaign.name + '_gpu' + str(index) + '.json')
os.environ['CUDA_VISIBLE_DEVICES'] = uuid
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, data):
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)


state = json.loads(registry.read_text())
state.update(pid=os.getpid(), process_start_ticks=Path('/proc/self/stat').read_text().split()[21], heartbeat_utc=now())
save(registry, state)
child = None
stop = False


def stop_signal(signum, frame):
    global stop
    stop = True
    # Only the exact child started here is signalled, never a process pattern.
    if child is not None and child.poll() is None:
        child.terminate()


signal.signal(signal.SIGTERM, stop_signal)
signal.signal(signal.SIGINT, stop_signal)

try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.cuda.device_count() == 1
    standby = torch.empty(1024 * 1024, dtype=torch.uint8, device='cuda:0')
    torch.cuda.synchronize()
    state.update(status='standby', cuda_device_name=torch.cuda.get_device_name(0), standby_bytes=1048576)
    save(registry, state)
    while not stop:
        pending = sorted(queue.glob('*.json'))
        selected = None
        for path in pending:
            spec = json.loads(path.read_text())
            if not all(Path(p).is_file() for p in spec.get('ready_files', [])):
                continue
            dependencies = [lane / (job + '.done.json') for job in spec.get('after', [])]
            if not all(p.is_file() for p in dependencies):
                continue
            if any(json.loads(p.read_text())['exit_code'] != 0 for p in dependencies):
                blocked = lane / (spec['job_id'] + '.blocked.json')
                save(blocked, {'job_id': spec['job_id'], 'status': 'blocked_by_failed_dependency', 'utc': now()})
                path.rename(lane / (spec['job_id'] + '.blocked_spec.json'))
                continue
            for file, expected in spec.get('source_sha256', {}).items():
                assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == expected, file
            selected = path, spec
            break
        if selected is None:
            state.update(status='standby', current_job=None, heartbeat_utc=now())
            save(registry, state)
            if index == 5 and (campaign / 'campaign_finalized').exists():
                break
            time.sleep(10)
            continue
        path, spec = selected
        job_id = spec['job_id']
        assert isinstance(spec['argv'], list) and all(isinstance(a, str) for a in spec['argv'])
        output = Path(spec['output_dir']).resolve()
        assert output.is_relative_to(campaign)
        output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(spec.get('env', {}))
        assert env['CUDA_VISIBLE_DEVICES'] == uuid
        current = {'job_id': job_id, 'phase': spec['phase'], 'started_utc': now(),
                   'argv': spec['argv'], 'output_dir': str(output), 'status': 'running'}
        path.rename(lane / (job_id + '.running_spec.json'))
        with (output / 'stdout.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(spec['argv'], cwd=spec.get('cwd', str(campaign)),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, close_fds=True)
            current['pid'] = child.pid
            current['process_start_ticks'] = Path('/proc', str(child.pid), 'stat').read_text().split()[21]
            save(lane / (job_id + '.live.json'), current)
            state.update(status=spec['phase'], current_job=current, heartbeat_utc=now())
            save(registry, state)
            print(json.dumps(current), flush=True)
            while child.poll() is None:
                state['heartbeat_utc'] = now()
                save(registry, state)
                time.sleep(15)
            current.update(status='complete' if child.returncode == 0 else 'failed',
                           exit_code=child.returncode, finished_utc=now())
        save(lane / (job_id + '.done.json'), current)
        state['last_completed_job'] = current
        print(json.dumps(current), flush=True)
        child = None
    state.update(status='released', released_utc=now())
    save(registry, state)
except BaseException:
    state.update(status='failed', error=traceback.format_exc(), heartbeat_utc=now())
    save(registry, state)
    traceback.print_exc()
    raise
