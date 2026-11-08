"""One atomic GPU2 lease; four bounded synthetic-only paired worker phases."""
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


UUID = 'GPU-a02affe0-7598-579e-ca39-f551f35ad6e9'
ROOT_NAME = '20261108-recon-l1-quarter-paired-gpu2-v1'
PHASES = [('smoke_eventgenesis', 'eventgenesis', 'smoke'),
          ('smoke_v2e', 'v2e', 'smoke'),
          ('study_eventgenesis', 'eventgenesis', 'study'),
          ('study_v2e', 'v2e', 'study')]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop_child(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def verify_sources(src, manifest):
    for filename, digest in manifest.items():
        assert Path(filename).name == filename and len(digest) == 64
        assert sha(src/filename) == digest, ('Source bytes changed', filename)


def signal_exit(signum, frame):
    raise SystemExit('Manager received signal '+str(signum))


def main():
    entered = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--campaign', required=True, type=Path)
    args = parser.parse_args()
    root, campaign = args.root.resolve(), args.campaign.resolve()
    assert socket.gethostname() == 'g1'
    assert root.name == ROOT_NAME and root.parent == campaign.parent and root != campaign
    assert root.is_dir() and (root/'sources').is_dir()
    # Reject an already-used directory before any receipt mutation.
    assert not any((root/name).exists() for name in ['logs', 'completion.json', 'failure.json', 'resource_release.json'])
    project = campaign.parents[2]
    src = root/'sources'
    completed, state, child = [], None, None
    gpu_lock = None
    record = project/'.gpu_reservations'/(ROOT_NAME+'_gpu2.json')
    deadline = entered+2*3600
    signal.signal(signal.SIGTERM, signal_exit)
    signal.signal(signal.SIGINT, signal_exit)
    try:
        manifest = json.loads((src/'source_manifest.json').read_text())
        verify_sources(src, manifest)
        registration = json.loads((src/'registered_protocol.json').read_text())
        assert registration['arms'] == ['control', 'photometric'] and registration['arm_count'] == 2
        assert registration['auxiliary_l1_weight'] == .25
        assert registration['total_main_optimizer_updates'] == 1200
        assert registration['overall_timeout_seconds'] == 7200
        assert registration['no_long_queue'] and registration['no_real_scoring']
        assert registration['protocol_sha256'] == sha(src/'PROTOCOL.md')
        observed = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).strip()
        assert observed == UUID, ('GPU index/UUID mismatch', observed)
        registry = project/'.gpu_reservations'
        gpu_lock = (registry/('.gpu-'+UUID+'.lock')).open('a+')
        with (registry/'.gpu_reservations.lock').open('a+') as global_lock:
            fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True, timeout=15)
                assert UUID not in apps, 'GPU2 has active compute processes'
                for path in registry.glob('*.json'):
                    existing = json.loads(path.read_text())
                    if existing.get('gpu_uuid') == UUID:
                        assert str(existing.get('status', '')).startswith(('released', 'complete')), str(path)
                assert not record.exists(), 'Do not overwrite an earlier own lease'
                state = dict(owner='sim-to-real research chat', host='g1', gpu_index=2, gpu_uuid=UUID,
                             run_id=ROOT_NAME, pid=os.getpid(), process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                             status='reserved', output_path=str(root), started_utc=now(),
                             purpose='single-device serial L1=.25 paired short diagnosis; 1200 main+4 smoke updates; no real scoring')
                save(record, state)
            finally:
                fcntl.flock(global_lock, fcntl.LOCK_UN)
        logs = root/'logs'
        logs.mkdir(exist_ok=False)
        for method in ['eventgenesis', 'v2e']:
            (root/method).mkdir(exist_ok=False)
        save(root/'protocol.json', dict(registration, host='g1', gpu_index=2, gpu_uuid=UUID,
                                       source_manifest_sha256=sha(src/'source_manifest.json'),
                                       runtime_root=str(root), started_utc=now(),
                                       worker_sequence=[label for label, _, _ in PHASES]))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='1', NVIDIA_TF32_OVERRIDE='0', PYTHONDONTWRITEBYTECODE='1')
        for label, method, phase in PHASES:
            assert time.monotonic() < deadline, 'Overall two-hour budget exhausted'
            verify_sources(src, manifest)
            if phase == 'study':
                paired = json.loads((root/'cross_generator_smoke_pairing.json').read_text())
                assert paired['passed'] and completed[:2] == ['smoke_eventgenesis', 'smoke_v2e']
            command = [sys.executable, '-u', str(src/'l1_quarter_diagnostic.py'), '--campaign', str(campaign),
                       '--out', str(root/method/phase), '--method', method]
            if phase == 'smoke':
                command.append('--smoke-only')
            receipt = dict(phase=label, method=method, kind=phase, argv=command, started_utc=now(), status='running')
            with (logs/(label+'.log')).open('x') as log:
                child = subprocess.Popen(command, env=env, cwd=str(src), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                receipt['pid'] = child.pid
                state.update(status='running', current_job=receipt, heartbeat_utc=now())
                save(record, state)
                save(root/'live.json', state)
                while child.poll() is None:
                    if time.monotonic() > deadline:
                        stop_child(child)
                        raise TimeoutError('Overall paired task exceeded two-hour bound')
                    state['heartbeat_utc'] = now()
                    save(record, state)
                    save(root/'live.json', state)
                    time.sleep(min(5., max(.05, deadline-time.monotonic())))
                receipt.update(exit_code=child.returncode, status='exited', finished_utc=now())
            save(root/(label+'_exit.json'), receipt)
            assert child.returncode == 0, (label, child.returncode)
            completion = json.loads((root/method/phase/'completion.json').read_text())
            assert completion['status'] == 'complete' and not completion.get('real_data_used', False)
            if phase == 'smoke':
                assert len(completion['results']) == 2
                assert [r['arm'] for r in completion['results']] == ['control', 'photometric']
                assert all(r['update']['optimizer_update'] == 4801 for r in completion['results'])
            else:
                assert completion['pairing_passed'] and completion['method'] == method
                assert len(completion['arms']) == 2
                assert [r['arm'] for r in completion['arms']] == ['control', 'photometric']
                assert all(r['additional_updates'] == 300 and r['absolute_update'] == 5100 for r in completion['arms'])
            completed.append(label)
            child = None
            if label == 'smoke_v2e':
                smoke = {m: json.loads((root/m/'smoke/completion.json').read_text()) for m in ['eventgenesis', 'v2e']}
                labels = {m: [r['update']['labels'] for r in smoke[m]['results']] for m in smoke}
                assert labels['eventgenesis'] == labels['v2e'], 'Cross-generator label/augmentation pairing failed'
                assert labels['eventgenesis'][0] == labels['eventgenesis'][1]
                save(root/'cross_generator_smoke_pairing.json', dict(passed=True, methods=['eventgenesis', 'v2e'],
                     labels=labels, labels_sha256=hashlib.sha256(json.dumps(labels['eventgenesis']).encode()).hexdigest(),
                     completed_smoke_updates=4, real_data_used=False))
        assert completed == [label for label, _, _ in PHASES]
        save(root/'completion.json', dict(status='complete', completed=completed, main_optimizer_updates=1200,
             smoke_optimizer_updates=4, real_data_used=False, real_scoring_enabled=False, finished_utc=now()))
    except BaseException:
        save(root/'failure.json', dict(status='failed', completed=completed, error=traceback.format_exc(), utc=now()))
        raise
    finally:
        if child is not None and child.poll() is None:
            stop_child(child)
        if state is not None:
            state.update(status='released', current_job=None, released_utc=now(), completed=completed)
            save(record, state)
            save(root/'resource_release.json', state)
        if gpu_lock is not None:
            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
            gpu_lock.close()


if __name__ == '__main__':
    main()
