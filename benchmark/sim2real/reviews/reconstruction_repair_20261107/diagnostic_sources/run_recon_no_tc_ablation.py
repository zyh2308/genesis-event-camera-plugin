"""Run the bounded synthetic reconstruction diagnosis under the project's atomic GPU reservation."""
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


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def stop_child(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--gpu-index',type=int,default=0)
    parser.add_argument('--gpu-uuid',default='GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23')
    parser.add_argument('--training-only',action='store_true')
    parser.add_argument('--anchored',action='store_true')
    parser.add_argument('--pose-weight',type=float,default=0)
    parser.add_argument('--seeds',nargs='+',type=int,choices=(0,1,2),default=[0,1,2])
    parser.add_argument('--interpolate-from',type=Path)
    parser.add_argument('--interpolation-refinement-cache',type=Path)
    parser.add_argument('--physical-validation',type=Path)
    args = parser.parse_args()
    root, campaign = args.root.resolve(), args.campaign.resolve()
    assert socket.gethostname() == 'g1'
    assert root.parent == campaign.parent and root != campaign
    if args.interpolate_from:
        deadline=time.monotonic()+2*3600
        while not (args.interpolate_from/'resource_release.json').exists():
            assert not (args.interpolate_from/'failure.json').exists()
            if time.monotonic()>deadline:raise TimeoutError('Source interpolation study did not finish')
            time.sleep(5)
        assert json.loads((args.interpolate_from/'completion.json').read_text())['status']=='complete'
    upstream=campaign.parent/'20261107-recon-range-short-v1'
    deadline=time.monotonic()+2*3600
    save(root/'waiting.json',dict(upstream=str(upstream),GPU_reserved=False,utc=now()))
    while not (upstream/'resource_release.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Own fixed-batch study did not finish')
        time.sleep(5)
    assert json.loads((upstream/'completion.json').read_text())['status']=='complete'
    uuid = args.gpu_uuid
    assert uuid.startswith('GPU-') and 0<=args.gpu_index<6
    observed=subprocess.check_output(['nvidia-smi','-i',str(args.gpu_index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert observed==uuid
    project = campaign.parents[2]
    src = root/'sources'
    manifest = json.loads((src/'source_manifest.json').read_text())
    for file, digest in manifest.items():
        assert hashlib.sha256((src/file).read_bytes()).hexdigest() == digest
    logs = root/'logs'
    logs.mkdir(exist_ok=False)
    registry = project/'.gpu_reservations'
    record = registry/(root.name+'_gpu'+str(args.gpu_index)+'.json')
    lock = (registry/('.gpu-'+uuid+'.lock')).open('a+')
    global_lock = (registry/'.gpu_reservations.lock').open('a+')
    fcntl.flock(global_lock, fcntl.LOCK_EX)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        assert uuid not in apps, 'Reserved GPU has active compute processes'
        for path in registry.glob('*.json'):
            existing = json.loads(path.read_text())
            if existing.get('gpu_uuid') == uuid:
                assert str(existing.get('status','')).startswith(('released','complete')), str(path)
        assert not record.exists()
        state = dict(owner='sim-to-real research chat', host='g1', gpu_index=args.gpu_index, gpu_uuid=uuid,
                     run_id=root.name, pid=os.getpid(), process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                     status='reserved', output_path=str(root), started_utc=now(), purpose='bounded synthetic reconstruction root-cause probes and paired fixed-batch short training')
        save(record, state)
    finally:
        fcntl.flock(global_lock, fcntl.LOCK_UN)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', NVIDIA_TF32_OVERRIDE='0', PYTHONDONTWRITEBYTECODE='1')
    specs=[('fixed_no_tc','recon_no_tc_ablation.py',['--campaign',str(campaign),'--out',str(root/'fixed_no_tc')])]
    completed = []
    child = None
    try:
        for label, filename, options in specs:
            for file, digest in manifest.items():
                assert hashlib.sha256((src/file).read_bytes()).hexdigest() == digest
            command = [sys.executable,'-u',str(src/filename),*options]
            receipt = dict(phase=label, argv=command, started_utc=now(), status='running')
            with (logs/(label+'.log')).open('x') as log:
                child = subprocess.Popen(command, env=env, cwd=str(src), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                receipt['pid'] = child.pid
                state.update(status='running', current_job=receipt, heartbeat_utc=now())
                save(record,state)
                save(root/'live.json',state)
                deadline=time.monotonic()+3*3600
                while child.poll() is None:
                    if time.monotonic() > deadline:
                        stop_child(child)
                        raise TimeoutError('Own study phase exceeded bounded runtime')
                    state['heartbeat_utc']=now()
                    save(record,state)
                    save(root/'live.json',state)
                    time.sleep(5)
                receipt.update(exit_code=child.returncode, finished_utc=now())
            save(root/(label+'_exit.json'),receipt)
            assert child.returncode == 0, (label,child.returncode)
            completion=json.loads((root/label/'completion.json').read_text())
            assert completion['status']=='complete' and not completion.get('real_data_used',False)
            if label=='smoke':assert len(completion['results'])==2
            completed.append(label)
            child=None
        save(root/'completion.json',dict(status='complete', completed=completed,real_data_used=False,finished_utc=now()))
    except BaseException:
        save(root/'failure.json',dict(status='failed',completed=completed,error=traceback.format_exc(),utc=now()))
        raise
    finally:
        if child is not None and child.poll() is None:
            stop_child(child)
        state.update(status='released',current_job=None,released_utc=now(),completed=completed)
        save(record,state)
        save(root/'resource_release.json',state)
        fcntl.flock(lock,fcntl.LOCK_UN)


if __name__=='__main__':main()
