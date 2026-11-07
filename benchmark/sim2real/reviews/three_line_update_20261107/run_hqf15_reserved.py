"""One authorized HQF15 evaluation using the existing project GPU lease contract."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--lease-helpers',type=Path,required=True)
    p.add_argument('--gpu-index',type=int,required=True)
    p.add_argument('--gpu-uuid',required=True)
    args=p.parse_args()
    root=args.root.resolve();project=root.parents[2]
    assert str(project)=='/home/zhaoyuhan/Eventbased_WAM' and socket.gethostname()=='g1'
    manifest=json.loads((args.lease_helpers.parent/'source_manifest.json').read_text())
    assert hashlib.sha256(args.lease_helpers.read_bytes()).hexdigest()==manifest[args.lease_helpers.name]
    spec=importlib.util.spec_from_file_location('existing_pose_lease_helpers',args.lease_helpers)
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    protocol=root/'protocol/hqf15_protocol.json';authorization=root/'authorization.json'
    metadata=json.loads(protocol.read_text());auth=json.loads(authorization.read_text())
    assert auth['authorized'] and auth['protocol_sha256']==hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert json.loads((root/'runtime_preflight/completion.json').read_text())['status']=='passed'
    bridge=Path(metadata['bridge_source']['path'])
    assert hashlib.sha256(bridge.read_bytes()).hexdigest()==metadata['bridge_source']['sha256']
    observed=subprocess.check_output(['nvidia-smi','-i',str(args.gpu_index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert observed==args.gpu_uuid
    registry=project/'.gpu_reservations'
    lease=(registry/('.gpu-'+args.gpu_uuid+'.lock')).open('a+')
    global_lock=(registry/'.gpu_reservations.lock').open('a+')
    record=registry/(root.name+'_gpu'+str(args.gpu_index)+'.json')
    fcntl.flock(global_lock,fcntl.LOCK_EX)
    try:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        assert args.gpu_uuid not in apps,'GPU has another compute process'
        for path in registry.glob('*.json'):
            existing=json.loads(path.read_text())
            if existing.get('gpu_uuid')==args.gpu_uuid:
                assert str(existing.get('status','')).startswith(('released','complete')),'GPU has another reservation'
        assert not record.exists()
        state=dict(status='reserved',owner='sim-to-real research chat',run_id=root.name,host='g1',
            gpu_index=args.gpu_index,gpu_uuid=args.gpu_uuid,pid=os.getpid(),
            process_start_ticks=Path('/proc/self/stat').read_text().split()[21],started_utc=helpers.now(),
            real_data_used_in_training=False,real_scoring_enabled=True,output=str(root))
        helpers.save(record,state)
    finally:fcntl.flock(global_lock,fcntl.LOCK_UN)
    child=None;code=None
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        assert args.gpu_uuid not in apps
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',NVIDIA_TF32_OVERRIDE='0')
        command=[sys.executable,'-u',str(bridge),'run','--protocol',str(protocol),
            '--authorization',str(authorization),'--out',str(root/'results'),'--device','cuda:0','--enable-real-scoring']
        with (root/'evaluation.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running',child_pid=child.pid,argv=command)
            deadline=time.monotonic()+12*3600
            while child.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Owned HQF evaluation exceeded bounded runtime')
                state['heartbeat_utc']=helpers.now();helpers.save(record,state);helpers.save(root/'live.json',state)
                time.sleep(5)
            code=child.returncode
        assert code==0,'Evaluation failed; preserve output and logs'
        result=json.loads((root/'results/run_manifest.json').read_text())
        assert result['status']=='complete' and result['real_scores_computed']
        helpers.save(root/'completion.json',dict(status='complete',exit_code=code,
            real_data_used_in_training=False,real_scores_computed=True,finished_utc=helpers.now()))
    finally:
        if child is not None and child.poll() is None:helpers.stop_child(child)
        if child is not None:code=child.returncode
        state.update(status='released_complete' if code==0 else 'released_failed',exit_code=code,finished_utc=helpers.now())
        helpers.save(record,state);helpers.save(root/'exit.json',state)
        fcntl.flock(lease,fcntl.LOCK_UN)


if __name__=='__main__':main()
