"""Task-local GPU launcher with atomic project reservation and owned cleanup."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

BASE=Path('/home/zhaoyuhan/Eventbased_WAM')

def active(r):
    return not (r.get('gpu_reserved') is False or str(r.get('status','')).lower().startswith(('released','failed','cancelled','completed','finished')) or r.get('status')=='complete')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
    job=json.loads(args.job.read_text());root=args.job.parent;recordpath=None;child=None;start=time.monotonic()
    try:
        os.nice(10);allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[-2:])
        required=job.get('wait_for_file')
        while required and not Path(required).exists():
            if time.monotonic()-start>1200:raise TimeoutError('Preceding stage did not finish')
            time.sleep(5)
        records_root=BASE/'.gpu_reservations'
        with (records_root/'.gpu_reservations.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=20));records=[]
            for directory in (records_root,BASE/'.resource_allocations',BASE/'runs/robot_scenes/.resource_allocations'):
                for p in directory.glob('*.json'):
                    r=json.loads(p.read_text())
                    if active(r):records.append(r)
            allocation=None
            for index in job.get('gpu_candidates',[1,0]):
                g=xml.findall('gpu')[index];uuid=g.findtext('uuid')
                if g.findall('processes/process_info') or int(g.findtext('fb_memory_usage/used').split()[0])>=100:continue
                if any(r.get('gpu_uuid')==uuid for r in records):continue
                allocation=dict(host=socket.gethostname(),owner='robot closed-loop research chat',run_id=job['name'],pid=os.getpid(),
                    process_start_ticks=Path('/proc/self/stat').read_text().split()[21],gpu_index=index,gpu_uuid=uuid,gpu_reserved=True,
                    status='running',output_path=str(root),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                recordpath=records_root/(job['name']+f'_gpu{index}.json');recordpath.write_text(json.dumps(allocation,indent=2));break
            if allocation is None:raise RuntimeError('No genuinely free unreserved GPU in allowed list')
            fresh=ET.fromstring(subprocess.check_output(['nvidia-smi','-i',str(index),'-q','-x'],text=True)).find('gpu')
            assert not fresh.findall('processes/process_info'),'New process appeared before launch'
        cpu_start=max(0,len(allowed)-2*(index+1));cpu_pair=allowed[cpu_start:cpu_start+2]
        os.sched_setaffinity(0,cpu_pair);allocation['cpu_affinity']=cpu_pair
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                 CUDA_VISIBLE_DEVICES='',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
                 EGL_DEVICE_ID=str(index),MUJOCO_EGL_DEVICE_ID=str(index),**job.get('env',{}))
        (root/(job['name']+'.allocation.json')).write_text(json.dumps(allocation,indent=2))
        with (root/(job['name']+'.log')).open('w') as log:
            child=subprocess.Popen(job['command'],cwd=job['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
            allocation['child_pid']=child.pid;recordpath.write_text(json.dumps(allocation,indent=2))
            code=child.wait(timeout=job.get('timeout_s',900))
        result=dict(status='completed' if code==0 else 'failed',exit_code=code,wall_s=time.monotonic()-start)
    except BaseException:result=dict(status='failed',error=traceback.format_exc(),wall_s=time.monotonic()-start)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        (root/(job['name']+'.result.json')).write_text(json.dumps(result,indent=2))
        if recordpath:
            with (recordpath.parent/'.gpu_reservations.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);record=json.loads(recordpath.read_text())
                if record.get('pid')==os.getpid():
                    record.update(status='released_after_'+job['name'],gpu_reserved=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());recordpath.write_text(json.dumps(record,indent=2))
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
