"""Bounded sharing of this study's own GPU reservation, never another task's."""
from pathlib import Path
import argparse, datetime, fcntl, json, os, signal, socket, subprocess, time, traceback
import xml.etree.ElementTree as ET

BASE=Path('/home/zhaoyuhan/Eventbased_WAM')
child=None


def active(r):
    return not (r.get('gpu_reserved') is False or str(r.get('status','')).lower().startswith(
        ('released','failed','cancelled','completed','finished')) or r.get('status')=='complete')


def stop(signum,frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=35)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise SystemExit(128+signum)


def descendants(roots):
    pairs={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            stat=(p/'stat').read_text();tail=stat.rsplit(')',1)[1].split()
            pairs[int(p.name)]=int(tail[1])
        except (OSError,ValueError,IndexError):pass
    owned=set(roots)
    while True:
        added={pid for pid,parent in pairs.items() if parent in owned}
        if added<=owned:return owned
        owned.update(added)


def main():
    global child
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args()
    job=json.loads(a.job.read_text());root=a.job.parent;start=time.monotonic();recordpath=None
    try:
        os.nice(10)
        cpu=job['cpu_affinity'];assert len(cpu)==2 and len(set(cpu))==2
        os.sched_setaffinity(0,cpu)
        assert os.getloadavg()[0]<64,'CPU load is too high for extra workers'
        mem={line.split(':',1)[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        assert mem['MemAvailable']>16*1024*1024,'Need16GiB host headroom'
        records_root=BASE/'.gpu_reservations'
        with (records_root/'.gpu_reservations.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            parent=json.loads(Path(job['parent_reservation']).read_text())
            assert active(parent) and parent['run_id']==job['parent_run_id']
            assert parent['owner']=='robot closed-loop research chat'
            assert Path(parent['output_path']).resolve()==Path(job['study_stage_root']).resolve()
            assert Path('/proc/'+str(parent['pid'])+'/stat').read_text().split()[21]==parent['process_start_ticks']
            index=parent['gpu_index'];uuid=parent['gpu_uuid'];records=[]
            for directory in (records_root,BASE/'.resource_allocations',BASE/'runs/robot_scenes/.resource_allocations'):
                for pth in directory.glob('*.json'):
                    rec=json.loads(pth.read_text())
                    if active(rec) and rec.get('gpu_uuid')==uuid:records.append(rec)
            assert len(records)<job.get('maximum_slots_per_gpu',2),'Owned shared slot limit reached'
            assert all(rec.get('owner')==parent['owner'] and Path(rec['output_path']).resolve()==Path(job['study_stage_root']).resolve() for rec in records)
            owned=descendants([rec['pid'] for rec in records])
            g=ET.fromstring(subprocess.check_output(['nvidia-smi','-i',str(index),'-q','-x'],text=True,timeout=20)).find('gpu')
            assert g.findtext('uuid')==uuid
            actual={int(v.findtext('pid')) for v in g.findall('processes/process_info')}
            assert actual<=owned,('Unowned GPU process appeared',sorted(actual-owned))
            used=int(g.findtext('fb_memory_usage/used').split()[0]);total=int(g.findtext('fb_memory_usage/total').split()[0])
            assert total-used>job.get('extra_memory_budget_mib',7000)
            record=dict(host=socket.gethostname(),owner=parent['owner'],run_id=job['name'],pid=os.getpid(),
                process_start_ticks=Path('/proc/self/stat').read_text().split()[21],gpu_index=index,gpu_uuid=uuid,
                gpu_reserved=True,status='running_shared_owned',output_path=job['study_stage_root'],
                parent_run_id=parent['run_id'],parent_pid=parent['pid'],cpu_affinity=cpu,
                extra_memory_budget_mib=job.get('extra_memory_budget_mib',7000),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            recordpath=records_root/(job['name']+'_gpu'+str(index)+'.json')
            assert not recordpath.exists();recordpath.write_text(json.dumps(record,indent=2))
            (root/(job['name']+'.allocation.json')).write_text(json.dumps(record,indent=2))
        env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
            'CUDA_VISIBLE_DEVICES':'','MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl',
            'EGL_DEVICE_ID':str(index),'MUJOCO_EGL_DEVICE_ID':str(index),**job.get('env',{})}
        with (root/(job['name']+'.log')).open('x') as log:
            child=subprocess.Popen(job['command'],cwd=job['cwd'],env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['child_pid']=child.pid;recordpath.write_text(json.dumps(record,indent=2))
            code=child.wait(timeout=job.get('timeout_s',43200))
        result=dict(status='completed' if code==0 else 'failed',exit_code=code,wall_s=time.monotonic()-start)
    except BaseException:
        result=dict(status='failed',error=traceback.format_exc(),wall_s=time.monotonic()-start)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=35)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        (root/(job['name']+'.result.json')).write_text(json.dumps(result,indent=2))
        if recordpath:
            with (recordpath.parent/'.gpu_reservations.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);rec=json.loads(recordpath.read_text())
                if rec['pid']==os.getpid():
                    rec.update(status='released_after_'+job['name'],gpu_reserved=False,
                        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    recordpath.write_text(json.dumps(rec,indent=2))
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main()
