"""One bounded, sequential data-only retry for the observed slow SSH handshake."""
import importlib.util,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path('/mnt/c/Users/zhaoyuhan/Documents/ChatGPT/New project/outputs/mainline_audit_20261105/flow')
OLD=ROOT/'transfer_run_20261107_v2'
NEW=ROOT/'transfer_run_20261107_v3_long_handshake'
BASE=ROOT/'resume_mvsec_transfer_g1.py'
OUT=Path(__file__).with_name('indoor_sequential_handoff')

def save(name,data):
    p=OUT/name;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)

def ticks(pid):
    p=Path('/proc',str(pid),'stat')
    return p.read_text().split()[21] if p.exists() else None

def main():
    OUT.mkdir(exist_ok=False)
    initial=json.loads((OLD/'launcher.json').read_text())
    parent=int(initial['pid']);start=ticks(parent)
    assert start is not None
    save('launch.json',dict(pid=os.getpid(),previous_pid=parent,previous_start_ticks=start,
        policy='wait; retry once only for recorded SSH timeout; do not interrupt active transfer',started_unix=time.time()))
    deadline=time.monotonic()+16*3600
    while ticks(parent)==start:
        if time.monotonic()>deadline:raise TimeoutError('Existing transfer still active; no replacement launched')
        time.sleep(30)
    final=json.loads((OLD/'exit.json').read_text())
    if final['success']:
        save('completion.json',dict(status='previous_transfer_completed',replacement_started=False));return
    failure=json.loads((OLD/'failure.json').read_text())
    errors='\n'.join(p.read_text(errors='replace')[-3000:] for p in OLD.glob('*.stderr.log'))
    assert 'failed with exit 255' in failure['error'] and any(s in errors.lower() for s in ('timed out','timeout')), 'Not a confirmed SSH timeout: stop for inspection'
    for line in (OLD/'processes.jsonl').read_text().splitlines():
        d=json.loads(line)
        if 'exit_code' not in d:
            assert not Path('/proc',str(d['pid'])).exists(),'An original child still exists; no second writer'
    time.sleep(max(0,60-(time.time()-failure['time_unix'])))
    spec=importlib.util.spec_from_file_location('transfer_long_wait',BASE)
    transfer=importlib.util.module_from_spec(spec);spec.loader.exec_module(transfer)
    assert 'ConnectTimeout=20' in transfer.SSH_OPTIONS
    transfer.SSH_OPTIONS=[s.replace('ConnectTimeout=20','ConnectTimeout=360') for s in transfer.SSH_OPTIONS]
    guard="""import fcntl,json
from pathlib import Path
p=Path('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-public-v2/indoor_flying')
assert p.is_dir() and str(p.resolve())==str(p)
lock=(p/'.stage9_mvsec_transfer.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
print(json.dumps(dict(root=str(p),writer_lock_free=True)))
"""
    check=subprocess.run([transfer.SSH,*transfer.SSH_OPTIONS,transfer.HOST,transfer.ssh_program(guard)],
        stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=420)
    (OUT/'writer_check.stdout.log').write_text(check.stdout)
    (OUT/'writer_check.stderr.log').write_text(check.stderr)
    assert check.returncode==0 and json.loads(check.stdout)['writer_lock_free']
    save('replacement.json',dict(started_unix=time.time(),connect_timeout_seconds=360,previous_failure=failure,
        does_not_modify_existing_launcher=True))
    sys.argv=[str(BASE),'--registration',str(ROOT/'stage9_mvsec_indoor_unscored_registration.json'),
        '--execute','--writer-check-confirmed','--bwlimit-kib','16384','--out',str(NEW)]
    transfer.main()
    save('completion.json',dict(status='replacement_completed',replacement_started=True,finished_unix=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():save('failure.json',dict(type=type(exc).__name__,error=str(exc),time_unix=time.time()))
        raise
