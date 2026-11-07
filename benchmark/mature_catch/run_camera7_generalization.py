"""Evaluate a preregistered case shard; never change perception or physics code."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

child = None

def stop(signum, frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
    raise SystemExit(128+signum)

def sha(path, canonical=False):
    data=Path(path).read_bytes()
    if canonical:data=data.replace(b"\r\n",b"\n")
    return hashlib.sha256(data).hexdigest()

def main():
    global child
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--shard',type=int,required=True);args=parser.parse_args()
    spec=json.loads(args.protocol.read_text());root=Path(spec['root']);root.mkdir(parents=True,exist_ok=True)
    verification={}
    for name,expected in spec['runtime_files'].items():
        f=Path(spec['source'])/name;actual=sha(f);assert actual==expected,(str(f),actual,expected);verification[str(f)]=actual
    for name,expected in spec['core_files'].items():
        f=Path(spec['core_root'])/'genesis_event_plugin'/name
        if not f.exists():f=Path(spec['core_root'])/name
        actual=sha(f,True);assert actual==expected,(str(f),actual,expected);verification[str(f)]=actual
    for f,expected in [(spec['checkpoint'],spec['checkpoint_sha256']),(spec['profile_file'],spec['profile_sha256'])]:
        actual=sha(f);assert actual==expected,(f,actual,expected);verification[f]=actual
    audit=dict(host=socket.gethostname(),shard=args.shard,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(args.protocol),verified=verification)
    (root/f'shard{args.shard}_preflight.json').write_text(json.dumps(audit,indent=2))
    results=[];template=spec['template']['command']
    for case in spec['cases']:
        if case['shard']!=args.shard:continue
        for mode in ['event','state_reference']:
            out=root/'cases'/case['case_id']/mode
            if out.exists():raise RuntimeError('Refusing to overwrite existing case '+str(out))
            cmd=list(template)
            for flag,value in [('--output',str(out)),('--seeds',str(case['seed'])),('--horizontal-speed',str(case['horizontal_speed_m_s'])),('--lighting',case['lighting'])]:
                cmd[cmd.index(flag)+1]=value
            if mode=='state_reference':
                cmd=[x for x in cmd if x not in ['--event-control','--rgb-initialize','--rgbd-initialize','--stop-events-after-grasp']]
            log=root/f"{case['case_id']}_{mode}.log"
            record=dict(case=case,mode=mode,command=cmd,output=str(out),host=socket.gethostname(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            (root/f'shard{args.shard}_running.json').write_text(json.dumps(record,indent=2))
            start=time.monotonic()
            with log.open('w') as stream:
                child=subprocess.Popen(cmd,cwd=spec['source'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                record['pid']=child.pid;print(json.dumps(dict(status='started',**record)),flush=True)
                code=child.wait()
            child=None
            record.update(exit_code=code,wall_s=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if (out/'episodes.json').exists():record['episodes']=json.loads((out/'episodes.json').read_text())
            results.append(record);(root/f'shard{args.shard}_progress.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(dict(status='finished',**record)),flush=True)
            if code!=0:raise RuntimeError('Infrastructure/runtime failure; retained and stopped shard: '+str(out))
    for path,expected in verification.items():
        actual=sha(path,'genesis_event_plugin' in path);assert actual==expected,('postflight changed',path)
    (root/f'shard{args.shard}_complete.json').write_text(json.dumps(dict(episodes=len(results),all_exit_zero=True,postflight_hashes_pass=True),indent=2))

if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main()
