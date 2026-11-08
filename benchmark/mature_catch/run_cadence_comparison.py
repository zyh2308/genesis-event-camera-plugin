"""Evaluate a preregistered shard; every task failure is retained and continued."""
import argparse,datetime,hashlib,json,os,signal,socket,subprocess,time
from pathlib import Path
from frontend_config import resolve_frontend

child=None

def stop(signum,frame):
    global child
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise SystemExit(128+signum)

def sha(p,canonical=False):
    b=Path(p).read_bytes()
    if canonical:b=b.replace(b'\r\n',b'\n')
    return hashlib.sha256(b).hexdigest()

def verify(spec):
    for n,digest in spec['runtime_files'].items():assert sha(Path(spec['source'])/n)==digest,n
    for n,digest in spec['core_files'].items():assert sha(Path(spec['core_root'])/'genesis_event_plugin'/n,True)==digest,n
    assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
    assert sha(spec['v2e_checkpoint'])==spec['v2e_checkpoint_sha256']
    assert sha(spec['profile_file'])==spec['profile_sha256']

def main():
    global child
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    spec=json.loads(a.protocol.read_text());root=Path(spec['root']);verify(spec)
    (root/f'shard{a.shard}_preflight.json').write_text(json.dumps({'host':socket.gethostname(),'protocol_sha256':sha(a.protocol),'frozen_inputs_pass':True,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
    completed=[]
    for case in spec['cases']:
        if case['shard']!=a.shard:continue
        for method in case['method_order']:
            output=root/'cases'/case['case_id']/method
            if output.exists():raise RuntimeError('Refuse to overwrite '+str(output))
            command=list(spec['template']);command+=['--output',str(output),'--seeds',str(case['seed']),'--launch-config',str(root/'launch_configs'/str(case['case_id']+'.json')),'--lighting',case['lighting']]
            env=dict(os.environ,CATCHIT_EVENT_GENERATOR='v2e' if method=='v2e' else 'ours',CATCHIT_EVENT_ENDPOINT_HZ=str(case['endpoint_hz']),CATCHIT_V2E_PARITY_RECEIPT=spec['v2e_parity_receipts'][str(case['endpoint_hz'])])
            frontend=resolve_frontend(spec,case,method)
            env.update(frontend)
            if method=='state_reference':command+=['--sensor-probe']
            else:
                command+=['--event-control','--rgbd-initialize','--stop-events-after-grasp']
                if method=='init_only':command+=['--initialization-only']
            record={'case_id':case['case_id'],'condition':case['condition'],'seed':case['seed'],'method':method,'endpoint_hz':case['endpoint_hz'],'frontend_overrides':frontend,'host':socket.gethostname(),'command':command,'output':str(output),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            (root/f'shard{a.shard}_running.json').write_text(json.dumps(record,indent=2));start=time.monotonic()
            with (root/f"{case['case_id']}_{method}.log").open('w') as log:
                child=subprocess.Popen(command,cwd=spec['source'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record['pid']=child.pid;print(json.dumps({'status':'started',**record}),flush=True)
                try:code=child.wait(timeout=spec.get('episode_timeout_s',1200))
                except subprocess.TimeoutExpired:stop(signal.SIGTERM,None)
            child=None;record.update(exit_code=code,wall_s=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if (output/'episodes.json').exists():record['episodes']=json.loads((output/'episodes.json').read_text())
            completed.append(record);(root/f'shard{a.shard}_progress.json').write_text(json.dumps(completed,indent=2));print(json.dumps({'status':'finished',**record}),flush=True)
            if code!=0:raise RuntimeError('Infrastructure failure retained; stop before invalid comparison: '+str(output))
    verify(spec);(root/f'shard{a.shard}_complete.json').write_text(json.dumps({'completed':len(completed),'all_exit_zero':True,'postflight_hashes_pass':True}))

if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main()
