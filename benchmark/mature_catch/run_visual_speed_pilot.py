from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,socket,subprocess,time

def sha(p,canonical=False):
    x=Path(p).read_bytes()
    if canonical:x=x.replace(b'\r\n',b'\n')
    return hashlib.sha256(x).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();spec=json.loads(a.spec.read_text());root=Path(spec['root']);done=[]
    for name,expected in spec['runtime_files'].items():assert sha(Path(spec['source'])/name)==expected,name
    for name,expected in spec['core_files'].items():assert sha(Path(spec['core_root'])/'genesis_event_plugin'/name,True)==expected,name
    assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
    for case in spec['cases']:
        if case['shard']!=a.shard:continue
        output=root/'pilot'/case['case_id']
        if output.exists():raise RuntimeError('Refusing overwrite '+str(output))
        command=list(spec['template']);command+=['--output',str(output),'--seeds',str(case['seed']),'--launch-config',str(root/'launch_configs'/str(case['case_id']+'.json'))]
        record={'case':case,'command':command,'host':socket.gethostname(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(root/('pilot_shard'+str(a.shard)+'_running.json')).write_text(json.dumps(record,indent=2))
        start=time.monotonic()
        with (root/str(case['case_id']+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=spec['source'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=child.pid
            try:code=child.wait(timeout=600)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        record.update(exit_code=code,wall_s=time.monotonic()-start,output=str(output))
        if (output/'episodes.json').exists():record['episodes']=json.loads((output/'episodes.json').read_text())
        done.append(record);(root/('pilot_shard'+str(a.shard)+'_progress.json')).write_text(json.dumps(done,indent=2));print(json.dumps(record),flush=True)
        if code!=0:raise RuntimeError('Pilot infrastructure failed; retained '+str(output))
    for name,expected in spec['runtime_files'].items():assert sha(Path(spec['source'])/name)==expected,name
    assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
    (root/('pilot_shard'+str(a.shard)+'_complete.json')).write_text(json.dumps({'episodes':len(done),'postflight_pass':True}))

if __name__=='__main__':main()
