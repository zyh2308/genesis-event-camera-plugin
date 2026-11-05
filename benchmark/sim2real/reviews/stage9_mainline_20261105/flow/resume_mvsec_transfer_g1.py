"""Resume the seven registered indoor MVSEC INPUT files; default is offline plan.

Run in WSL. Uses Windows OpenSSH and its existing config/known_hosts. Execution
is explicit, g1-only, and limited to one fixed data-input directory. No weights,
training, download, dataset content parsing, deletion or configuration changes.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,os,shlex,shutil,subprocess,sys,tempfile,time
from pathlib import Path

HOST='zhaoyuhan@g1.pkuml.org'
TARGET='/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-public-v2/indoor_flying'
SOURCE='/home/科研/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-download-v2/indoor_flying'
SSH='/mnt/c/Windows/System32/OpenSSH/ssh.exe'
REG_SHA='46e0141bde84b8c3c3cc96615cec247f44bb8488ae33bdaf626d4f23d8dca02f'
NAMES=tuple(f'indoor_flying{i}_{kind}.hdf5' for i in (1,2,3) for kind in ('data','gt'))+('indoor_flying_calib.zip',)
SSH_OPTIONS=['-p','22','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','StrictHostKeyChecking=yes',
             '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3']

def digest(path,limit=None):
    h=hashlib.sha256();left=limit
    with Path(path).open('rb') as f:
        while left is None or left>0:
            data=f.read(8*1024**2 if left is None else min(8*1024**2,left))
            if not data:break
            h.update(data)
            if left is not None:left-=len(data)
    if left not in (None,0):raise ValueError('Local file shorter than requested prefix')
    return h.hexdigest()

def save(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)

def specs(registration,selected):
    if digest(registration)!=REG_SHA:raise ValueError('Immutable window registration SHA mismatch')
    d=json.loads(Path(registration).read_text());result=[]
    for name in selected:
        if name not in NAMES:raise ValueError('Unregistered filename')
        if name.endswith('.hdf5'):
            seq,kind=name[:-5].rsplit('_',1);r=d['files'][seq]
            item=dict(name=name,source=str(Path(SOURCE)/name),target=str(Path(TARGET)/name),
                      expected_bytes=r[kind+'_size_bytes'],expected_sha256=r[kind+'_sha256'])
            if r[kind+'_path_on_g1']!=item['target']:raise ValueError('HDF5 target differs from immutable registration')
        else:
            r=d['calibration_files']['indoor_flying']
            item=dict(name=name,source=str(Path(SOURCE)/name),target=str(Path(TARGET)/name),
                      expected_bytes=r['size_bytes'],expected_sha256=r['sha256'],
                      calibration_original_path_metadata=r['path_on_g1'],calibration_ascii_path_mapping_only=True)
        result.append(item)
    return result

def classify(item,before,local_prefix_sha=None):
    if before.get('symlink'):raise ValueError('Refuse remote symlink')
    if not before['exists']:return 'create_missing_input'
    size=before['size_bytes']
    if size==item['expected_bytes']:
        if before['sha256']==item['expected_sha256']:return 'skip_correct_input'
        raise ValueError('Existing complete-size input has wrong SHA; preserve and investigate')
    if size>item['expected_bytes']:raise ValueError('Existing input larger than registered source; preserve')
    if before['sha256']!=local_prefix_sha:raise ValueError('Partial input is not an exact source prefix; preserve')
    return 'resume_verified_partial_input'

def remote_audit_program(name):
    if name not in NAMES:raise ValueError('Remote filename is not registered')
    return """import hashlib,json,os
from pathlib import Path
root=Path(%r)
if not root.is_dir() or str(root.resolve())!=str(root):raise RuntimeError('Expected exact existing input directory')
p=root/%r
if p.is_symlink():print(json.dumps(dict(exists=True,symlink=True)));raise SystemExit(0)
if not p.exists():print(json.dumps(dict(exists=False,symlink=False)));raise SystemExit(0)
if not p.is_file():raise RuntimeError('Input path is not a regular file')
before=p.stat();h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
after=p.stat()
fields=('st_size','st_mtime_ns','st_ino','st_dev')
if any(getattr(before,k)!=getattr(after,k) for k in fields):raise RuntimeError('Remote input changed during audit')
print(json.dumps(dict(exists=True,symlink=False,size_bytes=after.st_size,sha256=h.hexdigest(),mtime_ns=after.st_mtime_ns,inode=after.st_ino,device=after.st_dev)))
"""%(TARGET,name)

def ssh_program(program):
    encoded=base64.b64encode(program.encode()).decode()
    # Base64 and shlex prevent source paths or payloads from becoming shell code.
    return 'python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+encoded+"'))")

def rsync_command(item,bwlimit):
    remote_shell=shlex.join([SSH,*SSH_OPTIONS])
    # flock is held by the remote rsync for the entire write. It does not claim
    # to coordinate legacy writers, hence explicit writer-check flag is required.
    lock=str(Path(TARGET)/'.stage9_mvsec_transfer.lock')
    cmd=['rsync','-rt','--partial','--append-verify','--protect-args','--info=progress2',
         '--rsync-path','flock -n '+shlex.quote(lock)+' rsync','-e',remote_shell]
    if bwlimit:cmd+=['--bwlimit='+str(bwlimit)]
    return cmd+[item['source'],HOST+':'+item['target']]

def execute(a,items):
    if not a.writer_check_confirmed:raise ValueError('Execution requires --writer-check-confirmed after root checks active transfers/writers')
    if not Path(SSH).is_file() or not shutil.which('rsync'):raise RuntimeError('Existing Windows OpenSSH and WSL rsync are required; nothing will be installed')
    if a.out is None:raise ValueError('--out must name a new local receipt directory')
    a.out.mkdir(parents=True,exist_ok=False)
    save(a.out/'launcher.json',dict(pid=os.getpid(),started_unix=time.time(),argv=sys.argv,
         target=TARGET,host=HOST,registration_sha256=REG_SHA,writer_check_confirmed=True,
         uses_existing_windows_ssh_config_and_known_hosts=True,does_not_touch_training=True))
    (a.out/'launcher.pid').write_text(str(os.getpid())+'\n')
    events=(a.out/'processes.jsonl').open('x',buffering=1)
    results=[];success=False
    def run(cmd,label):
        with (a.out/(label+'.stdout.log')).open('xb') as out,(a.out/(label+'.stderr.log')).open('xb') as err:
            proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
            events.write(json.dumps(dict(label=label,pid=proc.pid,started_unix=time.time(),command=cmd))+'\n')
            rc=proc.wait()
            events.write(json.dumps(dict(label=label,pid=proc.pid,exit_code=rc,ended_unix=time.time()))+'\n')
        if rc:raise RuntimeError(label+' failed with exit '+str(rc)+'; preserved logs and current input bytes')
        return (a.out/(label+'.stdout.log')).read_text()
    def audit(name,label):
        raw=run([SSH,*SSH_OPTIONS,HOST,ssh_program(remote_audit_program(name))],label)
        return json.loads(raw)
    try:
        for index,item in enumerate(items):
            print('SOURCE_VERIFY',item['name'],flush=True)
            path=Path(item['source']);st=path.stat()
            if path.is_symlink() or st.st_size!=item['expected_bytes'] or digest(path)!=item['expected_sha256']:
                raise ValueError('Local registered source changed: '+item['name'])
            if st.st_mtime_ns!=path.stat().st_mtime_ns:raise ValueError('Local source changed during full hash')
            label=f'{index:02d}_{item["name"]}'
            before=audit(item['name'],label+'_before')
            prefix=digest(path,before['size_bytes']) if before.get('exists') and not before.get('symlink') and before['size_bytes']<item['expected_bytes'] else None
            action=classify(item,before,prefix)
            record=dict(**item,before=before,action=action,local_full_sha256_verified=True)
            save(a.out/(label+'_before.json'),record)
            print('TRANSFER_ACTION',item['name'],action,flush=True)
            if action!='skip_correct_input':run(rsync_command(item,a.bwlimit_kib),label+'_rsync')
            after=audit(item['name'],label+'_after')
            if not after.get('exists') or after.get('symlink') or after['size_bytes']!=item['expected_bytes'] or after['sha256']!=item['expected_sha256']:
                raise RuntimeError('Post-transfer complete SHA/size mismatch: '+item['name'])
            record.update(after=after,verified_complete=True);save(a.out/(label+'_complete.json'),record);results.append(record)
            print('VERIFIED_COMPLETE',item['name'],after['sha256'],flush=True)
        success=True
    except BaseException as e:
        save(a.out/'failure.json',dict(type=type(e).__name__,error=str(e),time_unix=time.time(),completed_files=len(results)))
        raise
    finally:
        events.close();save(a.out/'exit.json',dict(success=success,exit_code=0 if success else 1,
            ended_unix=time.time(),completed_files=len(results),expected_files=len(items),
            real_scores_computed=False,training_or_result_paths_modified=False,files=results))
    print('MVSEC_INPUT_TRANSFER_VERIFIED',len(results),'files',flush=True)

def selftest():
    item=dict(name=NAMES[0],source='/tmp/source',target=TARGET+'/'+NAMES[0],expected_bytes=6,expected_sha256=hashlib.sha256(b'abcdef').hexdigest())
    tests=[]
    assert classify(item,dict(exists=False))=='create_missing_input';tests.append('missing_create')
    assert classify(item,dict(exists=True,size_bytes=6,sha256=item['expected_sha256']))=='skip_correct_input';tests.append('correct_skip')
    prefix=hashlib.sha256(b'abc').hexdigest()
    assert classify(item,dict(exists=True,size_bytes=3,sha256=prefix),prefix)=='resume_verified_partial_input';tests.append('verified_prefix_resume')
    for label,before,local in [('wrong_complete',dict(exists=True,size_bytes=6,sha256='0'*64),None),
           ('wrong_prefix',dict(exists=True,size_bytes=3,sha256='0'*64),prefix),
           ('oversize',dict(exists=True,size_bytes=7,sha256='0'*64),None),
           ('symlink',dict(exists=True,symlink=True),None)]:
        try:classify(item,before,local)
        except ValueError:tests.append(label+'_rejected')
        else:raise AssertionError('Unsafe state accepted: '+label)
    with tempfile.TemporaryDirectory(prefix='mvsec_transfer_test_') as td:
        f=Path(td)/'tiny';f.write_bytes(b'abcdef')
        assert digest(f)==item['expected_sha256'] and digest(f,3)==prefix;tests.append('actual_prefix_full_hash')
    cmd=rsync_command(item,1000)
    assert '--append-verify' in cmd and '--partial' in cmd and '--inplace' not in cmd and '--delete' not in cmd
    assert cmd[-1]==HOST+':'+TARGET+'/'+NAMES[0] and SSH in cmd[cmd.index('-e')+1]
    tests.append('fixed_target_windows_ssh_resumable_command')
    try:remote_audit_program('../unsafe')
    except ValueError:tests.append('unregistered_path_rejected')
    else:raise AssertionError('Unsafe remote filename accepted')
    print(json.dumps(dict(passed=True,checks=tests,SSH_used=False,real_data_used=False),indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,default=Path(__file__).with_name('stage9_mvsec_indoor_unscored_registration.json'))
    p.add_argument('--file',choices=NAMES,action='append',dest='files')
    p.add_argument('--execute',action='store_true');p.add_argument('--writer-check-confirmed',action='store_true')
    p.add_argument('--self-test',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--bwlimit-kib',type=int,default=0,help='0 retains rsync default; root may cap shared bandwidth')
    a=p.parse_args()
    if a.bwlimit_kib<0:p.error('Bandwidth limit cannot be negative')
    if a.self_test:
        if a.execute:p.error('Self-test and execute are mutually exclusive')
        selftest();return
    names=a.files or list(NAMES)
    if len(set(names))!=len(names):p.error('Duplicate selected filename')
    items=specs(a.registration,names)
    if not a.execute:
        print(json.dumps(dict(mode='OFFLINE_PLAN_ONLY',SSH_used=False,files=items,
              execute_requires='--execute --writer-check-confirmed --out NEW_DIRECTORY'),indent=2,ensure_ascii=False));return
    execute(a,items)

if __name__=='__main__':main()
