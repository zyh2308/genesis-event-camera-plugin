"""Durable local transfer supervisor. Run under WSL; no training or scoring."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import os,json,zipfile,hashlib,subprocess,base64,shlex,time,fcntl,traceback
SRC=Path('/mnt/e/EPOSE_Downloads/28106717.zip')
LOCAL=SRC.parent/'server_transfer';LOCAL.mkdir(exist_ok=True)
PARTS=LOCAL/'rar';PARTS.mkdir(exist_ok=True)
BASE='/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1'
ZIP_SHA='81fd5ef872ad5551bd170b0451eb3df3e4536a7634202b431d322e05d60e5884'
SSH=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','Hostname=10.129.161.238','-o','HostKeyAlias=g1.pkuml.org','zhaoyuhan@g1.pkuml.org']
SSH_RSYNC=shlex.join(SSH[:-1])
ORDER=['original_data.rar','block.rar','mustard.rar','bleach.rar','cracker box.rar','drill.rar','five_obj.rar','five_obj_clut.rar','hammer.rar','nine_hole_peg.rar','potted_meat_can.rar','rubik.rar','scissors.rar','sugar_box.rar','three_obj.rar','three_obj_clut.rar','timer.rar','wrench.rar']
def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def status(phase,**kw):atomic(LOCAL/'transfer_status.json',{'phase':phase,'observed_utc':now(),'completed_archives':list(completed),'total_archives':18,'server_base':BASE,'source_zip_sha256':ZIP_SHA,**kw})
def remote_python(code):
    encoded=base64.b64encode(code.encode()).decode()
    command=shlex.join(['/usr/bin/python3','-c','import base64;exec(base64.b64decode('+repr(encoded)+'))'])
    return subprocess.run(SSH+[command],check=True,timeout=90,capture_output=True,text=True)
def extract(name):
    p=PARTS/name;receipt=PARTS/(name+'.json')
    if receipt.exists() and p.exists():
        meta=json.loads(receipt.read_text())
        if meta['source_zip_sha256']==ZIP_SHA and p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256']:return meta
    if p.exists():raise FileExistsError('unregistered local RAR '+str(p))
    temp=p.with_name(p.name+'.preparing');h=hashlib.sha256();size=0
    with zipfile.ZipFile(SRC) as z,z.open(name) as src,temp.open('wb') as dst:
        zi=z.getinfo(name)
        while True:
            block=src.read(4*1024*1024)
            if not block:break
            dst.write(block);h.update(block);size+=len(block)
    if size!=zi.file_size:raise ValueError('local ZIP member size mismatch')
    temp.rename(p)
    meta={'name':name,'bytes':size,'sha256':h.hexdigest(),'zip_crc32':f'{zi.CRC:08x}','source_zip_sha256':ZIP_SHA,'zip_member_crc_checked':True,'prepared_utc':now()}
    atomic(receipt,meta);return meta
completed=[]
lock=(LOCAL/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
atomic(LOCAL/'supervisor_process.json',{'pid':os.getpid(),'started_utc':now(),'script':str(Path(__file__))})
try:
    if SRC.stat().st_size!=29759156265:raise ValueError('source ZIP changed size')
    with zipfile.ZipFile(SRC) as z:
        if sorted(z.namelist())!=sorted(ORDER):raise ValueError('ZIP member manifest changed')
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures={name:pool.submit(extract,name) for name in ORDER}
        for name in ORDER:
            status('preparing_local_rar',archive=name)
            meta=futures[name].result();p=PARTS/name
            status('uploading_rar',archive=name,archive_bytes=meta['bytes'])
            log=LOCAL/(Path(name).stem+'.rsync.log')
            cmd=['rsync','-t','--protect-args','--partial','--append-verify','--info=progress2','--human-readable','-e',SSH_RSYNC,str(p),'zhaoyuhan@g1.pkuml.org:'+BASE+'/archives/'+name+'.uploading']
            for attempt in range(1,4):
                with log.open('ab') as stream:
                    stream.write(f'\nAttempt {attempt} {now()}\n'.encode());stream.flush()
                    proc=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
                if proc.returncode==0:break
                if attempt==3:raise RuntimeError('rsync failed for '+name+': '+str(proc.returncode))
                time.sleep(10)
            payload=base64.b64encode(json.dumps(meta).encode()).decode()
            remote_python('from pathlib import Path;import base64\np=Path('+repr(BASE+'/manifests/'+Path(name).stem+'.transfer.json')+');t=p.with_suffix(".tmp");t.write_bytes(base64.b64decode('+repr(payload)+'));t.replace(p)')
            completed.append(name);status('archive_uploaded',archive=name)
            print(json.dumps({'uploaded':name,'completed':len(completed),'utc':now()}),flush=True)
    status('all_archives_uploaded',note='Server extraction continues independently; check server manifests/ingestion_status.json')
except Exception as e:
    status('failed',error=repr(e));traceback.print_exc();raise
