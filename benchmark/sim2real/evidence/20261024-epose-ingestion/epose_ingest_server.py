"""Verify and unpack authorized public E-POSE archives without changing experiments."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import os,sys,json,time,hashlib,shutil,gzip,collections,traceback,fcntl

BASE=Path('/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1')
EXPECTED_ZIP_SHA='81fd5ef872ad5551bd170b0451eb3df3e4536a7634202b431d322e05d60e5884'
ORDER=['original_data.rar','block.rar','mustard.rar','bleach.rar','cracker box.rar','drill.rar','five_obj.rar','five_obj_clut.rar','hammer.rar','nine_hole_peg.rar','potted_meat_can.rar','rubik.rar','scissors.rar','sugar_box.rar','three_obj.rar','three_obj_clut.rar','timer.rar','wrench.rar']
sys.path.insert(0,str(BASE/'tools/python'))
os.environ['LIBARCHIVE']='/home/zhaoyuhan/miniconda3/lib/libarchive.so'
import libarchive,rarfile

def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,d):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def safe_path(name):
    name=name.replace('\\','/');p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or ':' in p.parts[0]:raise ValueError('unsafe archive path '+repr(name))
    return p
def status(phase,**kw):
    d={'schema':'epose_ingest_status_v1','observed_utc':now(),'phase':phase,'completed_archives':list(completed),'total_archives':len(ORDER),'source_zip_sha256':EXPECTED_ZIP_SHA,**kw}
    atomic(BASE/'manifests/ingestion_status.json',d)

completed=[];receipts=[]
lock=(BASE/'logs/ingest.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    atomic(BASE/'manifests/ingestion_process.json',{'pid':os.getpid(),'started_utc':now(),'script_sha256':sha(Path(__file__))})
    for name in ORDER:
        stem=Path(name).stem;ready=BASE/'manifests'/f'{stem}.transfer.json';receipt_path=BASE/'manifests'/f'{stem}.extracted.json'
        while not ready.exists():
            status('waiting_for_transfer',archive=name);time.sleep(20)
        meta=json.loads(ready.read_text())
        if meta['source_zip_sha256']!=EXPECTED_ZIP_SHA or meta['name']!=name:raise ValueError('transfer binding mismatch')
        archive=BASE/'archives'/name;incoming=BASE/'archives'/(name+'.uploading')
        source=archive if archive.exists() else incoming
        if source.stat().st_size!=meta['bytes']:raise ValueError('archive size mismatch '+name)
        status('verifying_archive',archive=name)
        digest=sha(source)
        if digest!=meta['sha256']:raise ValueError('archive SHA256 mismatch '+name)
        if source==incoming:incoming.rename(archive)
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if receipt['archive_sha256']!=digest or not Path(receipt['extracted_path']).is_dir():raise ValueError('stale extraction receipt')
            receipts.append(receipt);completed.append(name);continue
        status('auditing_rar_headers',archive=name)
        with rarfile.RarFile(archive) as rf:
            if rf.needs_password():raise ValueError('encrypted archive '+name)
            infos=rf.infolist();total_bytes=sum(i.file_size for i in infos if not i.isdir());expected_files=sum(not i.isdir() for i in infos)
            for info in infos:
                safe_path(info.filename)
                if hasattr(info,'is_symlink') and info.is_symlink():raise ValueError('symlink in archive')
        if shutil.disk_usage(BASE).free<total_bytes+5*1024**3:raise ValueError('insufficient extraction space')
        target=BASE/'extracted'/stem;stage=BASE/'extracted'/('.'+stem+'.extracting')
        if target.exists():raise FileExistsError('unregistered existing extraction '+str(target))
        if not stage.resolve().is_relative_to((BASE/'extracted').resolve()):raise ValueError('staging directory escapes intended root')
        resuming=stage.exists();stage.mkdir(exist_ok=True)
        inventory=BASE/'manifests'/(stem+'.files.jsonl.gz')
        files=0;written=0;extensions=collections.Counter();last=time.monotonic();began=now()
        made_dirs={stage};seen_files=set()
        with gzip.open(inventory,'wt',encoding='utf-8') as inv,libarchive.file_reader(str(archive)) as reader:
            for entry in reader:
                relative=safe_path(entry.pathname);dest=stage.joinpath(*relative.parts)
                # safe_path rejects absolute paths and '..'; all archive links
                # are rejected below. The controlled staging root was resolved
                # once above, avoiding millions of NFS stat calls per import.
                if entry.issym or entry.islnk:raise ValueError('archive link rejected')
                if entry.isdir:
                    if dest not in made_dirs:dest.mkdir(parents=True,exist_ok=True);made_dirs.add(dest)
                    continue
                if not entry.isfile:raise ValueError('nonregular archive member')
                if str(relative) in seen_files:raise ValueError('duplicate archive member')
                seen_files.add(str(relative))
                if dest.parent not in made_dirs:dest.parent.mkdir(parents=True,exist_ok=True);made_dirs.add(dest.parent)
                h=hashlib.sha256();count=0
                if resuming and dest.exists():
                    for block in entry.get_blocks():h.update(block);count+=len(block)
                    if dest.stat().st_size!=count or sha(dest)!=h.hexdigest():raise ValueError('partial extraction differs '+str(relative))
                else:
                    tmp=dest.with_name(dest.name+'.epose-writing')
                    with tmp.open('xb') as f:
                        for block in entry.get_blocks():f.write(block);h.update(block);count+=len(block)
                    if entry.size is not None and count!=entry.size:raise ValueError('extracted size mismatch')
                    tmp.rename(dest)
                files+=1;written+=count;extensions[dest.suffix.lower()]+=1
                inv.write(json.dumps({'path':str(relative),'bytes':count,'sha256':h.hexdigest()},ensure_ascii=False)+'\n')
                if time.monotonic()-last>20:
                    status('extracting_rar',archive=name,files=files,expected_files=expected_files,bytes=written,expected_bytes=total_bytes);last=time.monotonic()
        if files!=expected_files or written!=total_bytes:raise ValueError('extraction totals differ from RAR headers')
        stage.rename(target)
        receipt={'archive':name,'archive_sha256':digest,'source_zip_sha256':EXPECTED_ZIP_SHA,'archive_bytes':meta['bytes'],'extracted_path':str(target),'files':files,'uncompressed_bytes':written,'extensions':dict(extensions),'inventory_path':str(inventory),'inventory_sha256':sha(inventory),'started_utc':began,'completed_utc':now(),'reader':'libarchive-c 5.3; conda libarchive.so.13.8.7; rarfile 4.5 header audit','real_scoring_performed':False}
        atomic(receipt_path,receipt);receipts.append(receipt);completed.append(name)
        status('archive_complete',archive=name,files=files)
        print(json.dumps({'completed':name,'files':files,'bytes':written,'utc':now()}),flush=True)
    result={'schema':'epose_ingestion_receipt_v1','status':'complete','completed_utc':now(),'server':'g1','base_path':str(BASE),'source_local_zip':'E:\\EPOSE_Downloads\\28106717.zip','source_zip_sha256':EXPECTED_ZIP_SHA,'source_zip_bytes':29759156265,'transport':'18 original RAR members from the downloaded ZIP; per-member ZIP CRC and SHA256 verified; outer ZIP retained locally','archives':receipts,'total_files':sum(r['files'] for r in receipts),'total_uncompressed_bytes':sum(r['uncompressed_bytes'] for r in receipts),'real_scoring_performed':False}
    atomic(BASE/'manifests/ingestion_receipt.json',result);status('complete',total_files=result['total_files'],total_bytes=result['total_uncompressed_bytes'])
except Exception as e:
    status('failed',error=repr(e));traceback.print_exc();raise
