"""Finish authorized E-POSE ingestion with official UNRAR; preserve old evidence."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import os,sys,json,time,hashlib,shutil,gzip,collections,traceback,fcntl,subprocess
BASE=Path('/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1')
EXPECTED_ZIP_SHA='81fd5ef872ad5551bd170b0451eb3df3e4536a7634202b431d322e05d60e5884'
ORDER=['original_data.rar','block.rar','mustard.rar','bleach.rar','cracker box.rar','drill.rar','five_obj.rar','five_obj_clut.rar','hammer.rar','nine_hole_peg.rar','potted_meat_can.rar','rubik.rar','scissors.rar','sugar_box.rar','three_obj.rar','three_obj_clut.rar','timer.rar','wrench.rar']
sys.path.insert(0,str(BASE/'tools/python'))
import rarfile
UNRAR=BASE/'tools/rar-official-7.23/unrar'
def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def safe_path(name):
 p=PurePosixPath(name.replace('\\','/'))
 if p.is_absolute() or '..' in p.parts or not p.parts or ':' in p.parts[0]:raise ValueError('unsafe path '+repr(name))
 return p
completed=[];receipts=[]
def status(phase,**kw):
 atomic(BASE/'manifests/ingestion_status.json',{'schema':'epose_ingest_status_v1','observed_utc':now(),'phase':phase,'completed_archives':list(completed),'total_archives':len(ORDER),'source_zip_sha256':EXPECTED_ZIP_SHA,**kw})
lock=(BASE/'logs/ingest.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 old=BASE/'manifests/ingestion_status.json'
 if old.exists():shutil.copy2(old,BASE/'manifests/libarchive_failure_preserved.json')
 atomic(BASE/'manifests/ingestion_process.json',{'pid':os.getpid(),'started_utc':now(),'script_sha256':sha(Path(__file__)),'decoder':'official UNRAR 7.23'})
 for name in ORDER:
  stem=Path(name).stem;receipt_path=BASE/'manifests'/f'{stem}.extracted.json'
  meta=json.loads((BASE/'manifests'/f'{stem}.transfer.json').read_text())
  assert meta['name']==name and meta['source_zip_sha256']==EXPECTED_ZIP_SHA
  archive=BASE/'archives'/name;incoming=BASE/'archives'/(name+'.uploading')
  source=archive if archive.exists() else incoming
  assert source.stat().st_size==meta['bytes']
  status('verifying_archive',archive=name)
  digest=sha(source)
  assert digest==meta['sha256'], 'archive SHA256 mismatch'
  if source==incoming:incoming.rename(archive)
  if receipt_path.exists():
   receipt=json.loads(receipt_path.read_text())
   assert receipt['archive_sha256']==digest and Path(receipt['extracted_path']).is_dir()
   receipts.append(receipt);completed.append(name);continue
  with rarfile.RarFile(archive) as rf:
   assert not rf.needs_password()
   infos=rf.infolist();seen=set();files_info=[]
   for i in infos:
    rel=safe_path(i.filename)
    assert not i.is_symlink()
    if i.isdir():continue
    assert i.is_file(), 'special archive member'
    assert str(rel) not in seen, 'duplicate member'
    seen.add(str(rel));files_info.append((rel,i.file_size,i.CRC))
  total_bytes=sum(s for _,s,_ in files_info);expected_files=len(files_info)
  assert shutil.disk_usage(BASE).free>total_bytes+5*1024**3
  target=BASE/'extracted'/stem
  assert not target.exists(),'unregistered existing target'
  stage=BASE/'extracted'/('.'+stem+'.unrar-'+str(os.getpid())+'.extracting')
  assert stage.parent.resolve()==(BASE/'extracted').resolve()
  stage.mkdir(exist_ok=False)
  began=now();log_path=BASE/'logs'/(stem+'.unrar.log')
  with log_path.open('ab') as log:
   p=subprocess.Popen([str(UNRAR),'x','-y','-o-','-idq','-p-','-mt2',str(archive),str(stage)+'/'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   while p.poll() is None:
    status('extracting_rar',archive=name,expected_files=expected_files,expected_bytes=total_bytes,decoder='official UNRAR 7.23',extractor_pid=p.pid,log_path=str(log_path))
    time.sleep(20)
   if p.returncode!=0:raise RuntimeError('UNRAR exit '+str(p.returncode)+'; '+str(log_path))
  inventory=BASE/'manifests'/(stem+'.files.jsonl.gz')
  written=0;extensions=collections.Counter();last=time.monotonic()
  with gzip.open(inventory,'wt') as inv:
   for n,(rel,size,crc) in enumerate(files_info,1):
    dest=stage.joinpath(*rel.parts)
    assert not dest.is_symlink() and dest.is_file() and dest.stat().st_size==size,str(rel)
    written+=size;extensions[dest.suffix.lower()]+=1
    inv.write(json.dumps({'path':str(rel),'bytes':size,'rar_crc32':f'{crc:08x}' if crc is not None else None,'integrity':'verified by UNRAR extraction CRC; exit 0'})+'\n')
    if time.monotonic()-last>20:
     status('verifying_extracted_files',archive=name,files=n,expected_files=expected_files,bytes=written,expected_bytes=total_bytes);last=time.monotonic()
  assert written==total_bytes
  stage.rename(target)
  receipt={'archive':name,'archive_sha256':digest,'source_zip_sha256':EXPECTED_ZIP_SHA,'archive_bytes':meta['bytes'],'extracted_path':str(target),'files':expected_files,'uncompressed_bytes':written,'extensions':dict(extensions),'inventory_path':str(inventory),'inventory_sha256':sha(inventory),'started_utc':began,'completed_utc':now(),'reader':'official UNRAR 7.23; exit 0 verifies archive CRC; rarfile 4.5 path/header audit; all output sizes checked','unrar_exit_code':p.returncode,'log_path':str(log_path),'real_scoring_performed':False}
  atomic(receipt_path,receipt);receipts.append(receipt);completed.append(name)
  status('archive_complete',archive=name,files=expected_files)
  print(json.dumps({'completed':name,'files':expected_files,'bytes':written,'utc':now()}),flush=True)
 result={'schema':'epose_ingestion_receipt_v1','status':'complete','completed_utc':now(),'server':'g1','base_path':str(BASE),'source_local_zip':'E:\\EPOSE_Downloads\\28106717.zip','source_zip_sha256':EXPECTED_ZIP_SHA,'source_zip_bytes':29759156265,'transport':'18 original ZIP members; ZIP CRC and per-member SHA256 verified','archives':receipts,'total_files':sum(r['files'] for r in receipts),'total_uncompressed_bytes':sum(r['uncompressed_bytes'] for r in receipts),'real_scoring_performed':False}
 atomic(BASE/'manifests/ingestion_receipt.json',result);status('complete',total_files=result['total_files'],total_bytes=result['total_uncompressed_bytes'])
except Exception as e:
 status('failed',error=repr(e));traceback.print_exc();raise
