import hashlib,json,sys,tarfile,zipfile
from pathlib import Path
base=Path(sys.argv[1]);dest=Path(sys.argv[2]);rows=[]
with tarfile.open(dest,'w:gz',compresslevel=1) as tar:
 f=base/'vid2e_SuperSloMo.ckpt';tar.add(f,arcname='vid2e/SuperSloMo.ckpt');rows.append({'path':'vid2e/SuperSloMo.ckpt','sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'size_bytes':f.stat().st_size})
 for zipname,prefix,selector in [('film_official_pretrained_models.zip','film',lambda n:n.startswith('pretrained_models/film_net/Style/saved_model/')),('rife_paper_model.zip','rife',lambda n:n=='train_log/flownet.pkl')]:
  with zipfile.ZipFile(base/zipname) as z:
   for entry in z.infolist():
    if entry.is_dir() or not selector(entry.filename):continue
    name=prefix+'/'+entry.filename;info=tarfile.TarInfo(name);info.size=entry.file_size;info.mode=0o644
    with z.open(entry) as stream:tar.addfile(info,stream)
    with z.open(entry) as stream:
     h=hashlib.sha256()
     for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    rows.append({'path':name,'sha256':h.hexdigest(),'size_bytes':entry.file_size})
manifest={'schema':'stage9_official_baseline_weights_v1','files':rows,'downloads':json.loads((base/'acquisition_summary.json').read_text()),'archive_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
dest.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(dest.stat().st_size)
