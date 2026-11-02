"""Acquire official public interpolation weights into an isolated directory."""
import concurrent.futures,hashlib,json,sys,time,urllib.request,zipfile
from pathlib import Path
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
items=[('vid2e_SuperSloMo.ckpt','https://rpg.ifi.uzh.ch/data/VID2E/SuperSloMo.ckpt'),('film_official_pretrained_models.zip','https://rpg.ifi.uzh.ch/data/VID2E/pretrained_models.zip'),('rife_paper_model.zip','https://drive.usercontent.google.com/download?id=1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX&export=download&confirm=t')]
def one(item):
 name,url=item;target=out/name;partial=out/(name+'.partial');begin=time.monotonic();row={'file':name,'official_url':url,'status':'downloading'}
 try:
  assert not target.exists() and not partial.exists()
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=40) as r,partial.open('xb') as f:
   assert r.headers.get_content_type()!='text/html','Server returned HTML instead of weights'
   total=0;digest=hashlib.sha256()
   while True:
    chunk=r.read(4*1024*1024)
    if not chunk:break
    f.write(chunk);digest.update(chunk);total+=len(chunk)
    (out/(name+'.progress.json')).write_text(json.dumps({'downloaded_bytes':total,'elapsed_s':time.monotonic()-begin})+'\n')
  assert total>1_000_000
  if name.endswith('.zip'):
   with zipfile.ZipFile(partial) as z:assert z.testzip() is None
  partial.rename(target);row.update(status='complete',size_bytes=total,sha256=digest.hexdigest(),elapsed_s=time.monotonic()-begin)
 except Exception as exc:row.update(status='failed',error=repr(exc),elapsed_s=time.monotonic()-begin)
 (out/(name+'.receipt.json')).write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,items))
(out/'acquisition_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
