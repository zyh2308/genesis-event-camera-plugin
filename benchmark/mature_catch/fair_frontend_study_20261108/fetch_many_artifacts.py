"""Fetch bounded chunks through our existing SSH relay, with final SHA256 checks."""
import argparse,base64,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--prefix',required=True);p.add_argument('--queue',default='remote_queue');a=p.parse_args();r=Path(__file__).resolve().parent;q=r/a.queue;plan=json.loads(a.plan.read_text(encoding='utf-8'))
def request(identifier,code):
 response=q/(identifier+'.response.json')
 if not response.exists():
  dest=q/(identifier+'.request.py')
  if not dest.exists():draft=q/(identifier+'.draft');draft.write_text(code,encoding='utf-8');draft.rename(dest)
  deadline=time.monotonic()+55
  while not response.exists() and time.monotonic()<deadline:time.sleep(.4)
  if not response.exists():raise TimeoutError(identifier+' remains queued; rerun to resume without replay')
 envelope=json.loads(response.read_text(encoding='utf-8'));assert envelope['ok'],envelope
 return json.loads(envelope['output'])
receipts=[]
for i,item in enumerate(plan):
 local=Path(item['local']);local.parent.mkdir(parents=True,exist_ok=True);offset=local.stat().st_size if local.exists() else 0
 metadata=request(a.prefix+'_%03d_meta'%i,'from pathlib import Path\nimport hashlib,json\np=Path('+repr(item['remote'])+');h=hashlib.sha256()\nwith p.open("rb") as f:\n for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)\nprint(json.dumps({"size":p.stat().st_size,"sha256":h.hexdigest()}))\n')
 assert offset<=metadata['size']
 while offset<metadata['size']:
  identifier=a.prefix+'_%03d_%d'%(i,offset)
  code='from pathlib import Path\nimport base64,json\np=Path('+repr(item['remote'])+');f=p.open("rb");f.seek('+str(offset)+');b=f.read(1048576);print(json.dumps({"base64":base64.b64encode(b).decode()}))\n'
  result=request(identifier,code);chunk=base64.b64decode(result['base64']);assert chunk
  with local.open('ab') as f:f.write(chunk)
  offset+=len(chunk)
 h=hashlib.sha256(local.read_bytes()).hexdigest();assert h==metadata['sha256'],str(local)
 receipts.append(dict(item,**metadata));print(json.dumps({'file':str(local),'bytes':offset,'sha256_verified':True},ensure_ascii=False),flush=True)
a.plan.with_suffix('.receipt.json').write_text(json.dumps(receipts,indent=2,ensure_ascii=False),encoding='utf-8')
