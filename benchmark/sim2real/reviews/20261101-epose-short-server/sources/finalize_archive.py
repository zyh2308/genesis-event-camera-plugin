from pathlib import Path
import json,hashlib,shutil,zipfile
W=Path('/mnt/c/Users/zhaoyuhan/Documents/ChatGPT/New project');O=W/'outputs/EPOSE_服务器短训初测_20261101';R=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real/benchmark/sim2real/reviews/20261101-epose-short-server');D=Path('/mnt/c/Users/zhaoyuhan/Desktop/EPOSE_服务器短训初测_20261101')
for n in ['EPOSE_初测指标.csv','EPOSE_短训结果说明.md']:
 p=O/n;s=p.read_text(encoding='utf-8-sig').rstrip()+'\n';p.write_text(s,encoding='utf-8-sig' if n.endswith('.csv') else 'utf-8');shutil.copy2(p,R/n);shutil.copy2(p,D/n)
for p in (W/'artifacts/epose_short_server').glob('*.py'):shutil.copy2(p,R/'sources'/p.name)
receipt=json.loads((R/'archive_receipt.json').read_text());receipt['artifacts']={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in R.rglob('*') if p.is_file() and p.name!='archive_receipt.json'};(R/'archive_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
with zipfile.ZipFile(D/'完整结果与核验记录.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in R.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(R))
print('Archive refreshed; original machine evidence remains byte-identical.')
