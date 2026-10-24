from pathlib import Path
import subprocess,shlex,base64,json
CODE=r'''from pathlib import Path
import os,json,subprocess,datetime,hashlib
base=Path('/home/zhaoyuhan/Eventbased_WAM')
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'server':subprocess.check_output(['hostname'],text=True).strip()}
def listing(root,depth=2):
 rows=[]
 if not root.exists():return {'missing':str(root)}
 for p in sorted(root.iterdir()):
  row={'name':p.name,'directory':p.is_dir()}
  if p.is_file():row['bytes']=p.stat().st_size
  if p.is_dir() and depth>1:row['children']=listing(p,depth-1)
  rows.append(row)
 return rows
bank=base/'data/sim2real/stage5/20261021-stage5-flow-coverage-v3-eight-rollouts'
run=base/'runs/sim2real/20261021-stage5-flow-coverage-v3-eight-rollouts-eg'
out['flow_bank']=listing(bank,2);out['flow_run']=listing(run,1)
out['flow_run_text']={}
if run.exists():
 for p in run.iterdir():
  if p.is_file() and p.suffix in ('.json','.log','.txt') and p.stat().st_size<40*1024*1024:
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-10000));out['flow_run_text'][p.name]=f.read().decode(errors='replace')
recon=base/'runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0'
out['reconstruction_tree']=listing(recon,2)
out['reconstruction_summary']={}
if recon.exists():
 for p in recon.rglob('*summary*.json'):
  if p.stat().st_size<200000:out['reconstruction_summary'][str(p.relative_to(recon))]=json.loads(p.read_text())
epose=base/'data/real/epose/figshare_28106717_v1/manifests'
out['epose_status']=json.loads((epose/'ingestion_status.json').read_text())
receipt=epose/'ingestion_receipt.json'
if receipt.exists():out['epose_receipt']=json.loads(receipt.read_text())
out['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
ps=subprocess.check_output(['ps','-u','zhaoyuhan','-o','pid,etimes,args'],text=True)
out['relevant_processes']=[r for r in ps.splitlines() if any(w in r.lower() for w in ('flow_','steflow','stage5','e2vid','epose','unrar'))][-25:]
print(json.dumps(out))
'''
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','Hostname=10.129.161.238','-o','HostKeyAlias=g1.pkuml.org','zhaoyuhan@g1.pkuml.org']
cmd=shlex.join(['/usr/bin/python3','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(ssh+[cmd],input=base64.b64encode(CODE.encode()),capture_output=True,timeout=90)
if r.returncode:print(r.stderr.decode());r.check_returncode()
d=json.loads(r.stdout)
dst=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real/benchmark/sim2real/evidence/20261024-research-audit')
dst.mkdir(exist_ok=True)
(dst/'server_refresh.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
print(json.dumps(d,ensure_ascii=False,indent=2))
