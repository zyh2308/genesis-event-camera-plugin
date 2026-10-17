from pathlib import Path
import base64,json,shlex,subprocess
O=Path(__file__).resolve().parent
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode((O/'remote_preflight_audit.py').read_bytes()).decode(),capture_output=True,text=True,timeout=600)
(O/'remote_preflight.json').write_text(r.stdout,encoding='utf-8');(O/'remote_preflight.stderr.txt').write_text(r.stderr,encoding='utf-8')
if r.returncode:raise RuntimeError(r.stderr)
d=json.loads(r.stdout)
print(json.dumps({'checked_at_UTC':d['checked_at_UTC'],'runtime_seconds':d['runtime_seconds'],'checks':list(d['checks']),'source_count':len(d['source_hashes'])}))
