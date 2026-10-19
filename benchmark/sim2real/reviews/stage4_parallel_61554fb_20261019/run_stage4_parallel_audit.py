from pathlib import Path
import subprocess,base64,shlex,json
L=Path(__file__).parent
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['wsl','-d','Ubuntu-22.04','--','ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','zhaoyuhan@g1.pkuml.org',cmd],input=base64.b64encode((L/'stage4_parallel_audit_remote.py').read_bytes()).decode(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=140)
(L/'stage4_parallel_audit.json').write_text(r.stdout,encoding='utf-8');(L/'stage4_parallel_audit.stderr.txt').write_text(r.stderr,encoding='utf-8')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
d=json.loads(r.stdout)
print(json.dumps({'observed_utc':d['observed_utc'],'arms':{k:{'count':v['count'],'last':v['last'],'validation':v['validation'],'checkpoint':v['checkpoint'],'gaps':v['gaps'],'bad':v['bad_rows']} for k,v in d['arms'].items()},'pairing':d['paired_clips'],'git_head':d['git_head'],'hashes':d['code_hashes']},indent=2))
