from pathlib import Path
import subprocess,base64,shlex,json
root=Path(__file__).parent
command=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
p=subprocess.run(['wsl','-d','Ubuntu-22.04','--','ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','zhaoyuhan@g1.pkuml.org',command],input=base64.b64encode((root/'stage4_motion_review_remote.py').read_bytes()).decode(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=350)
(root/'stage4_motion_review_remote.json').write_text(p.stdout,encoding='utf-8')
(root/'stage4_motion_review_remote.stderr.txt').write_text(p.stderr,encoding='utf-8')
if p.returncode:print(p.stderr);raise SystemExit(p.returncode)
d=json.loads(p.stdout)
print(json.dumps({'observed_utc':d['observed_utc'],'source_hash_matches':d['source_hash_matches_receipt'],'checkpoints':d['v2e_oracle_checkpoints'],'completion':d['completion_files'],'replay':d['replay']},indent=2))
