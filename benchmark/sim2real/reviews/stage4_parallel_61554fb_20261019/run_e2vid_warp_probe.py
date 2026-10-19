from pathlib import Path
import subprocess,base64,shlex
L=Path(__file__).parent
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['wsl','-d','Ubuntu-22.04','--','ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode((L/'e2vid_warp_identity_probe.py').read_bytes()).decode(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=55)
(L/'e2vid_warp_identity_probe.json').write_text(r.stdout,encoding='utf-8');(L/'e2vid_warp_identity_probe.stderr.txt').write_text(r.stderr,encoding='utf-8')
print(r.stdout if r.returncode==0 else r.stderr)
raise SystemExit(r.returncode)
