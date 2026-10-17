from pathlib import Path
import base64,json,shlex,subprocess
O=Path(__file__).resolve().parent
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=0','PYOPENGL_PLATFORM=egl','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=2','PYTHONPATH=/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2:/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode((O/'remote_render_minimal.py').read_bytes()).decode(),capture_output=True,text=True,timeout=180)
(O/'render_minimal.stdout.txt').write_text(r.stdout,encoding='utf-8');(O/'render_minimal.stderr.txt').write_text(r.stderr,encoding='utf-8')
if r.returncode:raise RuntimeError(r.stderr[-4000:])
line=next(s for s in r.stdout.splitlines() if s.startswith('AUDIT_JSON='))
d=json.loads(line.split('=',1)[1]);(O/'render_minimal.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8');print(json.dumps(d))
