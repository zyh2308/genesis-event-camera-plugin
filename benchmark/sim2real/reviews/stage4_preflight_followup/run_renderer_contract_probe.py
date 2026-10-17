from pathlib import Path
import base64,shlex,subprocess
O=Path(__file__).resolve().parent
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode((O/'remote_renderer_contract_probe.py').read_bytes()).decode(),capture_output=True,text=True,timeout=60)
(O/'renderer_contract_probe.json').write_text(r.stdout,encoding='utf-8')
if r.returncode:raise RuntimeError(r.stderr)
print(r.stdout)
