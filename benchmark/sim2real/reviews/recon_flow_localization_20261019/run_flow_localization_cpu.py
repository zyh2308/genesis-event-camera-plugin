from pathlib import Path
import subprocess,shlex,base64,json,sys
L=Path(__file__).parent
source=Path(sys.argv[1]) if len(sys.argv)>1 else L/'flow_localization_cpu_remote.py'
output=L/'flow_gradient_precision.json' if len(sys.argv)>1 else L/'flow_localization_cpu.json'
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['wsl','-d','Ubuntu-22.04','--','ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode(source.read_bytes()).decode(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=165)
output.write_text(r.stdout,encoding='utf-8');output.with_suffix('.stderr.txt').write_text(r.stderr,encoding='utf-8')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
d=json.loads(r.stdout)
print(json.dumps(d if len(sys.argv)>1 else {'loss_oracle':d['loss_oracle'],'rows':len(d['temporal_support_rows']),'finished_utc':d['finished_utc']},indent=2))
