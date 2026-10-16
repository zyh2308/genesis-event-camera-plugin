from pathlib import Path
import base64,json,shlex,subprocess
O=Path(__file__).resolve().parent
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-p','22','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode((O/'remote_targeted_program.py').read_bytes()).decode(),capture_output=True,text=True,timeout=180)
(O/'targeted_stdout.json').write_text(r.stdout);(O/'targeted_stderr.txt').write_text(r.stderr)
if r.returncode:print(json.dumps({'exit':r.returncode,'stderr':r.stderr,'stdout_tail':r.stdout[-1000:]}))
else:
    d=json.loads(r.stdout);print(json.dumps({k:v for k,v in d.items() if k!='pose_actual_sample_checks'},ensure_ascii=False));print('pose_samples_checked',len(d['pose_actual_sample_checks']))
