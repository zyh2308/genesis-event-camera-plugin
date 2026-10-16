from pathlib import Path
import base64,json,shlex,subprocess
OUT=Path(__file__).resolve().parent
code='import base64,sys; exec(compile(base64.b64decode(sys.stdin.buffer.read()),"<research-audit>","exec"))'
command=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c',code])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-p','22','zhaoyuhan@10.129.163.68',command],input=base64.b64encode((OUT/'remote_program.py').read_bytes()).decode(),text=True,capture_output=True,timeout=240)
(OUT/'remote_audit_stdout.json').write_text(r.stdout)
(OUT/'remote_audit_stderr.txt').write_text(r.stderr)
if r.returncode: print(json.dumps({'exit':r.returncode,'stderr':r.stderr,'stdout_tail':r.stdout[-2000:]}))
else:
    d=json.loads(r.stdout)
    print(json.dumps({k:d[k] for k in ['host','head','status','core_diff','checkpoints','paired_data','exit_codes','STE_return_source']},ensure_ascii=False))
