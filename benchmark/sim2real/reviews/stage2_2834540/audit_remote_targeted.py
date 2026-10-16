from pathlib import Path
import base64,json,shlex,subprocess
OUT=Path(__file__).resolve().parent
code='import base64,sys; exec(compile(base64.b64decode(sys.stdin.buffer.read()),"<targeted-review>","exec"))'
command=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c',code])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-p','22','zhaoyuhan@10.129.163.68',command],input=base64.b64encode((OUT/'remote_targeted_program.py').read_bytes()).decode(),text=True,capture_output=True,timeout=180)
(OUT/'targeted_stdout.txt').write_text(r.stdout)
(OUT/'targeted_stderr.txt').write_text(r.stderr)
if r.returncode: print(json.dumps({'exit':r.returncode,'stderr':r.stderr,'stdout_tail':r.stdout[-1200:]}))
else:
    d=json.loads(r.stdout.strip().splitlines()[-1]); (OUT/'targeted_verification.json').write_text(json.dumps(d,indent=2,ensure_ascii=False))
    print(json.dumps(d,ensure_ascii=False))
