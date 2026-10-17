from pathlib import Path
import base64,json,shlex,subprocess
O=Path(__file__).resolve().parent
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-p','22','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode((O/'remote_segmentation_program.py').read_bytes()).decode(),capture_output=True,text=True,timeout=90)
(O/'segmentation_verification.json').write_text(r.stdout);(O/'segmentation_stderr.txt').write_text(r.stderr)
if r.returncode:print(r.stderr)
else:
    d=json.loads(r.stdout);print(json.dumps({'controlled_label_checks':d['controlled_label_checks'],'renderer_source_hits':d['renderer_source_hits'][:20]},ensure_ascii=False))
