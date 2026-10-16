from pathlib import Path
import base64,json,shlex,subprocess
O=Path(__file__).resolve().parent
code='import base64,sys;exec(compile(base64.b64decode(sys.stdin.buffer.read()),"<stage3-review>","exec"))'
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c',code])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-p','22','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode((O/'remote_program.py').read_bytes()).decode(),capture_output=True,text=True,timeout=240)
(O/'remote_audit_stdout.json').write_text(r.stdout);(O/'remote_audit_stderr.txt').write_text(r.stderr)
if r.returncode:print(json.dumps({'exit':r.returncode,'stderr':r.stderr,'stdout_tail':r.stdout[-1000:]}))
else:
    d=json.loads(r.stdout);print(json.dumps({'host':d['host'],'head':d['head'],'status':d['status'],'data_count':len(d['data']),'data_pass':all(v['sha_match'] and v['size_match'] for v in d['data']),'checkpoint_count':len(d['checkpoints']),'checkpoints_pass':all(x['sha_match'] and x['size_match'] and x['parameters_finite'] for x in d['checkpoints']),'manifest_checks':[{'task':x['task'],'sha_match':x['sha_match'],'size_match':x['size_match']} for x in d['inference_manifests']],'mvsec_output_count':sum(x['count'] for x in d['MVSEC_saved_outputs'])},ensure_ascii=False))
