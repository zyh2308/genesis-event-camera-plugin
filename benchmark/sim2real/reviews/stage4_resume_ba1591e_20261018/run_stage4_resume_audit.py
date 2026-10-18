from pathlib import Path
import subprocess,base64,json,shlex
import sys
root=Path(__file__).parent
details='--details' in sys.argv
remote=root/('stage4_resume_details_remote.py' if details else 'stage4_resume_audit_remote.py')
command=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['wsl','-d','Ubuntu-22.04','--','ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','zhaoyuhan@g1.pkuml.org',command],input=base64.b64encode(remote.read_bytes()).decode(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=170)
stem='stage4_resume_details' if details else 'stage4_resume_audit'
(root/(stem+'.json')).write_text(r.stdout,encoding='utf-8')
(root/(stem+'.stderr.txt')).write_text(r.stderr,encoding='utf-8')
if r.returncode: print(r.stderr);raise SystemExit(r.returncode)
d=json.loads(r.stdout)
if details:
    print(json.dumps(d,indent=2));raise SystemExit(0)
print(json.dumps({'observed_utc':d['observed_utc'],'arms':{k:{'updates':v['latest_update']['optimizer_update'],'validation_epochs':[x['epoch'] for x in v['validation']],'bad_updates':v['bad_numeric_updates'],'gaps':v['update_gaps'],'process':v['process']} for k,v in d['arms'].items()},'pairing':d['paired_input_comparison'],'gpus':d['gpus'],'exit_records':d['exit_records']},indent=2))
