from pathlib import Path
import subprocess,base64,shlex,json
L=Path(__file__).parent
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['wsl','-d','Ubuntu-22.04','--','ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode((L/'stage5_review_remote.py').read_bytes()).decode(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=160)
(L/'stage5_review_remote.json').write_text(r.stdout,encoding='utf-8');(L/'stage5_review_remote.stderr.txt').write_text(r.stderr,encoding='utf-8')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
d=json.loads(r.stdout)
print(json.dumps({'observed_utc':d['observed_utc'],'arms':{k:{'count':v['count'],'last':v['last'],'validation_count':v['validation_count'],'last_validation':v['last_validation'],'gaps':v['gaps'],'bad_updates':v['bad_updates']} for k,v in d['arms'].items()},'pairing':d['pairing'],'gpus':d['gpus'],'coverage':d['coverage'],'exec_head':d['exec_head'],'exec_status':d['exec_status'],'tc_test_rc':d['tc_geometry_rerun']['rc'],'independent_tc':d['independent_tc_closed_form']},ensure_ascii=False,indent=2))
