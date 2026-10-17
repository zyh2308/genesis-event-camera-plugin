from pathlib import Path
import base64,json,shlex,subprocess
code=r'''
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import json,math,subprocess
r=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/flow_dt1_seed0')
out={'checked_at_JST':datetime.now(timezone.utc).astimezone(ZoneInfo('Asia/Tokyo')).isoformat(),'arms':{}}
for gen in ['eventgenesis','v2e']:
    rows=[]
    for line in (r/gen/'training_metrics.jsonl').read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    updates=[x for x in rows if 'optimizer_update' in x];vals=[x for x in rows if x.get('event')=='synthetic_validation'];last=updates[-1]
    out['arms'][gen]={'update':last['optimizer_update'],'epoch':last['epoch'],'updated_utc':last['updated_utc'],'budget':8910,'completed_validation_epochs':[v['epoch'] for v in vals],'loss_finite':all(math.isfinite(x['loss']) for x in updates),'checkpoint20_exists':(r/gen/'update_000020.pt').is_file(),'stderr_tail':(r/gen/'train.stderr.log').read_text(errors='replace')[-600:]}
out['processes']=subprocess.run(['pgrep','-af','[t]rain_steflow_stage4.py'],capture_output=True,text=True).stdout.strip()
out['GPU']=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True).stdout.strip()
print(json.dumps(out,ensure_ascii=False))
'''
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-p','22','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode(code.encode()).decode(),capture_output=True,text=True,timeout=35)
print(p.stdout);print(p.stderr)
if p.returncode==0:
    (Path(__file__).resolve().parent/'latest_progress_refresh.json').write_text(p.stdout)
