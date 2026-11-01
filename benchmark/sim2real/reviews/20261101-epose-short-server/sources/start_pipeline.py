import os,json,subprocess,hashlib,fcntl
from pathlib import Path
from datetime import datetime,timezone
R=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-epose-wood-short-research-v1');S=R/'sources'
for n in ['run_v2e_bank.py','train_matched_pair.py','evaluate_real_epose.py','continue_pipeline.py']:assert (S/n).is_file()
cmd=f'''#!/usr/bin/env bash
export EPOSE_SHORT_RUN='{R}'
export CUDA_VISIBLE_DEVICES=GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH='{S}:{S}/benchmark/sim2real/scripts'
exec /home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python -u '{S}/continue_pipeline.py' > '{R}/pipeline.stdout.log' 2> '{R}/pipeline.stderr.log'
'''
(R/'continue.sh').write_text(cmd);subprocess.run(['tmux','new-session','-d','-s','sim2real_epose_short_pair_20261101',f'bash {R}/continue.sh'],check=True)
pid=int(subprocess.check_output(['tmux','display-message','-p','-t','sim2real_epose_short_pair_20261101','#{pane_pid}'],text=True))
reg=Path(json.loads((R/'reservation_paths.json').read_text())['registry'])
with (reg.parent/'.gpu_reservations.lock').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);d=json.loads(reg.read_text());assert d['run_id']==R.name and d['status']=='reserved';d.update({'pid':pid,'status':'running','phase':'generation -> paired V2E -> short train -> fixed real test'});reg.write_text(json.dumps(d,indent=2))
(R/'pipeline_pid.txt').write_text(str(pid));print(json.dumps({'started':True,'pid':pid,'tmux':'sim2real_epose_short_pair_20261101','run':str(R)}))
