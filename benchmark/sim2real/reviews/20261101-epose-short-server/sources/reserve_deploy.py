import os,json,socket,fcntl,subprocess,time,tarfile
from pathlib import Path
from datetime import datetime,timezone
P=Path('/home/zhaoyuhan/Eventbased_WAM');RUN='20261101-epose-wood-short-research-v1';R=P/'runs/sim2real'/RUN
UUID='GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460';reg=P/'.gpu_reservations';reg.mkdir(exist_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
with (reg/'.gpu_reservations.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for p in reg.glob('*.json'):
  d=json.loads(p.read_text());s=str(d.get('status','')).lower()
  if d.get('gpu_uuid')==UUID and not (s.startswith('released') or d.get('ended_at_utc') or d.get('finished_utc')):raise RuntimeError(f'GPU already reserved: {p}')
 row=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
 if int(row[0])>128 or int(row[1])>5:raise RuntimeError(f'GPU not idle: {row}')
 R.mkdir(exist_ok=False)
 legacy=P/'runs/sim2real/resource_reservations'/f'g1_{UUID}.lock';legacy.mkdir(parents=True,exist_ok=False)
 d={'schema':'sim2real_gpu_reservation_v1','status':'reserved','run_id':RUN,'owner':'Sim-to-Real research chat; user explicitly requested server short training + real E-POSE pilot','host':socket.gethostname(),'gpu_index':1,'gpu_uuid':UUID,'created_utc':now(),'run_path':str(R),'pid':None,'cpu_threads':2,'phase':'preparing matched CAD synthetic bank'}
 (reg/f'{RUN}.json').write_text(json.dumps(d,indent=2));(legacy/'resource_reservation.json').write_text(json.dumps(d,indent=2));(R/'resource_reservation.json').write_text(json.dumps(d,indent=2))
 (R/'reservation_paths.json').write_text(json.dumps({'registry':str(reg/f'{RUN}.json'),'legacy_lock':str(legacy)},indent=2))
 (R/'sources').mkdir()
print(json.dumps({'run':str(R),'reservation':d,'python':'/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python','genesis_runtime':str(P/'runtime/sim2real/genesis_hdr_v2')},indent=2))
