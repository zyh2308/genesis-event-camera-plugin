from pathlib import Path
import os,sys,subprocess,json,time,datetime
root=Path(__file__).resolve().parent
source=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_sparse_cuda_acceptance_20261102c')
accepted=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_r1_5_sparse_time_repair_20261101a')
reservation=Path('/home/zhaoyuhan/Eventbased_WAM/.gpu_reservations/robot_cuda_r2_server_20261102a.json')
python='/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python'
stages=[('cuda_controls',[python,'-u',str(source/'validate_cuda.py'),'--devices','cpu,cuda:0','--output',str(root/'cuda_controls')]),
 ('cuda_cache',[python,'-u',str(source/'replay_sparse_cache.py'),'--devices','cpu,cuda:0','--input',str(source/'cpu_cache_replay/sparse_endpoint_blocks.npz'),'--output',str(root/'cuda_cache')]),
 ('pixel_equivalence',[python,'-u',str(source/'local_cuda_20261102a/check_pixel_equivalence.py'),'--replay',str(root/'cuda_cache'),
  '--model-source',str(accepted/'source/benchmark/robot_closed_loop/event_state_model.py'),'--checkpoint',str(accepted/'fit/spatial_event_state.pt'),
  '--input',str(source/'cpu_cache_replay/sparse_endpoint_blocks.npz')])]
report={'code_commit':'6ac417c7a03e125f40ebf967796bf28191d55859','stages':[],'status':'running'}
try:
 for name,command in stages:
  started=time.monotonic();report['stage']=name
  (root/'progress.json').write_text(json.dumps(report,indent=2))
  with (root/(name+'.log')).open('w') as log:
   proc=subprocess.Popen(command,env=os.environ,cwd=source,stdout=log,stderr=subprocess.STDOUT)
   report['child_pid']=proc.pid;(root/'progress.json').write_text(json.dumps(report,indent=2))
   code=proc.wait()
  report['stages'].append(dict(name=name,command=command,exit_code=code,wall_s=time.monotonic()-started))
  (root/(name+'.exit')).write_text(str(code)+chr(10))
  if code:raise RuntimeError(name+' failed: '+str(code))
 report['status']='cuda_validation_passed_R2_pending'
except Exception as exc:
 report['status']='failed';report['error']=repr(exc)
finally:
 report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (root/'progress.json').write_text(json.dumps(report,indent=2))
 record=json.loads(reservation.read_text());record.update(stage=report['status'],validation_pid=os.getpid())
 # Retain our explicitly recorded GPU reservation for the immediate R2 phase.
 if report['status']=='failed':record['status']='released_validation_failed'
 reservation.write_text(json.dumps(record,indent=2))
