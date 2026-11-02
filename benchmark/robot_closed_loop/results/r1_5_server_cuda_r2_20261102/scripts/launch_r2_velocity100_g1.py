"""One matched development rollout: only causal velocity window changes."""
import pathlib,json,os,signal,datetime,fcntl,subprocess,time,hashlib,xml.etree.ElementTree as ET
base=pathlib.Path('/home/zhaoyuhan/Eventbased_WAM');registry=base/'.gpu_reservations'
holderroot=base/'runs/robot_gpu2_standby_20261102a';recordpath=registry/(holderroot.name+'.json')
root=base/'runs/robot_R2_velocity100_20261102c';uuid='GPU-a02affe0-7598-579e-ca39-f551f35ad6e9'
original=base/'runs/robot_cuda_r2_server_20261102a'
def gpu():
 return next(g for g in ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],universal_newlines=True,timeout=15)).findall('gpu') if g.findtext('uuid')==uuid)
with (registry/'.gpu_reservations.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);record=json.loads(recordpath.read_text())
 assert record['status']=='standby' and record['last_task_result']['status']=='cuda_validation_passed_R2_pending'
 pid=record['pid'];cmd=(pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes().decode()
 assert str(holderroot/'hold_gpu.py') in cmd
 assert [int(p.findtext('pid')) for p in gpu().findall('processes/process_info')]==[pid]
 assert not root.exists();root.mkdir()
 controlpath=base/'runs/robot_r1_5_sparse_time_repair_20261101a/source/benchmark/robot_closed_loop/configs/shared_control_r0_8_monitor60_diagnostic.json'
 control=json.loads(controlpath.read_text());old=control['close_time_estimator']['velocity_window_s'];assert old==.05
 control['close_time_estimator']['velocity_window_s']=.1
 newcontrol=root/'shared_control_velocity100ms.json';newcontrol.write_text(json.dumps(control,indent=2))
 plan=json.loads((original/'R2_plan.json').read_text());command=plan['command']
 def setarg(flag,value):command[command.index(flag)+1]=str(value)
 setarg('--control-config',newcontrol);setarg('--out',root/'R2_velocity100ms')
 setarg('--speeds','.3')
 command[command.index('--reference-y-values=0,-0.055,0.055')]='--reference-y-values=0'
 plan.update(command=command,control_sha256=hashlib.sha256(newcontrol.read_bytes()).hexdigest(),
  conditions=[dict(seed=2026110200,speed_m_s=.3,y_m=0.)],controller_unchanged=False,
  development_only=True,only_change={'parameter':'close_time_estimator.velocity_window_s','before':old,'after':.1},
  original_control=str(controlpath),original_control_sha256=hashlib.sha256(controlpath.read_bytes()).hexdigest(),
  rationale='Completed episode causal replay showed lower velocity error with 100ms history; test action consequence without changing core, model, sensor, or strict score')
 (root/'R2_plan.json').write_text(json.dumps(plan,indent=2))
 runner=(original/'run_R2.py').read_text().replace("robot_cuda_r2_server_20261102a.json",holderroot.name+'.json')
 start=runner.index(" record=json.loads(recordpath.read_text());record.update")
 runner=runner[:start]+''' with (recordpath.parent/'.gpu_reservations.lock').open('a+') as lock:
  import fcntl
  fcntl.flock(lock,fcntl.LOCK_EX)
  record=json.loads(recordpath.read_text());record.update(last_task_result=report,last_task_path=str(root),status='starting_standby')
  holderroot=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_gpu2_standby_20261102a')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-a02affe0-7598-579e-ca39-f551f35ad6e9')
  with (holderroot/'holder_after_R2.log').open('w') as log:
   p=subprocess.Popen(['/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-u',str(holderroot/'hold_gpu.py')],cwd=holderroot,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  record.update(status='standby',pid=p.pid);recordpath.write_text(json.dumps(record,indent=2))
'''
 compile(runner,'run_R2.py','exec');(root/'run_R2.py').write_text(runner)
 record.update(status='deploying_task',task_run_id=root.name,task_path=str(root),holder_pid_before_task=pid)
 recordpath.write_text(json.dumps(record,indent=2));os.kill(pid,signal.SIGTERM)
for _ in range(100):
 p=pathlib.Path('/proc')/str(pid)
 if not p.exists() or 'State:\tZ' in (p/'status').read_text():break
 time.sleep(.1)
else:raise RuntimeError('Own holder did not exit; task not launched')
with (registry/'.gpu_reservations.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not gpu().findall('processes/process_info')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EGL_DEVICE_ID='2',PYOPENGL_PLATFORM='egl',
  PYTHONPATH=plan['core_source']+':'+plan['benchmark_source'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
  ROBOT_EXECUTION_CODE_COMMIT=plan['code_commit'])
 with (root/'launcher.log').open('w') as log:
  p=subprocess.Popen([command[0],'-u',str(root/'run_R2.py')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record.update(status='running',pid=p.pid,task_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='One matched .3m/s development R2 rollout; 100ms velocity window only; restore standby afterward')
 recordpath.write_text(json.dumps(record,indent=2))
print(json.dumps({'task':root.name,'pid':p.pid,'GPU':2,'control_sha256':plan['control_sha256']}))
