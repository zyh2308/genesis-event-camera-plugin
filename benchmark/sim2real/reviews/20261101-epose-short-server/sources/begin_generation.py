import os,json,subprocess,tarfile,hashlib
from pathlib import Path
from datetime import datetime,timezone
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261101-epose-wood-short-research-v1';S=R/'sources'
with tarfile.open(S/'source_base.tar') as t:
 for m in t.getmembers():
  assert not m.name.startswith('/') and '..' not in Path(m.name).parts
 t.extractall(S)
config=json.loads((S/'benchmark/sim2real/configs/pose_stage4_seed0_eval_stable.json').read_text());config['training']['windows_per_physical_rollout']=4;config['scope']='exact E-POSE wood-block paired synthetic-only short training, fixed final checkpoint';(R/'short_config.json').write_text(json.dumps(config,indent=2))
protocol={'run_id':R.name,'scope':'server short training and first fixed-sequence E-POSE real pilot; NOT full E-POSE benchmark','frozen_before_training_utc':datetime.now(timezone.utc).isoformat(),'execution_host':'g1','device':'GPU 1 RTX3090','epochs':31,'train_rollouts':4,'validation_rollouts':2,'windows_per_rollout':4,'optimizer_updates_per_arm':496,'confidence_supervised_updates_per_arm':256,'batch_size':1,'seed':0,'initialization':'identical random seed; no real pretrained weights','checkpoint_selection':'fixed final update496; no real or synthetic-best selection','input':'full-frame event-derived ON/OFF/total log counts; no RGB, GT mask/crop/box/pose at inference','real_sequence':'block/r_01_gl','real_sample_end_indices':list(range(3,603,3)),'real_window':'each MAT raw event window only, approximately10ms; no union of adjacent windows','real_coordinates':'raw events undistorted with official K and distortion, then binned; PnP uses ideal K','distortion':[-.362,.115,.0009576,.00000455,.130],'keypoints':'original CAD AABB centroid then eight lexicographic min/max corners; not legacy stored UV','pnp':'fixed OpenCV SOLVEPNP_EPNP, eight corners, no GT choice of cells/solutions','primary_metric':'ADD-S mean and success at0.1*exact mesh diameter; all200 denominator, PnP failures count as unsuccessful','other_metrics':['ADD','translation error','raw (not symmetry adjusted) rotation error','2D corner EPE'],'metric_scope':'derived corrected geometry pilot; not a claimed reproduction of official full E-POSE leaderboard protocol','training_gt':'shared synthetic physical poses only','real_data_training_or_selection':False,'core':'e79e656ac00f7847b98db9e12bf43b093f3afff1','base_commit':'7b8a6d4','deadline_policy':'generation600s; each V2E episode180s; training1200s; real eval300s; finite budgets, failures retained'}
(R/'protocol.json').write_text(json.dumps(protocol,indent=2))
cmd=f'''#!/usr/bin/env bash
set +e
export EPOSE_SHORT_RUN='{R}'
export CUDA_VISIBLE_DEVICES=GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460
export EGL_DEVICE_ID=1
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export QD_NUM_THREADS=2 QD_OFFLINE_CACHE=1 QD_OFFLINE_CACHE_FILE_PATH='{R}/cache/quadrants'
export GS_CACHE_FILE_PATH='{R}/cache/genesis'
export PYTHONPATH='{P}/runtime/sim2real/genesis_hdr_v2:{S}:{S}/benchmark/sim2real/scripts'
date -u +%FT%TZ > '{R}/generation_started_utc.txt'
timeout 600s /home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python -u '{S}/generate_matched_bank.py' > '{R}/generation.stdout.log' 2> '{R}/generation.stderr.log'
echo $? > '{R}/generation_exit_code.txt'
date -u +%FT%TZ > '{R}/generation_finished_utc.txt'
'''
(R/'generate.sh').write_text(cmd)
resources=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);assert all(int(x.strip())<10 for x in resources.split(',')),resources
subprocess.run(['tmux','new-session','-d','-s','sim2real_epose_short_gen_20261101',f"bash {R}/generate.sh"],check=True)
print(json.dumps({'started':'generator','run':str(R),'tmux':'sim2real_epose_short_gen_20261101','source_base_sha256':hashlib.sha256((S/'source_base.tar').read_bytes()).hexdigest(),'prelaunch_gpu':resources}))
