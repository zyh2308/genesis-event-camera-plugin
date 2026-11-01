from pathlib import Path
import os,sys,json,subprocess,time,datetime,hashlib,csv,concurrent.futures
ROOT=Path(__file__).resolve().parent
BASE=Path('/home/zhaoyuhan/Eventbased_WAM');SOURCE=ROOT/'source';CORE=ROOT/'core'
PY=BASE/'repositories/gec-robot/.venv-robot/bin/python'
CFG=SOURCE/'benchmark/robot_closed_loop/configs'
RECORD=BASE/'runs/robot_scenes/.resource_allocations'/(ROOT.name+'.json')
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
     'PYTHONPATH':str(SOURCE)+':'+str(CORE),'PYTHONUNBUFFERED':'1',
     'ROBOT_EXECUTION_CODE_COMMIT':'sparse-time-candidate-v1; exact hashes in SOURCE_SHA256SUMS'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def status(stage,**kw):
    d=json.loads(RECORD.read_text());d.update(status='running',stage=stage,pid=os.getpid(),**kw);dump(RECORD,d)
def execute(name,command,threads=1):
    dump(ROOT/(name+'_command.json'),list(map(str,command)))
    env={**ENV,'OMP_NUM_THREADS':str(threads),'MKL_NUM_THREADS':str(threads),'OPENBLAS_NUM_THREADS':str(threads)}
    with (ROOT/(name+'.log')).open('w') as log:
        p=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        rc=p.wait()
    (ROOT/(name+'.exit')).write_text(str(rc))
    if rc:raise RuntimeError(name+' failed; see its log')
def gen(i):
    execute('generate_%03d'%i,[PY,'-u',ROOT/'generate_sparse.py',str(i)])
    return json.loads((ROOT/'sparse_events/episodes'/('episode_%03d'%i)/'complete.json').read_text())

code=1;started=time.perf_counter()
try:
    status('controls')
    execute('sparse_controls',[PY,ROOT/'check_sparse.py'])
    execute('pixel_controls',[PY,SOURCE/'benchmark/robot_closed_loop/diagnostics/check_pixel_localization_metrics.py'])
    # One complete real cache episode first, to catch input/API failures cheaply.
    status('sparse_generation')
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(gen,range(9)):
            results.append(result);print(json.dumps(result),flush=True)
    dump(ROOT/'generation_summary.json',dict(episodes=results,renderer_calls=0,dense_HDR_used=False,
        sparse_public_geometry_path=True,wall_s=time.perf_counter()-started))
    data=ROOT/'sparse_events'
    (data/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(data).as_posix()+'\n' for p in sorted(data.rglob('*')) if p.is_file()))
    status('same_budget_training')
    execute('fit',[PY,'-u','-m','benchmark.robot_closed_loop.diagnostics.train_event_state_development',
        '--config',CFG/'scene_pilot_r0_5_candidate.json','--source-results',data,'--out',ROOT/'fit',
        '--train-episodes','0,1,2,3,7,8','--validation-episodes','4,5,6','--epochs','40','--patience','40',
        '--batch-size','32','--seed','606','--threads','2','--architecture','heatmap_v2',
        '--source-label','100Hz native HDR endpoint-only public geometric reconstruction, coverage/time log fusion'],threads=2)
    checkpoint=ROOT/'fit/spatial_event_state.pt'
    # Checkpoint chosen solely on training-bank validation, before any reference replay.
    dump(ROOT/'checkpoint_frozen_before_evaluation.json',dict(sha256=sha(checkpoint),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),train=[0,1,2,3,7,8],validation=[4,5,6]))
    status('pixel_evaluation')
    v2e=BASE/'repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/paired_r1/v2e_fit/spatial_event_state.pt'
    refs={
      'primary':BASE/'repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016',
      'secondary':BASE/'repositories/genesis-event-camera-plugin-r0_7-heatmap-20261017/benchmark/robot_closed_loop/results/r0_7_heatmap_event_state_closed_loop_3seed_20261017'}
    baseline_root=BASE/'runs/robot_r1_5_pixel_objective_20261101a'
    baseline_files={'primary':baseline_root/'summary.json','secondary':baseline_root/'secondary_confirmation/summary.json'}
    decel=json.loads((CFG/'scene_pilot_r0_5_candidate.json').read_text())['controller']['initial_deceleration_assumption_m_s2']
    summaries={};decisions={}
    for name,ref in refs.items():
        dest=ROOT/'evaluation'/name
        execute(name+'_eval',[PY,'-u','-m','benchmark.robot_closed_loop.diagnostics.evaluate_r1_common_direct_hdr',
            '--scene-config',CFG/'scene_pilot_r0_5_candidate.json','--control-config',CFG/'shared_control_r0_8_candidate.json',
            '--reference-root',ref,'--ours-checkpoint',checkpoint,'--v2e-checkpoint',v2e,'--output',dest,'--deceleration',str(decel)],threads=2)
        j=json.loads((dest/'r1_common_direct_hdr_summary.json').read_text());baseline=json.loads(baseline_files[name].read_text())
        current=j['overall']['ours'];old=baseline['models']['old_ours']
        decision={key:current['metrics']['pixel_euclidean_error_px'][key]<old['pixel'][key] for key in ['p50','p95']}
        decision['fewer_over_10px_or_missing']=current['pixel_localization_failures']['thresholds_px']['10.0']['failed_or_missing']<old['failures']['thresholds_px']['10.0']['failed_or_missing']
        summaries[name]=dict(candidate=current,old=old,v2e=j['overall']['v2e'],by_episode=j['by_episode'],
            reference_manifest_sha256=sha(ref/'SHA256SUMS'),checkpoint_sha256=sha(checkpoint))
        decisions[name]=decision
        print(json.dumps(dict(bank=name,pixel=current['metrics']['pixel_euclidean_error_px'],acceptance=decision)),flush=True)
    passed=all(all(d.values()) for d in decisions.values())
    dump(ROOT/'summary.json',dict(status='PASS' if passed else 'FAILED_CANDIDATE',
        purpose='Repair sparse reconstruction; localization pixels are acceptance, not image similarity',
        dense_HDR_used_in_candidate=False,acceptance=decisions,results=summaries,
        checkpoint_sha256=sha(checkpoint),wall_s=time.perf_counter()-started))
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name not in ['worker.log','SHA256SUMS','pipeline.exit'] and 'source' not in p.relative_to(ROOT).parts and 'core' not in p.relative_to(ROOT).parts]
    (ROOT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in sorted(files)))
    code=0
except Exception as exc:
    print(type(exc).__name__,str(exc),flush=True);raise
finally:
    (ROOT/'pipeline.exit').write_text(str(code))
    d=json.loads(RECORD.read_text());d.update(status='completed' if code==0 else 'failed',exit_code=code,
        ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());dump(RECORD,d)
