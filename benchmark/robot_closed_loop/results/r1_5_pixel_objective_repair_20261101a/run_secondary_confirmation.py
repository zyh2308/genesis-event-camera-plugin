"""Frozen-model confirmation on three archived episodes outside the fit/primary bank."""
from pathlib import Path
import os,json,subprocess,hashlib,datetime,csv,time

ROOT=Path(__file__).resolve().parent
BASE=Path('/home/zhaoyuhan/Eventbased_WAM')
SOURCE=ROOT/'source'
PRIOR=BASE/'runs/robot_r1_5_research_repair_demo_20261101a'
REF=BASE/'repositories/genesis-event-camera-plugin-r0_7-heatmap-20261017/benchmark/robot_closed_loop/results/r0_7_heatmap_event_state_closed_loop_3seed_20261017'
PY=BASE/'repositories/gec-robot/.venv-robot/bin/python'
OUT=ROOT/'secondary_confirmation'
ENV={**os.environ,'PYTHONPATH':str(SOURCE),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','ROBOT_EXECUTION_CODE_COMMIT':'403426700ff279555ddbb060cec8c43094e693f2'}
RECORD=BASE/'runs/robot_scenes/.resource_allocations'/(ROOT.name+'_secondary.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def update(**kw):
    d=json.loads(RECORD.read_text());d.update(kw);dump(RECORD,d)

code=1;started=time.perf_counter()
try:
    OUT.mkdir(exist_ok=False)
    primary=json.loads((ROOT/'summary.json').read_text())
    reference=json.loads((REF/'r0_5_reference_summary.json').read_text())
    source_seeds=primary['training']['training_source_seeds']
    old_test_seeds=[e['seed'] for e in primary['reference_provenance']]
    seeds=[e['seed'] for e in reference['episodes']]
    assert seeds==[2026103900,2026103901,2026103902]
    assert not set(seeds)&set(source_seeds+old_test_seeds)
    checkpoints={
        'old_ours':BASE/'repositories/genesis-event-camera-plugin-r1-nativehdr-20261019a/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/ours_fit/spatial_event_state.pt',
        'coverage_only':next(p for p in (BASE/'runs/robot_r1_5_coverage_candidate_retrain_20261101a').rglob('spatial_event_state.pt') if sha(p)==primary['models']['coverage_only']['checkpoint_sha256']),
        'observed_candidate':PRIOR/'observed_fit/spatial_event_state.pt',
    }
    v2e=BASE/'repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/paired_r1/v2e_fit/spatial_event_state.pt'
    registration={'secondary_episode_seeds':seeds,'training_or_primary_episode_overlap':[],
        'checkpoint_sha256':{k:sha(v) for k,v in checkpoints.items()},'v2e_checkpoint_sha256':sha(v2e),
        'reference_manifest_sha256':sha(REF/'SHA256SUMS'),
        'scope':'archived episodes newly evaluated by these frozen checkpoints; historical development episodes, not a freshly collected formal test',
        'checkpoint_selection':'fixed by training validation before secondary evaluation; no training, fitting, tuning or selection on this bank',
        'decision':'confirm median/P95 and >10px failures improve over old Ours, with all-poll denominators',
        'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    for k,cp in checkpoints.items():assert sha(cp)==primary['models'][k]['checkpoint_sha256']
    dump(OUT/'registration.json',registration)
    update(status='running',pid=os.getpid(),started_utc=registration['registered_utc'])
    cfg=SOURCE/'benchmark/robot_closed_loop/configs'
    decel=json.loads((cfg/'scene_pilot_r0_5_candidate.json').read_text())['controller']['initial_deceleration_assumption_m_s2']
    summaries={};control=None;all_rows=[]
    for name,cp in checkpoints.items():
        dest=OUT/name
        cmd=[str(PY),'-u','-m','benchmark.robot_closed_loop.diagnostics.evaluate_r1_common_direct_hdr','--scene-config',str(cfg/'scene_pilot_r0_5_candidate.json'),'--control-config',str(cfg/'shared_control_r0_8_candidate.json'),'--reference-root',str(REF),'--ours-checkpoint',str(cp),'--v2e-checkpoint',str(v2e),'--output',str(dest),'--deceleration',str(decel)]
        dump(OUT/(name+'_command.json'),cmd)
        with (OUT/(name+'.log')).open('w') as log:
            p=subprocess.Popen(cmd,cwd=SOURCE,env=ENV,stdout=log,stderr=subprocess.STDOUT);update(stage=name,child_pid=p.pid);rc=p.wait()
        (OUT/(name+'.exit')).write_text(str(rc))
        if rc:raise RuntimeError(name+' evaluation failed')
        j=json.loads((dest/'r1_common_direct_hdr_summary.json').read_text());summaries[name]=j
        if control is None:control=j['overall']['v2e']
        else:assert control==j['overall']['v2e']
        with (dest/'r1_common_direct_hdr_polls.csv').open(newline='') as f:
            current=list(csv.DictReader(f))
        for r in current:
            if r['source']=='ours':r['method']=name;all_rows.append(r)
        if name=='observed_candidate':
            for r in current:
                if r['source']=='v2e':r['method']='v2e';all_rows.append(r)
        print(name,j['overall']['ours']['metrics']['pixel_euclidean_error_px'],flush=True)
    models={name:{'pixel':j['overall']['ours']['metrics']['pixel_euclidean_error_px'],
            'failures':j['overall']['ours']['pixel_localization_failures'],
            'world_xy_m':j['overall']['ours']['metrics']['position_xy_error_m'],
            'by_episode':{k:e['ours']['metrics']['pixel_euclidean_error_px'] for k,e in j['by_episode'].items()}}
            for name,j in summaries.items()}
    models['v2e']={'pixel':control['metrics']['pixel_euclidean_error_px'],'failures':control['pixel_localization_failures'],
        'world_xy_m':control['metrics']['position_xy_error_m'],
        'by_episode':{k:e['v2e']['metrics']['pixel_euclidean_error_px'] for k,e in summaries['observed_candidate']['by_episode'].items()}}
    with (OUT/'all_models_pixel_polls.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    new,old=models['observed_candidate'],models['old_ours']
    decision={k:new['pixel'][k]<old['pixel'][k] for k in ['p50','p95']}
    decision['fewer_over_10px_or_missing']=new['failures']['thresholds_px']['10.0']['failed_or_missing']<old['failures']['thresholds_px']['10.0']['failed_or_missing']
    dump(OUT/'summary.json',{'status':'PASS','models':models,'registration':registration,'fixed_v2e_control_identical':True,'acceptance':decision,'wall_s':time.perf_counter()-started})
    files=[p for p in OUT.rglob('*') if p.is_file()]
    (OUT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(OUT).as_posix()+'\n' for p in sorted(files)))
    code=0
finally:
    (OUT/'pipeline.exit').write_text(str(code))
    update(status='completed' if code==0 else 'failed',exit_code=code,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
