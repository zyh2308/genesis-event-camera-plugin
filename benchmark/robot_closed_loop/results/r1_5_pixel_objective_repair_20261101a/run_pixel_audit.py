"""Replay fixed checkpoints on identical cached inputs; no training or rendering."""
from pathlib import Path
import os, sys, subprocess, json, csv, hashlib, datetime, time

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/zhaoyuhan/Eventbased_WAM')
PRIOR = BASE/'runs/robot_r1_5_research_repair_demo_20261101a'
SOURCE = ROOT/'source'
PY = BASE/'repositories/gec-robot/.venv-robot/bin/python'
REF = BASE/'repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016'
CFG = SOURCE/'benchmark/robot_closed_loop/configs'
CODE = '403426700ff279555ddbb060cec8c43094e693f2'
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2',
       'OPENBLAS_NUM_THREADS':'2', 'PYTHONPATH':str(SOURCE), 'PYTHONUNBUFFERED':'1',
       'ROBOT_EXECUTION_CODE_COMMIT':CODE}
RECORD = BASE/'runs/robot_scenes/.resource_allocations'/(ROOT.name+'.json')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d): Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def update(**kw):
    d=json.loads(RECORD.read_text());d.update(kw);dump(RECORD,d)
def execute(name,cmd):
    dump(ROOT/(name+'_command.json'),cmd)
    with (ROOT/(name+'.log')).open('w') as log:
        child=subprocess.Popen(cmd,cwd=SOURCE,env=ENV,stdout=log,stderr=subprocess.STDOUT)
        update(stage=name,child_pid=child.pid)
        rc=child.wait()
    (ROOT/(name+'.exit')).write_text(str(rc))
    if rc: raise RuntimeError(name+' failed; see log')
def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def key(r):return r['source'],int(r['episode_id']),round(float(r['poll_s']),9)


code=1;started=time.perf_counter()
try:
    update(status='running',pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    assert sha(REF/'SHA256SUMS')=='86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c'
    execute('unit_controls',[str(PY),str(SOURCE/'benchmark/robot_closed_loop/diagnostics/check_pixel_localization_metrics.py')])
    v2e=BASE/'repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/paired_r1/v2e_fit/spatial_event_state.pt'
    old=BASE/'repositories/genesis-event-camera-plugin-r1-nativehdr-20261019a/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/ours_fit/spatial_event_state.pt'
    coverage_root=BASE/'runs/robot_r1_5_coverage_candidate_retrain_20261101a'
    coverage=[p for p in coverage_root.rglob('spatial_event_state.pt') if sha(p)=='eb671a19c70bb9f004cc4c630764f497d6c2ed9b4361ef7904665fddce3cd796']
    assert len(coverage)==1,coverage
    methods={'old_ours':old,'coverage_only':coverage[0],'observed_candidate':PRIOR/'observed_fit/spatial_event_state.pt'}
    decel=json.loads((CFG/'scene_pilot_r0_5_candidate.json').read_text())['controller']['initial_deceleration_assumption_m_s2']
    outputs={};all_rows=[];provenance=None;v2e_reference=None;audits={}
    for name,cp in methods.items():
        dest=ROOT/'evaluation'/name
        execute(name,[str(PY),'-u','-m','benchmark.robot_closed_loop.diagnostics.evaluate_r1_common_direct_hdr',
                      '--scene-config',str(CFG/'scene_pilot_r0_5_candidate.json'),'--control-config',str(CFG/'shared_control_r0_8_candidate.json'),
                      '--reference-root',str(REF),'--ours-checkpoint',str(cp),'--v2e-checkpoint',str(v2e),'--output',str(dest),'--deceleration',str(decel)])
        summary=json.loads((dest/'r1_common_direct_hdr_summary.json').read_text())
        current=rows(dest/'r1_common_direct_hdr_polls.csv')
        assert len(current)==1162 and summary['overall']['ours']['poll_count']==581
        if provenance is None:provenance=summary['provenance']
        else:assert summary['provenance']==provenance
        control={key(r):r for r in current if r['source']=='v2e'}
        if v2e_reference is None:v2e_reference=control
        else:
            assert control.keys()==v2e_reference.keys()
            for k,r in control.items():
                for metric in ['position_xy_error_m','pixel_euclidean_error_px']:
                    assert abs(float(r[metric])-float(v2e_reference[k][metric]))<1e-10
        historical_name={'old_ours':'old_native_model','observed_candidate':'new_observed_model'}.get(name)
        if historical_name:
            old_rows={key(r):r for r in rows(PRIOR/'common_reference_eval'/historical_name/'r1_common_direct_hdr_polls.csv')}
            assert old_rows.keys()=={key(r) for r in current}
            worst=0.
            for r in current:
                for metric in ['position_xy_error_m','velocity_xy_error_m_s']:
                    delta=abs(float(r[metric])-float(old_rows[key(r)][metric]));worst=max(worst,delta)
                    assert delta<1e-7,(name,key(r),metric,delta)
            audits[name]={'historical_world_scores_reproduced':True,'maximum_delta_m_or_m_per_s':worst}
        outputs[name]=summary
        for r in current:
            if r['source']=='ours':r['method']=name;all_rows.append(r)
        print(json.dumps({'method':name,'pixel':summary['overall']['ours']['metrics']['pixel_euclidean_error_px'],'failures':summary['overall']['ours']['pixel_localization_failures']}),flush=True)
    for r in v2e_reference.values():r['method']='v2e';all_rows.append(r)
    with (ROOT/'all_models_pixel_polls.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    table={k:{'pixel':v['overall']['ours']['metrics']['pixel_euclidean_error_px'],
              'failures':v['overall']['ours']['pixel_localization_failures'],
              'world_xy_m':v['overall']['ours']['metrics']['position_xy_error_m'],
              'checkpoint_sha256':v['checkpoint_sha256']['ours'],
              'by_episode':{i:e['ours']['metrics']['pixel_euclidean_error_px'] for i,e in v['by_episode'].items()}}
           for k,v in outputs.items()}
    last=outputs['observed_candidate'];table['v2e']={'pixel':last['overall']['v2e']['metrics']['pixel_euclidean_error_px'],
          'failures':last['overall']['v2e']['pixel_localization_failures'],'world_xy_m':last['overall']['v2e']['metrics']['position_xy_error_m'],
          'checkpoint_sha256':last['checkpoint_sha256']['v2e'],
          'by_episode':{i:e['v2e']['metrics']['pixel_euclidean_error_px'] for i,e in last['by_episode'].items()}}
    dump(ROOT/'summary.json',{'purpose':'pixel-objective development audit, not event-similarity optimization or formal ranking',
          'status':'PASS','code_commit':CODE,'resolution_wh':[320,240],'models':table,'world_score_audits':audits,
          'all_models_use_same_test_input':True,'v2e_repeated_controls_exact':True,
          'reference_manifest_sha256':sha(REF/'SHA256SUMS'),'reference_provenance':provenance,
          'training':json.loads((PRIOR/'data_split_provenance.json').read_text()),
          'candidate_scope':'explicit dense actual-HDR observation input, not repaired sparse warp',
          'wall_s':time.perf_counter()-started})
    files=[p for p in ROOT.rglob('*') if p.is_file() and 'source' not in p.relative_to(ROOT).parts and p.name!='worker.log']
    (ROOT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in sorted(files)))
    code=0
except Exception as e:
    print(type(e).__name__,str(e),flush=True)
    raise
finally:
    (ROOT/'pipeline.exit').write_text(str(code))
    update(status='completed' if code==0 else 'failed',exit_code=code,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
