from pathlib import Path
import os,sys,time,subprocess,json,csv,hashlib,datetime
import numpy as np
import h5py

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1]
PY=BASE/'repositories/gec-robot/.venv-robot/bin/python'
RECORD=BASE/'runs/robot_scenes/.resource_allocations'/(ROOT.name+'_audit.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def update(**kw):
    d=json.loads(RECORD.read_text());d.update(**kw);dump(RECORD,d)
def read_csv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))

code=1
try:
    update(status='waiting_for_pipeline',pid=os.getpid(),cpu_threads=0)
    deadline=time.monotonic()+2400
    while not (ROOT/'pipeline.exit').exists():
        if time.monotonic()>deadline:raise TimeoutError('pipeline completion not observed')
        time.sleep(2)
    assert (ROOT/'pipeline.exit').read_text()=='0'
    update(status='running',cpu_threads=2)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
        'OPENBLAS_NUM_THREADS':'2','PYTHONPATH':str(ROOT/'source')+':'+str(ROOT/'core')}
    for name in ['check_sparse_additional','make_evidence']:
        with (ROOT/(name+'.log')).open('w') as log:
            process=subprocess.run([str(PY),str(ROOT/(name+'.py'))],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:raise RuntimeError(name+' failed')
    summary=json.loads((ROOT/'summary.json').read_text())
    per_bank={}
    prior=BASE/'runs/robot_r1_5_pixel_objective_20261101a'
    for name,n in [('primary',581),('secondary',193)]:
        raw=read_csv(ROOT/'evaluation'/name/'r1_common_direct_hdr_polls.csv')
        ours=[r for r in raw if r['source']=='ours'];control=[r for r in raw if r['source']=='v2e']
        assert len(ours)==len(control)==n
        e=np.array([float(r['pixel_euclidean_error_px']) for r in ours])
        recomputed=np.array([np.hypot(float(r['pred_u_px'])-float(r['gt_u_px']),
            float(r['pred_v_px'])-float(r['gt_v_px'])) for r in ours])
        assert np.all(np.isfinite(e)) and np.max(np.abs(recomputed-e))<1e-9
        metric=summary['results'][name]['candidate']['metrics']['pixel_euclidean_error_px']
        for key,val in [('mean',e.mean()),('p50',np.median(e)),('p95',np.percentile(e,95))]:
            assert abs(metric[key]-float(val))<1e-9
        if name=='primary':old_csv=prior/'all_models_pixel_polls.csv'
        else:old_csv=prior/'secondary_confirmation/all_models_pixel_polls.csv'
        previous=[r for r in read_csv(old_csv) if r.get('method')=='v2e']
        lookup={(int(r['episode_id']),round(float(r['poll_s']),9)):r for r in previous}
        assert len(lookup)==n
        for row in control:
            old=lookup[int(row['episode_id']),round(float(row['poll_s']),9)]
            for key in ['pixel_euclidean_error_px','position_xy_error_m']:
                assert abs(float(old[key])-float(row[key]))<1e-10
        per_bank[name]=dict(polls=n,pixel_csv_max_residual=float(np.max(np.abs(e-recomputed))),
            frozen_v2e_control_identical=True)
    data_audits=[]
    for folder in sorted((ROOT/'sparse_events/episodes').iterdir()):
        audit=json.loads((folder/'reconstruction_audit.json').read_text())
        source=Path(audit['source_path'])
        with h5py.File(source/'trajectory_cache.h5','r') as h:
            timestamps=np.asarray(h['timestamps_s'][:],dtype=np.float64)
        callbacks=[a for a in audit['access'] if a['kind']=='frame']
        assert len(callbacks)==len(timestamps)
        assert all(x['index']==i and x['t']==float(timestamps[i]) for i,x in enumerate(callbacks))
        assert audit['renderer_calls']==audit['dense_radiance_reads']==0
        assert audit['intermediate_state_provider'] is None
        assert audit['source_cache_sha256']==sha(source/'trajectory_cache.h5')
        assert sha(folder/'state_control_trace.jsonl')==sha(source/'oracle_trace.jsonl')
        data_audits.append(dict(episode=folder.name,actual_endpoints=len(timestamps),
            endpoint_reads=len(callbacks),source_cache_sha256=audit['source_cache_sha256']))
    report=dict(status='PASS',candidate_acceptance=summary['acceptance'],
        pixel_recomputation=per_bank,sparse_input_audit=data_audits,
        dense_observation_api_present=False,
        candidate_core_sha256={p.name:sha(p) for p in (ROOT/'core/genesis_event_plugin').glob('*.py')},
        code_commit='32567603c6d2dff01993b8ba89befb46f43a00ac',
        checkpoint_sha256=sha(ROOT/'fit/spatial_event_state.pt'))
    assert 'def capture_observed' not in (ROOT/'core/genesis_event_plugin/physics_plugin.py').read_text()
    dump(ROOT/'independent_audit.json',report)
    print(json.dumps(dict(audit='PASS',candidate_status=summary['status'],acceptance=summary['acceptance'])),flush=True)
    code=0
finally:
    (ROOT/'postprocess.exit').write_text(str(code))
    update(status='completed' if code==0 else 'failed',exit_code=code,cpu_threads=0,
        ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
