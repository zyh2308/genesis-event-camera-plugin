"""Pure Census short-study contracts; no data loading, CUDA or optimizer."""
import hashlib,json,math,statistics
from pathlib import Path
METHODS=('eventgenesis','v2e');ARMS=('native_control','census')
STRATA=('camera_motion_static_scene','independent_rigid_motion_static_camera','camera_and_independent_rigid_motion','combined_motion_with_occlusion')
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def record(path):
    p=Path(path).resolve(strict=True);return dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p))
def check(r):
    p=Path(r['path']).resolve(strict=True)
    need(p.is_file() and p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256'],'Changed bound file: '+str(p));return p
def validate(c):
    need(c['schema']=='flow_census_paired_short_v1' and c['methods']==list(METHODS) and c['seed']==0 and c['arms']==list(ARMS),'Wrong canonical study identity')
    need(c['updates_per_arm']==200 and c['effective_batch']==8 and c['warm_update']==8910 and c['smoke_updates_total']==4 and c['main_updates_total']==800,'Changed fixed budget')
    need(c['real_data_used'] is False and c['real_scores_used'] is False and c['GT_flow_optimizer_supervision'] is False,'Synthetic-only isolation required')
    need(c['sampling']=='original_control_schedule_only' and c['original_smoothness_coefficient']==10. and c['patch_size']==7,'Unexpected sampling/loss intervention')
    need(c['calibration']==dict(batches_per_method=8,gradient_space='all_requires_grad_parameters_zero_unused',native_non_none_expected=70,formula='1/median(norm_census/norm_native_photo)',train_only=True,GT_in_lambda=False),'Changed shared lambda calibration')
    need(c['quality_gate']==dict(primary_all_valid_improves=True,common_active_improves=True,maximum_relative_stratum_epe_increase=.05,both_generators_required=True),'Quality gate changed')
    need(c['frozen_inputs']['methods']==list(METHODS) and c['frozen_inputs']['seed']==0,'Frozen input identities differ')
    return c
def paired_replay(rollout_strata):
    from sampling import schedules
    old=schedules(rollout_strata)
    replay=dict(schema='flow_census_original_control_replay_v1',seed=0,updates=200,batch=8,samples=old['samples'],
        augmentation_seed_sequences=old['augmentation_seed_sequences'],arms={a:old['arms']['control'].copy() for a in ARMS},
        counts=old['counts']['control'],stratum_order=list(STRATA),historical_sampler_replay_sha256=old['replay_sha256'],
        intervention='No motion sampler values used; both arms exactly old control order')
    replay['replay_sha256']=hashlib.sha256(json.dumps(replay,sort_keys=True).encode()).hexdigest();return replay
def raw_endpoint_domain(data):
    import numpy as np
    report={}
    for name in ('gray_prev','gray_next'):
        array=data[name]
        need(np.issubdtype(array.dtype,np.floating) and np.isfinite(array).all() and array.min()>=0 and array.max()<=1,
             'Raw NPZ endpoints must be strictly finite [0,1]; no internal-interpolation tolerance for raw encoding')
        report[name]=dict(dtype=str(array.dtype),minimum=float(array.min()),maximum=float(array.max()))
    return report
def shared_lambda(records,replay_hash):
    expected={(m,i) for m in METHODS for i in range(8)}
    need(len(records)==16 and {(r['method'],r['batch']) for r in records}==expected,'Require fixed8 TRAIN batches for both generators')
    need(len({r['parameter_signature_sha256'] for r in records})==1,'Methods must use the same full parameter inventory')
    ratios=[];snapshots={};labels={}
    for r in records:
        need(r['split']=='train' and r['optimizer_updates']==0 and r['GT_in_lambda'] is False and r['replay_sha256']==replay_hash,'Non-TRAIN or adaptive calibration')
        need(r['gradient_space']=='all_requires_grad_parameters_zero_unused' and r['native_finite_gradient_tensors']==70 and r['census_finite_gradient_tensors']==70,
             'Need the full trainable parameter set, preserving the verified70 connected finite gradients')
        need(r['total_requires_grad_parameters']>=70 and len(r['parameter_signature_sha256'])==64,'Full parameter signature is missing')
        x,y=r['native_photo_gradient_l2'],r['census_photo_gradient_l2']
        need(all(isinstance(v,(float,int)) and math.isfinite(v) and v>0 for v in (x,y)),'Zero/nonfinite calibration gradients')
        need(r['census_to_native_ratio']==y/x,'Incorrect parameter-gradient ratio')
        if r['method'] in snapshots:need(snapshots[r['method']]==(r['model_state_sha256'],r['parameter_signature_sha256']),'Calibration model/signature drift')
        snapshots[r['method']]=(r['model_state_sha256'],r['parameter_signature_sha256'])
        if r['batch'] in labels:need(labels[r['batch']]==r['labels_and_augmentation_sha256'],'Unpaired calibration batch')
        labels[r['batch']]=r['labels_and_augmentation_sha256'];ratios.append(y/x)
    median=statistics.median(ratios);value=1/median
    need(math.isfinite(value) and value>0,'Invalid shared coefficient')
    return dict(shared_census_weight=value,median_ratio=median,ratio_count=16,formula='1/median(norm_census/norm_native_photo)',
        gradient_space='all_requires_grad_parameters_zero_unused',native_non_none_expected=70,GT_in_lambda=False,validation_in_lambda=False,real_in_lambda=False,
        records=records,replay_sha256=replay_hash)
def quality(results):
    need(len(results)==4 and {(r['method'],r['arm']) for r in results}=={(m,a) for m in METHODS for a in ARMS},'Incomplete paired quality matrix')
    checks=[]
    for m in METHODS:
        control=next(r['validation'] for r in results if (r['method'],r['arm'])==(m,'native_control'))
        candidate=next(r['validation'] for r in results if (r['method'],r['arm'])==(m,'census'))
        for v in (control,candidate):need(v['window_count']==528 and set(v['strata'])==set(STRATA),'Incomplete full528/4-strata validation')
        for name in ('all_physical_valid','common_event_active_valid'):
            x,y=control['regions'][name]['epe_px'],candidate['regions'][name]['epe_px']
            finite=x is not None and y is not None and math.isfinite(x) and math.isfinite(y)
            checks.append(dict(method=m,region=name,control=x,candidate=y,passed=bool(finite and y<x)))
        for st in STRATA:
            x,y=control['strata'][st]['all_physical_valid']['epe_px'],candidate['strata'][st]['all_physical_valid']['epe_px']
            finite=x is not None and y is not None and math.isfinite(x) and math.isfinite(y)
            checks.append(dict(method=m,stratum=st,control=x,candidate=y,threshold=1.05*x if x is not None else None,passed=bool(finite and y<=1.05*x)))
    return dict(passed=all(r['passed'] for r in checks),checks=checks,single_seed_development=True,automatic_long_training_or_real_scoring=False)
