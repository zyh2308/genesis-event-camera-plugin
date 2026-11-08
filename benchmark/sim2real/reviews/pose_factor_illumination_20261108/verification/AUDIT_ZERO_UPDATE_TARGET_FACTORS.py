"""Independently recompute local teacher-UV and predicted depth-factor evidence."""
import argparse
import hashlib
import itertools
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

METHODS=('eventgenesis','v2e');SEEDS=(0,1,2);LIGHTS=('normal','half','eighth')
TEACHER_METRICS=('raw_teacher_center_error_px','baf_teacher_center_error_px','teacher_target_shift_px')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def jsonl(p):return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
def close(x,y,atol=2e-12,rtol=2e-12):
    assert math.isfinite(float(x)) and math.isfinite(float(y))
    assert math.isclose(float(x),float(y),abs_tol=atol,rel_tol=rtol),(x,y)


def verify_index(root,name):
    index=read(root/name)
    for name,rec in index['files'].items():
        p=root/name
        assert p.stat().st_size==rec['size_bytes'] and sha(p)==rec['sha256'],str(p)
    return index


def teacher_audit(review):
    root=review/'teacher_target_diagnostic'; index=verify_index(root,'TEACHER_EVIDENCE_INDEX.json')
    assert len(index['files'])==14 and index['optimizer_updates']==0 and index['GPU_used'] is False and index['exit_code']==0
    manifest=read(root/'sources/source_manifest.json')
    assert sha(root/'sources/source_manifest.json')==index['actual_source_manifest_sha256']
    for name,digest in manifest.items():assert sha(root/'sources'/name)==digest
    completion=read(root/'results/completion.json')
    assert completion['status']=='complete' and completion['rows']==1728 and completion['optimizer_updates']==0
    assert completion['GPU_used'] is False and completion['real_data_used'] is False and completion['teacher_modified'] is False
    assert completion['raw_encode_exact_parity'] is True and completion['GT_used_only_in_diagnostic_errors'] is True
    assert completion['C1_endpoint_UV_parity']=='not_applicable_different_weights'
    assert 'NOT_C1' in completion['model_role']
    baf=review/'baf_center_execution';bafindex=read(baf/'BAF_FINAL_INDEX.json')
    admission_path=baf/'study/pre_update_admission.json'
    expected=bafindex['files']['study/pre_update_admission.json']
    assert sha(admission_path)==expected['sha256'] and admission_path.stat().st_size==expected['size_bytes']
    admission=read(admission_path)
    assert admission['status']=='passed' and admission['optimizer_updates']==0
    ad={(r['method'],int(r['seed'])):r for r in admission['full288_admissions']}
    models={(r['method'],int(r['seed'])):r for r in completion['models']}
    expected_models=set(itertools.product(METHODS,SEEDS))
    assert set(ad)==set(models)==expected_models
    source_records={r['path']:r for r in completion['source_records']}
    for key,model in models.items():
        assert model['state_sha256']==ad[key]['source_state_sha256']
        assert model['path']==ad[key]['source_model']['path'] and model['unchanged'] is True
        assert source_records[model['path']]['sha256']==ad[key]['source_model']['sha256']
    rows=jsonl(root/'results/teacher_targets.jsonl')
    assert len(rows)==1728
    seen=set();groups=defaultdict(lambda:defaultdict(list));identities={};maxdelta=0.
    for r in rows:
        key=(r['method'],int(r['seed']),r['split'],r['condition'])
        unique=key+(r['rid'],r['sample_id'])
        assert unique not in seen;seen.add(unique)
        assert r['model_role']=='frozen_source_init_teacher_NOT_C1_endpoint'
        assert r['start_s']<r['end_s'] and 0<=r['retained_events']<=r['raw_events']
        uv=[r[n] for n in ('raw_teacher_uv','baf_teacher_uv','uv_gt')]
        assert all(len(v)==2 and all(math.isfinite(float(x)) for x in v) for v in uv)
        distance=lambda a,b:math.hypot((a[0]-b[0])*346.,(a[1]-b[1])*260.)
        expected_values=dict(zip(TEACHER_METRICS,(distance(uv[0],uv[2]),distance(uv[1],uv[2]),distance(uv[0],uv[1]))))
        for name,value in expected_values.items():close(r[name],value);maxdelta=max(maxdelta,abs(r[name]-value))
        groups[key][r['rid']].append(expected_values)
        geometry_key=(r['split'],r['rid'],r['sample_id'])
        geometry=(r['uv_gt'],r['start_s'],r['end_s'])
        if geometry_key in identities:assert geometry==identities[geometry_key]
        identities[geometry_key]=geometry
    assert set(groups)==set(itertools.product(METHODS,SEEDS,('train','validation'),LIGHTS))
    computed={}
    inventories={}
    for key,rollouts in groups.items():
        train=key[2]=='train';assert len(rollouts)==(8 if train else 4)
        assert all(len(values)==(4 if train else 16) for values in rollouts.values())
        computed[key]={name:statistics.mean(statistics.mean(r[name] for r in values) for values in rollouts.values()) for name in TEACHER_METRICS}
        computed[key].update(windows=sum(map(len,rollouts.values())),rollouts=len(rollouts))
        inv={(u[-2],u[-1]) for u in seen if u[:4]==key}
        split=key[2]
        if split in inventories:assert inv==inventories[split]
        inventories[split]=inv
    supplied=read(root/'results/summary.json')
    assert len(supplied)==36
    for r in supplied:
        key=(r['method'],int(r['seed']),r['split'],r['condition'])
        for metric in (*TEACHER_METRICS,'windows','rollouts'):close(r[metric],computed[key][metric])
    means=read(root/'derived/three_seed_mean.json');assert len(means)==12
    means_map={}
    for r in means:
        key=(r['method'],r['split'],r['condition']);assert key not in means_map
        means_map[key]=r
        for metric in TEACHER_METRICS:close(r[metric],statistics.mean(computed[(key[0],s,key[1],key[2])][metric] for s in SEEDS))
    assert set(means_map)==set(itertools.product(METHODS,('train','validation'),LIGHTS))
    initial_log=(root/'actual_server_cpu_tests.log').read_text();retry_log=(root/'actual_server_cpu_tests_retry1.log').read_text()
    assert 'ModuleNotFoundError' in initial_log and 'FAILED (errors=1)' in initial_log
    assert 'Ran 4 tests' in retry_log and '\nOK' in retry_log
    for artifact in completion['artifacts']:
        p=root/'results'/Path(artifact['path']).name
        assert sha(p)==artifact['sha256'] and p.stat().st_size==artifact['size_bytes']
    return dict(status='passed',index_sha256=sha(root/'TEACHER_EVIDENCE_INDEX.json'),files_verified=14,
                manifest_members_verified=len(manifest),rows_recomputed=1728,per_seed_macros_recomputed=36,three_seed_means_recomputed=12,
                source_init_states_bound=6,raw_to_native_pixel_max_abs_recompute_error=maxdelta,
                TRAIN_normal_teacher_means={m:{k:means_map[(m,'train','normal')][k] for k in TEACHER_METRICS} for m in METHODS},
                CPU_tests=dict(initial_attempt='failed_wrong_test_module_name_preserved',retry='4/4 passed'),
                optimizer_updates=0,GPU_used=False,real_data_used=False,model_role='source_init_teacher_not_C1_endpoint',
                limitation='Local raw UV/errors and summaries recomputed. Source-init state/file SHA associations cross-checked in receipts; server-only weight bytes not loaded by this audit. Dark teacher responses are diagnostic, not historical dark KD training targets.')


def depth_audit(review):
    root=review/'depth_factor_cross';index=verify_index(root,'DEPTH_CROSS_INDEX.json')
    assert len(index['files'])==4
    completion=read(root/'completion.json')
    assert completion['status']=='complete' and completion['diagnostic_only'] is True
    assert completion['optimizer_updates']==0 and completion['new_model_forward']==0 and completion['real_data_used'] is False
    assert completion['GT_used_only_for_error'] is True and completion['uses_two_frozen_predicted_models'] is True
    assert completion['source_code_sha256']==sha(root/'depth_predicted_cross.py')
    old=read(review/'evidence/EVIDENCE_INDEX.json')
    source=completion['source'];assert old['external_artifacts'][source['path']]['sha256']==source['sha256']
    geom=read(review/'evidence/geometry/completion.json');assert geom['inputs']['predictions']==source
    rows=jsonl(root/'factor_inputs_and_predictions.jsonl');assert len(rows)==completion['per_window_records']==1152
    arms=('C0','C1','C0_size_C1_shape','C1_size_C0_shape')
    seen=set();groups=defaultdict(lambda:defaultdict(list));source_z_max=0.;mixed_max=0.;physical={}
    for r in rows:
        key=(r['method'],int(r['seed']),r['condition']);unique=key+(r['rid'],r['sample_id'])
        assert unique not in seen;seen.add(unique)
        assert all(math.isfinite(float(r[n])) for n in ('gt_z_m','focal_log','C0_log_size','C0_log_shape','C1_log_size','C1_log_shape'))
        assert r['gt_z_m']>0 and set(r['predicted_depth_m'])==set(arms)
        computed=dict(C0=math.exp(r['C0_log_shape']+r['focal_log']-r['C0_log_size']),
                      C1=math.exp(r['C1_log_shape']+r['focal_log']-r['C1_log_size']),
                      C0_size_C1_shape=math.exp(r['C1_log_shape']+r['focal_log']-r['C0_log_size']),
                      C1_size_C0_shape=math.exp(r['C0_log_shape']+r['focal_log']-r['C1_log_size']))
        for arm,value in computed.items():
            supplied=r['predicted_depth_m'][arm]
            assert math.isfinite(supplied) and supplied>0
            error=abs(value-supplied)
            if arm in ('C0','C1'):
                assert error<2e-6;source_z_max=max(source_z_max,error)
            else:
                close(value,supplied);mixed_max=max(mixed_max,error)
            groups[key][r['rid']].append((arm,abs(supplied-r['gt_z_m']),supplied-r['gt_z_m']))
        physical_key=(r['rid'],r['sample_id']);label=(r['gt_z_m'],r['focal_log'])
        if physical_key in physical:assert physical[physical_key]==label
        physical[physical_key]=label
    assert set(groups)==set(itertools.product(METHODS,SEEDS,LIGHTS))
    seed_means={}
    for key,rollouts in groups.items():
        assert len(rollouts)==4 and all(len(v)==64 for v in rollouts.values())
        seed_means[key]={}
        for arm in arms:
            perrid=[]
            for values in rollouts.values():
                matched=[v for v in values if v[0]==arm];assert len(matched)==16
                perrid.append((statistics.mean(v[1] for v in matched),statistics.mean(v[2] for v in matched)))
            seed_means[key][arm]=dict(depth_mae_cm=100*statistics.mean(v[0] for v in perrid),depth_bias_cm=100*statistics.mean(v[1] for v in perrid))
    means=read(root/'three_seed_matched_summary.json');assert len(means)==6
    mean_map={}
    for r in means:
        key=(r['method'],r['condition']);assert key not in mean_map;mean_map[key]=r['arms']
        for arm in arms:
            for metric in ('depth_mae_cm','depth_bias_cm'):
                close(r['arms'][arm][metric],statistics.mean(seed_means[(key[0],s,key[1])][arm][metric] for s in SEEDS))
    assert set(mean_map)==set(itertools.product(METHODS,LIGHTS))
    old_matched={(r['method'],r['arm'],r['condition']):r['metrics'] for r in read(review/'evidence/matched_synthetic_summary.json')}
    for (method,condition),values in mean_map.items():
        for arm in ('C0','C1'):
            close(values[arm]['depth_mae_cm'],100*old_matched[(method,arm,condition)]['depth_mae_m'])
            close(values[arm]['depth_bias_cm'],100*old_matched[(method,arm,condition)]['depth_bias_m'])
    half=mean_map[('eventgenesis','half')];dark=mean_map[('eventgenesis','eighth')]
    return dict(status='passed',index_sha256=sha(root/'DEPTH_CROSS_INDEX.json'),files_verified=4,per_window_factor_records_recomputed=1152,
                per_seed_condition_macros_recomputed=18,three_seed_condition_means_recomputed=6,
                source_z_decode_tolerance_m=2e-6,source_z_decode_max_abs_m=source_z_max,mixed_exp_max_abs_m=mixed_max,
                EventGenesis_half=half,EventGenesis_dark=dark,
                EventGenesis_half_C1_minus_C0=dict(depth_mae_cm=half['C1']['depth_mae_cm']-half['C0']['depth_mae_cm'],
                                                 depth_bias_cm=half['C1']['depth_bias_cm']-half['C0']['depth_bias_cm']),
                source_predictions_sha256=source['sha256'],optimizer_updates=0,new_model_forward=0,real_data_used=False,
                limitation='Controlled crossing of two models predicted output factors, not unique weight-level causal attribution or an accepted deployable method. Original 3456 prediction bytes remain server-only; their SHA receipt binding, not full byte provenance, was independently verified locally.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review',type=Path,required=True);args=parser.parse_args()
    teacher=teacher_audit(args.review);depth=depth_audit(args.review)
    output=dict(schema='zero_update_teacher_depth_factor_independent_audit_v1',status='passed',teacher_target=teacher,depth_factor=depth)
    dest=args.review/'verification';dest.mkdir(exist_ok=True);script=dest/'AUDIT_ZERO_UPDATE_TARGET_FACTORS.py'
    if Path(__file__).resolve()!=script.resolve():shutil.copyfile(__file__,script)
    output['audit_script_sha256']=sha(script)
    (dest/'ZERO_UPDATE_TARGET_FACTOR_AUDIT.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,allow_nan=False))
