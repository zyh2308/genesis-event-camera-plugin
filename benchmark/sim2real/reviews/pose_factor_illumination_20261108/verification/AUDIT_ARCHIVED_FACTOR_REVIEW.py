"""Offline audit of the archived scalar evidence, not a raw prediction replay."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ('depth_mae_m','depth_bias_m','log_size_mae','log_size_bias','center_px','success','auc')
METHODS = ('eventgenesis','v2e')
ARMS = ('C0','C1','slow')
LIGHTS = ('normal','half','eighth')
SEEDS = (0,1,2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def close(a,b,tol=1e-12):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a),float(b),abs_tol=tol,rel_tol=tol),(a,b)


def csv_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def run(root):
    evidence=root/'evidence'; tables=root/'tables'
    index=load(evidence/'EVIDENCE_INDEX.json')
    assert index['new_real_scores'] is False and index['optimizer_updates_in_diagnostics']==0
    verified=[]
    for name,rec in index['files'].items():
        path=evidence/name
        assert path.stat().st_size==rec['size_bytes'],name
        assert sha(path)==rec['sha256'],name
        verified.append(name)
    assert len(verified)==29
    manifests={}
    for group in ('geometry','diagnostic','teacher_projection_v2'):
        source=root/'sources'/group
        manifest=load(source/'source_manifest.json')
        for name,digest in manifest.items():
            assert sha(source/name)==digest,(group,name)
        manifests[group]=dict(member_count=len(manifest),manifest_sha256=sha(source/'source_manifest.json'))

    native=load(evidence/'export/native_scores.json')
    expected=set(itertools.product(METHODS,SEEDS,ARMS,LIGHTS))
    native_map={(r['method'],int(r['seed']),r['arm'],r['condition']):r['score'] for r in native}
    assert len(native)==len(native_map)==54 and set(native_map)==expected
    rids={'validation_000','validation_031','validation_063','validation_095'}
    for score in native_map.values():
        assert set(score['by_rollout'])==rids
        for metric in METRICS:
            close(score['summary'][metric],statistics.mean(r[metric] for r in score['by_rollout'].values()))
    matched=load(evidence/'matched_synthetic_summary.json')
    mm={(r['method'],r['arm'],r['condition']):r['metrics'] for r in matched}
    assert len(matched)==len(mm)==18 and set(mm)==set(itertools.product(METHODS,ARMS,LIGHTS))
    for key,metrics in mm.items():
        for name,value in metrics.items():
            close(value,statistics.mean(native_map[(key[0],s,key[1],key[2])]['summary'][name] for s in SEEDS))
    matched_csv=csv_rows(tables/'matched_pose_three_illuminations.csv')
    assert len(matched_csv)==18
    for row in matched_csv:
        metrics=mm[(row['method'],row['arm'],row['condition'])]
        close(row['depth_mae_cm'],metrics['depth_mae_m']*100)
        close(row['center_epe_native_px'],metrics['center_px'])
        close(row['ADD_S_success_percent'],metrics['success']*100)

    macro=csv_rows(evidence/'geometry/macro_scores.csv')
    gm={(r['method'],int(r['seed']),r['condition'],r['variant']):r for r in macro}
    assert len(macro)==len(gm)==630
    variants={'actual/'+a for a in ARMS}
    variants.update(f'oracle/{a}/u{u}z{z}r{r}' for a in ARMS for u,z,r in itertools.product((0,1),repeat=3))
    variants.update(f'cross/u{u}_z{z}_r{r}' for u,z,r in itertools.product(('C0','C1'),repeat=3))
    assert len(variants)==35
    assert set(gm)==set(itertools.product(METHODS,SEEDS,LIGHTS,variants))
    assert all(int(r['sample_count'])==64 and int(r['rollout_count'])==4 for r in macro)
    for (method,seed,arm,light),score in native_map.items():
        row=gm[(method,seed,light,'actual/'+arm)]
        for metric in METRICS:
            assert math.isclose(float(row[metric]),score['summary'][metric],abs_tol=2e-7,rel_tol=2e-6)

    oracle=load(evidence/'factor_oracle_summary_C1.json')
    oracle_map={(r['method'],r['condition']):r['success_percent'] for r in oracle}
    assert len(oracle)==len(oracle_map)==6
    names={'actual':'actual/C1','GTcenter':'oracle/C1/u1z0r0','GTdepth':'oracle/C1/u0z1r0',
           'GTrotation':'oracle/C1/u0z0r1','GTcenter_depth':'oracle/C1/u1z1r0','allGT':'oracle/C1/u1z1r1'}
    for (method,light),values in oracle_map.items():
        for name,variant in names.items():
            close(values[name],100*statistics.mean(float(gm[(method,s,light,variant)]['success']) for s in SEEDS))
        assert values['allGT']==100
    oracle_csv=csv_rows(tables/'GT_factor_diagnostic_only.csv')
    assert len(oracle_csv)==6
    for row in oracle_csv:
        for name in names:
            close(row[name],oracle_map[(row['method'],row['condition'])][name])
    allgt=[r for r in macro if r['variant'].endswith('/u1z1r1')]
    assert len(allgt)==54 and all(float(r['success'])==1. for r in allgt)

    parity=load(evidence/'geometry/native_parity.json')
    assert parity['status']=='passed' and parity['checked_cells']==54 and len(parity['checks'])==1890
    seen=set(); maxerror=0.
    for check in parity['checks']:
        method,seed,light,arm=check['cell']; scope=check['scope']; metric=check['metric']
        key=(method,seed,light,arm,scope,metric)
        assert key not in seen
        seen.add(key)
        score=native_map[(method,seed,arm,light)]
        expected_native=score['summary'][metric] if scope=='summary' else score['by_rollout'][scope][metric]
        close(check['native'],expected_native)
        if scope=='summary': close(check['actual'],gm[(method,seed,light,'actual/'+arm)][metric])
        close(check['abs_error'],abs(check['actual']-check['native']))
        passed=(check['actual']==check['native'] if metric=='success' else math.isclose(check['actual'],check['native'],abs_tol=parity['atol'],rel_tol=parity['rtol']))
        assert passed and check['passed'] is True
        maxerror=max(maxerror,check['abs_error'])
    assert seen==set((m,s,l,a,scope,k) for m,s,a,l in expected for scope in ('summary',*sorted(rids)) for k in METRICS)
    export=load(evidence/'export/completion.json')
    geom=load(evidence/'geometry/completion.json')
    assert export['optimizer_updates']==0 and export['real_data_used'] is False and export['models_and_sources_unchanged'] is True
    assert geom['physical_records']==1152 and geom['flat_predictions']==3456 and geom['scored_variants']==40320 and geom['macro_cells']==630
    assert geom['gt_oracles_deployable'] is False and geom['real_data_used'] is False
    assert 'distinct' in export['matched_normal_scope'] and '384' in export['matched_normal_scope']
    oldparity=load(evidence/'export/native_parity.json')
    clean=[r for r in oldparity if r['condition']=='clean384']
    assert len(clean)==18 and all(len(r['score']['by_rollout'])==96 and r['passed'] for r in clean)
    assert not any(r['condition']=='clean384' for r in native)
    for key in ('predictions','vertices','keypoints','native_scores'):
        rec=geom['inputs'][key]
        if key=='native_scores': assert rec['sha256']==sha(evidence/'export/native_scores.json')
        elif key=='predictions': assert index['external_artifacts'][rec['path']]['sha256']==rec['sha256']
        else: assert export['assets'][key]['sha256']==rec['sha256']

    flow=load(evidence/'flow/completion_summary.json')
    assert flow['status']=='complete' and flow['mode']=='study' and flow['pairing_passed'] is True
    assert all(flow[k] is False for k in ('formal_weights_modified','GT_flow_optimizer_supervision','real_data_used','real_scores_used','automatic_long_training_or_real_scoring'))
    results={(r['method'],r['arm']):r for r in flow['results']}
    assert len(results)==4 and set(results)==set(itertools.product(METHODS,('native_control','census')))
    for row in results.values():
        assert row['seed']==0 and row['additional_updates']==200 and row['optimizer_update']==9110
        assert row['finite_gradient_tensors']==70 and row['independent_model_and_Adam_restore_passed']
        assert row['validation']['window_count']==528 and row['validation']['source_rollout_count']==8
        assert row['real_data_used'] is False and row['validation']['real_data_used'] is False
        close(row['shared_census_weight'],flow['shared_census_weight'])
    for method in METHODS:
        control=results[(method,'native_control')]; candidate=results[(method,'census')]
        for field in ('source_checkpoint','initial_learning_rates','initial_rng_fingerprint','shared_calibration',
                      'source_manifest_sha256','label_sequence_sha256','sample_sequence_sha256','augmentation_sequence_sha256'):
            assert control[field]==candidate[field],(method,field)
    assert len({r['label_sequence_sha256'] for r in results.values()})==1
    assert len({r['sample_sequence_sha256'] for r in results.values()})==1
    assert len({r['augmentation_sequence_sha256'] for r in results.values()})==1
    checks=flow['development_quality_gate']['checks']
    assert len(checks)==12
    for check in checks:
        m=check['method']; v={a:results[(m,a)]['validation'] for a in ('native_control','census')}
        if 'region' in check:
            actual={a:x['regions'][check['region']]['epe_px'] for a,x in v.items()}
            passed=actual['census']<actual['native_control']
        else:
            actual={a:x['strata'][check['stratum']]['all_physical_valid']['epe_px'] for a,x in v.items()}
            close(check['threshold'],1.05*actual['native_control'])
            passed=actual['census']<=1.05*actual['native_control']
        close(check['control'],actual['native_control']);close(check['candidate'],actual['census'])
        assert check['passed']==passed
    assert sum(r['passed'] for r in checks)==3 and flow['development_quality_gate']['passed'] is False
    flow_table=csv_rows(tables/'flow_census_short_development.csv')
    assert len(flow_table)==4
    differences=[]
    for row in flow_table:
        rec=results[(row['method'],row['arm'])]
        assert int(row['additional_updates'])==200 and int(row['windows'])==528 and int(row['seed'])==0
        close(row['all_valid_EPE_px_per_30ms'],rec['validation']['regions']['all_physical_valid']['epe_px'])
        close(row['common_active_EPE_px_per_30ms'],rec['validation']['regions']['common_event_active_valid']['epe_px'])
    for method in METHODS:
        metrics={}
        for region in ('all_physical_valid','common_event_active_valid'):
            c=results[(method,'native_control')]['validation']['regions'][region]['epe_px']
            x=results[(method,'census')]['validation']['regions'][region]['epe_px']
            metrics[region]=dict(control=c,census=x,delta=x-c,relative_worse_percent=100*(x/c-1))
        differences.append(dict(method=method,metrics=metrics))
    done=load(evidence/'queue/155_flow_census_paired_study_20261108.done.json')
    assert done['exit_code']==0 and done['status']=='complete'
    return dict(schema='pose_factor_archive_independent_audit_v1',status='passed',evidence_index_sha256=sha(evidence/'EVIDENCE_INDEX.json'),
                local_evidence_files_verified=len(verified),source_manifest_members_verified=sum(x['member_count'] for x in manifests.values()),
                source_groups=manifests,native_cells=54,matched_three_seed_mean_cells=18,matched_csv_rows=18,
                geometry_macro_cells=630,oracle_C1_cells=6,all_GT_success_macro_cells=54,all_GT_success_percent=100,
                geometry_parity_checks_recomputed=1890,geometry_parity_max_abs_error=maxerror,
                scoped_native_clean384_cells=18,scopes_separated=True,
                flow=dict(arms=4,additional_updates_each=200,windows_each=528,process_exit_code=0,checks_passed=3,checks_failed=9,
                          quality_gate_passed=False,paired_replay_and_init_verified=True,differences=differences),
                gt_oracles_are_diagnostic_only=True,new_real_scores=False,
                scope_limitations=['No SSH or GPU inference or training was performed by this audit.',
                                   'Raw 3456 predictions and 40320 per-window diagnostic records remain server-only; their indexed SHA links are checked, not their bytes or per-window geometry independently replayed locally.',
                                   '1890 checks are recomputed from archived scalar parity/native tables; per-rollout actual entries are not independently reconstructed from server-only raw predictions.',
                                   'Candidate smoke/study added after this index snapshot is outside this audit.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',required=True,type=Path)
    args=parser.parse_args()
    output=run(args.review)
    dest=args.review/'verification';dest.mkdir(exist_ok=True)
    target=dest/'AUDIT_ARCHIVED_FACTOR_REVIEW.py'
    if Path(__file__).resolve()!=target.resolve():shutil.copyfile(__file__,target)
    output['audit_script_sha256']=sha(target)
    (dest/'INDEPENDENT_ARCHIVED_FACTOR_AUDIT.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,allow_nan=False))
