"""Strict read-only summary of pinned actual GPU3 smoke/study JSON receipts."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
METHODS=('eventgenesis','v2e');SEEDS=(0,1,2);CONDITIONS=('clean','matched_normal','half','eighth')
ARMS=('raw_C1','aux_visible_mask_center')
SCIENCE='d2860889e582223bf248d7b8e8e6fcc8f80aa71041a371243fdae06010e312ff'
SOURCES='4819053bf08966a153055b81eab86c0f70f519109afcb104cf71059cf23800c5'
RESOURCE='c1dc975cd625a5b865e184368aacd576d09b07ca82937079f7468c1af37cfacf'
GPU3='GPU-253ab9b8-a0e5-4f12-0579-261818cf31dd';GPU4='GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3'
METRICS=('depth_mae_m','depth_bias_m','log_size_mae','log_size_bias','center_px','success','auc')


def need(ok,message):
    if not ok:raise ValueError(message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sha_string(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def pinned(path,sha):
    need(sha_string(sha) and digest(path)==sha,'External receipt SHA mismatch');return json.loads(Path(path).read_text())
def record(path):return dict(path=str(Path(path).resolve()),size_bytes=Path(path).stat().st_size,sha256=digest(path))


def verify_common(c,phase):
    need(c['schema']=='pose_aux_visible_mask_center_v1' and c['status']=='complete_synthetic_only' and c['phase']==phase,'Incomplete/wrong phase receipt')
    need(c['canonical_candidate']=='aux_visible_mask_center' and c['contract_sha256']==SCIENCE and c['source_manifest_sha256']==SOURCES and c['runtime_resource_contract_sha256']==RESOURCE,'Frozen contract/source/resource mismatch')
    need(c['execution_gpu_uuid']==GPU3 and c['original_frozen_source_gpu_uuid']==GPU4,'Runtime/provenance GPU confused')
    need(c['real_scores_computed'] is False and c['GT_forward_inputs'] is False and c['pose_prediction_uses_auxiliary_head'] is False,'Inference/real isolation violated')
    need(c['resume_supported'] is False and c['checkpoint_verification_scope']=='model_auxiliary_optimizer_readback_only_no_resume_claim','Unsupported resume claim')
    for k in ('original30_gate','original8_confirmation'):need(c[k]['status']=='not_retested_not_passed_by_this_probe' and c[k]['thresholds_unchanged'] is True,'Formal gate misrepresented')
    expected={(m,s) for m in METHODS for s in ([0] if phase=='smoke' else SEEDS)};rows=c['records'];need(len(rows)==len(expected) and {(r['method'],r['seed']) for r in rows}==expected,'Incomplete/duplicate run matrix')
    updates=1 if phase=='smoke' else 288;need(c['new_optimizer_updates']==updates*len(expected),'Wrong aggregate optimizer budget')
    for r in rows:
        need(r['arm']=='aux_visible_mask_center' and r['branch']=='center' and r['updates']==r['new_optimizer_updates']==updates,'Wrong branch/budget')
        need(all(r[k] is True for k in ('teacher_unchanged','size_unchanged','independent_model_optimizer_rebuild','optimizer_state_equal')),'Frozen/rebuild proof missing')
        need(r['first_native_loss_oracle']['passed'] is True and all(finite(v) and v>=0 for v in r['first_native_loss_oracle']['absolute_errors'].values()),'First native scalar oracle missing')
        need(all(sha_string(r[k]) for k in ('source_state_sha256','final_state_sha256','auxiliary_state_sha256','labels_sha256','checkpoint_sha256')),'Model/label proof hashes missing')
        n=32 if phase=='smoke' else 1529*6;need(r['exposure']['normal']==n and r['exposure']['half']+r['exposure']['eighth']==n,'Exposure budget differs')
    admissions=c['full288_admissions'];need(len(admissions)==6 and {(r['method'],r['seed']) for r in admissions}=={(m,s) for m in METHODS for s in SEEDS},'Incomplete six full tapes')
    need(all(r['full288_replay_exact'] is True and sha_string(r['labels_sha256']) and sha_string(r['trajectory_sha256']) for r in admissions),'Full tape proof missing')
    for seed in SEEDS:need(len({r['labels_sha256'] for r in admissions if r['seed']==seed})==1,'Cross-generator labels differ')
    masks=c['mask_labels'];need(len(masks)==2 and {r['method'] for r in masks}==set(METHODS),'Paired physical mask proof missing')
    for k in ('normal_sha256','auxiliary_sha256'):need(len({r[k] for r in masks})==1 and sha_string(masks[0][k]),'Physical mask labels differ')
    caches=c['input_caches'];need(len(caches)==2 and all(sha_string(r['sha256']) and r['size_bytes']>0 for r in caches),'RAW cache proof missing')
    baseline=c['runtime_reference_baseline'];need(baseline['status']=='passed' and baseline['optimizer_updates']==0 and baseline['real_data_used'] is False and baseline['all_models_unchanged'] is True and baseline['prediction_windows']==1152,'Numerical baseline missing')
    checks=baseline['checks'];need(len(checks)==18 and {(r['method'],r['seed'],r['condition']) for r in checks}=={(m,s,k) for m in METHODS for s in SEEDS for k in ('normal','half','eighth')},'Baseline18 matrix incomplete')
    for r in checks:
        need(r['windows']==64 and r['passed'] is True and r['native_summary_and_by_rollout_parity'] is True,'Baseline parity failure')
        values=list(r['max_abs'].values())+[r['rotation_max_abs']];need(len(values)==7 and all(finite(v) and 0<=v<=2e-6 for v in values),'Baseline tolerance exceeded')
    for k in ('resource_start','resource_end'):
        r=c[k];need(r['passed'] is True and r['gpu_uuid']==GPU3 and r['gpu_index']==3 and r['cuda_device_count']==1 and r['cuda_visible_devices']==GPU3 and r['queue_parent_pid']==335161 and r['queue_parent_start_ticks']==53273268,'Runtime admission identity differs')
        need(set(r['active_gpu_pids'])<={335161,r['worker_pid']} and r['resource_contract']['sha256']==RESOURCE,'Runtime exclusive/resource proof missing')
    sources=c['source_records'];need(isinstance(sources,list) and len(sources)>0,'Source/input receipt inventory missing');seen={}
    for r in sources:
        need(sha_string(r['sha256']) and isinstance(r['size_bytes'],int) and r['size_bytes']>=0,'Bad source record')
        if r['path'] in seen:need(seen[r['path']]==r,'Conflicting source record')
        seen[r['path']]=r
    return dict(status='passed',phase=phase,runs=len(rows),optimizer_updates=c['new_optimizer_updates'],source_record_count=len(sources),external_source_and_checkpoint_bytes_rehashed=False)


def independent_checks(results):
    out={g:[] for g in ('center_stage36','matchednormal12')}
    for run in sorted(results,key=lambda r:(r['method'],r['seed'])):
        for condition in ('clean','half','eighth','matched_normal'):
            a=run['arms']['raw_C1'][condition];b=run['arms']['aux_visible_mask_center'][condition];g='matchednormal12' if condition=='matched_normal' else 'center_stage36'
            for metric,passed in [('center_epe_px',b['center_epe_px']<a['center_epe_px'] if condition=='eighth' else b['center_epe_px']<=a['center_epe_px']),('success',b['pose']['summary']['success']-a['pose']['summary']['success']>=(-.02 if condition=='clean' else 0.))]:
                out[g].append(dict(method=run['method'],seed=run['seed'],condition=condition,metric=metric,passed=passed))
    return out


def metric_row(method,seed,condition,arm,v):
    s=v['pose']['summary'];need(set(s)==set(METRICS) and all(finite(s[k]) for k in METRICS),'Nonfinite/incomplete native summary')
    need(finite(v['center_epe_px']) and abs(v['center_epe_px']-s['center_px'])<=2e-6,'Center/native metric scope mismatch')
    need(0<=s['success']<=1 and 0<=s['auc']<=1 and s['depth_mae_m']>=0 and s['center_px']>=0,'Invalid native metric range')
    return dict(method=method,seed=seed,condition=condition,arm=arm,windows=v['windows'],rollouts=v['rollouts'],center_epe_px=v['center_epe_px'],depth_mae_cm=s['depth_mae_m']*100,depth_bias_cm=s['depth_bias_m']*100,success_pct=s['success']*100,auc=s['auc'],log_size_mae=s['log_size_mae'],log_size_bias=s['log_size_bias'],auxiliary_mask_diagnostic=v['auxiliary_mask_diagnostic'])


def summarize(smoke,study=None,smoke_sha=None):
    proof=dict(smoke=verify_common(smoke,'smoke'));need(smoke['results']==[] and smoke['stage_gates'] is None,'Smoke must not claim study scores')
    summary=dict(schema='pose_aux_mask_actual_summary_v1',proof=proof,status='smoke_complete_study_not_supplied',new_real_scores=False,formal30_and8='not_retested_not_passed_by_this_probe',post_training_wrong_peak_measurement_available=False,
        wrong_peak_limitation='This frozen producer reports aggregate UV/native pose and auxiliary mask BCE/IoU, not candidate per-window attention/peak or no-target-cell counts; it cannot establish that the28 wrong peaks were fixed.',external_artifact_bytes_verified=False)
    if study is None:return summary,[],[]
    proof['study']=verify_common(study,'study');need(study['input_caches']==smoke['input_caches'],'Study cache is not the actual smoke cache')
    if smoke_sha is not None:need(any(r['sha256']==smoke_sha and Path(r['path']).name=='completion.json' for r in study['source_records']),'Study did not bind supplied actual smoke')
    results=study['results'];need(len(results)==6 and {(r['method'],r['seed']) for r in results}=={(m,s) for m in METHODS for s in SEEDS},'Study6 metrics matrix incomplete');rows=[];rollouts=[]
    for run in results:
        need(set(run['arms'])==set(ARMS),'Unexpected/missing scientific arms')
        for arm,views in run['arms'].items():
            need(set(views)==set(CONDITIONS),'Condition matrix incomplete')
            for condition,v in views.items():
                n,nr=(384,96) if condition=='clean' else (64,4);ids=v['window_ids'];need(v['windows']==n and v['rollouts']==nr and len(ids)==n and len(set(ids))==n,'Window inventory/scope differs')
                need(len(v['pose']['by_rollout'])==nr,'Rollout score inventory differs');rows.append(metric_row(run['method'],run['seed'],condition,arm,v))
                for rid,s in v['pose']['by_rollout'].items():
                    need(set(s)==set(METRICS) and all(finite(s[k]) for k in METRICS),'Nonfinite per-rollout native score');rollouts.append(dict(method=run['method'],seed=run['seed'],condition=condition,arm=arm,rid=rid,**s))
        for condition in CONDITIONS:
            a=run['arms']['raw_C1'][condition];b=run['arms']['aux_visible_mask_center'][condition]
            need(a['window_ids']==b['window_ids'],'Control/candidate physical windows differ')
            for metric in ('depth_mae_m','depth_bias_m','log_size_mae','log_size_bias'):need(a['pose']['summary'][metric]==b['pose']['summary'][metric],'Frozen RAW depth/size scores changed')
    independent=independent_checks(results);producer=study['stage_gates'];gate={};run_checks={}
    for name,n in (('center_stage36',36),('matchednormal12',12)):
        actual={(r['method'],r['seed'],r['condition'],r['metric']):r['passed'] for r in producer[name]['checks']};expected={(r['method'],r['seed'],r['condition'],r['metric']):r['passed'] for r in independent[name]}
        need(len(producer[name]['checks'])==len(actual)==len(expected)==n and actual==expected,'Producer gate differs from independent scalar recomputation')
        passed=all(expected.values());need(producer[name]['passed'] is passed,'Producer overall gate false claim');gate[name]=dict(passed=passed,passed_checks=sum(expected.values()),total_checks=n,checks=independent[name])
        for key,value in expected.items():run_checks.setdefault(key[:2],[]).append(value)
    means=[];paired=[];improvements=[]
    for method in METHODS:
        for condition in CONDITIONS:
            for arm in ARMS:
                rr=[r for r in rows if (r['method'],r['condition'],r['arm'])==(method,condition,arm)];need({r['seed'] for r in rr}==set(SEEDS),'Missing three-seed aggregate')
                metrics={k:dict(mean=statistics.mean(r[k] for r in rr),minimum=min(r[k] for r in rr),maximum=max(r[k] for r in rr)) for k in ('center_epe_px','depth_mae_cm','success_pct','auc')}
                means.append(dict(method=method,condition=condition,arm=arm,seeds=list(SEEDS),physical_rollouts=rr[0]['rollouts'],windows_per_seed=rr[0]['windows'],metrics=metrics))
            a=[r for r in rows if (r['method'],r['condition'],r['arm'])==(method,condition,'raw_C1')];b=[r for r in rows if (r['method'],r['condition'],r['arm'])==(method,condition,'aux_visible_mask_center')]
            aa={r['seed']:r for r in a};bb={r['seed']:r for r in b};improvements.append(dict(method=method,condition=condition,mean_center_delta_px=statistics.mean(bb[s]['center_epe_px']-aa[s]['center_epe_px'] for s in SEEDS),mean_success_delta_pp=statistics.mean(bb[s]['success_pct']-aa[s]['success_pct'] for s in SEEDS),strict_center_improved_seeds=sum(bb[s]['center_epe_px']<aa[s]['center_epe_px'] for s in SEEDS),pose_nonregressed_seeds=sum(bb[s]['success_pct']>=aa[s]['success_pct'] for s in SEEDS)))
    for condition in CONDITIONS:
        for arm in ARMS:
            a=next(r for r in means if (r['method'],r['condition'],r['arm'])==('eventgenesis',condition,arm));b=next(r for r in means if (r['method'],r['condition'],r['arm'])==('v2e',condition,arm));metrics={}
            for k,lower in (('center_epe_px',True),('depth_mae_cm',True),('success_pct',False),('auc',False)):
                x=a['metrics'][k]['mean'];y=b['metrics'][k]['mean'];winner='tie' if abs(x-y)<=1e-12 else ('eventgenesis' if ((x<y) if lower else (x>y)) else 'v2e');metrics[k]=dict(eventgenesis=x,v2e=y,eventgenesis_minus_v2e=x-y,winner=winner)
            paired.append(dict(condition=condition,arm=arm,metrics=metrics))
    each=[dict(method=m,seed=s,passed_checks=sum(run_checks[(m,s)]),total_checks=8,all_registered_center_checks_passed=all(run_checks[(m,s)])) for m in METHODS for s in SEEDS]
    summary.update(status='study_complete',stage_gates=gate,run_gate_results=each,all_six_runs_failed_registered_center_checks=not any(r['all_registered_center_checks_passed'] for r in each),
        both_gate_groups_passed=gate['center_stage36']['passed'] and gate['matchednormal12']['passed'],per_run_condition_arm=rows,three_seed_means_and_ranges=means,paired_generator_comparisons=paired,within_method_candidate_deltas=improvements,
        aggregation='Each native summary already averages its rollouts; all3 seeds are then equally weighted. Clean384 covers96 rollouts; other conditions cover fixed4 rollouts64windows. Adjacent windows and model seeds are not independent scene replicates; no p-values.')
    return summary,rows,rollouts


def write_csv(path,rows):
    if not rows:return
    with path.open('x',newline='') as f:
        fieldnames=[k for k in rows[0] if k!='auxiliary_mask_diagnostic'];w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore');w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',type=Path,required=True);p.add_argument('--smoke-sha256',required=True);p.add_argument('--study',type=Path);p.add_argument('--study-sha256');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    need(not a.out.exists() and a.out.parent.is_dir(),'Fresh output directory required');s=pinned(a.smoke,a.smoke_sha256);t=None
    if a.study is not None:need(a.study_sha256 is not None,'External study SHA required');t=pinned(a.study,a.study_sha256)
    report,rows,rr=summarize(s,t,a.smoke_sha256);a.out.mkdir();report['inputs']=[record(a.smoke)]+([record(a.study)] if a.study else []);report['summarizer_source_sha256']=digest(__file__)
    (a.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');write_csv(a.out/'per_run_condition_arm.csv',rows);write_csv(a.out/'per_run_rollout.csv',rr)
    manifest={q.name:dict(size_bytes=q.stat().st_size,sha256=digest(q)) for q in a.out.iterdir() if q.is_file()};(a.out/'artifact_index.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],stage_gates={k:dict(passed=v['passed'],passed_checks=v['passed_checks'],total_checks=v['total_checks']) for k,v in report.get('stage_gates',{}).items()},wrong_peak_available=report['post_training_wrong_peak_measurement_available']),indent=2))


if __name__=='__main__':main()
