"""Read archived pose-factor evidence only; no SSH/GPU/experiment mutation."""
import argparse,collections,datetime,hashlib,json,math,statistics,traceback
from pathlib import Path

CHECKS=[]
def need(ok,message,details=None):
    CHECKS.append(dict(passed=bool(ok),check=message,details=details))
    if not ok:raise AssertionError(message+': '+str(details))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def close(a,b):return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10)
def record(p):return dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size)
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def scalar_audit(raw):
    m=raw['mixture'];groups=('full','features','center','size');comparison=m['comparisons']
    need(m['weights']==dict(normal=.5,half=.375,eighth=.125,normal_teacher=2.),'Canonical C1/slow expected mixture weights')
    P=m['physical_expected'];T=m['normal_teacher_weighted'];G=m['actual_expected_mixture'];project=m['projected_teacher_diagnostic']
    p=P['full']['norm'];t=T['full']['norm'];d=comparison['physical_vs_teacher']['full']['dot'];alpha=min(d/(p*p),0.) if p else 0.
    need(close(alpha,project['full_space_removed_negative_coefficient']),'Global one-sided coefficient independently recomputed')
    for group in groups:
        pp=P[group]['norm'];tt=T[group]['norm'];pt=comparison['physical_vs_teacher'][group]
        dot=pt['dot'];gg=G[group]['norm'];want=(max(0.,pp*pp+tt*tt+2*dot))**.5
        need(close(want,gg),'Actual mixture norm equals norm(P+2teacher)',group)
        need(close(pt['left_norm'],pp) and close(pt['right_norm'],tt) and close(pt['cosine'],dot/(pp*tt)),'Raw P/T cosine and full-prefix norm identity',group)
        need(close(gg/(pp+tt),m['cancellation_ratio'][group]),'Independent cancellation ratio',group)
        need(close(tt,2*raw['losses_and_gradients']['normal_teacher']['gradients'][group]['norm']),'T already includes original teacher weight2',group)
        n=raw['losses_and_gradients']['normal/supervised']['gradients'][group]['norm'];h=raw['losses_and_gradients']['half/supervised']['gradients'][group]['norm'];e=raw['losses_and_gradients']['eighth/supervised']['gradients'][group]['norm']
        pn2=.5**2*n*n+.375**2*h*h+.125**2*e*e
        pn2+=2*(.5*.375*comparison['normal_vs_half'][group]['dot']+.5*.125*comparison['normal_vs_eighth'][group]['dot']+.375*.125*comparison['half_vs_eighth'][group]['dot'])
        need(close(max(pn2,0.)**.5,pp),'Physical expected mixture norm independently reconstructed from three tasks',group)
        proj_t2=tt*tt-2*alpha*dot+alpha*alpha*pp*pp
        proj_g2=(1-alpha)**2*pp*pp+tt*tt+2*(1-alpha)*dot
        need(close(max(proj_t2,0.)**.5,project['projected_teacher'][group]['norm']) and close(max(proj_g2,0.)**.5,project['projected_mixture'][group]['norm']),'One full-space projection scalar preserves prefix algebra',group)
        for condition in ('normal','half','eighth'):
            pc=comparison['physical_vs_'+condition][group];tc=comparison['teacher_vs_'+condition][group]
            actual=comparison['actual_expected_mixture_vs_'+condition][group];projected=comparison['projected_teacher_mixture_vs_'+condition][group]
            need(close(actual['dot'],pc['dot']+tc['dot']),'Actual final gradient-to-task dot reconstructed',dict(group=group,condition=condition))
            need(close(actual['cosine'],actual['dot']/(gg*actual['right_norm'])),'Actual final task cosine consistent with norm/dot',dict(group=group,condition=condition))
            need(close(projected['dot'],(1-alpha)*pc['dot']+tc['dot']),'Projected direction-to-task dot reconstructed',dict(group=group,condition=condition))
    for name in ('physical_expected','normal_teacher_weighted','actual_expected_mixture'):
        values=m[name];need(close(values['full']['norm']**2,sum(values[k]['norm']**2 for k in groups[1:])),'Full vector covers features/center/size norm decomposition',name)

def summary_from_raw(raw):
    m=raw['mixture'];c=m['comparisons'];full=lambda key:c[key]['full']['cosine']
    return dict(method=raw['method'],seed=raw['seed'],arm=raw['arm'],branch=raw['branch'],
        P_T_cos=full('physical_vs_teacher'),Pnorm=m['physical_expected']['full']['norm'],Tnorm=m['normal_teacher_weighted']['full']['norm'],cancel=m['cancellation_ratio']['full'],
        final_H_cos=full('actual_expected_mixture_vs_half'),final_D_cos=full('actual_expected_mixture_vs_eighth'),
        P_H_cos=full('physical_vs_half'),P_D_cos=full('physical_vs_eighth'),project_H_cos=full('projected_teacher_mixture_vs_half'),project_D_cos=full('projected_teacher_mixture_vs_eighth'))

def run(root):
    e=root/'evidence';index=read(e/'EVIDENCE_INDEX.json')
    need(not index['new_real_scores'] and index['optimizer_updates_in_diagnostics']==0,'Evidence snapshot has no new real scores or optimizer updates')
    for name,b in index['files'].items():
        p=(e/name).resolve();need(p.is_relative_to(e.resolve()) and p.is_file() and sha(p)==b['sha256'] and p.stat().st_size==b['size_bytes'],'Archived evidence bytes match index',name)
    sources=[]
    for manifest in root.glob('sources/**/source_manifest.json'):
        value=read(manifest);files=value.get('files',value)
        for name,b in files.items():
            p=manifest.parent/name;digest=b if isinstance(b,str) else b['sha256'];need(Path(name).name==name and p.is_file() and sha(p)==digest,'Final source manifest bytes match',str(p))
        sources.append(record(manifest))
    completion=read(e/'train_probe/completion.json');summary=read(e/'train_gradient_summary.json')
    need(completion['status']=='complete' and completion['models']==36 and completion['optimizer_updates']==0 and not completion['real_scores_read'] and not completion['Adam_direction_computed'],'Actual36-endpoint TRAIN diagnostic complete and zero update')
    need(not completion['validation_gradients_used'] and not completion['real_gradients_used'] and not completion['affine_applied'],'Probe uses fixed unaugmented TRAIN, no validation/real tuning')
    expected={(m,s,a,b) for m in ('eventgenesis','v2e') for s in (0,1,2) for a in ('C0','C1','slow') for b in ('depth','center')}
    keys=lambda rows:{(x['method'],x['seed'],x['arm'],x['branch']) for x in rows}
    need(len(summary)==len(completion['results'])==36 and keys(summary)==keys(completion['results'])==expected,'All36 endpoint summaries/receipts retained')
    external=index['external_artifacts'];need(isinstance(external,dict),'External raw endpoint index is explicit')
    artifact_bindings={**external,**{v['server_path']:v for v in index['files'].values()}}
    for result in completion['results']:
        need(artifact_bindings[result['path']]['sha256']==result['sha256'],'Each raw endpoint SHA bound by completion and external artifact index',result['path'])
    need(completion['contract_sha256']==sha(root/'sources/diagnostic/contract.json') and completion['worker_sha256']==sha(root/'sources/diagnostic/probe_pose_training_losses.py'),'Actual TRAIN probe binds archived source/contract')
    labels={v['labels_sha256'] for v in completion['cohort'].values()};need(len(labels)==1 and all(v['windows']==32 and len(v['rollouts'])==8 and len(v['sample_ids'])==32 and not v['validation_samples_used'] and not v['affine_applied'] for v in completion['cohort'].values()),'TRAIN32 is matching first8rollouts/4windows with paired labels')
    audited=[]
    for method in ('eventgenesis','v2e'):
        path=e/f'train_probe/results/{method}_seed0_C1_depth.json';raw=read(path);slot=next(v for v in summary if (v['method'],v['seed'],v['arm'],v['branch'])==(method,0,'C1','depth'))
        entry=next(v for v in completion['results'] if (v['method'],v['seed'],v['arm'],v['branch'])==(method,0,'C1','depth'))
        need(sha(path)==entry['sha256'],'Fixed raw C1 seed0 example bytes match36-endpoint receipt',method)
        need(raw['optimizer_updates']==0 and not raw['Adam_direction_computed'] and raw['training_windows']==32 and raw['cohort_labels_sha256'] in labels,'Raw diagnostic obeys fixed cohort and no-Adam scope',method)
        need(raw['model_state_before_sha256']==raw['model_state_after_sha256']==raw['source_training_record']['final_state_sha256'] and raw['teacher_state_sha256']==raw['source_training_record']['source_state_sha256'],'Raw model/teacher states align with actual endpoint training receipts',method)
        need(completion['verified_original_file_sha256'][raw['model_path']]==raw['model_file_sha256'] and completion['verified_original_file_sha256'][raw['teacher_path']]==raw['teacher_file_sha256'],'Raw model/teacher byte hashes align with final source check',method)
        need(len(raw['parameter_inventory'])==24 and canonical(raw['parameter_inventory'])==raw['parameter_inventory_sha256'],'Complete24 named parameters represented',method)
        scalar_audit(raw);rebuilt=summary_from_raw(raw)
        need(all(slot[k]==v if isinstance(v,str) else close(slot[k],v) for k,v in rebuilt.items()),'Raw P/T norms/cancel/final/projected directions exactly align with compact summary',method)
        audited.append(dict(method=method,raw=record(path),recomputed=rebuilt))
    branch=lambda m,b:[next(v for v in summary if (v['method'],v['seed'],v['arm'],v['branch'])==(m,s,'C1',b)) for s in range(3)]
    eg=branch('eventgenesis','depth');v2=branch('v2e','depth');centers=branch('eventgenesis','center')
    need(all(v['P_T_cos']<-.9 and v['cancel']<.25 for v in eg),'EG C1 depth three endpoint summaries show strong conflict and cancellation')
    need(all(v['Tnorm']/v['Pnorm']<.2 for v in v2),'V2E teacher is relatively weak, not universally conflict-free')
    need(all(v['final_H_cos']>.987 and v['final_D_cos']>.962 for v in v2),'V2E actual expected mixture remains aligned with both relit TRAIN tasks (exact minimum half 0.987787, dark 0.962460)')
    need(all(v['final_H_cos']>.85 and v['final_D_cos']>.79 for v in centers),'EG center final direction does not show the depth reversal pattern')
    export=read(e/'export/completion.json');scores=read(e/'export/native_scores.json');parity=read(e/'export/native_parity.json')
    need(export['status']=='complete' and export['optimizer_updates']==0 and not export['real_data_used'] and export['models_and_sources_unchanged'] and export['native_parity_passed'],'Forward export is frozen/no-training and native parity passed')
    need(export['counters']['predictions']==3456 and len(scores)==54 and len(parity)==54 and all(v['passed'] for v in parity),'Complete54 matched64 predictions and separate54 native parity cases')
    normal=[v for v in scores if v['condition']=='normal'];normal_ids={tuple(sorted(v['score']['by_rollout'])) for v in normal}
    need(len(normal)==18 and len(normal_ids)==1 and len(next(iter(normal_ids)))==4,'Matched normal is the same4 trajectories across models')
    need(all(tuple(sorted(v['score']['by_rollout'])) in normal_ids for v in scores),'Matched half/eighth share the same4 trajectory IDs')
    clean384=[v for v in parity if v['condition']=='clean384'];need(len(clean384)==18 and all(len(v['score']['by_rollout'])==96 for v in clean384),'Normal clean384/96rollout parity remains distinct from matchednormal64')
    matched=read(e/'matched_synthetic_summary.json')
    for slot in matched:
        ss=[v['score']['summary'] for v in scores if v['method']==slot['method'] and v['arm']==slot['arm'] and v['condition']==slot['condition']]
        need(len(ss)==3 and all(close(value,statistics.mean(x[k] for x in ss)) for k,value in slot['metrics'].items()),'Matched summary is the full3-seed mean, not clean384',dict(method=slot['method'],arm=slot['arm'],condition=slot['condition']))
    gains=read(e/'count_gain_summary_C1.json');attention=read(e/'attention_decode_summary.json');context=read(e/'context_swap_summary_C1.json')
    need(len(gains)==12 and {(x['method'],x['condition'],x['gain']) for x in gains}=={(m,c,g) for m in ('eventgenesis','v2e') for c in ('normal','half','eighth') for g in (.5,2.)},'Both gain tests and every lighting/method retained')
    c1=lambda m,c:next(v['metrics'] for v in matched if v['method']==m and v['arm']=='C1' and v['condition']==c)
    gap=c1('eventgenesis','eighth')['center_px']-c1('v2e','eighth')['center_px'];eg_gain=[v for v in gains if v['method']=='eventgenesis']
    need(max(v['center_change_px'] for v in eg_gain)<1 and max(v['center_change_px'] for v in eg_gain)<gap/5,'Tested scalar count gains change center far less than observed dark gap')
    local=next(v for v in attention if (v['method'],v['arm'],v['condition'])==('eventgenesis','C1','eighth'))
    need(local['local3_px']>local['global_px'],'Dark local3 attention alternative worsens center; negative result retained')
    need(len(context)==12,'All same-geometry cross-light global context substitutions retained')
    projection=root/'sources/teacher_projection_v2';cpu=read(projection/'CPU_TEST_RESULT.json')
    need(sha(projection/'projection.py')=='71f115263723d997979a37512211e2bd3d944886d19564df7fcb1e1c35cae9d2','One-sided projection scientific source is exact frozen71f module')
    need(cpu['status']=='passed' and cpu['tests']==13 and cpu['projection_tests']==7 and cpu['actual_schema_admission_tests']==3 and not cpu['GPU_used'],'Actual13 CPU checks including preserved admission failure and pure projection passed')
    reg=read(projection/'contract.json')
    need(reg['candidate_center']=='reuse_C1_unchanged' and reg['feature_learning_rate']==reg['head_learning_rate']==.00015 and reg['half_probability']==.75 and reg['normal_teacher_weight']==2 and reg['auxiliary_teacher_weight']==0 and reg['new_depth_optimizer_updates']==1728,'Registered intervention is depth-only with original LR/mixture/labels/control reuse')
    need(reg['train_probe_completion_sha256']==sha(e/'train_probe/completion.json') and reg['forward_completion_sha256']==sha(e/'export/completion.json'),'Projection registration binds these exact diagnostics, no new real feedback')
    return dict(evidence_files_verified=len(index['files']),source_groups=sources,raw_seed0_cases=audited,
        other_endpoint_bound_descriptors=34,other_raw_endpoint_JSON_replayed=False,raw_gradient_vectors_recomputed=False,
        three_seed_C1_depth=dict(eventgenesis=eg,v2e=v2),matched_validation=dict(trajectories=4,windows=64,clean384_separate=True),
        negative_controls=dict(count_gain_EG_max_center_change_px=max(v['center_change_px'] for v in eg_gain),count_gain_EG_max_depth_change_cm=max(v['abs_z_change_cm'] for v in eg_gain),
            C1_dark_center_gap_px=gap,EG_local3_dark_px=local['local3_px'],EG_softargmax_dark_px=local['global_px'],
            EG_max_context_depth_change_cm=max(v['abs_z_change_cm'] for v in context if v['method']=='eventgenesis'),V2E_max_context_depth_change_cm=max(v['abs_z_change_cm'] for v in context if v['method']=='v2e')),
        interpretation=dict(supported='Fixed TRAIN32/3seed C1 depth endpoints support a shared depth-only one-sided teacher-projection short experiment.',
            caveats=['Reverse supervised/regularizer gradients can occur naturally at a trained stationary compromise.','Local Euclidean gradients and expected mixture are not the realized AdamW momentum/preconditioned update.','Summary covers36 endpoints, but only2 fixed raw endpoint JSONs were copied locally; scalar consistency is not an independent GPU replay.','At C0 the canonical C1 mixture is counterfactual, not the historical normal-only objective.','Scalar gain/global-context/local3 controls are narrow negative tests, not a rejection of every density/noise/feature hypothesis.','Four matched validation trajectories are a development cohort; no population significance or new real-domain advantage.','Projection improves an instantaneous aggregate supervised direction algebraically, not necessarily each task, validation30+8 gate, or performance relative toV2E.'],
            forbidden_claims=['Unique cause proved','Adam steps proved wrong','All density/noise explanations excluded','Projection already winsV2E or guarantees success']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,default=Path('/home/科研/Eventbased_WAM/repositories/gec-mainline-resume-20261105/benchmark/sim2real/reviews/pose_factor_illumination_20261108'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.review.resolve(strict=True);need(not a.out.exists() and a.out.resolve().is_relative_to(root/'verification'),'Unique verification output only')
    result=dict(schema='pose_factor_independent_scalar_review_v1',status='failed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_script=record(Path(__file__).resolve()),checks=CHECKS,no_SSH=True,no_training_or_optimizer_updates=True,no_source_report_status_git_changes=True)
    try:result['audit']=run(root);result['status']='passed'
    except BaseException:result['error']=traceback.format_exc()
    result['passed_checks']=sum(v['passed'] for v in CHECKS);result['failed_checks']=sum(not v['passed'] for v in CHECKS)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],passed_checks=result['passed_checks'],failed_checks=result['failed_checks'],error=result.get('error'),output=str(a.out)),indent=2))
    if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
