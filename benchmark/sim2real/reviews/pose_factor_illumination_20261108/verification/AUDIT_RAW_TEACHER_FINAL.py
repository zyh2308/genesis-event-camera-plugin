"""Read only the new RAW-teacher evidence; no SSH, optimizer or old audit rerun."""
import argparse,datetime,hashlib,json,math,statistics,traceback
from pathlib import Path
SOURCE='bdd07e6d176374f9f37dad34731e894a33b17307cfaaa4e0b1e93991e46eddc3'
POLICY='9e713cc63aefe10c53b27505101f1a4271483a20a1657b5c64ca2de838d9aa04'
ORIGINAL='e1883b68715ecf067a9f4dccbd7a61af5f5dc26b557e061859f4066e7de8d38c'
CONTRACT='d682813400a1709c90b47b4dcc73a83c8f7ea3fe830e9d837316555dea11ca41'
PREVIOUS_SOURCE='db07f866cc0ce6535a2f1917ea324b954bf5b3bc3d96966e9553bc56be4b8037'
CHECKS=[]
def need(ok,msg):
    CHECKS.append(dict(check=msg,passed=bool(ok)))
    if not ok:raise AssertionError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
def draw(r):return {k:r[k] for k in ('update','normal_indices','auxiliary_indices','half_mask')}
def drawsha(rows):return hashlib.sha256(json.dumps([draw(x) for x in rows]).encode()).hexdigest()
def old_parity(a,b):return a.keys()==b.keys() and all(math.isclose(a[k],v,rel_tol=1e-6,abs_tol=1e-8) for k,v in b.items())
def audit(root,sources):
    index=read(root/'RAW_TEACHER_FINAL_INDEX.json');previous=read(root.parent/'baf_center_execution/study/completion.json')
    old_smoke=read(root.parent/'baf_center_execution/smoke/completion.json')
    need(index['schema']=='actual_RAW_teacher_BAF_center_short_final_v1' and not index['real_scores_computed'],'New evidence index isolated from real scoring')
    need(index['actual_source_manifest_sha256']==SOURCE and sha(sources/'source_manifest.json')==SOURCE,'Actual reviewed bdd07 source freeze')
    for name,r in index['files'].items():
        p=(root/name).resolve();need(p.is_relative_to(root.resolve()) and p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256'],'New immutable artifact: '+name)
    for name,digest in read(sources/'source_manifest.json').items():need(Path(name).name==name and sha(sources/name)==digest,'Exact frozen source: '+name)
    need(sha(sources/'contract.json')==CONTRACT and sha(sources/'raw_teacher_policy.py')==POLICY and sha(sources/'pose_priority_policy.py')==ORIGINAL,'d682 contract and both policy SHA pins')
    derivation=read(sources/'POLICY_DERIVATION.json');data=(sources/'raw_teacher_policy.py').read_bytes()
    need(len(derivation['replacements'])==2 and derivation['original_sha256']==ORIGINAL and derivation['derived_sha256']==POLICY,'Only two registered policy replacements')
    for item in reversed(derivation['replacements']):
        new,old=item['new'].encode(),item['old'].encode();need(data.count(new)==item['count']==1,'Unique derived line: '+item['new']);data=data.replace(new,old)
    need(data==(sources/'pose_priority_policy.py').read_bytes(),'Reverse two changes restores every original e188 byte')
    cpu=read(root/'cpu_admission.json');log=(root/'actual_server_cpu_tests.log').read_text(encoding='utf-8')
    need(cpu['status']=='passed' and cpu['actual_server_tests']==23 and cpu['exit_code']==0 and cpu['source_manifest_sha256']==SOURCE and cpu['contract_sha256']==CONTRACT and cpu['log_sha256']==sha(root/'actual_server_cpu_tests.log'),'Actual23 server CPU tests bound to final source/log')
    need('Ran 23 tests' in log and log.rstrip().endswith('OK'),'Actual server test output is complete23, not local2 AST-only')
    for job in ('164_pose_baf_raw_teacher_smoke_20261108','165_pose_baf_raw_teacher_study_20261108'):
        q=read(root/'queue'/(job+'.done.json'));need(q['status']=='complete' and q['exit_code']==0 and q['job_id']==job,'Actual child exit0: '+job)
    smoke,study=[read(root/p/'completion.json') for p in ('smoke','study')];external={r['path']:r for r in index['external_checkpoints']}
    for phase,c in [('smoke',smoke),('study',study)]:
        need(c['status']=='complete_synthetic_only' and c['phase']==phase and c['schema']=='pose_baf_raw_teacher_adaptation_v1','Complete new phase: '+phase)
        need(c['contract_sha256']==CONTRACT and c['source_manifest_sha256']==SOURCE and c['policy_sha256']==POLICY and not c['real_scores_computed'],'Phase science/isolation pins: '+phase)
        need(c['student_input']=='unchanged_previous_fixed_BAF128_cache' and c['normal_teacher_input']=='RAW_normal_paired_same_augmentation','Student/teacher roles explicit')
        need(c['checkpoint_verification_scope']=='model_optimizer_readback_only_no_resume_claim' and c['resume_supported'] is False,'No RNG/cursor continuation claim')
        need(c['original30_gate']['status']==c['original8_confirmation']['status']=='not_evaluated' and c['original30_gate']['thresholds_unchanged'] and c['original8_confirmation']['thresholds_unchanged'],'Original global30+8 unchanged and closed')
        admissions=c['full288_admissions'];need(len(admissions)==6 and {(a['method'],a['seed']) for a in admissions}=={(m,s) for m in ('eventgenesis','v2e') for s in range(3)} and all(a['full288_replay_exact'] and a['raw_teacher_paired_replay_exact'] for a in admissions),'Original and paired-RAW all-six288 pre-update replay complete')
        pre=read(root/phase/'pre_update_admission.json');need(pre['status']=='passed' and pre['optimizer_updates']==0 and pre['admissions']==admissions,'Full admissions preserved before updates')
        for a in admissions:
            old=next(x for x in previous['full288_admissions'] if (x['method'],x['seed'])==(a['method'],a['seed']))
            need(all(a[k]==old[k] for k in old),'Original C1 init/log/label/draw admission remains exactly previous BAF')
        for seed in range(3):need(len({a['labels_sha256'] for a in admissions if a['seed']==seed})==len({a['trajectory_sha256'] for a in admissions if a['seed']==seed})==1,'Cross-generator label/draw pairing: '+str(seed))
        need(c['input_caches']==previous['input_caches']==old_smoke['input_caches'],'Exactly previous fixed BAF student caches, not regenerated')
        need(c['filtered_teacher_control']['immutable_reuse'] and c['filtered_teacher_control']['new_optimizer_updates']==0 and c['filtered_teacher_control']['sha256']==sha(root.parent/'baf_center_execution/study/completion.json'),'Failed old filtered teacher is immutable reuse with zero new updates')
    need(smoke['full288_admissions']==study['full288_admissions'],'Study admissions exactly equal actual smoke')
    need(smoke['raw_teacher_caches']==study['raw_teacher_caches'] and len(study['raw_teacher_caches'])==2,'Two immutable new RAW-normal teacher caches reused from smoke')
    need(index['external_input_caches']==study['input_caches']+study['raw_teacher_caches'],'All four RAW/BAF caches externally indexed')
    need(smoke['new_optimizer_updates']==2 and len(smoke['records'])==2 and all(r['updates']==1 for r in smoke['records']),'Two discarded one-update smoke branches')
    need(study['new_optimizer_updates']==1728 and len(study['records'])==6 and all(r['updates']==288 for r in study['records']),'Exactly six new center288 branches; no depth training')
    for phase,c in [('smoke',smoke),('study',study)]:
        for r in c['records']:
            m,s=r['method'],r['seed'];a=next(a for a in c['full288_admissions'] if (a['method'],a['seed'])==(m,s))
            need(r['branch']=='center' and r['canonical_arm']=='raw_teacher_baf_center' and r['teacher_input']=='RAW_normal_paired_affine' and r['student_input']=='unchanged_fixed_BAF' and r['source_state_sha256']==a['source_state_sha256'],'Single-route center initialization and roles')
            need(r['teacher_unchanged'] and r['independent_model_optimizer_rebuild'] and r['optimizer_state_equal'],'Native teacher fixed/model+Adam reconstruction proof')
            need(external[r['checkpoint_path']]['sha256']==r['checkpoint_sha256'] and external[r['final_model']['path']]==r['final_model'],'Actual new model/checkpoint byte bindings')
            p=root/phase/'training_evidence'/f'{m}_seed{s}_physical_supervised_center.jsonl';rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
            need(len(rows)==r['updates'] and [x['update'] for x in rows]==list(range(1,r['updates']+1)) and sha(p)==r['training_log']['sha256'],'New update log contiguous, bounded, immutable')
            bad=[]
            for x in rows:
                finite=all(math.isfinite(x[k]) for k in ('supervised','clean_retention','half_same_view_retention','loss','gradient_l2')) and x['gradient_l2']>0 and x['finite_gradient_tensors']==20
                objective=x['half_same_view_retention']==0 and math.isclose(x['loss'],x['supervised']+2*x['clean_retention'],rel_tol=1e-6,abs_tol=1e-6)
                paired=len(x['normal_indices'])==len(x['auxiliary_indices'])==len(x['half_mask']) and 0<len(x['normal_indices'])<=32 and all(0<=i<1529 for i in x['normal_indices']) and all(0<=i<3058 for i in x['auxiliary_indices']) and all((i<1529)==v for i,v in zip(x['auxiliary_indices'],x['half_mask']))
                if not (finite and objective and paired and x['seed']==s and x['branch']=='center' and x['arm']=='physical_supervised'):bad.append(x['update'])
            need(not bad,'Every actual update finite/original objective/paired bounds: '+phase+m+str(s)+' bad='+str(bad))
            if phase=='study':need(r['labels_sha256']==a['labels_sha256'] and drawsha(rows)==a['trajectory_sha256'],'All288 new labels/draws equal original C1 and failed filtered-teacher study')
    rb=read(root/'actual_smoke_readback.json');need(rb['status']=='passed' and rb['new_optimizer_updates']==0 and rb['GPU_used'] is False and rb['resume_supported'] is False and rb['receipt_sha256']==sha(root/'smoke/completion.json') and rb['source_manifest_sha256']==SOURCE and rb['verifier_sha256']==sha(root/'verify_actual_smoke.py'),'Independent actual paired smoke readback byte/source bindings')
    need(len(rb['checks'])==2,'Both smoke methods independently rebuilt')
    for x in rb['checks']:
        r=next(r for r in smoke['records'] if (r['method'],r['seed'])==(x['method'],x['seed']))
        need(x['Adam_steps']==[1] and x['frozen_size_exact'] and x['CPU_forward_max_abs']==0 and x['model_state_sha256']==r['final_state_sha256'] and x['checkpoint_sha256']==r['checkpoint_sha256'],'Actual smoke model/Adam/size/fixed forward match')
    rebuilt=[];depthkeys=('depth_mae_m','depth_bias_m','log_size_mae','log_size_bias');results=study['results']
    need(len(results)==6 and {(r['method'],r['seed']) for r in results}=={(m,s) for m in ('eventgenesis','v2e') for s in range(3)},'Six complete evaluated method/seed branches')
    for r in results:
        old=next(x for x in previous['results'] if (x['method'],x['seed'])==(r['method'],r['seed']))
        need(set(r['arms'])=={'raw_C1','frozen_C1_BAF','raw_teacher_baf_center','filtered_teacher_BAF'},'Four role-labelled arms, no per-condition model switching')
        need(r['arms']['filtered_teacher_BAF']==old['arms']['baf_center'],'Old failed filtered-teacher stats reused exactly, including every scope/trajectory')
        for cond in ('clean','matched_normal','half','eighth'):
            arms=r['arms'];before=arms['raw_C1'][cond];after=arms['raw_teacher_baf_center'][cond]
            for arm in arms:
                v=arms[arm][cond];need(v['windows']==(384 if cond=='clean' else 64) and v['rollouts']==(96 if cond=='clean' else 4),'Original clean384 vs matched4x64 scope')
                need(all(v['pose']['summary'][k]==before['pose']['summary'][k] for k in depthkeys) and all(all(v['pose']['by_rollout'][rid][k]==x[k] for k in depthkeys) for rid,x in before['pose']['by_rollout'].items()),'RAW depth/size outputs unchanged for all four arms and every trajectory')
            for arm in ('raw_C1','frozen_C1_BAF'):
                a,b=arms[arm][cond],old['arms'][arm][cond]
                need(math.isclose(a['center_epe_px'],b['center_epe_px'],rel_tol=1e-6,abs_tol=1e-8) and old_parity(a['pose']['summary'],b['pose']['summary']),'Recomputed fixed baseline retains original native tolerance')
            delta=r['teacher_route_effect'][cond]
            need(close(delta['center_epe_delta_px'],after['center_epe_px']-old['arms']['baf_center'][cond]['center_epe_px']) and close(delta['pose_success_delta'],after['pose']['summary']['success']-old['arms']['baf_center'][cond]['pose']['summary']['success']),'Reported new-versus-old teacher effect arithmetic')
            if cond=='matched_normal':continue
            rebuilt.append(dict(method=r['method'],seed=r['seed'],condition=cond,metric='center_epe_px',passed=after['center_epe_px']<before['center_epe_px'] if cond=='eighth' else after['center_epe_px']<=before['center_epe_px']))
            rebuilt.append(dict(method=r['method'],seed=r['seed'],condition=cond,metric='success',passed=after['pose']['summary']['success']-before['pose']['summary']['success']>=(-.02 if cond=='clean' else 0.)))
    gate=study['center_stage_gate'];need(gate['scope']=='center_stage_only_36_checks_not_global30' and gate['checks']==rebuilt and len(rebuilt)==36 and sum(x['passed'] for x in rebuilt)==19 and not gate['passed'],'All36 original center-stage checks independently recomputed:19pass17fail')
    table=read(root/'derived/rt_summary.json');ridtable=read(root/'derived/rt_rid_summary.json');need(len(table)==8 and len(ridtable)==216,'All8 method/light means and216trajectory rows retained')
    for collection,perrid in [(table,False),(ridtable,True)]:
        for row in collection:
            cells=[r for r in results if r['method']==row['method']]
            for arm,metrics in row['arms'].items():
                scores=[r['arms'][arm][row['condition']]['pose']['by_rollout'][row['rid']] if perrid else r['arms'][arm][row['condition']]['pose']['summary'] for r in cells]
                expected=dict(center_px=statistics.mean(x['center_px'] for x in scores),depth_cm=statistics.mean(x['depth_mae_m'] for x in scores)*100,success_percent=statistics.mean(x['success'] for x in scores)*100)
                need(metrics.keys()==expected.keys() and all(close(metrics[k],v) for k,v in expected.items()),'Independent3seed recomputation: '+('trajectory' if perrid else 'condition'))
    # Frozen rotation uses the same RAW model/source but predicted center crop;
    # predicted R is expected to vary across center arms, and is not forced equal.
    oldpins={r['path']:r['sha256'] for r in previous['source_records']};newpins={r['path']:r['sha256'] for r in study['source_records']}
    frozen=[p for p in oldpins if p.endswith('.pt') and ('_rotation.pt' in p or '_physical_supervised.pt' in p)]
    need(frozen and all(newpins.get(p)==oldpins[p] for p in frozen),'Frozen RAW depth/rotation checkpoint bytes retain old source hashes')
    return dict(indexed_artifacts=len(index['files']),source_manifest_sha256=SOURCE,actual_server_CPU_tests=23,new_center_updates=1728,new_depth_updates=0,center_stage=dict(passed=19,failed=17,adopted=False),global30_and8='not_evaluated',means=table,trajectory_rows=216,limitations=['This is a local scalar/hash/receipt audit, not a new torch/GPU reconstruction or cache tensor replay.','Model/checkpoint/raw teacher cache bytes remain externally indexed; actual server independent smoke readback is verified by its pinned script and receipt.','Native RNG/Adam snapshots do not establish resumed-next-step or whole-training trajectory equivalence.','Rotation weights/RAW input remain frozen; predicted rotations can change with predicted-center crop/ray frame.','Failed center gate does not authorize long training or real E-POSE scoring.'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--sources',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();need(not a.out.exists(),'Unique audit output')
    d=dict(schema='RAW_teacher_BAF_final_independent_audit_v1',status='failed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_script_sha256=sha(Path(__file__)),no_SSH=True,no_training=True,checks=CHECKS)
    try:d['audit']=audit(a.evidence,a.sources);d['status']='passed'
    except BaseException:d['error']=traceback.format_exc()
    d.update(passed_checks=sum(c['passed'] for c in CHECKS),failed_checks=sum(not c['passed'] for c in CHECKS));a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps({k:d[k] for k in ('status','passed_checks','failed_checks')}));print(d.get('error',''))
    if d['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
