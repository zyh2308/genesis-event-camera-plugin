"""Read immutable BAF short-trial receipts/logs; no SSH, torch or training."""
import argparse,datetime,hashlib,json,math,statistics,traceback
from pathlib import Path

SOURCE='db07f866cc0ce6535a2f1917ea324b954bf5b3bc3d96966e9553bc56be4b8037'
CONTRACT='60db65c84cba5c6045542eca90640017f4e863e5e8ad9cab774dae778deee762'
CHECKS=[]
def need(ok,msg):
    CHECKS.append({'check':msg,'passed':bool(ok)})
    if not ok:raise AssertionError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10)
def draw(row):return {k:row[k] for k in ('update','normal_indices','auxiliary_indices','half_mask')}
def canonical_draws(rows):return hashlib.sha256(json.dumps([draw(x) for x in rows]).encode()).hexdigest()
def numeric_equal(a,b,where):
    need(a.keys()==b.keys(),'Same scalar metric inventory: '+where)
    for k in a:need(a[k]==b[k],'Fixed depth scalar exact: '+where+'/'+k)

def run(root,sources):
    index=read(root/'BAF_FINAL_INDEX.json')
    need(index['schema']=='actual_BAF_center_short_final_v1' and not index['real_scores_computed'],'Final trial index isolation')
    need(index['actual_source_manifest_sha256']==SOURCE and sha(sources/'source_manifest.json')==SOURCE,'Final actual source manifest matches reviewed source')
    files=index['files']
    need(len(files)==42,'All42 indexed artifacts plus index are present')
    for name,rec in files.items():
        p=(root/name).resolve();need(p.is_relative_to(root.resolve()) and p.is_file() and p.stat().st_size==rec['size_bytes'] and sha(p)==rec['sha256'],'Indexed actual bytes: '+name)
    for name,digest in read(sources/'source_manifest.json').items():need(Path(name).name==name and sha(sources/name)==digest,'Reviewed scientific source exact: '+name)
    need(sha(sources/'contract.json')==CONTRACT,'Original registered contract unchanged')
    cpu=read(root/'cpu_admission.json');testlog=(root/'actual_server_cpu_tests.log').read_text(encoding='utf-8')
    need(cpu['status']=='passed' and cpu['actual_server_tests']==15 and cpu['exit_code']==0 and cpu['source_manifest_sha256']==SOURCE and cpu['contract_sha256']==CONTRACT and cpu['log_sha256']==sha(root/'actual_server_cpu_tests.log'),'Actual15 server CPU checks bind final source/contract/log')
    need('Ran 15 tests' in testlog and testlog.rstrip().endswith('OK') and 'test_input_lineage_checks_actual_bytes_not_just_cached_source_name' in testlog,'Actual newly added lineage regression ran, not old14 receipt')
    for job in ('162_pose_baf_center_smoke_20261108','163_pose_baf_center_study_20261108'):
        q=read(root/'queue'/(job+'.done.json'))
        need(q['status']=='complete' and q['exit_code']==0 and q['job_id']==job,'Actual queue child exited0: '+job)
    external={x['path']:x for x in index['external_checkpoints']}
    need(len(external)==len(index['external_checkpoints']),'External checkpoint descriptors unique')
    smoke,study=[read(root/x/'completion.json') for x in ('smoke','study')]
    for phase,c in [('smoke',smoke),('study',study)]:
        need(c['status']=='complete_synthetic_only' and c['schema']=='pose_baf_center_adaptation_v1' and c['phase']==phase,'Complete expected phase: '+phase)
        need(c['source_manifest_sha256']==SOURCE and c['contract_sha256']==CONTRACT and not c['real_scores_computed'],'Phase science/source/real flags: '+phase)
        need(c['checkpoint_verification_scope']=='model_optimizer_readback_only_no_resume_claim' and c['resume_supported'] is False,'Checkpoint scope accurately limited: '+phase)
        need(c['original30_gate']['status']==c['original8_confirmation']['status']=='not_evaluated' and c['original30_gate']['thresholds_unchanged'] and c['original8_confirmation']['thresholds_unchanged'],'Original global30+8 remains closed: '+phase)
        need(c['frozen_depth_input']=='raw' and c['frozen_rotation_input']=='raw with predicted center crop/ray frame','RAW depth/rotation input contract: '+phase)
        a=c['full288_admissions'];need(len(a)==6 and {(x['method'],x['seed']) for x in a}=={(m,s) for m in ('eventgenesis','v2e') for s in range(3)} and all(x['full288_replay_exact'] for x in a),'Six full288 pre-update admissions: '+phase)
        pre=read(root/phase/'pre_update_admission.json')
        need(pre['status']=='passed' and pre['optimizer_updates']==0 and pre['full288_admissions']==a,'Admissions recorded before any optimizer update: '+phase)
        for seed in range(3):need(len({x['labels_sha256'] for x in a if x['seed']==seed})==1 and len({x['trajectory_sha256'] for x in a if x['seed']==seed})==1,'Cross-generator labels/draws paired: '+phase+str(seed))
    need(study['full288_admissions']==smoke['full288_admissions'],'Study matches smoke complete original288 admissions')
    prior=read(root.parent/'followup/projection/execution_receipt.json')
    prior_scores=read(root.parent/'followup/projection/completion_summary.json')['results']
    prior_normal=read(root.parent/'evidence/export/native_scores.json')
    for a in study['full288_admissions']:
        original=next(x for x in prior['records'] if (x['method'],x['seed'],x['arm'],x['branch'])==(a['method'],a['seed'],'physical_supervised','center'))
        need(a['original_log']==original['original_log'] and a['source_state_sha256']==original['source_state_sha256'] and a['labels_sha256']==original['labels_sha256'],'Full288 admission anchored to previously archived immutable C1 controls')
    need(len(smoke['records'])==2 and smoke['new_optimizer_updates']==2 and all(x['updates']==x['new_optimizer_updates']==1 for x in smoke['records']),'Actual smoke two one-update methods')
    need(len(study['records'])==6 and study['new_optimizer_updates']==1728 and all(x['updates']==x['new_optimizer_updates']==288 for x in study['records']),'Exactly6x288 new center optimizer updates')
    need({(x['method'],x['seed']) for x in study['records']}=={(m,s) for m in ('eventgenesis','v2e') for s in range(3)},'All six paired center branches trained')
    need(study['input_caches']==smoke['input_caches']==index['external_input_caches'],'Same hash-bound smoke input caches reused for study')
    original_bindings={x['path']:x for x in study['source_records']}
    need(len(original_bindings)>96*2,'Actual bank/NPZ lineage present; source SHA is not substituted for input SHA')
    lineage_counts={suffix:sum(Path(p).suffix==suffix for p in original_bindings) for suffix in ('.npz','.h5','.npy')}
    need(lineage_counts['.npz']==192 and lineage_counts['.h5']>=392 and lineage_counts['.npy']>=784,'Actual192 normalNPZ and all usedphysical/event files bound')
    for phase,c in [('smoke',smoke),('study',study)]:
        for row in c['records']:
            m,s=row['method'],row['seed'];a=next(x for x in c['full288_admissions'] if (x['method'],x['seed'])==(m,s))
            need(row['branch']=='center' and row['canonical_arm']=='fixed_baf_center' and row['source_state_sha256']==a['source_state_sha256'],'Center from original source initialization: '+phase+m+str(s))
            need(row['teacher_unchanged'] and row['independent_model_optimizer_rebuild'] and row['optimizer_state_equal'],'Native actual model/Adam readback flags: '+phase+m+str(s))
            need(external[row['checkpoint_path']]['sha256']==row['checkpoint_sha256'],'Training checkpoint external SHA matches record')
            need(external[row['final_model']['path']]==row['final_model'],'Final model external descriptor matches record')
            logfile=root/phase/'training_evidence'/f'{m}_seed{s}_physical_supervised_center.jsonl'
            rows=[json.loads(x) for x in logfile.read_text(encoding='utf-8').splitlines()]
            need(len(rows)==row['updates'] and [x['update'] for x in rows]==list(range(1,row['updates']+1)),'Actual contiguous log budget: '+phase+m+str(s))
            need(sha(logfile)==row['training_log']['sha256'] and logfile.stat().st_size==row['training_log']['size_bytes'],'Training log immutable receipt binding')
            for x in rows:
                need(x['branch']=='center' and x['arm']=='physical_supervised' and x['seed']==s,'Native log role clear (legacy slot, not original control)')
                need(all(math.isfinite(x[k]) for k in ('supervised','clean_retention','half_same_view_retention','loss','gradient_l2')) and x['gradient_l2']>0 and x['finite_gradient_tensors']==20,'Finite actual center update with native20 trainable parameter tensors')
                # JSON components are individual FP32 scalar readbacks; summing
                # those Python floats need not equal the rounded FP32 total.
                need(x['half_same_view_retention']==0 and math.isclose(x['loss'],x['supervised']+2*x['clean_retention'],rel_tol=1e-6,abs_tol=1e-6),'Original center objective retained (FP32 scalar rounding), no new auxiliary teacher term')
                need(len(x['normal_indices'])==len(x['auxiliary_indices'])==len(x['half_mask']) and 0<len(x['normal_indices'])<=32,'Matched batch dimensions')
                need(all(0<=i<1529 for i in x['normal_indices']) and all(0<=i<3058 for i in x['auxiliary_indices']),'Original TRAIN index bounds')
                need(all((i<1529)==v for i,v in zip(x['auxiliary_indices'],x['half_mask'])),'Original half/eighth index and draw-mask consistency')
            native=read(logfile.with_name(logfile.stem+'_receipt.json'))
            need(all(native[k]==row[k] for k in native),'Standalone native Fitter receipt matches extended parent record')
            if phase=='study':need(canonical_draws(rows)==a['trajectory_sha256'] and row['labels_sha256']==a['labels_sha256'],'All288 actual draw hash/augmented-label hash equals original C1 admission')
            else:
                actual=next(x for x in study['records'] if (x['method'],x['seed'])==(m,s))
                studylog=root/'study/training_evidence'/Path(logfile).name
                first=json.loads(studylog.read_text(encoding='utf-8').splitlines()[0])
                need(draw(first)==draw(rows[0]) and row['source_state_sha256']==actual['source_state_sha256'],'Smoke discarded; study starts same source and first draw (not smoke resume)')
    readback=read(root/'actual_smoke_readback.json')
    need(readback['status']=='passed' and readback['new_optimizer_updates']==0 and not readback['GPU_used'] and not readback['resume_supported'] and readback['source_manifest_sha256']==SOURCE,'Actual independent CPU smoke readback scope and source')
    need(readback['receipt_sha256']==sha(root/'smoke/completion.json') and len(readback['checks'])==2,'Independent readback binds actual paired smoke receipt')
    need(readback['verifier_sha256']==sha(root/'verify_actual_smoke.py'),'Actual independent readback verifier bytes bound')
    for x in readback['checks']:
        r=next(r for r in smoke['records'] if (r['method'],r['seed'])==(x['method'],x['seed']))
        need(x['Adam_steps']==[1] and x['frozen_size_exact'] and x['CPU_forward_max_abs']==0 and x['model_state_sha256']==r['final_state_sha256'] and x['checkpoint_sha256']==r['checkpoint_sha256'],'Independent model/Adam/size/forward readback matches actual smoke')
    results=study['results'];need(len(results)==6,'Six evaluated center branches')
    rebuilt=[];metric_keys=('depth_mae_m','depth_bias_m','log_size_mae','log_size_bias')
    for r in results:
        before,after=r['arms']['raw_C1'],r['arms']['baf_center']
        need(set(r['arms'])=={'raw_C1','frozen_C1_BAF','baf_center'},'Three distinct inference roles preserved')
        oldrun=next(x for x in prior_scores if (x['method'],x['seed'])==(r['method'],r['seed']))
        for cond in ('clean','matched_normal','half','eighth'):
            values=[r['arms'][arm][cond] for arm in r['arms']]
            need(all(v['windows']==(384 if cond=='clean' else 64) and v['rollouts']==(96 if cond=='clean' else 4) for v in values),'clean384 distinct from matchedfour64 validation')
            if cond=='matched_normal':reference=next(x['score']['summary'] for x in prior_normal if (x['method'],x['seed'],x['arm'],x['condition'])==(r['method'],r['seed'],'C1','normal'))
            else:reference=oldrun['arms']['physical_supervised'][cond]['summary']
            current=before[cond]['pose']['summary']
            need(reference.keys()==current.keys() and all(math.isclose(current[k],v,rel_tol=1e-6,abs_tol=1e-8) for k,v in reference.items()),'RAW C1 metrics preserve original native tolerance (rtol1e-6,atol1e-8)')
            for other in values[1:]:
                numeric_equal({k:values[0]['pose']['summary'][k] for k in metric_keys},{k:other['pose']['summary'][k] for k in metric_keys},str((r['method'],r['seed'],cond)))
                for rid,metrics in values[0]['pose']['by_rollout'].items():numeric_equal({k:metrics[k] for k in metric_keys},{k:other['pose']['by_rollout'][rid][k] for k in metric_keys},str((r['method'],r['seed'],cond,rid)))
            if cond=='matched_normal':continue
            x,y=before[cond],after[cond]
            rebuilt.append(dict(method=r['method'],seed=r['seed'],condition=cond,metric='center_epe_px',passed=y['center_epe_px']<x['center_epe_px'] if cond=='eighth' else y['center_epe_px']<=x['center_epe_px']))
            rebuilt.append(dict(method=r['method'],seed=r['seed'],condition=cond,metric='success',passed=y['pose']['summary']['success']-x['pose']['summary']['success']>=(-.02 if cond=='clean' else 0.)))
    gate=study['center_stage_gate'];need(gate['scope']=='center_stage_only_36_checks_not_global30' and gate['checks']==rebuilt and len(rebuilt)==36,'All36 center-stage checks independently rebuilt with original rules')
    need(sum(x['passed'] for x in rebuilt)==20 and not gate['passed'],'Quality result exactly20pass16fail; candidate not adopted')
    mean=[]
    for method in ('eventgenesis','v2e'):
        for cond in ('clean','matched_normal','half','eighth'):
            rows=[r for r in results if r['method']==method]
            mean.append(dict(method=method,condition=cond,arms={arm:dict(center_epe_px=statistics.mean(r['arms'][arm][cond]['center_epe_px'] for r in rows),pose_success_percent=statistics.mean(r['arms'][arm][cond]['pose']['summary']['success'] for r in rows)*100) for arm in ('raw_C1','frozen_C1_BAF','baf_center')}))
    table=read(root/'derived/bt_summary.json');need(len(table)==8,'All eight method/condition three-seed table rows retained')
    for row in table:
        runs=[x for x in results if x['method']==row['method']]
        for arm,metrics in row['arms'].items():
            scores=[x['arms'][arm][row['condition']] for x in runs]
            expected=dict(center_px=statistics.mean(x['center_epe_px'] for x in scores),depth_cm=statistics.mean(x['pose']['summary']['depth_mae_m'] for x in scores)*100,success_percent=statistics.mean(x['pose']['summary']['success'] for x in scores)*100)
            need(metrics.keys()==expected.keys() and all(close(metrics[k],v) for k,v in expected.items()),'Derived full method/condition mean table independently recomputed')
    ridtable=read(root/'derived/bt_rid_summary.json');need(len(ridtable)==24,'All2methods x3matchedlights x4trajectory rows retained')
    for row in ridtable:
        runs=[x for x in results if x['method']==row['method']]
        for arm,metrics in row['arms'].items():
            scores=[x['arms'][arm][row['condition']]['pose']['by_rollout'][row['rid']] for x in runs]
            expected=dict(center_px=statistics.mean(x['center_px'] for x in scores),depth_cm=statistics.mean(x['depth_mae_m'] for x in scores)*100,success_percent=statistics.mean(x['success'] for x in scores)*100)
            need(metrics.keys()==expected.keys() and all(close(metrics[k],v) for k,v in expected.items()),'Derived per-trajectory three-seed table independently recomputed')
    return dict(indexed_files=len(files),external_checkpoint_descriptors=len(external),lineage_file_counts=lineage_counts,actual_new_center_updates=1728,center_gate_pass=20,center_gate_fail=16,global30_and8='not_evaluated',adopted=False,three_seed_means=mean,per_trajectory_rows=len(ridtable),limitations=['No new model/checkpoint or input-cache bytes downloaded; hashes/actual server readback receipts checked, no independent torch/GPU replay here.','Native Fitter saves RNG but this one-shot experiment supports only model/Adam readback, not resumed next-step equivalence.','Labels are verified against original pre-update replay and actual receipt hashes, without recomputing augmentation tensors locally.','Center-stage failure prohibits claiming global30+8 stability or real E-POSE improvement.'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--sources',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    need(not a.out.exists(),'Unique audit result only')
    d=dict(schema='BAF_center_independent_final_audit_v1',status='failed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_source_sha256=sha(Path(__file__)),no_SSH=True,no_training=True,checks=CHECKS)
    try:d['audit']=run(a.evidence,a.sources);d['status']='passed'
    except BaseException:d['error']=traceback.format_exc()
    d.update(passed_checks=sum(x['passed'] for x in CHECKS),failed_checks=sum(not x['passed'] for x in CHECKS));a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps({k:d[k] for k in ('status','passed_checks','failed_checks')}));print(d.get('error',''))
    if d['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
