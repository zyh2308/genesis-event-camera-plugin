"""Independent local scalar audit; no SSH, torch, optimizer, or result edits."""
import argparse,datetime,hashlib,json,math,statistics,traceback
from pathlib import Path

CHECKS=[]
def need(ok,message):
    CHECKS.append({'passed':bool(ok),'check':message})
    if not ok:raise AssertionError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,rtol=1e-10,atol=1e-10):return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=rtol,abs_tol=atol)
def fact(x):return {k:x[k] for k in ('method','seed','condition','passed')}

def audit(root):
    index=read(root/'FOLLOWUP_INDEX.json')
    need(not index['real_data_used_for_tuning'],'No new real scores used for selection')
    for name,r in index['files'].items():
        p=(root/name).resolve();need(p.is_relative_to(root.resolve()) and p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256'],'Immutable indexed bytes: '+name)
    source_groups=[]
    for m in root.glob('sources/**/source_manifest.json'):
        files=read(m)
        for name,digest in files.items():need(Path(name).name==name and sha(m.parent/name)==digest,'Pinned followup source: '+str(m.parent.name)+'/'+name)
        source_groups.append({'path':str(m),'sha256':sha(m),'members':len(files)})
    external={r['path']:r for r in index['external_artifacts']}
    need(len(external)==len(index['external_artifacts']),'External descriptors are uniquely keyed')
    cp=read(root/'projection/completion_summary.json');receipt=read(root/'projection/execution_receipt.json')
    need(cp['status']=='complete' and not cp['real_data_used'] and len(cp['results'])==6,'Projection complete six paired synthetic-only runs')
    need(cp['source']==external[cp['source']['path']],'Full unabridged completion remains externally hash-bound')
    need(receipt['status']=='complete_synthetic_only' and receipt['new_optimizer_updates']==1728 and not receipt['real_scores_computed'],'Exactly1728 new depth updates, no real scoring')
    need(receipt['candidate_center']=='reuse_C1_unchanged' and receipt['original_thirty_gate_unchanged'] and receipt['original_eight_confirmation_unchanged'],'Projection changes depth only; original gates unchanged')
    need(receipt['thirty_gate']==cp['gate'] and receipt['original_confirmation']==cp['original_confirmation'],'Full receipt and abridged completion gates identical')
    rebuilt=[]
    for run in cp['results']:
        control,candidate=run['arms']['clean_control'],run['arms']['paired_consistency']
        for condition in ('clean','half','eighth'):
            before,after=control[condition]['summary'],candidate[condition]['summary']
            if condition=='clean':rebuilt.append(dict(method=run['method'],seed=run['seed'],condition=condition,passed=after['success']>=before['success']-.02))
            else:
                rebuilt.append(dict(method=run['method'],seed=run['seed'],condition=condition,passed=after['depth_mae_m']<before['depth_mae_m']))
                rebuilt.append(dict(method=run['method'],seed=run['seed'],condition=condition+'_pose',passed=after['success']>=before['success']))
        for condition,metrics in run['arms']['physical_supervised'].items():
            need(candidate[condition]['summary']['center_px']==metrics['summary']['center_px'],'Candidate center exactly reused: '+str((run['method'],run['seed'],condition)))
    need(rebuilt==[fact(x) for x in cp['gate']['checks']] and len(rebuilt)==30,'All original30 checks independently rebuilt from original macro metrics')
    need(sum(x['passed'] for x in rebuilt)==21 and not cp['gate']['passed'],'Projection21/30 is a failed overall gate, not an adopted model')
    confirmation=cp['original_confirmation'];need(len(confirmation['comparisons'])==8,'Original eight confirmation comparisons retained')
    for x in confirmation['comparisons']:
        passed=all(v>0 for v in x['seed_improvement']) and x['ci'][0]>0
        need(x['passed']==passed and len(x['seed_improvement'])==3 and len(x['ci'])==2,'Original8 stored CI and per-seed pass rule: '+str((x['method'],x['baseline'],x['metric'])))
    need(confirmation['passed']==all(x['passed'] for x in confirmation['comparisons']),'Eight confirmation aggregate consistent')
    records=receipt['records'];need(len(records)==36,'Full source/control/candidate36 branch records retained')
    need(sum(r['new_optimizer_updates'] for r in records)==1728,'New updates summed from all36 individual records')
    for r in records:
        prefix='projection/training_evidence/'+r['method']+'_seed'+str(r['seed'])+'_'+r['arm']+'_'+r['branch']
        if r['arm']=='paired_consistency' and r['branch']=='center':name=prefix+'_C1_reuse_receipt.json'
        elif r['arm']=='paired_consistency':name=prefix+'_receipt.json'
        else:name=prefix+'_reused_receipt.json'
        need(read(root/name)==r,'Every branch receipt equals parent execution record: '+name)
        if r['arm']=='paired_consistency' and r['branch']=='depth':
            need(r['new_optimizer_updates']==r['updates']==288 and r['depth_teacher_projection'] and r['labels_and_draws_identical_to_C1'],'New depth288 label/draw-parity receipt: '+str((r['method'],r['seed'])))
            need(external[r['checkpoint_path']]['sha256']==r['checkpoint_sha256'],'New training checkpoint externally hash-bound')
        else:need(r['new_optimizer_updates']==0,'Reused branch adds zero optimizer updates')
        if r['arm']=='paired_consistency' and r['branch']=='center':
            control=next(x for x in records if (x['method'],x['seed'],x['arm'],x['branch'])==(r['method'],r['seed'],'physical_supervised','center'))
            need(r['reused_C1_center'] and not r['candidate_center_optimizer_used'] and all(r[k]==control[k] for k in ('source_state_sha256','final_state_sha256','labels_sha256','checkpoint_sha256')),'Candidate center state/labels/optimizer exactlyC1: '+str((r['method'],r['seed'])))
    derived=read(root/'derived/projection_summary.json')
    for row in derived:
        runs=[r for r in cp['results'] if r['method']==row['method']]
        for short,long in [('C0','clean_control'),('C1','physical_supervised'),('projection','paired_consistency')]:
            values=[r['arms'][long][row['condition']]['summary'] for r in runs]
            for field,key,multiplier in [('depth_mae_cm','depth_mae_m',100),('success_percent','success',100),('center_px','center_px',1)]:
                need(close(row['arms'][short][field],statistics.mean(x[key] for x in values)*multiplier),'Projection three-seed summary independently averaged')
    b=read(root/'baf_diagnostic/completion.json');summ=read(root/'baf_diagnostic/summary.json');ds=read(root/'derived/baf_summary.json')
    need(b['status']=='complete' and b['optimizer_updates']==0 and not b['real_data_used'] and b['models_unchanged'],'BAF is zero-update fixed-model diagnostic only')
    need(b['filter_threshold_us']==2000 and b['initial_time_surface']==0 and not b['polarity_used'] and b['causal_whole_episode'] and not b['GT_filter_input'],'BAF fixed causal2ms unprivileged rule')
    need(b['zero_input_control'] and not b['zero_control_uses_GT'] and b['rows']==1728 and len(summ)==36,'BAF zero-input control and complete TRAIN32/VAL64 matrix retained')
    need(len(b['parity'])==18 and all(x['passed'] and x['max_abs_normalized_uv']<=2e-6 for x in b['parity']),'All18 RAW C1 reference parities passed')
    for r in ds:
        rows=[x for x in summ if (x['method'],x['split'],x['condition'])==(r['method'],r['split'],r['condition'])]
        need(len(rows)==3 and {x['seed'] for x in rows}=={0,1,2},'BAF summary averages all three seeds')
        for a,key in [('raw_px','raw_center_error_px'),('filtered_px','filtered_center_error_px'),('zero_px','zero_center_error_px'),('retention_ratio','event_retention_ratio')]:
            need(close(r[a],statistics.mean(x[key] for x in rows)),'BAF derived raw scalar mean: '+a)
    eg=[x for x in ds if x['method']=='eventgenesis']
    # Report values; these are not new candidate-quality thresholds.
    egval=[x for x in eg if x['split']=='validation']
    need(len(egval)==3,'EG three validation-light BAF comparisons retained')
    crop=read(root/'rotation_crops/completion.json');parity=read(root/'rotation_crops/native_parity.json');ref=read(root/'rotation_crops/gttranslation_reference_parity.json');rot=read(root/'rotation_crops/gtboth_rotation_consistency.json')
    need(crop['status']=='complete_diagnostic_only' and crop['optimizer_updates']==crop['model_forward_calls']==0 and not crop['real_data_used'] and not crop['GT_oracles_deployable'],'Rotation crop CPU oracle is not deployable model performance')
    need(crop['flat_predictions']==3456 and crop['C1_cohort']==1152 and crop['scored_variants']==9216 and crop['macro_cells']==144 and crop['three_seed_summary_cells']==48,'Complete matched4trajectory64window crop factorial inventory')
    for p,countkey in [(parity,'checked_cells'),(ref,'cells')]:
        need(p['status']=='passed' and p[countkey]==18 and all(x['passed'] for x in p['checks']),'All18 crop native/reference cells passed')
    for x in parity['checks']:
        need(close(x['actual'],x['native'],rtol=parity['rtol'],atol=parity['atol']) and close(abs(x['actual']-x['native']),x['abs_error']),'Crop native stored scalar parity independently checked')
    need(rot['case_count']==1152 and rot['pair_count']==3456 and rot['all_bitwise_equal'] and rot['max_abs']==0 and all(p['bitwise_equal'] and p['max_abs']==0 for c in rot['cases'] for p in c['pairs']),'Saved GT-both R consistency retains all1152 cases/3456 pairs')
    cs=read(root/'rotation_crops/three_seed_summary.json');macro=read(root/'rotation_crops/macro_scores.json')
    for row in cs:
        variant='rotation_crop/'+row['translation_condition']+'/'+row['rotation_factor']
        cells=[x for x in macro if x['method']==row['method'] and x['condition']==row['condition'] and x['variant']==variant]
        need(len(cells)==3 and {x['seed'] for x in cells}=={0,1,2},'Crop three-seed cell inventory')
        for key,value in row['metrics'].items():
            target=statistics.mean(x['summary']['success'] for x in cells)*100 if key=='success_percent' else statistics.mean(x['summary'][key] for x in cells)
            need(close(value,target),'Crop macro three-seed mean: '+key)
    return dict(archive_files=len(index['files']),source_groups=source_groups,external_descriptors=len(external),projection=dict(passed_30=sum(x['passed'] for x in rebuilt),total_30=30,passed_8=sum(x['passed'] for x in confirmation['comparisons']),new_depth_updates=1728,center_new_updates=0,adopted=False),BAF_EG_validation=egval,crop=dict(scored_variants=9216,native_cells=18),limitations=['External models/raw JSONLs were hash-bound but not downloaded or independently rerun.','Original8 bootstrap CI values were retained and pass algebra checked; omitted per-rollout tables prevent independently rerunning bootstrap here.','BAF numbers are fixed-model zero-update sensitivity results, not newly trained center or final pose scores.','Crop interventions use privileged synthetic GT and cannot be reported as deployable performance.'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).parent/'followup_local');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    need(not a.out.exists(),'Unique new verification output')
    result=dict(schema='pose_factor_followup_independent_scalar_audit_v1',status='failed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_script_sha256=sha(Path(__file__)),checks=CHECKS,no_SSH=True,no_training=True)
    try:result['audit']=audit(a.root);result['status']='passed'
    except BaseException:result['error']=traceback.format_exc()
    result.update(passed_checks=sum(x['passed'] for x in CHECKS),failed_checks=sum(not x['passed'] for x in CHECKS));a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','passed_checks','failed_checks')}));print(result.get('error',''))
    if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
