"""Independent stdlib-only evidence/source audit. No SSH, CUDA or training.

Checks archived bytes and independently recomputes scalar claims from exported
receipts. External model/data/large per-example artifacts remain explicitly
unread; a matching index is not described as a fresh GPU or filesystem replay.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,math,statistics,sys,traceback
from pathlib import Path

METHODS=('eventgenesis','v2e','vid2e','film_esim','rife_esim')
PAIR=('eventgenesis','v2e')
STRATA=('camera_motion_static_scene','independent_rigid_motion_static_camera','camera_and_independent_rigid_motion','combined_motion_with_occlusion')
CHECKS=[]

def check(ok,message,details=None):
    CHECKS.append(dict(passed=bool(ok),check=message,details=details))
    if not ok:raise AssertionError(message+(': '+str(details) if details is not None else ''))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('Duplicate JSON key: '+k)
        out[k]=v
    return out
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique)
def jhash(value,compact=False):
    options=dict(sort_keys=True)
    if compact:options['separators']=(',',':')
    return hashlib.sha256(json.dumps(value,**options).encode()).hexdigest()
def close(a,b,tol=1e-12):return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol)
def record(path):return dict(path=str(path),size_bytes=path.stat().st_size,sha256=sha(path))
def safe_member(root,name):
    relative=Path(name)
    check(not relative.is_absolute() and '..' not in relative.parts,'Safe relative manifest member',name)
    target=(root/relative).resolve();check(target.is_relative_to(root.resolve()),'Manifest member stays inside archived directory',name)
    return target

def source_manifests(root):
    rows=[]
    paths=sorted(root.rglob('source_manifest.json'));check(len(paths)>=10,'All ten source groups archived',len(paths))
    for path in paths:
        value=read(path);files=value.get('files',value)
        check(isinstance(files,dict) and bool(files),'Recognized nonempty source-manifest mapping',str(path))
        n=0
        for name,binding in files.items():
            p=safe_member(path.parent,name);expected=binding if isinstance(binding,str) else binding['sha256']
            check(p.is_file() and sha(p)==expected,'Archived source bytes match SHA',str(p))
            if isinstance(binding,dict):check(p.stat().st_size==binding['size_bytes'],'Archived source size matches',str(p))
            n+=1
        rows.append(dict(**record(path),members_verified=n))
    return rows

def evidence_index(directory,name,expected_count):
    p=directory/name;index=read(p);files=index['files'];check(len(files)==expected_count,'Evidence index has expected full count',dict(index=name,count=len(files)))
    check(index.get('real_scores_read') is False,'Evidence export did not read real scores',name)
    for name,binding in files.items():
        member=safe_member(directory,name)
        check(member.is_file() and sha(member)==binding['sha256'] and member.stat().st_size==binding['size_bytes'],'Evidence bytes match indexed size/SHA',str(member))
    return index

def queue_sources(queue,source_dir):
    for server,digest in queue['source_sha256'].items():
        local=source_dir/Path(server).name
        check(local.is_file() and sha(local)==digest,'Actual queued source matches archived source',server)

def photo(root,r):
    source=root/'flow_diagnostic_v2/sources';contract=read(source/'contract.json');bindings=contract['frozen_inputs']['methods_binding']
    complete=read(r/'flow_photo_v2/results/completion.json');manifest_sha=sha(source/'source_manifest.json')
    check(complete['status']=='complete' and complete['optimizer_updates']==0 and not complete['real_data_used'] and not complete['real_scores_used'] and not complete['automatic_training'],'Photo v2 complete with zero updates and no real scoring')
    check(complete['source_manifest_sha256']==manifest_sha and complete['contract_sha256']==sha(source/'contract.json'),'Photo actual completion binds final archived source/contract')
    check(complete['strict_raw_support_completed'] and complete['NN_and_strict_parameter_oracles_complete'] and complete['paired_physical_labels'],'Photo strict/raw pairing and both parameter oracles complete')
    check(read(r/'queue/151_flow_photometric_alignment_20261108.done.json')['exit_code']==1,'Original151 exit1 failure retained')
    check(read(r/'queue/152_flow_photometric_alignment_v2_20261108.done.json')['exit_code']==0,'Repaired152 actual exit0')
    failure=r/'preserved_failures/photo_v1/failure.json';old=read(failure)
    check(old['status']=='failed' and 'Expected70 unchanged model parameter tensors' in old['error'] and complete['preserved151_failure_sha256']==sha(failure),'Old count-error failure preserved and bound by v2')
    queue_sources(read(r/'flow_photo_v2/queue_submission.json'),source)
    artifacts=read(r/'FULL_PHOTO_DIAGNOSTIC_ARTIFACT_INDEX.json')['files'];check(len(artifacts)==88,'Full photo external artifact index covers88 files')
    summaries={m:read(r/f'flow_photo_v2/results/{m}_summary.json') for m in PAIR};strict_rows={};models={};example_count=0
    for m,summary in summaries.items():
        check(summary in complete['methods'],'Completion embeds the same exported method summary',m)
        check(summary['checkpoint_sha256']==bindings[m]['last_checkpoint_sha256'] and summary['checkpoint_path']==bindings[m]['last_checkpoint'],'Photo uses exact original complete8910 last checkpoint',m)
        check(summary['optimizer_updates']==0 and summary['model_state_unchanged'] and summary['source_checkpoint_unchanged'] and summary['model_state_before_sha256']==summary['model_state_after_sha256'],'Photo model state and source checkpoint unchanged',m)
        models[m]=summary['model_state_before_sha256']
        rows=summary['examples'];check(len(rows)==40 and sum(x['split']=='train' for x in rows)==8 and sum(x['split']=='validation' for x in rows)==32,'Photo complete fixed8TRAIN+32VAL per method',m)
        for row in rows:
            name=Path(row['path']).name;check(artifacts[name]['sha256']==row['sha256'] and artifacts[name]['server_path']==row['path'],'Each photo example receipt matches external artifact index',name)
        example_count+=len(rows)
        parameter=read(r/f'flow_photo_v2/results/{m}_parameter_gradient_oracle.json');check(parameter['status']=='complete' and parameter['optimizer_updates']==0,'Photo NN/strict parameter receipt complete',m)
        strict_rows[m]=parameter['strict_statistics'];check(len(strict_rows[m])==8,'Eight raw strict-support sample statistics',m)
        for kind in ('NN_oracle','strict_oracle'):
            oracle=parameter[kind];inventory=oracle['parameter_inventory'];names={v['name'] for v in inventory}
            check(oracle['status']=='complete' and len(inventory)==82 and oracle['named_parameter_tensors']==82 and oracle['trainable_parameter_tensors']==82 and all(v['requires_grad'] for v in inventory),'All82 model parameter tensors included',dict(method=m,oracle=kind))
            check(jhash(inventory,compact=True)==oracle['parameter_inventory_sha256'],'Independent parameter inventory canonical SHA',dict(method=m,oracle=kind))
            check(len(oracle['gradient_records'])==4,'Four actual parameter gradient oracles recorded',dict(method=m,oracle=kind))
            for g in oracle['gradient_records']:
                check(g['nonNone_count']==70 and len(g['nonNone_names'])==70 and len(g['unused_names'])==12 and set(g['nonNone_names'])|set(g['unused_names'])==names and g['finite_nonNone'],'70 involved/12 unused finite gradients distinguished',dict(method=m,oracle=kind,loss=g['loss_index']))
            for cmp in oracle['comparisons'].values():
                check(len(cmp['per_parameter'])==82 and {x['parameter'] for x in cmp['per_parameter']}==names,'Gradient cosine keeps all82 parameter entries')
                a=sum(x['photo_gradient_l2']**2 for x in cmp['per_parameter'])**.5;b=sum(x['GT_gradient_l2']**2 for x in cmp['per_parameter'])**.5
                dot=sum((x.get('cosine') or 0)*x['photo_gradient_l2']*x['GT_gradient_l2'] for x in cmp['per_parameter'])
                check(close(a,cmp['photo_gradient_l2'],1e-10) and close(b,cmp['GT_gradient_l2'],1e-10) and close(dot,cmp['dot'],1e-10),'Parameter cosine/norm independently recomputed from per-parameter receipts')
                check(close(cmp['cosine'],cmp['dot']/(a*b),1e-10),'Global parameter cosine scalar consistency')
    check(example_count==80,'Both generators total80 fixed examples')
    for a,b in zip(strict_rows['eventgenesis'],strict_rows['v2e']):
        check(all(a[k]==b[k] for k in ('rollout','index','sample_id','augmentation_seed','raw_valid_sha256','grid_sha256','strict_mask_sha256','source_npz_sha256')),'Strict-support raw labels/grid/mask paired across generators',a['sample_id'])
    return dict(example_receipts_verified=80,external_artifact_descriptors=88,model_state_sha256=models,raw_example_JSON_replayed_locally=False)

def census(root,r,photo_result):
    source=root/'flow_census/sources';contract=read(source/'contract.json');bindings=contract['frozen_inputs']['methods_binding'];source_sha=sha(source/'source_manifest.json')
    cpu=read(r/'census/cpu/completion.json');cal=read(r/'census/phase0/completion.json');smoke=read(r/'census/smoke/completion.json')
    check(cpu['status']=='passed' and cpu['source_manifest_sha256']==source_sha and cpu['optimizer_updates']==0 and not cpu['CUDA_initialized'] and cpu['total_named_parameters']==82 and cpu['total_requires_grad_parameters']==82,'Census actual CPU admission is correct82/noGPU/no updates')
    check(cal['status']=='calibrated_train_only' and cal['optimizer_updates']==0 and not cal['GT_in_lambda'] and not cal['validation_in_lambda'] and not cal['real_in_lambda'] and cal['source_manifest_sha256']==source_sha,'Actual Census coefficient is TRAIN-only full parameter calibration')
    check(len(cal['records'])==16 and {(x['method'],x['batch']) for x in cal['records']}=={(m,i) for m in PAIR for i in range(8)},'Census exactly16 fixed pairedTRAIN calibration batches')
    replay=read(r/'census/phase0/control_replay.json');expected_hash=replay.pop('replay_sha256');check(jhash(replay)==expected_hash,'Independent original-control replay SHA')
    check(replay['arms']['native_control']==replay['arms']['census'],'Census and native arms share every original control index')
    ratios=[];labels={};signature_sha=None
    for row in cal['records']:
        check(row['split']=='train' and row['optimizer_updates']==0 and not row['GT_in_lambda'] and row['gradient_space']=='all_requires_grad_parameters_zero_unused' and row['total_requires_grad_parameters']==82,'Calibration records use all82 parameters, no GT/updates')
        check(row['native_finite_gradient_tensors']==70 and row['census_finite_gradient_tensors']==70 and len(row['native_unused_parameter_names'])==12 and len(row['census_unused_parameter_names'])==12,'Both losses retain70 finite connected+12 unused gradients')
        check(row['source_checkpoint']['sha256']==bindings[row['method']]['last_checkpoint_sha256'] and row['source_checkpoint']['model_state_sha256']==photo_result['model_state_sha256'][row['method']],'Calibration CP/source state agrees with independent photo diagnostic')
        check(row['replay_sha256']==expected_hash,'Calibration uses frozen original-control replay')
        ratio=row['census_photo_gradient_l2']/row['native_photo_gradient_l2'];check(close(ratio,row['census_to_native_ratio']),'Actual gradient ratio scalar consistency');ratios.append(ratio)
        if row['batch'] in labels:check(labels[row['batch']]==row['labels_and_augmentation_sha256'],'Calibration physical labels/augmentation paired')
        labels[row['batch']]=row['labels_and_augmentation_sha256']
        signature=read(r/f"census/phase0/{row['method']}_parameter_signature.json")
        check(len(signature['parameters'])==82 and jhash(signature['parameters'])==signature['sha256']==row['parameter_signature_sha256'],'Full actual calibration parameter inventory SHA')
        if signature_sha is not None:check(signature_sha==signature['sha256'],'Same architecture signature across generators')
        signature_sha=signature['sha256']
    coefficient=1/statistics.median(ratios)
    check(close(coefficient,cal['shared_census_weight'],1e-15) and close(statistics.median(ratios),cal['median_ratio'],1e-15),'One shared coefficient independently recomputed as1/median16 model-norm ratios')
    lock=read(r/'census/phase0/calibration_lock.json');calpath=r/'census/phase0/completion.json'
    check(lock['status']=='frozen_before_optimizer' and lock['completion']['sha256']==sha(calpath) and lock['completion']['size_bytes']==calpath.stat().st_size and close(lock['shared_census_weight'],coefficient),'Coefficient locked to actual completion bytes before optimizer')
    check(smoke['status']=='complete' and smoke['mode']=='smoke' and smoke['pairing_passed'] and smoke['source_manifest_sha256']==source_sha and len(smoke['results'])==4,'All four actual paired smoke updates completed')
    check({(x['method'],x['arm']) for x in smoke['results']}=={(m,a) for m in PAIR for a in ('native_control','census')},'Complete two-generator/two-arm smoke matrix')
    for x in smoke['results']:
        check(x['additional_updates']==1 and x['optimizer_update']==8911 and x['finite_gradient_tensors']==70 and x['independent_model_and_Adam_restore_passed'] and x['checkpoint']['model_rebuild_passed'] and x['checkpoint']['optimizer_state_equal'],'Each smoke has exactly1 actual restored update')
        check(x['source_checkpoint']['sha256']==bindings[x['method']]['last_checkpoint_sha256'] and x['shared_calibration']['sha256']==sha(calpath) and close(x['shared_census_weight'],coefficient),'Smoke shares fixed CP/calibration coefficient')
    for field in ('label_sequence_sha256','sample_sequence_sha256','augmentation_sequence_sha256'):check(len({x[field] for x in smoke['results']})==1,'Four-arm smoke exact sample/label/augmentation pairing',field)
    live=read(r/'queue/155_flow_census_paired_study_20261108.live.json');check('exit_code' not in live and not (r/'census/study/completion.json').exists(),'Archived Census full short-study result is not complete; no quality claim')
    return dict(shared_weight=coefficient,calibration_batches=16,actual_smoke_updates=4,study_gate_status='NOT_YET_AVAILABLE_IN_THIS_SNAPSHOT')

def pose_gate(rows,arm):
    checks=[]
    for row in rows:
        a=row['arms']['clean_control'];b=row['arms'][arm]
        checks.append(dict(method=row['method'],seed=row['seed'],condition='clean',passed=b['clean']['success']-a['clean']['success']>=-.02))
        for c in ('half','eighth'):
            checks.append(dict(method=row['method'],seed=row['seed'],condition=c,passed=b[c]['depth_mae_m']<a[c]['depth_mae_m']))
            checks.append(dict(method=row['method'],seed=row['seed'],condition=c+'_pose',passed=b[c]['success']>=a[c]['success']))
    return checks

def pose(root,r,initial):
    soft=read(r/'pose_soft_feature_summary.json');receipt=read(r/'pose_soft/results/feature_retention_execution_receipt.json');rows=soft['per_seed']
    check(len(rows)==6 and {(x['method'],x['seed']) for x in rows}=={(m,s) for m in PAIR for s in range(3)},'Pose soft complete2generators x3seed cohort')
    c1=pose_gate(rows,'physical_supervised');new=pose_gate(rows,'paired_consistency')
    check(sum(x['passed'] for x in c1)==22 and sum(x['passed'] for x in new)==22 and len(new)==30,'Independent original30-rule count: C1=22,soft=22')
    check(new==soft['gate']['checks']==receipt['thirty_gate']['checks'] and not soft['gate']['passed'] and not receipt['thirty_gate']['passed'],'Soft gate fails original unchanged30 rules; not adopted')
    check(sha(r/'pose_soft/results/feature_retention_execution_receipt.json')==soft['source_receipt_sha256'],'Soft compact binds actual execution receipt')
    check(receipt['source_manifest_sha256']==sha(root/'pose_feature_retention/sources/source_manifest.json') and not receipt['real_scores_computed'] and receipt['actual_half_teacher_weight']==0 and not receipt['dark_feature_retention'],'Pose actual feature-retention source and synthetic-only recipe binding')
    previous=read(root.parent/'parallel_priority_20261108/evidence/pose_trial/study/completion_compact.json')
    by_key={(x['method'],x['seed']):x for x in previous['per_seed']};metric_count=0
    for row in rows:
        old=by_key[(row['method'],row['seed'])]
        for arm in ('clean_control','physical_supervised'):
            for condition,metrics in row['arms'][arm].items():
                for name,value in metrics.items():
                    expected=old['arms'][arm][condition][name]
                    check(math.isclose(value,expected,rel_tol=1e-6,abs_tol=1e-8),'Pose immutable control metric matches original independently',dict(method=row['method'],seed=row['seed'],arm=arm,condition=condition,metric=name));metric_count+=1
    source_controls=receipt['source_controls'];check(source_controls[0]['sha256']==previous['source_completion']['sha256'],'Pose source-control completion SHA matches prior archived compact')
    records=receipt['records'];check(sum(x['new_optimizer_updates'] for x in records)==soft['new_optimizer_updates']==receipt['new_optimizer_updates']==3456,'Only new candidate updates counted; reused controls add0')
    for x in records:
        if x.get('reused_immutable_control'):check(x['new_optimizer_updates']==0 and x['label_and_augmentation_replay_exact'],'Reused controls retain exact labels and0 newupdates')
        check(x['teacher_unchanged'] and x['independent_model_optimizer_rebuild'] and x['optimizer_state_equal'],'Pose model/optimizer reconstruction receipts passed')
    calibration=read(initial/'pose/phase0/calibration.json');check(sha(initial/'pose/phase0/calibration.json')==receipt['calibration_sha256'] and close(calibration['coefficient'],receipt['feature_coefficient']) and close(soft['feature_coefficient'],receipt['feature_coefficient']),'Pose coefficient bound to prior TRAIN calibration bytes')
    return dict(C1_passed=22,soft_passed=22,total_checks=30,adopted=False,immutable_control_metrics_rechecked=metric_count,new_candidate_updates=3456)

def data_v3(root,r):
    v=read(r/'pose_audit_v3/data_contract.json')
    check(v['status']=='passed' and v['selected_rollouts']==96 and v['all_train_selected'] and not v['missing_train'] and not v['foreign_selected'],'Pose v3 audited all96 training trajectories')
    check(not v['real_data_used'] and not v['model_inference'] and not v['GPU_used'] and v['optimizer_updates']==0 and v['representations_executed'],'Pose data audit is full CPU encoding audit only')
    for m in PAIR:
        x=v['representation'][m];check(x['executed'] and x['normal_npz_vs_fresh_base_max_abs']==0 and x['mismatch_windows']==0,'Fresh raw-event encoding equals normal NPZ exactly',m)
        check(v['normal_adapter'][m]['rollouts']==96 and v['normal_adapter'][m]['same_ids_as_fresh_base'],'Adapter covers same96 trajectories',m)
        check(v['event_generator_contracts'][m]['invariant_settings_equal'],'Generator/noise contracts invariant across relighting',m)
    info=v['input_information'];sat=[x['metrics']['actual_model128_exact_one_rate'] for x in info['summary'] if 'actual_model128_exact_one_rate' in x['metrics']]
    check(len(sat)==126 and all(all(float(x)==0 for x in row.values()) for row in sat),'All126 actual128-input saturation summaries are exactly0')
    evidence=info['raw_event_evidence']
    for m in PAIR:
        for level in ('normal','half','eighth'):
            rows=[x for x in evidence if x['method']==m and x['level']==level and x['split']=='train'];check(len(rows)==96 and len({x['rollout'] for x in rows})==96,'Raw event hashes cover all96 train trajectories/level',dict(method=m,level=level))
    provenance=v['serialization_provenance'];src=root/'pose_data_audit_v3/sources'
    check(provenance['wrapper_sha256']==sha(src/'run_serializable_audit.py') and provenance['original_v2_script_sha256']==sha(src/'audit_pose_data_contract_v2.py') and provenance['source_manifest_sha256']==sha(src/'source_manifest.json') and provenance['preserved_original_failure'],'Data v3 changed JSON serialization only; original v2 failure remains bound')
    return dict(selected_train_rollouts=96,raw_NPZ_max_difference=0,actual128_saturation_groups=126,all_saturation_zero=True)

def full15(root,r):
    value=read(r/'full15/merged_matrix.json');cells=value['cells'];contract=read(root/'full_synthetic_singlecell/sources/contract.json');binding=contract['checkpoint_binding']
    check(value['status']=='complete' and value['cell_count']==len(cells)==15 and not value['checkpoint_selection_changed'] and value['optimizer_updates']==0 and not value['real_data_used'] and not value['real_scores_used'],'Full15 frozen best checkpoints merged without training/selection change')
    check({(x['method'],x['seed']) for x in cells}=={(m,s) for m in METHODS for s in range(3)},'Complete original5methods x3seed matrix')
    for cell in cells:
        check(cell['windows']==528 and cell['checkpoint_sha256']==binding[f"{cell['method']}/seed{cell['seed']}"]['selected_checkpoint_sha256'] and set(cell['strata'])==set(STRATA),'Each merged cell has528windows and exact originalbest CP',dict(method=cell['method'],seed=cell['seed']))
    supplement=read(r/'full15/single_cell/completion.json');check(supplement['status']=='complete' and supplement['seeds']==[1] and supplement['shard']=='rife_seed1' and supplement['cells']==1 and supplement['windows_per_checkpoint']==528 and supplement['optimizer_updates']==0 and not supplement['real_data_used'] and not supplement['real_scores_used'],'LastRIFE seed1 explicit singlecell is complete zero-update inference')
    check(supplement['source_manifest_sha256']==sha(root/'full_synthetic_singlecell/sources/source_manifest.json') and supplement['contract_sha256']==sha(root/'full_synthetic_singlecell/sources/contract.json'),'Supplement actual sources match archived final package')
    check(supplement['results'][0]['method']=='rife_esim' and supplement['results'][0]['seed']==1,'Supplement is not disguised as seed2')
    old=read(r/'preserved_failures/full15_original_timeout/resource_release.json');check(old['status']=='released' and old['stopped_all_own_group_members'] and not old['successful'] and old['optimizer_updates']==0,'Original60minute timeout and safe release remain recorded')
    summary=read(r/'full15_synthetic_decomposition_summary.json');check(summary['source']['sha256']==sha(r/'full15/merged_matrix.json'),'Full15 summary binds actual merge bytes')
    computed=[]
    for m in METHODS:
        rows=[x for x in cells if x['method']==m];a=[x['regions']['all_physical_valid']['epe_px'] for x in rows];b=[x['regions']['common_event_active_valid']['epe_px'] for x in rows]
        computed.append(dict(method=m,overall_mean=statistics.mean(a),overall_sd=statistics.stdev(a),active_mean=statistics.mean(b),active_sd=statistics.stdev(b)))
    by_method={x['method']:x for x in summary['methods']}
    for row in computed:
        for key,value in row.items():
            if key!='method':check(close(value,by_method[row['method']][key]),'Complete three-seed mean/sampleSD independently recomputed',dict(method=row['method'],metric=key))
    return dict(cells=15,windows_per_cell=528,seeds=3,methods=computed,external_per_cell_JSON_replayed_locally=False)

def tables(root,r):
    full=read(r/'full15_synthetic_decomposition_summary.json');pose=read(r/'pose_soft_feature_summary.json')
    arm_mapping=read(r/'pose_soft/results/feature_retention_execution_receipt.json')['canonical_arm_mapping']
    reverse_arm_mapping={v:k for k,v in arm_mapping.items()}
    check(len(reverse_arm_mapping)==3 and reverse_arm_mapping.get('feature_retention')=='paired_consistency','Pose CSV uses explicit receipt canonical arm mapping')
    expected={x['method']:x for x in full['methods']};count=0
    with (root/'tables/full15_synthetic_methods.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f));check(len(rows)==5,'Method CSV has five complete methods')
        for row in rows:
            for key,value in row.items():
                if key!='method':check(close(value,expected[row['method']][key]),'Method CSV agrees with actual three-seed evidence');count+=1
    cells=read(r/'full15/merged_matrix.json')['cells']
    with (root/'tables/full15_synthetic_strata_EG_V2E.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f));check(len(rows)==8,'Four-stratum/two-region CSV complete')
        for row in rows:
            values={m:[next(x for x in cells if x['method']==m and x['seed']==s)['strata'][row['stratum']][row['region']]['epe_px'] for s in range(3)] for m in PAIR}
            a,b=statistics.mean(values['eventgenesis']),statistics.mean(values['v2e'])
            check(close(row['EG_mean'],a) and close(row['V2E_mean'],b) and int(row['EG_winning_seeds'])==sum(x<y for x,y in zip(values['eventgenesis'],values['v2e'])) and close(row['EG_relative_reduction'],(b-a)/b),'Stratum CSV means/wins/relative difference independently recomputed');count+=4
    values={(x['method'],x['seed']):x['arms'] for x in pose['per_seed']}
    with (root/'tables/pose_soft_per_seed.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f));check(len(rows)==72,'Pose CSV preserves6seeds x3arms x4conditions')
        for row in rows:
            check(row['arm'] in reverse_arm_mapping,'Pose CSV arm belongs to frozen canonical mapping',row['arm'])
            v=values[(row['method'],int(row['seed']))][reverse_arm_mapping[row['arm']]][row['condition']]
            check(close(row['depth_mae_cm'],100*v['depth_mae_m']) and close(row['center_epe_px'],v['center_px']) and close(row['ADD_S_success_percent'],100*v['success']),'Pose table units and values match raw compact evidence');count+=3
    return dict(numeric_table_fields_verified=count)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-root',type=Path,default=Path('/home/科研/Eventbased_WAM/repositories/gec-mainline-resume-20261105/benchmark/sim2real/reviews/flow_pose_short_20261108'))
    p.add_argument('--zip',type=Path,default=Path('/mnt/c/Users/zhaoyuhan/Documents/ChatGPT/New project/outputs/flow_pose_short_20261108/round2_evidence.zip'))
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.review_root.resolve(strict=True)
    check(not a.out.exists() and a.out.resolve().is_relative_to(root/'verification'),'New verification-only output; no source/report/index mutation')
    result=dict(schema='flow_pose_short_independent_archive_audit_v1',status='failed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_script=record(Path(__file__).resolve()),review_root=str(root),checks=CHECKS,
        no_SSH=True,no_training=True,no_real_scores_read=True,scope='Archived bytes and scalar/metadata consistency only; no independent GPU replay, raw bank or large checkpoint download')
    try:
        check(a.zip.is_file() and a.zip.stat().st_size==304763 and sha(a.zip)=='2a16ba7e58a8210765c6201c1504885718cc141b5db3d8f0ef5efbea79ec8bac','Original round2 ZIP exact304763B/SHA verified')
        result['source_manifests']=source_manifests(root)
        initial=root/'evidence/initial';r=root/'evidence/round2'
        i=evidence_index(initial,'INITIAL_EVIDENCE_INDEX.json',9);j=evidence_index(r,'ROUND2_EVIDENCE_INDEX.json',62)
        check(j['source_core']=='62da4f35ce3c5c96f04268bd1687a27657af9238','Frozen simulator core unchanged')
        result['photo']=photo(root,r);result['census']=census(root,r,result['photo']);result['pose']=pose(root,r,initial)
        result['pose_data']=data_v3(root,r);result['full_synthetic']=full15(root,r);result['tables']=tables(root,r)
        slow_live=read(r/'queue/157_pose_slow_features_study_20261108.live.json')
        check('exit_code' not in slow_live and not (r/'pose_slow/study/completion.json').exists(),'Archived slow-feature pose study is incomplete; no adoption claim')
        result.update(status='passed',initial_evidence_files=9,round2_evidence_files=62,external_artifacts_not_replayed=j['external_artifacts'],incomplete_studies=['Census200updatepairedstudy','pose_slow_features_study'])
    except BaseException:result['error']=traceback.format_exc()
    result['passed_checks']=sum(x['passed'] for x in CHECKS);result['failed_checks']=sum(not x['passed'] for x in CHECKS)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],passed_checks=result['passed_checks'],failed_checks=result['failed_checks'],output=str(a.out),error=result.get('error')),ensure_ascii=False,indent=2))
    if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
