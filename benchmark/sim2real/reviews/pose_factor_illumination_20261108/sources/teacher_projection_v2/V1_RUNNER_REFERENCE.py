"""Depth-only teacher conflict filtering with immutable C1 centers and native gates."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch

sys.dont_write_bytecode=True
CONTRACT_SHA='90eec5f59cc31d8543a6de5aad2638fbc2b8618338e7c2f1b7b0154168f1c222'
ARM_MAPPING={'clean_control':'clean_control','physical_supervised':'physical_supervised','paired_consistency':'teacher_projection'}


def validate_candidate_contract(c):
    assert c['schema']=='pose_depth_teacher_projection_v1'
    assert c['candidate_name']=='teacher_projection' and c['candidate_center']=='reuse_C1_unchanged'
    assert c['new_depth_optimizer_updates']==1728 and c['updates_per_depth_branch']==288
    assert c['epochs']==6 and c['batch_size']==32 and c['half_probability']==.75
    assert c['feature_learning_rate']==c['head_learning_rate']==.00015
    assert c['normal_teacher_weight']==2 and c['auxiliary_teacher_weight']==0
    assert c['methods']==['eventgenesis','v2e'] and c['seeds']==[0,1,2]


def verify_smoke(path,digest,manifest_sha,contract_sha,sha):
    if path is None or digest is None:raise ValueError('Actual SHA-bound teacher-projection smoke is required')
    assert sha(path)==digest
    s=json.loads(Path(path).read_text())
    assert s['status']=='complete_synthetic_only' and s['phase']=='smoke' and not s['real_scores_computed']
    assert s['source_manifest_sha256']==manifest_sha and s['contract_sha256']==contract_sha
    assert s['canonical_candidate']=='teacher_projection' and s['new_optimizer_updates']==10
    rows=s['records'];expected={(m,0,a,b) for m in ('eventgenesis','v2e') for a in ARM_MAPPING for b in ('depth','center')}
    assert len(rows)==12 and {(r['method'],r['seed'],r['arm'],r['branch']) for r in rows}==expected
    for r in rows:
        assert r['teacher_unchanged'] and r['independent_model_optimizer_rebuild'] and r['optimizer_state_equal']
        if r['arm']=='paired_consistency' and r['branch']=='center':
            assert r['new_optimizer_updates']==0 and r['reused_C1_center'] and r['candidate_center_state_equal_C1'] and not r['candidate_center_optimizer_used']
        else:assert r['updates']==r['new_optimizer_updates']==1
        if r['arm']=='paired_consistency' and r['branch']=='depth':
            assert r['depth_teacher_projection'] and r['canonical_arm']=='teacher_projection'
            assert r['feature_learning_rate']==r['head_learning_rate']==.00015 and r['normal_teacher_weight']==2 and r['actual_half_teacher_output_weight']==0
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--phase',choices=('smoke','study'),required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke-receipt',type=Path);p.add_argument('--smoke-receipt-sha256');args=p.parse_args()
    from reuse_controls import sha,record,PriorControls
    assert sha(args.contract)==CONTRACT_SHA
    c=json.loads(args.contract.read_text());validate_candidate_contract(c)
    assert args.out.is_absolute() and args.out.parent.is_dir() and not args.out.exists() and not args.out.is_symlink()
    diagnostic_sources=Path(c['diagnostic_source_dir']);mp=diagnostic_sources/'source_manifest.json'
    assert sha(mp)==c['diagnostic_source_manifest_sha256']
    for name,digest in json.loads(mp.read_text()).items():assert Path(name).name==name and sha(diagnostic_sources/name)==digest
    spec=importlib.util.spec_from_file_location('teacher_projection_checked_export',diagnostic_sources/'export_pose_factors.py')
    E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
    base=dict(c);base['schema']='pose_factor_conflict_diagnostic_v1'
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    ctx=E.load_context(base,check_gpu=True)
    for path,digest in [(args.contract,CONTRACT_SHA),(mp,c['diagnostic_source_manifest_sha256']),
                        (c['forward_completion'],c['forward_completion_sha256']),
                        (c['train_probe_completion'],c['train_probe_completion_sha256'])]:E.remember(ctx.source_records,path,digest)
    for key in ('forward_completion','train_probe_completion'):
        value=json.loads(Path(c[key]).read_text());assert value['status']=='complete' and value['optimizer_updates']==0 and not value['real_data_used']
    own=Path(__file__).resolve().parent;manifest_path=own/'source_manifest.json'
    own_manifest=json.loads(manifest_path.read_text())
    for name,digest in own_manifest.items():
        assert Path(name).name==name;E.remember(ctx.source_records,own/name,digest)
    E.remember(ctx.source_records,manifest_path)
    manifest_sha=sha(manifest_path)
    if args.phase=='study':verify_smoke(args.smoke_receipt,args.smoke_receipt_sha256,manifest_sha,CONTRACT_SHA,sha)
    prior=PriorControls(Path(c['prior_study']),Path(c['prior_diagnostic']))
    assert sha(Path(c['prior_study'])/'completion.json')==c['prior_study_completion_sha256']
    assert sha(Path(c['prior_study'])/'priority_execution_receipt.json')==c['prior_execution_receipt_sha256']
    assert sha(c['prior_diagnostic'])==c['prior_diagnostic_sha256']
    for path,digest in [(Path(c['prior_study'])/'completion.json',c['prior_study_completion_sha256']),
                        (Path(c['prior_study'])/'priority_execution_receipt.json',c['prior_execution_receipt_sha256']),
                        (c['prior_diagnostic'],c['prior_diagnostic_sha256'])]:E.remember(ctx.source_records,path,digest)
    native=ctx.native;original_write=native.shared.write
    from pose_teacher_projection_policy import TeacherProjectionFitter
    smoke=args.phase=='smoke';fitter=TeacherProjectionFitter(native,args.out,smoke,None if smoke else prior)
    native.fit,native.fit_center=fitter.fit,fitter.fit_center
    start=time.monotonic()
    def write(path,value):
        if time.monotonic()-start>c['timeout_seconds']:raise TimeoutError('Projection phase exceeded its fixed runtime bound')
        path=Path(path)
        if path==args.out/'protocol.json':
            value=dict(value,source_sha256=sha(Path(__file__)),native_source_sha256=c['native_source_sha256'],
                native_main_and_evaluation_unchanged=True,canonical_arm_mapping=ARM_MAPPING,policy='depth_only_one_sided_teacher_gradient_projection',
                candidate_center='exact verified C1 localizer reused; no optimizer update',actual_half_teacher_weight=0,
                new_total_optimizer_updates=10 if smoke else 1728,new_candidate_center_optimizer_updates=0,
                new_candidate_depth_optimizer_updates_per_method_seed=1 if smoke else 288,
                teacher_clean_weight=2,feature_learning_rate=.00015,head_learning_rate=.00015,auxiliary_half_probability=.75,
                projected_gradient='P+Tprime; P=.5*grad(native_sup(clean)+native_sup(aux)); T=grad(2*normalteacher); Tprime=T-min(dot(T,P)/dot(P,P),0)*P',
                projection_space='one complete24-tensor depth vector; no per-layer projection; zero-P leaves T unchanged',
                loss_reporting='original raw scalar P_loss+T_loss is observed but is not the potential whose gradient is applied after projection',
                real_evaluation='disabled; original30 illumination+8 confirmation checks unchanged',contract_sha256=CONTRACT_SHA)
        if not smoke and path.parent==args.out and path.name in {f'{m}_seed{s}.json' for m in c['methods'] for s in c['seeds']}:
            prior.compare_control_summaries(value)
            for condition in value['arms']['paired_consistency']:
                a=value['arms']['paired_consistency'][condition]['summary']['center_px']
                b=value['arms']['physical_supervised'][condition]['summary']['center_px']
                assert abs(a-b)<1e-8,'Candidate center must remain exact C1'
        return original_write(path,value)
    native.shared.write=write
    original_argv=sys.argv
    try:
        sys.argv=[str(Path(c['prior_source_dir'])/'pose_two_branch_coverage.py'),'--campaign',c['campaign'],
            '--radiance-train',c['radiance_train'],'--radiance-validation',c['radiance_validation'],'--out',str(args.out)]+(['--smoke'] if smoke else [])
        native.main()
        assert len(fitter.records)==(12 if smoke else 36)
        assert sum(r['new_optimizer_updates'] for r in fitter.records)==(10 if smoke else 1728)
        assert sum(bool(r.get('reused_C1_center')) for r in fitter.records)==(2 if smoke else 6)
        for seed in ([0] if smoke else c['seeds']):assert len({r['labels_sha256'] for r in fitter.records if r['seed']==seed})==1
        result=json.loads((args.out/'completion.json').read_text());assert result['status']=='complete' and not result['real_data_used']
        if not smoke:
            assert len(result['gate']['checks'])==30 and len(result['original_confirmation']['comparisons'])==8
            expected=[]
            for run in result['results']:
                old=prior.runs[(run['method'],run['seed'])]['arms']['clean_control'];new=run['arms']['paired_consistency']
                expected.append(new['clean']['summary']['success']-old['clean']['summary']['success']>=-.02)
                for condition in ('half','eighth'):expected += [new[condition]['summary']['depth_mae_m']<old[condition]['summary']['depth_mae_m'],new[condition]['summary']['success']>=old[condition]['summary']['success']]
            assert expected==[r['passed'] for r in result['gate']['checks']]
        for item in ctx.source_records:assert sha(item['path'])==item['sha256']
        for r in fitter.records:
            if r.get('reused_immutable_control'):
                for key in ('original_model','original_log','original_receipt'):assert sha(r[key]['path'])==r[key]['sha256']
        original_write(args.out/'teacher_projection_execution_receipt.json',dict(status='complete_synthetic_only',phase=args.phase,
            canonical_candidate='teacher_projection',canonical_arm_mapping=ARM_MAPPING,contract_sha256=CONTRACT_SHA,
            source_manifest_sha256=manifest_sha,native_source_sha256=c['native_source_sha256'],real_scores_computed=False,
            new_optimizer_updates=sum(r['new_optimizer_updates'] for r in fitter.records),candidate_center='reuse_C1_unchanged',
            original_thirty_gate_unchanged=True,original_eight_confirmation_unchanged=True,prior_controls_unchanged=True,
            all_controls_metrics_verified_against_original=not smoke,raw_scalar_loss_is_applied_gradient_potential=False,
            thirty_gate=result.get('gate'),original_confirmation=result.get('original_confirmation'),records=fitter.records,
            source_records=ctx.source_records,elapsed_seconds=time.monotonic()-start,
            limitation='One-sided local Euclidean gradient filtering does not guarantee AdamW step alignment or pose/generalization improvement.'))
    finally:native.shared.write=original_write;sys.argv=original_argv


if __name__=='__main__':main()
