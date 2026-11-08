"""One bounded synthetic-only soft feature-retention trial; native full evaluator and gates unchanged."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
CORE = '62da4f35ce3c5c96f04268bd1687a27657af9238'
GPU4 = 'GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3'
NATIVE_SHA = '85951e6d43e59140b81d7303370a6b252fec902dc065ff17bb30faf8e4fc6186'
PRIOR_POLICY_SHA = 'e1883b68715ecf067a9f4dccbd7a61af5f5dc26b557e061859f4066e7de8d38c'
ARM_MAPPING = {'clean_control':'clean_control','physical_supervised':'physical_supervised',
               'paired_consistency':'feature_retention'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2),b''): value.update(block)
    return value.hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location('pose_feature_bound_native',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def function_ast_sha(path, name):
    node = next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','prior-source-dir','radiance-train','radiance-validation','prior-study','diagnostic','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prior-source-manifest-sha256',required=True)
    parser.add_argument('--phase',choices=('calibrate','smoke','study'),required=True)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--calibration-sha256')
    parser.add_argument('--smoke-receipt',type=Path)
    parser.add_argument('--smoke-receipt-sha256')
    args = parser.parse_args()
    assert args.out.is_absolute() and not args.out.exists() and not args.out.is_symlink()
    assert args.out.parent.is_dir()
    assert json.loads((args.campaign/'sources/training_recipe.json').read_text())['frozen_simulator_core']==CORE
    sources = args.prior_source_dir
    assert sources.resolve(strict=True)==sources
    assert sha(sources/'source_manifest.json')==args.prior_source_manifest_sha256
    manifest = json.loads((sources/'source_manifest.json').read_text())
    for name,digest in manifest.items():
        assert Path(name).name==name and sha(sources/name)==digest
    native_path=sources/'pose_two_branch_coverage.py'; assert sha(native_path)==NATIVE_SHA
    own=Path(__file__).resolve().parent
    own_manifest=json.loads((own/'source_manifest.json').read_text())
    for name,digest in own_manifest.items():
        assert Path(name).name==name and sha(own/name)==digest
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (GPU4,'4')
    actual=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).strip()
    assert actual==GPU4
    import torch
    assert torch.cuda.device_count()==1
    torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(sources))
    native=load(native_path)
    from pose_feature_retention_policy import FeatureRetentionFitter
    from reuse_controls import PriorControls, record
    if args.phase=='calibrate':
        torch.set_num_threads(2);torch.set_num_interop_threads(1);native.cv2.setNumThreads(1)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        from calibrate_feature_retention import calibrate
        calibrate(native,args,own,sha)
        return
    from calibration_contract import verify_calibration, verify_smoke
    calibration=verify_calibration(args.calibration,args.calibration_sha256,sha(own/'source_manifest.json'))
    coefficient=calibration['coefficient']
    if args.phase=='study': verify_smoke(args.smoke_receipt,args.smoke_receipt_sha256,calibration,args.calibration_sha256,sha(own/'source_manifest.json'))
    # All source controls and their actual metrics are fixed before any new update.
    prior=PriorControls(args.prior_study,args.diagnostic)
    assert prior.execution['prior_source_manifest_sha256']==args.prior_source_manifest_sha256
    smoke=args.phase=='smoke'
    fitter=FeatureRetentionFitter(native,args.out,smoke,coefficient,None if smoke else prior)
    native.fit,native.fit_center=fitter.fit,fitter.fit_center
    original_write=native.shared.write
    old_file_records={str(args.prior_study/'completion.json'):sha(args.prior_study/'completion.json'),
                      str(args.prior_study/'priority_execution_receipt.json'):sha(args.prior_study/'priority_execution_receipt.json')}
    def write(path,value):
        path=Path(path)
        if path==args.out/'protocol.json':
            value=dict(value,source_sha256=sha(Path(__file__).resolve()),native_source_sha256=NATIVE_SHA,
                native_main_ast_sha256=function_ast_sha(native_path,'main'),native_main_and_evaluation_unchanged=True,
                policy='protected_same_view_feature_retention',canonical_arm_mapping=ARM_MAPPING,
                candidate_legacy_slot='paired_consistency',candidate_canonical_name='feature_retention',
                teacher_clean_weight=2,paired_teacher_weight=0,actual_half_teacher_weight=0,
                candidate_trainable='same as C1; all feature parameters trainable; localizer size head remains frozen',
                features='soft same-view source retention on normal/half only; no dark retention or feature freezing',
                feature_coefficient=coefficient,calibration=record(args.calibration),calibration_source='TRAIN-only C1 gradient scale across all twelve branches',
                auxiliary_half_probability=.75,expected_supervision_fractions=dict(normal=.5,half=.375,eighth=.125),
                controls='fresh one-update smoke only' if smoke else 'immutable prior C0/C1 288-update checkpoints; complete trajectory replay and metric verification',
                prior_completion=record(args.prior_study/'completion.json'),
                prior_execution_receipt=record(args.prior_study/'priority_execution_receipt.json'),
                diagnostic=record(args.diagnostic),
                real_evaluation='disabled; unchanged original thirty illumination and eight confirmation checks')
        if not smoke and path.parent==args.out and path.name in {f'{m}_seed{s}.json' for m in native.shared.METHODS for s in range(3)}:
            prior.compare_control_summaries(value)
        return original_write(path,value)
    native.shared.write=write
    command=[str(native_path),'--campaign',str(args.campaign),'--radiance-train',str(args.radiance_train),
             '--radiance-validation',str(args.radiance_validation),'--out',str(args.out)]
    if smoke: command.append('--smoke')
    original_argv=sys.argv
    try:
        sys.argv=command; native.main()
        assert len(fitter.records)==(12 if smoke else 36)
        for seed in ([0] if smoke else range(3)):
            assert len({r['labels_sha256'] for r in fitter.records if r['seed']==seed})==1
        result=json.loads((args.out/'completion.json').read_text())
        assert result['status']=='complete' and not result['real_data_used']
        if not smoke:
            assert len(result['gate']['checks'])==30 and len(result['original_confirmation']['comparisons'])==8
            assert sum(r['new_optimizer_updates'] for r in fitter.records)==12*288
            # Recheck native gate against the immutable ORIGINAL clean-control values.
            expected=[]
            for run in result['results']:
                old=prior.runs[(run['method'],run['seed'])]['arms']['clean_control']
                new=run['arms']['paired_consistency']
                expected.append(new['clean']['summary']['success']-old['clean']['summary']['success']>=-.02)
                for condition in ('half','eighth'):
                    expected += [new[condition]['summary']['depth_mae_m']<old[condition]['summary']['depth_mae_m'],
                                 new[condition]['summary']['success']>=old[condition]['summary']['success']]
            assert expected==[r['passed'] for r in result['gate']['checks']]
        for r in fitter.records:
            if r.get('reused_immutable_control'):
                for key in ('original_model','original_log','original_receipt'):
                    assert sha(r[key]['path'])==r[key]['sha256']
        assert all(sha(path)==digest for path,digest in old_file_records.items())
        original_write(args.out/'feature_retention_execution_receipt.json',dict(
            status='complete_synthetic_only',phase=args.phase,real_scores_computed=False,
            canonical_arm_mapping=ARM_MAPPING,actual_half_teacher_weight=0,
            new_optimizer_updates=sum(r['new_optimizer_updates'] for r in fitter.records),
            source_manifest_sha256=sha(own/'source_manifest.json'),native_source_sha256=NATIVE_SHA,
            prior_controls_unchanged=True,all_controls_metrics_verified_against_original=not smoke,
            feature_coefficient=coefficient,calibration_sha256=args.calibration_sha256,
            dark_feature_retention=False,retention_views=['normal','half'],features_frozen=False,
            original_thirty_gate_unchanged=True,original_eight_confirmation_unchanged=True,
            thirty_gate=result.get('gate'),original_confirmation=result.get('original_confirmation'),
            records=fitter.records,source_controls=[record(Path(p)) for p in old_file_records]))
    finally:
        native.shared.write=original_write; sys.argv=original_argv


if __name__=='__main__': main()
