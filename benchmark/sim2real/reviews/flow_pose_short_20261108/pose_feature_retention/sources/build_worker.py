"""Surgical standalone extension of the prior guarded native-evaluator wrapper."""
from pathlib import Path
import hashlib
P=Path(__file__).resolve().parent
s=(P/'worker_reference.py').read_text()
assert hashlib.sha256((P/'worker_reference.py').read_bytes()).hexdigest()=='2cba8a4661eebdbd58d46b9023af436fb1c978806da090565be7d0a4fb131d19'
s=s.replace('run_pose_frozen_features','run_pose_feature_retention')
s=s.replace('FrozenFeaturesFitter','FeatureRetentionFitter').replace('pose_frozen_features_policy','pose_feature_retention_policy')
s=s.replace('frozen_features','feature_retention')
s=s.replace('One bounded synthetic-only head adaptation','One bounded synthetic-only soft feature-retention trial')
s=s.replace('pose_frozen_bound_native','pose_feature_bound_native')
s=s.replace("choices=('smoke','study')","choices=('calibrate','smoke','study')")
needle="    args = parser.parse_args()\n"
s=s.replace(needle,"    parser.add_argument('--calibration',type=Path)\n    parser.add_argument('--calibration-sha256')\n    parser.add_argument('--smoke-receipt',type=Path)\n    parser.add_argument('--smoke-receipt-sha256')\n"+needle)
needle="    from reuse_controls import PriorControls, record\n"
assert s.count(needle)==1
s=s.replace(needle,needle+'''    if args.phase=='calibrate':
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
''')
s=s.replace('fitter=FeatureRetentionFitter(native,args.out,smoke,None if smoke else prior)','fitter=FeatureRetentionFitter(native,args.out,smoke,coefficient,None if smoke else prior)')
s=s.replace("policy='feature_retention_head_only'","policy='protected_same_view_feature_retention'")
s=s.replace("candidate_trainable='center and size heads in depth branch; center head only in localizer branch'","candidate_trainable='same as C1; all feature parameters trainable; localizer size head remains frozen'")
s=s.replace("features='weights and buffers fixed to source teacher; eval mode (including any BN stats)'","features='soft same-view source retention on normal/half only; no dark retention or feature freezing',\n                feature_coefficient=coefficient,calibration=record(args.calibration),calibration_source='TRAIN-only C1 gradient scale across all twelve branches'")
s=s.replace("prior_controls_unchanged=True,all_controls_metrics_verified_against_original=not smoke,","prior_controls_unchanged=True,all_controls_metrics_verified_against_original=not smoke,\n            feature_coefficient=coefficient,calibration_sha256=args.calibration_sha256,\n            dark_feature_retention=False,retention_views=['normal','half'],features_frozen=False,")
(P/'run_pose_feature_retention.py').write_text(s)
print(hashlib.sha256((P/'run_pose_feature_retention.py').read_bytes()).hexdigest())
