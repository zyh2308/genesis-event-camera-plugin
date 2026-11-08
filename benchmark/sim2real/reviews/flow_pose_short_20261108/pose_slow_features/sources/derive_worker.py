"""Derive a new canonical worker; preserve native evaluator and original controls."""
from pathlib import Path
import hashlib

def main():
    here=Path(__file__).resolve().parent;source=here.parent/'pose_frozen_features_20261108/run_pose_frozen_features.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='2cba8a4661eebdbd58d46b9023af436fb1c978806da090565be7d0a4fb131d19'
    s=source.read_text().replace('One bounded synthetic-only head adaptation','One bounded synthetic-only fixed slow-feature adaptation')
    s=s.replace("'paired_consistency':'frozen_features'","'paired_consistency':'slow_features'")
    s=s.replace("'pose_frozen_bound_native'","'pose_slow_features_bound_native'")
    s=s.replace("    args = parser.parse_args()", "    parser.add_argument('--smoke-receipt',type=Path)\n    parser.add_argument('--smoke-receipt-sha256')\n    args = parser.parse_args()")
    s=s.replace('from pose_frozen_features_policy import FrozenFeaturesFitter','from pose_slow_features_policy import SlowFeaturesFitter')
    s=s.replace('fitter=FrozenFeaturesFitter(native,args.out,smoke,None if smoke else prior)',
        '''if not smoke:
        from smoke_contract import verify_smoke
        verify_smoke(args.smoke_receipt,args.smoke_receipt_sha256,sha(own/'source_manifest.json'))
    fitter=SlowFeaturesFitter(native,args.out,smoke,None if smoke else prior)''')
    s=s.replace("policy='frozen_features_head_only'","policy='features_lr_point1_same_C1_objective'")
    s=s.replace("candidate_canonical_name='frozen_features'","candidate_canonical_name='slow_features'")
    s=s.replace("candidate_trainable='center and size heads in depth branch; center head only in localizer branch'",
                "candidate_trainable='same as C1: trainable features and heads; original localizer size-head freeze only'")
    s=s.replace("features='weights and buffers fixed to source teacher; eval mode (including any BN stats)'",
                "features='trainable, same original mode as C1; AdamW lr1.5e-5 vs head lr1.5e-4; no feature KD or half teacher',\n                feature_KD_weight=0.,features_frozen=False,feature_learning_rate=.000015,head_learning_rate=.00015,feature_lr_ratio=.1")
    s=s.replace("args.out/'frozen_features_execution_receipt.json'", "args.out/'slow_features_execution_receipt.json'")
    s=s.replace("            canonical_arm_mapping=ARM_MAPPING,actual_half_teacher_weight=0,", "            canonical_arm_mapping=ARM_MAPPING,actual_half_teacher_weight=0,feature_KD_weight=0.,\n            canonical_candidate='slow_features',features_frozen=False,feature_learning_rate=.000015,head_learning_rate=.00015,feature_lr_ratio=.1,")
    target=here/'run_pose_slow_features.py'
    if target.exists():assert target.read_text()==s
    else:target.write_text(s)
    print(hashlib.sha256(target.read_bytes()).hexdigest())

if __name__=='__main__':main()
