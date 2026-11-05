"""Register a fixed nonrigid structure using raw native-label audit, no scores."""
import hashlib,json
from pathlib import Path
import numpy as np
here=Path(__file__).resolve().parent;source=here/'pose_rotation_native_diagnostic.json'
raw=json.loads(source.read_text());rows=raw['samples']
a=np.array([x['pose'] for x in rows],np.float64)[:,:3,:3]
gram=a.transpose(0,2,1)@a;sv=np.linalg.svd(a,compute_uv=False);det=np.linalg.det(a)
assert len(rows)==54 and len({Path(x['file']).parent.name for x in rows})==18
assert np.max(np.abs(gram-gram[0]))<1e-12 and np.max(np.abs(sv-1))<.01 and (det>0).all()
policy=dict(schema='stage9_epose_native_linear_audit_v1',reference_right_gram=gram[0].tolist(),
    native_gram_max_abs_residual_tolerance=1e-8,native_det_reference=float(det[0]),native_det_abs_tolerance=1e-8,
    max_singular_value_relative_strain=.01,strict_rigid_max_orthogonality=1e-4,strict_rigid_det_abs_tolerance=1e-4,
    audit_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),audited_packets=len(rows),audited_sequences=18,
    audited_sequence_names=sorted({Path(x['file']).parent.name for x in rows}),
    observed_right_gram_max_variation=float(np.max(np.abs(gram-gram[0]))),
    observed_singular_values_min=sv.min(0).tolist(),observed_singular_values_max=sv.max(0).tolist(),
    observed_max_relative_strain=float(np.max(np.abs(sv-1))),observed_det_min=float(det.min()),observed_det_max=float(det.max()),
    rationale='54 native labels in18 sequences have identical right Gram to2e-15 and0.5063% maximum strain. Accept only this audited fixed structure within1e-8 plus1% strain hard ceiling, or strict rigid inputs. This is not a generic1% nonrigid allowance.',
    causal_interpretation='fixed right-side linear distortion structure; physical/calibration cause unverified',
    ground_truth_policy='preserve published4x4 transform exactly for all main geometry; never repair/project mainGT')
with (here/'native_pose_audit_policy.json').open('x') as f:json.dump(policy,f,indent=2);f.write('\n')
print('NATIVE_POSE_POLICY_REGISTERED')
