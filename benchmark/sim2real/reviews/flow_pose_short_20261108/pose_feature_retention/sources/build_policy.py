"""Reproducible minimal derivation; never edits the existing deployed policy."""
from pathlib import Path
import hashlib

HERE=Path(__file__).resolve().parent
old=HERE/'pose_priority_policy.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='e1883b68715ecf067a9f4dccbd7a61af5f5dc26b557e061859f4066e7de8d38c'
s=old.read_text()
s=s.replace('class BoundedFitter:','class FeatureRetentionFitter:')
s=s.replace('def __init__(self, native, out, smoke):','def __init__(self, native, out, smoke, coefficient, prior=None):')
s=s.replace('self.records, self.parents, self.probes = [], {}, {}','self.records, self.parents, self.probes = [], {}, {}\n        self.prior, self.coefficient = prior, float(coefficient)\n        assert 1e-6 <= self.coefficient <= 1e6')
needle='        self.parents[key] = source_hash\n'
assert s.count(needle)==1
s=s.replace(needle,needle+'''        if self.prior is not None and arm != 'paired_consistency':
            return self.prior.reuse(self,method,seed,arm,branch,scale,state,
                                    normal,target,aux,aux_target,aux_clean)
        # Compatibility slot only; canonical candidate uses C1 loss + feature retention.
        pair_weight = 0
''')
needle='                    loss = supervised + 2 * clean_term + pair_weight * half_term\n'
assert s.count(needle)==1
s=s.replace(needle,'''                    from feature_retention import protected_feature_loss
                    feature_term = (protected_feature_loss(model,teacher,x,a,torch.as_tensor(half,device=normal.device))
                                    if arm == 'paired_consistency' else supervised*0)
                    loss = supervised + 2 * clean_term + self.coefficient*feature_term
''')
needle="                               gradient_l2=float(norm), finite_gradient_tensors=len(parameters))\n"
assert s.count(needle)==1
s=s.replace(needle,"                               gradient_l2=float(norm), finite_gradient_tensors=len(parameters),\n                               feature_retention=float(feature_term.detach()),\n                               feature_coefficient=self.coefficient if arm == 'paired_consistency' else 0)\n")
needle="                        update=updates, source_state_sha256=source_hash, real_data_used=False), checkpoint)\n"
assert s.count(needle)==1
s=s.replace(needle,"                        update=updates, source_state_sha256=source_hash, real_data_used=False,\n                        feature_coefficient=self.coefficient if arm == 'paired_consistency' else 0,\n                        canonical_arm='feature_retention' if arm == 'paired_consistency' else arm), checkpoint)\n")
needle="        assert state_sha(restored['model']) == state_sha(model.state_dict())\n"
s=s.replace(needle,needle+"        assert restored['feature_coefficient'] == (self.coefficient if arm == 'paired_consistency' else 0)\n")
needle="        self.shared.write(directory / (tag + '_receipt.json'), record); self.records.append(record)\n"
assert s.count(needle)==1
s=s.replace(needle,'''        record.update(canonical_arm='feature_retention' if arm == 'paired_consistency' else arm,
                      actual_half_teacher_output_weight=0,new_optimizer_updates=updates,
                      features_frozen=False,normal_teacher_weight=2,
                      feature_coefficient=self.coefficient if arm == 'paired_consistency' else 0,
                      retention_views=['normal','half'] if arm == 'paired_consistency' else [],
                      dark_feature_retention=False)
        if self.prior is not None and arm == 'paired_consistency':
            self.prior.check_candidate(record,directory / (tag + '.jsonl'))
        self.shared.write(directory / (tag + '_receipt.json'), record); self.records.append(record)
''')
s='"""C1 objective plus one calibrated protected-view feature-retention term."""\n'+s[s.index('from __future__'):]
(HERE/'pose_feature_retention_policy.py').write_text(s)
print(hashlib.sha256((HERE/'pose_feature_retention_policy.py').read_bytes()).hexdigest())
