"""Local, reproducible surgical derivation from the already audited fit policy."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'pose_priority_20261108/pose_priority_policy.py'
assert hashlib.sha256(OLD.read_bytes()).hexdigest() == 'e1883b68715ecf067a9f4dccbd7a61af5f5dc26b557e061859f4066e7de8d38c'
source = OLD.read_text()
source = source.replace('class BoundedFitter:', 'class FrozenFeaturesFitter:')
source = source.replace('def __init__(self, native, out, smoke):', 'def __init__(self, native, out, smoke, prior=None):')
source = source.replace('self.records, self.parents, self.probes = [], {}, {}', 'self.records, self.parents, self.probes = [], {}, {}\n        self.prior = prior')
needle = '        self.parents[key] = source_hash\n'
assert source.count(needle) == 1
source = source.replace(needle, needle + '''        if self.prior is not None and arm != 'paired_consistency':
            return self.prior.reuse(self, method, seed, arm, branch, scale, state,
                                    normal, target, aux, aux_target, aux_clean)
        # Legacy native slot is retained solely for the unchanged gate interface.
        # Canonical candidate = frozen_features; it has NO half teacher term.
        pair_weight = 0
''')
needle = "        if branch == 'center': model.size.requires_grad_(False)\n"
assert source.count(needle) == 1
source = source.replace(needle, needle + "        if arm == 'paired_consistency': model.features.requires_grad_(False); model.features.eval()\n")
needle = '        assert state_sha(teacher.state_dict()) == source_hash\n'
source = source.replace(needle, needle + "        if arm == 'paired_consistency': assert all(torch.equal(v.detach().cpu(), state[k].cpu()) for k, v in model.state_dict().items() if k.startswith('features.'))\n")
needle = "            if branch == 'center': rebuilt.size.requires_grad_(False)\n"
source = source.replace(needle, needle + "            if arm == 'paired_consistency': rebuilt.features.requires_grad_(False); rebuilt.features.eval()\n")
needle = '        self.shared.write(directory / (tag + \'_receipt.json\'), record); self.records.append(record)\n'
assert source.count(needle) == 1
source = source.replace(needle, '''        record.update(canonical_arm='frozen_features' if arm == 'paired_consistency' else arm,
                      actual_half_teacher_weight=0, new_optimizer_updates=updates,
                      features_frozen=arm == 'paired_consistency',
                      features_weights_and_buffers_unchanged=(all(torch.equal(v.detach().cpu(), state[k].cpu()) for k,v in model.state_dict().items() if k.startswith('features.')) if arm == 'paired_consistency' else None))
        if self.prior is not None and arm == 'paired_consistency':
            self.prior.check_candidate(record, directory / (tag + '.jsonl'))
        self.shared.write(directory / (tag + '_receipt.json'), record); self.records.append(record)
''')
source = '"""Derived C1 objective, with candidate features frozen. See DERIVATION.md."""\n' + source[source.index('from __future__'):]
(HERE / 'pose_frozen_features_policy.py').write_text(source)
print(hashlib.sha256((HERE / 'pose_frozen_features_policy.py').read_bytes()).hexdigest())
