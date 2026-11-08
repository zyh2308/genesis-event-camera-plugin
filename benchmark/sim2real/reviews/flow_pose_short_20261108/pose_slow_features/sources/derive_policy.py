"""Reproducible local derivation; never changes an old policy or starts a run."""
import ast,hashlib
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    here=Path(__file__).resolve().parent;old=here.parent/'pose_frozen_features_20261108'
    assert sha(old/'pose_frozen_features_policy.py')=='11523763629c0c72236d84c53c6fc391010ccce0bf3db6466ab3513270e0fefb'
    assert sha(old/'reuse_controls.py')=='9d4ef4612862408f5e60bd4cf8b12850c090e5529196980500b812b82859d080'
    original=(old/'pose_frozen_features_policy.py').read_text()
    s=original.replace('class FrozenFeaturesFitter:','class SlowFeaturesFitter:')
    s=s.replace('from torch.nn import functional as F','from torch.nn import functional as F\nfrom slow_feature_optimizer import make_optimizer,validate_optimizer')
    s=s.replace('# Canonical candidate = frozen_features; it has NO half teacher term.','# Canonical candidate = slow_features; half teacher0, feature KD0; only feature LR changes.')
    s=s.replace("        if arm == 'paired_consistency': model.features.requires_grad_(False); model.features.eval()\n",'')
    oldopt='        optimizer = torch.optim.AdamW(parameters, lr=.00015, weight_decay=.01)\n'
    assert s.count(oldopt)==1
    s=s.replace(oldopt,"        optimizer,group_signature = make_optimizer(model,arm == 'paired_consistency')\n")
    s=s.replace("        tag = f'{method}_seed{seed}_{arm}_{branch}'", "        canonical_arm = 'slow_features' if arm == 'paired_consistency' else arm\n        tag = f'{method}_seed{seed}_{canonical_arm}_{branch}'")
    oldfreeze="        if arm == 'paired_consistency': assert all(torch.equal(v.detach().cpu(), state[k].cpu()) for k, v in model.state_dict().items() if k.startswith('features.'))\n"
    assert s.count(oldfreeze)==1;s=s.replace(oldfreeze,'')
    s=s.replace("                        update=updates, source_state_sha256=source_hash, real_data_used=False), checkpoint)",
                "                        update=updates, source_state_sha256=source_hash, real_data_used=False,optimizer_group_signature=group_signature), checkpoint)")
    oldrebuild="            if arm == 'paired_consistency': rebuilt.features.requires_grad_(False); rebuilt.features.eval()\n"
    assert s.count(oldrebuild)==1;s=s.replace(oldrebuild,'')
    oldopt="            rebuilt_optimizer = torch.optim.AdamW([p for p in rebuilt.parameters() if p.requires_grad], lr=.00015, weight_decay=.01)"
    assert s.count(oldopt)==1
    s=s.replace(oldopt,"            rebuilt_optimizer,rebuilt_groups = make_optimizer(rebuilt,arm == 'paired_consistency')\n            assert rebuilt_groups == restored['optimizer_group_signature']")
    s=s.replace("            rebuilt_optimizer.load_state_dict(restored['optimizer'])", "            rebuilt_optimizer.load_state_dict(restored['optimizer'])\n            assert validate_optimizer(rebuilt,rebuilt_optimizer,arm == 'paired_consistency') == group_signature")
    start=s.index("        record.update(canonical_arm='frozen_features'")
    end=s.index("        if self.prior is not None and arm == 'paired_consistency':",start)
    s=s[:start]+'''        changed={name:[n for n,p in model.named_parameters() if p.requires_grad and (n.startswith('features.') if name=='features' else not n.startswith('features.')) and not torch.equal(p.detach().cpu(),state[n].cpu())] for name in ('features','heads')}
        if arm == 'paired_consistency': assert changed['features'] and changed['heads']
        record.update(canonical_arm=canonical_arm,actual_half_teacher_weight=0,actual_half_teacher_output_weight=0,
            feature_KD_weight=0.,normal_teacher_weight=2,new_optimizer_updates=updates,features_frozen=False,
            feature_learning_rate=.000015 if arm == 'paired_consistency' else .00015,head_learning_rate=.00015,
            optimizer_group_signature=group_signature,optimizer_groups_independently_rebuilt=True,changed_parameter_names=changed,
            candidate_single_factor='features LR0.1x; same C1 objective, all other settings retained')
''' + s[end:]
    s='"""Derived C1 physical+normalteacher2 with fixed feature LR0.1x only."""\n'+s[s.index('from __future__'):]
    # Every mature loss/sampling definition remains bytecode-AST-equivalent.
    def funcs(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    before,after=funcs(original),funcs(s)
    assert before==after
    target=here/'pose_slow_features_policy.py'
    if target.exists():assert target.read_text()==s
    else:target.write_text(s)
    # Controls' source reuse/metric comparisons remain unchanged; candidate admission only differs.
    reuse=(old/'reuse_controls.py').read_text().replace('from pose_frozen_features_policy import state_sha, draw_auxiliary','from pose_slow_features_policy import state_sha, draw_auxiliary')
    oldcheck="        assert candidate['features_frozen'] and candidate['features_weights_and_buffers_unchanged']\n"
    assert oldcheck in reuse
    reuse=reuse.replace(oldcheck,"        assert candidate['features_frozen'] is False and candidate['feature_KD_weight']==0\n        assert candidate['feature_learning_rate']==.000015 and candidate['head_learning_rate']==.00015\n        assert candidate['actual_half_teacher_output_weight']==0 and candidate['normal_teacher_weight']==2\n        assert candidate['optimizer_groups_independently_rebuilt'] is True\n")
    target=here/'reuse_controls.py'
    if target.exists():assert target.read_text()==reuse
    else:target.write_text(reuse)
    print(dict(policy=sha(here/'pose_slow_features_policy.py'),reuse=sha(target),inherited_top_level_functions_unchanged=True))

if __name__=='__main__':main()
