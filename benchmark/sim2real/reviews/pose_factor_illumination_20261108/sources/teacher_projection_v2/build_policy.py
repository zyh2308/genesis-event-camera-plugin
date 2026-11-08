"""Reproducible single-factor derivation from the frozen original C1 fit."""
from pathlib import Path
import hashlib
P=Path(__file__).resolve().parent
old=P/'pose_priority_policy.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='e1883b68715ecf067a9f4dccbd7a61af5f5dc26b557e061859f4066e7de8d38c'
s=old.read_text().replace('class BoundedFitter:','class TeacherProjectionFitter:')
s=s.replace('def __init__(self, native, out, smoke):','def __init__(self, native, out, smoke, prior=None):')
s=s.replace('self.records, self.parents, self.probes = [], {}, {}','self.records, self.parents, self.probes = [], {}, {}\n        self.prior, self.center_cache = prior, {}')
needle='        self.parents[key] = source_hash\n'
assert s.count(needle)==1
s=s.replace(needle,needle+'''        if arm == 'paired_consistency' and branch == 'center':
            return reuse_candidate_center(self,method,seed,scale,state,normal,target,aux,aux_target,aux_clean)
        if self.prior is not None and arm != 'paired_consistency':
            result=self.prior.reuse(self,method,seed,arm,branch,scale,state,normal,target,aux,aux_target,aux_clean)
            if arm == 'physical_supervised' and branch == 'center':cache_center(self,method,seed,*result,self.records[-1])
            return result
        # Legacy candidate name is only the unchanged native gate interface.
        pair_weight=0
''')
needle='                    optimizer.zero_grad(set_to_none=True); loss.backward()\n'
assert s.count(needle)==1
s=s.replace(needle,'''                    optimizer.zero_grad(set_to_none=True)
                    projection_evidence=None
                    if arm == 'paired_consistency':
                        assert branch == 'depth' and len(parameters)==24
                        from projection import project_teacher_gradients
                        Pgrad=torch.autograd.grad(supervised,parameters,retain_graph=True,allow_unused=True)
                        Tgrad=torch.autograd.grad(2*clean_term,parameters,allow_unused=True)
                        projected=project_teacher_gradients(parameters,Pgrad,Tgrad,
                            [name for name,p in model.named_parameters() if p.requires_grad])
                        for parameter,gradient in zip(parameters,projected['combined_grads']):parameter.grad=gradient.detach().clone()
                        projection_evidence=projected['diagnostics']
                    else:loss.backward()
''')
needle='                               gradient_l2=float(norm), finite_gradient_tensors=len(parameters))\n'
assert s.count(needle)==1
s=s.replace(needle,'''                               gradient_l2=float(norm), finite_gradient_tensors=len(parameters),
                               gradient_operation='one_sided_teacher_projection' if arm == 'paired_consistency' else 'ordinary_scalar_backward',
                               raw_scalar_loss_is_applied_gradient_potential=arm != 'paired_consistency',
                               projection=projection_evidence)
''')
needle='                        update=updates, source_state_sha256=source_hash, real_data_used=False), checkpoint)\n'
assert s.count(needle)==1
s=s.replace(needle,'''                        update=updates, source_state_sha256=source_hash, real_data_used=False,
                        canonical_arm='teacher_projection' if arm == 'paired_consistency' else arm,
                        gradient_operation='P+projected(2*normalteacher)' if arm == 'paired_consistency' else 'ordinary'), checkpoint)
''')
needle="        self.shared.write(directory / (tag + '_receipt.json'), record); self.records.append(record)\n"
assert s.count(needle)==1
s=s.replace(needle,'''        record.update(canonical_arm='teacher_projection' if arm == 'paired_consistency' else arm,
                      new_optimizer_updates=updates,normal_teacher_weight=2,actual_half_teacher_output_weight=0,
                      feature_learning_rate=.00015,head_learning_rate=.00015,
                      depth_teacher_projection=arm == 'paired_consistency',
                      projection_formula='Tprime=T-min(dot(T,P)/dot(P,P),0)*P; G=P+Tprime',
                      raw_scalar_loss_is_applied_gradient_potential=arm != 'paired_consistency')
        if self.prior is not None and arm == 'paired_consistency':check_candidate_trajectory(self,record,directory / (tag + '.jsonl'))
        self.shared.write(directory / (tag + '_receipt.json'), record); self.records.append(record)
        if arm == 'physical_supervised' and branch == 'center':cache_center(self,method,seed,model,history,record)
''')
s='"""Original C1 objective and sampler; only candidate DEPTH teacher gradient is projected."""\n'+s[s.index('from __future__'):]
s+='''

def cache_center(fitter,method,seed,model,history,record):
    key=(method,seed);assert key not in fitter.center_cache
    fitter.center_cache[key]=dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                 history=list(history),record=copy.deepcopy(record))


def check_candidate_trajectory(fitter,candidate,log_path):
    from reuse_controls import read_log,trajectory,sha
    reference=fitter.prior.records[(candidate['method'],candidate['seed'],'physical_supervised','depth')]
    assert candidate['labels_sha256']==reference['labels_sha256'] and candidate['source_state_sha256']==reference['source_state_sha256']
    assert candidate['updates']==reference['updates']==(1 if fitter.smoke else 288)
    original=fitter.prior.root/'training_evidence'/f"{candidate['method']}_seed{candidate['seed']}_physical_supervised_depth.jsonl"
    assert trajectory(read_log(log_path))==trajectory(read_log(original))
    candidate.update(labels_and_draws_identical_to_C1=True,matched_C1_trajectory_sha256=sha(original))


def reuse_candidate_center(fitter,method,seed,scale,state,normal,target,aux,aux_target,aux_clean):
    from reuse_controls import replay,read_log,trajectory,sha
    cached=fitter.center_cache[(method,seed)];original=cached['record']
    assert original['branch']=='center' and original['arm']=='physical_supervised'
    assert original['source_state_sha256']==state_sha(state) and state_sha(cached['state'])==original['final_state_sha256']
    labels,draws=replay(fitter,normal,target,aux,aux_target,aux_clean,seed,'center','paired_consistency')
    assert labels==original['labels_sha256']
    log=(fitter.prior.root if fitter.prior is not None else fitter.out)/'training_evidence'/f'{method}_seed{seed}_physical_supervised_center.jsonl'
    assert trajectory(read_log(log))==draws
    saved=torch.get_rng_state()
    try:
        model=scale.CenterScale();model.load_state_dict(cached['state'],strict=True);model.size.requires_grad_(False)
    finally:torch.set_rng_state(saved)
    assert state_sha(model.state_dict())==original['final_state_sha256']
    record=dict(original,arm='paired_consistency',canonical_arm='teacher_projection',new_optimizer_updates=0,
        reused_C1_center=True,reused_immutable_control=False,depth_teacher_projection=False,
        candidate_center_optimizer_used=False,center_training='none; exact verified C1 state and history reused',
        C1_center_source_record=original,labels_sha256=labels,label_and_augmentation_replay_exact=True,
        source_C1_center_log_sha256=sha(log),candidate_center_state_equal_C1=True,
        recovery_scope='C1 source optimizer independently verified when loaded/trained; new center raw model independently reconstructed')
    directory=fitter.out/'training_evidence';directory.mkdir(exist_ok=True)
    fitter.shared.write(directory/f'{method}_seed{seed}_paired_consistency_center_C1_reuse_receipt.json',record)
    fitter.records.append(record)
    return model.to(normal.device).eval(),cached['history']
'''
(P/'pose_teacher_projection_policy.py').write_text(s)
helper=(P/'reuse_controls.py').read_text().replace('from pose_frozen_features_policy import','from pose_teacher_projection_policy import')
if '    def check_candidate(' in helper:
    start=helper.index('    def check_candidate(');end=helper.index('    def compare_control_summaries(',start)
    helper=helper[:start]+helper[end:]
(P/'reuse_controls.py').write_text(helper)
print(hashlib.sha256((P/'pose_teacher_projection_policy.py').read_bytes()).hexdigest())
