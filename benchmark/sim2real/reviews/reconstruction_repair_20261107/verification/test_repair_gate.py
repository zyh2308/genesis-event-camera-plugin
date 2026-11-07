"""Negative launch-gate controls. Fixture numbers are procedural, never experimental results."""
import copy,json,py_compile
from pathlib import Path
from recon_repair_gate import decide

def fixture():
    rows=[]
    for method in ['eventgenesis','v2e']:
        for seed in [0,1,2]:
            metrics=dict(clipped_lpips=.3,raw_lpips=.3,zero_lpips=.7,gray_lpips=.5,mse=.02,zero_mse=.2,
                clipped_std=.06,target_std=.15,correlation=.5,raw_min=.1,raw_max=.9,negative_fraction=0.)
            evaluation=dict(equal_rollout_mean=metrics,sequence_length=200,degenerate_frame_fraction=0.,
                rollouts=[dict(rollout='test_fixture_'+str(i)) for i in range(8)])
            rows.append(dict(method=method,seed=seed,arm='sigmoid',fixed_train_batch=False,updates=120,real_data_used=False,
                before=dict(equal_rollout_mean=dict(clipped_lpips=.6)),after=evaluation,
                recovery_model_hash_match=True,optimizer_counters=[120],initial_state_sha256='fixture_initial_'+str(seed),
                sample_label_hashes_sha256='fixture_labels_'+str(seed)))
    return rows

checks=[]
def rejected(label,change):
    values=fixture();change(values)
    try:passed=decide(values)['passed']
    except (ValueError,KeyError):passed=False
    assert not passed,label;checks.append(label)
assert decide(fixture())['passed'];checks.append('valid_procedural_fixture')
rejected('missing_seed_cannot_be_excluded',lambda r:r.pop())
rejected('duplicate_cannot_replace_other_seed',lambda r:r.__setitem__(-1,copy.deepcopy(r[-2])))
rejected('real_training_rejected',lambda r:r[0].update(real_data_used=True))
rejected('one_window_fit_is_not_generalization',lambda r:r[0].update(fixed_train_batch=True))
rejected('truncated_validation_rejected',lambda r:r[0]['after'].update(sequence_length=40))
rejected('gray_constant_cannot_pass',lambda r:r[0]['after']['equal_rollout_mean'].update(clipped_std=0.,correlation=0.))
rejected('weak_seed_cannot_be_hidden_by_mean',lambda r:r[2]['after'].update(degenerate_frame_fraction=.9))
rejected('negative_output_rejected',lambda r:r[0]['after']['equal_rollout_mean'].update(raw_min=-.02,negative_fraction=.2))
rejected('raw_clipped_metric_mismatch_rejected',lambda r:r[0]['after']['equal_rollout_mean'].update(raw_lpips=.01))
rejected('different_initial_state_rejected',lambda r:r[-1].update(initial_state_sha256='wrong'))
rejected('different_labels_or_augmentation_rejected',lambda r:r[-1].update(sample_label_hashes_sha256='wrong'))
rejected('optimizer_counter_mismatch_rejected',lambda r:r[0].update(optimizer_counters=[119]))
rejected('no_initial_improvement_rejected',lambda r:r[0]['before']['equal_rollout_mean'].update(clipped_lpips=.2))
for name in ['train_e2vid_repaired.py','recon_repair_launch_contract.py','run_recon_repair_queue.py']:
    py_compile.compile(str(Path(__file__).parent/name),doraise=True)
(Path(__file__).parent/'repair_gate_preflight.json').write_text(json.dumps(dict(passed=True,checks=checks,fixture_only=True,real_data_used=False),indent=2)+'\n')
print('preflight',len(checks),'passed')
