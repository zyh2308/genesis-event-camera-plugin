# Single-factor slow-feature adaptation proposal

This package is prepared only. No deployment, calibration, GPU smoke, study, real score
or new experimental result is claimed. Deployment is held until the separate raw-to-cached
input / all96-window contract / count_scale8 saturation investigation is resolved.

The same C1 physical supervised loss and normal-view teacher weight2 remain unchanged.
There is no feature KD, no half-view teacher KD and no feature freezing. Only the AdamW
learning rate of trainable `features.*` parameters changes from1.5e-4 to1.5e-5;
trainable heads retain1.5e-4 and all weight-decay settings remain. The center branch's
original size-head freezing remains. Candidate features stay in their original training
mode, including any buffers; this is a different tradeoff from hard-frozen features.

The policy is derived locally from the already audited frozen-feature fitting wrapper,
removing its hard freeze and replacing only candidate optimizer construction. Inherited
supervised/teacher loss expressions, sampling, augmentation, original initialization,
batch32, six epochs/288 updates,3:1 auxiliary half/eighth ratio, rotation and full native
evaluation are retained. The 0.1 factor is fixed in advance, not chosen from real/validation
scores or a new coefficient calibration.

Canonical candidate is `slow_features`. The native evaluator's third
`paired_consistency` slot remains an explicitly documented legacy interface. It does not
mean half-teacher KD is active. The new policy, worker and receipts name the candidate
canonically; result mappings expose the legacy slot without changing frozen metrics.

Study reuses the same immutable24 C0/C1 branches and trains only12 candidate branches
(two methods × three seeds × depth/center), each288 updates. Their labels/affine draws
must exactly reproduce C1. Original controls are checked by source/hash/optimizer/replay
and compared numerically again after the unchanged evaluator. Original30 illumination
and8 confirmation checks are kept; nothing is relaxed after feature-retention failure.

The mature motivation is conserving an already useful representation during adaptation:
[Less-forgetting Learning](https://arxiv.org/pdf/1607.00122) describes the representation
drift issue, and [L2-SP](https://proceedings.mlr.press/v80/li18a.html) motivates retaining
the initial learned solution. Neither paper establishes this exact0.1 LR ratio as optimal
for event pose regression. This is a conservative separately testable optimization
choice, not a claimed paper-derived guarantee or a simulator change.
