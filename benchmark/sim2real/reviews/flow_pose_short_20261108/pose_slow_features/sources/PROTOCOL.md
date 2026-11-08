# Fixed single-factor backup protocol — no execution yet

Candidate: slow_features. Same original C1 physical loss+normal teacher2. Half teacher0,
feature KD0; no feature freeze. Fixed feature/head LR1.5e-5/1.5e-4, AdamW weight_decay.01.
All trainable named parameters must belong exactly once to the correct group; original
frozen localizer size parameters must belong to neither group. Both candidate groups must
actually update and independently restore their optimizer states and learning rates.

All source initialization,2 methods,3 seeds,3:1 auxiliary half/eighth draws, batch32,
6 epochs/288 updates per branch, augmentation, model, rotation and real isolation remain.
Study reuses the same hash-locked24 C0/C1 branches, trains12 new depth/localizer branches,
and requires identical C1 labels and affine draw trajectories. No new coefficient or
learning-rate sweep, no event re-encoding, no new dataset/subset and no real adaptation.

The legacy paired_consistency evaluation slot is mapped to canonical slow_features only
for the unchanged native evaluator. The actual half-teacher and feature-KD weights are
zero in all receipts. The whole native main/evaluation/gate AST remains externally pinned.

Same unchanged full original30 illumination and8 confirmation checks. Their values are
computed against immutable original controls, not weakened or replaced controls. Failed
checks remain failures; no new threshold is inferred from this candidate's result.

Deployment is not authorized by this preparation. First resolve the separate cached-input
contract probe and saturation diagnosis. Then an actual one-update GPU smoke is required
before study. This package creates no resource reservation or job. Real scoring remains
closed until original stability gates and research protocol permit it.
