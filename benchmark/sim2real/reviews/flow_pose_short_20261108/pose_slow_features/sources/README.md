# Prepared backup: slow-feature learning rate only

**Not deployed or run.** CPU fixtures use tiny artificial models only. The reported
feature-retention study's22/30 failures motivate keeping this option ready; they do not
establish that the0.1 LR factor will improve pose. The separate raw→cached representation,
all96-window contract and count_scale8 saturation investigation must be resolved before
the root decides whether to deploy this package. If that investigation requires changing
the input contract, old controls cannot be silently reused across incompatible inputs.

Candidate `slow_features` uses the unchanged C1 physical supervised objective plus
normal teacher weight2, no half-view teacher and no feature KD. Only trainable
`features.*` AdamW LR changes to1.5e-5; heads stay1.5e-4, weight decay.01. The original
center-branch size-head freeze, feature training mode, model, initialization,3:1 auxiliary
half/eighth draw, normal replay, augmentation, batch32,6 epochs/288 updates and rotation
remain. There is no coefficient search or phase0 calibration.

Study reuses24 immutable C0/C1 branches and runs only12 new branches:2 methods×3 seeds×
depth/center, each288 updates. Source hashes, complete labels/affine replay, source states,
optimizer recovery and control metrics are checked again. The native evaluator and
original30 illumination+8 confirmation checks are preserved without relaxing controls.

The worker and receipts use the canonical name. The native evaluator's historical third
`paired_consistency` slot and its model output filenames are retained as an explicitly
declared interface mapping to `slow_features`; that slot does **not** enable half KD.
New fitting logs/checkpoints use canonical `slow_features` tags.

CPU checks cover exhaustive/disjoint optimizer grouping, frozen parameters excluded,
both features and heads actually updating at different rates, two-group AdamW independent
save/restore plus a genuine next update, label/augmentation identity against C1, unchanged
immutable control files, rejection of degraded control metrics and incomplete/fake smoke.
The tiny CPU model updates are unit fixtures, not STE/pose experiment training or scores.

Future commands, only after input-contract review and an external safe resource decision:

```
python -B sources/run_pose_slow_features.py --campaign CAMPAIGN \
  --prior-source-dir ORIGINAL_SOURCES --prior-source-manifest-sha256 ORIGINAL_SHA \
  --radiance-train RADIANCE_TRAIN --radiance-validation RADIANCE_VALIDATION \
  --prior-study FIXED_C0_C1_STUDY --diagnostic FIXED_DIAGNOSTIC \
  --out NEW_ROOT/smoke --phase smoke
python -B sources/run_pose_slow_features.py ... --out NEW_ROOT/study --phase study \
  --smoke-receipt NEW_ROOT/smoke/slow_features_execution_receipt.json \
  --smoke-receipt-sha256 ACTUAL_SHA
```

Smoke has12 fresh one-update branches (seed0; all three legacy slots) so the actual
pipeline/restore gate can be checked. Study accepts only a hash-bound matching smoke
receipt, and then uses24 old controls rather than retraining them. No queue, lease,
automatic real test or long training is created by this package.

`derive_policy.py` / `derive_worker.py` record a surgical derivation from the previously
audited frozen-feature wrapper; old sources are never modified. See DERIVATION.md for
the primary-source motivation and the limits of extrapolating classification/domain-shift
work to event-camera pose regression.0.1 is a fixed conservative test value, not a claimed
optimal literature prescription.
