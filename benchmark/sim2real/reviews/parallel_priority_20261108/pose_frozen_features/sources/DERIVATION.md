# Minimal change audit

`pose_frozen_features_policy.py` is derived by `build_policy.py` from the SHA-locked prior policy e1883b68…8d38c. No original file was edited. Helpers, RNG/affine sequence, objectives and update loop are retained. Changes are:

- Constructor accepts immutable prior controls. C0/C1 invoke independent reuse/replay instead of optimizer updates.
- Only the native candidate slot sets half teacher weight from the legacy argument 2 to **actual zero**, matching C1's loss.
- Only that candidate freezes `model.features` before building AdamW, sets its eval mode, and repeats freeze configuration for independent optimizer reconstruction.
- Candidate feature weights and buffers are checked bitwise unchanged. Receipt exposes canonical arm and actual loss weight. Candidate labels/draws must exactly equal the existing C1's corresponding log.

`run_pose_frozen_features.py` reuses the exact original native `main` and its data, complete validation, rotation, metrics and 30+8 gate implementation. A write wrapper augments protocol metadata and verifies immutable controls; it does not replace metric or gate functions. The original main AST hash is recorded. Legacy output key is transparently mapped to canonical `frozen_features` in the protocol and independent execution receipt.

`pose_priority_policy.py` and `test_pose_priority_policy.py` are unchanged local source/CPU fixture references. No deployed campaign code, old output, or gate threshold was modified. New sources are supplementary standalone files, not additions to old running manifests.
