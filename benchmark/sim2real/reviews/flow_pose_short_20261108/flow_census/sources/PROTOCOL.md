# Paired Census short-train proposal — prepared, not deployed or executed

Methods EventGenesis/V2E, seed0, two canonical arms `native_control` and `census`.
Start each arm independently from its own unchanged formal last8910 checkpoint.
Both methods use the original epoch45/46 control sample order and shared affine seeds;
no motion reweighting, synthetic-bank change or generator-rule change. Each study arm
has exactly200 additional optimizer updates (800 total). Fixed last9110 is evaluated;
there is no best-checkpoint search or real scoring.

Phase0 measures exactly the first8 original-control training batches for both generators
without updating parameters. The complete `requires_grad` parameter set defines each
native-photometric and Census gradient norm; unused entries contribute zero. All70
connected finite-gradient tensors are checked separately from total named parameters.
Names/shapes/requires_grad/dtype and their SHA are recorded. For the16 batch/method
ratios r=||g_census||/||g_native_photo||,
freeze one shared lambda=1/median(r), before smoke/study. Zero/nonfinite norms are errors;
GT does not enter lambda. This deliberately replaces the earlier standalone prototype's
unused RMS-of-flow-output calibration example. No value from that example is adopted.

The modified objective is lambda*Census+10*original_smoothness. Census is the declared
new downstream objective, with original four native scale weights/pixel SUM and native
label resizing. It uses predicted support, detached complete-patch support and patch
borders; no GT mask, backward network, SSIM term or scene/sampling changes.

Smoke: four arms × one real update, each independently restored from last8910. Preserve
all70 finite gradients, native LR/Adam/scheduler state, labels/sample/augmentation hashes,
new checkpoint independent model+Adam restore, unchanged old CP/model-source hashes.
Study remains blocked until phase0 and all four actual smoke updates pass.

Study: four arms ×200, evaluate each fixed last point on all528 synthetic validation
windows (equal8-rollout primary, four equal strata, common event-active diagnostics).
Each method's candidate must improve all-physical-valid and common-active-valid EPE;
no stratum's all-valid EPE may worsen by more than5%. Both methods are required.
These gates and current real-test isolation are retained without relaxation.

Default warm parity is independently recomputed over528 before each generator's first
arm, and checked against original final8910 validation. Optional reuse is accepted only
if separately registered actual before/parity files are bound by SHA, source CP SHA,
model state SHA, original data/source recipe identity and full528 inventory; otherwise
recompute. No unbound warm snapshot is silently accepted.

All source modules, original14-file pinned downstream model tree, recipe, bank manifests,
CPs and replay/RNG are hash-bound. New outputs are separate; no old source or result is
overwritten. Helpers are imported unchanged with their historical implementation labels;
new experiment identity, arms, objective, lambda, receipts and result scope are canonical.
There is no automatic long training, adoption or real scoring. Actual phase0/GPU smoke,
study and quality gates are **not yet run** by preparing this bundle.
