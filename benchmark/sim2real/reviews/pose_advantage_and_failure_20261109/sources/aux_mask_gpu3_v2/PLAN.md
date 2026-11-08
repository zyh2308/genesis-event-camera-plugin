# Auxiliary visible-mask center short trial Implementation Plan

> For implementation: execute the registered steps with review checkpoints. No SSH, training, or deployment from this child session.

**Goal:** Test whether synthetic visible-target supervision reduces genuine wrong-target center peaks, with a single training-only head and the existing C1 recipe.

**Architecture:** Preserve native CenterScale forward and all original C1 supervision, teacher, source initialization, sampling, and augmentation draws. Add a separately checkpointed 1x1 mask head on student features; class-balanced BCE has fixed coefficient1. All masks are loss targets only. Frozen C1 depth and rotation run on RAW inputs and predicted center crops for evaluation.

**Tech stack:** Existing PyTorch/OpenCV/NumPy/HDF5 native pipeline; independently pinned e188 policy, E diagnostic context, and S visible-instance reader.

1. Implement balanced fractional-mask BCE, independent auxiliary initialization, shared-affine mask targets, and an equal-budget native center fitter with separate model/auxiliary/Adam independent readback.
2. Prepare RAW NPZ inputs with original y/IDs and RAW relit inputs; construct endpoint mask128 labels using the existing region mapping. Pin all NPZ, event, physical, source, and prior-control bytes.
3. Replay all six full288 tapes before any smoke update; compare to original C1 logs and label hashes. Admission must not consume actual training RNG.
4. Run local meaningful CPU fixtures: balanced loss/empty classes, mask/image affine synchronization and RNG preservation, real micro-updates/frozen size/teacher/Adam rebuild, and contract rejection. Parent performs actual server admission and two-update smoke, then six×288 study only after smoke passes.
5. Evaluate original C1 control parity and candidate full pose with the original crop/ray frame/rotation/scorer, report36 center-stage checks, preserve global30+8 as not retested and real scoring closed. Package source bytes and actual test evidence; parent alone deploys.
