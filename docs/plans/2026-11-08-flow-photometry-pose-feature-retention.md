# Flow photometry and pose feature retention implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether native flow photometric gradients reward wrong motion, and test one soft feature-retention pose candidate that can preserve half-light performance while adapting to dark light, with paired synthetic-only experiments.

**Architecture:** All work uses independent, immutable source snapshots and new output directories. Existing reconstruction lanes on g3 and the original frozen MVSEC matrix continue unchanged. g1's already owned GPU3/GPU4 queues execute bounded jobs after their current studies finish; failed dependencies block the next phase. No new real scoring is authorized by these workers.

**Tech Stack:** Pinned STE-FlowNet and native CenterScale downstream code, PyTorch 2.8, NumPy/OpenCV, Python unittest, existing UUID-locked g1 queues, SHA256 evidence indices, WSL review repository.

---

## Constraints and evidence before implementation

- Frozen simulator core: `62da4f35ce3c5c96f04268bd1687a27657af9238`; generator/noise/occlusion/physical labels/data split stay fixed.
- No true flow, object crop/mask, pose, or geometry is supplied as privileged model input. Flow GT is used only by diagnostic oracles, not as a new optimizer target or training mask.
- No new real results select loss, parameters, checkpoint, or scene. Already viewed E-POSE sequences remain development evidence.
- Same architecture, initialization lineage, physical sample/augmentation sequence, training budget, metrics and candidate for EventGenesis/V2E. Do not change to a favorable seed or final checkpoint after seeing scores.
- The preceding flow sampling study is not adopted: EventGenesis's fixed 200-update motion arm is worse on complete 528-window overall/active validation. Collect the remaining paired results without altering it.
- The complete pose hard-freeze study exits 0 but passes only 19/30 illumination checks. Its failure and all old results remain intact; original eight confirmations do not substitute for those thirty checks.

## Task 1: Collect the preceding immutable results

**Read:** server `20261108-flow-motion-sampling-paired-v1/study`, `20261108-pose-frozen-features-v1/study`, and their own queue done/live receipts.

1. Read concise completion/gate/phase status using the existing SSH RPC. Preserve raw result SHA and receipt SHA; never overwrite live or old artifacts.
2. Recheck g3 reconstruction processes, latest optimizer updates, and absence of GLOBAL_FAILURE without stopping or resetting them.
3. Archive compact previous results and exact machine evidence in a new WSL `benchmark/sim2real/reviews/flow_pose_short_20261108/` review.

## Task 2: Native flow gradient diagnostic, zero optimizer updates

**Create:** `outputs/flow_photometric_alignment_20261108/` source bundle, protocol, meaningful CPU tests and read-only worker. Copy the final locked bundle into the review.

1. Pin the exact original trainer, original `multiscaleloss.py`, warp, compatibility patch, complete 8910-update EG/V2E checkpoints and synthetic bank manifests. Native photo uses next(x+flow) to reconstruct previous image, OpenCV linear grayscale resizing, full-pixel Charbonnier sum and native four-scale aggregation; events are not an implicit training mask.
2. Verify signs, scale units, boundary behavior and native loss parity with ideal known translations. Add separately controlled occlusion and illumination changes with fixed geometry to distinguish implementation errors from objective assumptions.
3. Use the pre-existing fixed 8 training samples and 32 independent validation windows, balanced across four registered strata. At each native scale report zero/predicted/GT residuals, photo-gradient alignment with the direction that reduces true motion error, and centered finite differences toward/away from GT. Report physical-valid/common-event-active/texture/motion support explicitly; GT geometry masks are diagnostic only.
4. CPU checks must fail wrong warp sign/unit, empty support, or a changed native source. GPU admission verifies original loss parity before interpreting gradient signs. Compare full model state before/after; require zero optimizer updates and no source/checkpoint mutation.
5. Run on GPU3 after the prior sampling study exits successfully, with a bounded timeout and unique output. A scientific negative result is preserved as a completed diagnostic. Do not auto-start an altered loss trial until evidence identifies a justified, prediction-only change.

## Task 3: One pose soft feature-retention candidate

**Create:** `outputs/pose_feature_retention_20261108/` protocol, phase0 calibration, smoke/study worker and meaningful CPU tests. Reuse pinned native evaluation and unchanged 30+8 checks.

1. Considered alternatives: soft same-view source-feature retention; smaller feature learning rate; parameter-space source penalty. Select only soft feature retention, as it targets the observed representation drift while keeping dark feature adaptation possible.
2. Preserve original C1 physical supervision, normal teacher weight 2, 3:1 half/eighth sampling, six epochs, batch32, 288 updates per branch and original source teacher. Add normalized source-feature retention on identical normal and half inputs. No feature retention or bright-to-dark teacher target on the darkest view; features remain trainable.
3. Phase0 uses fixed TRAIN batches and the immutable already-completed C1 checkpoints to measure nonzero main/retention feature-gradient norms. Fix one scalar coefficient for both generators/all seeds/branches using the prespecified median ratio targeting 10% of main gradient norm. Record exact formula, all twelve observations, hashes and coefficient; no validation or real scores are read. Reject nonfinite/zero normalization, changed source or inconsistent pairing.
4. Smoke performs genuine isolated updates, independent model/Adam reconstruction, nonzero feature and head changes, frozen teacher and exact sample/augmentation-label pairing. CPU tests use same-package fixtures or explicit source paths, never workstation-specific sibling assumptions.
5. Study reuses the exact 24 old C0/C1 control branches and rechecks source weights, history, optimizer, metrics and replay trajectories. Only 12 new candidate branches x288 updates are trained. Fixed final step, no checkpoint sweep or loss-coefficient search.
6. Reevaluate all original conditions. Compare every seed to the immutable strong C0 baseline with the original thirty illumination checks plus eight confirmations. Any failure prevents adoption/new real evaluation. Averages do not mask a failed seed.

## Task 4: Review, publish and finish safe deployment

1. Independently review the proposed loss and feature retention against primary papers/author code. Explain deviations; do not claim the exact published method if only borrowing a mechanism.
2. Run appropriate CPU checks on the actual server. Preserve initial failures. Publish only verified ready files, source SHAs and phase dependencies.
3. Queue phase0 -> smoke -> pose study and the flow read-only diagnostic sequentially on their owned lanes. Bound own subprocess groups and never signal unrelated processes. Verify actual optimizer/forward evidence before claiming launch.
4. Hash all archived sources and evidence, check Git stages preserve locked bytes, commit and push to `exp/public-sim2real-v1`; verify remote HEAD and clean worktree. Report concrete results and unresolved limitations in plain Chinese. Long jobs may continue unattended once safe deployment is verified.
