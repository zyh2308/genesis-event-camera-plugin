# Selected reconstruction and flow/pose priority implementation plan

> Execution: existing authorized research chat, three parallel agents, isolated output roots. No additional approval gate is introduced by this plan.

**Goal:** Complete the repaired reconstruction comparison on three explicitly selected real sequences while prioritizing synthetic flow diagnostics and pose stability repair.

**Architecture:** Reuse the frozen simulator and physical/event banks. Keep original formal flow evaluation untouched. Schedule repaired reconstruction on free g3 devices; reuse the research chat's existing g1 GPU3/4 queue leases for flow/pose.

**Tech Stack:** Existing PyTorch 2.8/CUDA 12.6 environment, Python subprocess workers, atomic UUID leases, immutable JSON manifests, existing training/evaluation adapters.

## Scope and scientific interpretation

- User authorized complete reconstruction training now, superseding the earlier short-training-only restriction for this selected comparison.
- Selected real HQF sequences: `boxes`, `reflective_materials`, `still_life`. Selection was made after observing repaired seed-0 results. Report as a post-hoc selected development subset, not the complete HQF benchmark or a new blind test.
- Synthetic training/validation trajectories stay unchanged. No real events, frames or labels enter training, loss selection or checkpoint selection. Do not move selected real trajectories into the synthetic training bank.
- Reconstruction uses the common stable sigmoid output and original LPIPS-VGG=1, TC=2 recipe. L1 candidates failed the joint gate and are not adopted. All five methods, seeds 0/1/2, complete the original 12,000 optimizer-update / 200-epoch budget; checkpoints selected only on the registered synthetic validation.
- Preserve all historical 14-sequence scores, failed runs and diagnoses. Do not combine new repaired EventGenesis weights with old linear-output baselines in a five-method table.
- Frozen simulator core: `62da4f35ce3c5c96f04268bd1687a27657af9238`. Noise, occlusion, timestamps and physical labels remain unchanged.

## Task 1 — Resource verification and allocation

1. Reuse the healthy g1 SSH RPC; create a persistent g3 RPC without changing SSH configuration.
2. Verify both actual compute processes and UUID lease records. g1 GPU0/1 and g3 GPU0/1/3/4 belong to other work and are excluded.
3. Preserve g1 GPU2/5 fixed MVSEC evaluation. Existing own g1 GPU3/4 queue parents have no pending work and can accept bounded flow/pose commands.
4. Initially free g3 devices are GPU2 `GPU-b89ca413-eb9c-61a5-a89b-9c244d032fff`, GPU5 `GPU-dd93b58d-df3d-9d33-bffc-2203bb503e87`, GPU6 `GPU-48385047-4365-b814-dac9-7d0d2c63d9b0`. Recheck availability atomically when acquiring leases; never evict another job.
5. Use active queues and bounded preparation reservations, without dummy GPU computation. Release on completion or verified failure cleanup.

## Task 2 — Reconstruction complete-budget jobs

Owner: reconstruction agent; local staging `outputs/recon_selected_full_20261108/`.

1. Bind unchanged repaired trainer, recipe, event banks and data manifests; independently check data/model/optimizer/RNG recovery.
2. Resume two original seed-0 runs from their safety checkpoints (5264 and 5306 updates), preserving the previous synthetic best at update4800. Start the remaining 13 formal jobs from registered paired seed initialization.
3. Reuse exact matching completed 300-step stability evidence where valid. Complete missing baseline/seed stability checks; a shorter previously failed gate is not substituted. Only after the complete common gate may full training proceed.
4. Assign each seed's five methods to one g3 device, preserving same-device paired comparisons. Independent fresh output directories; source hashes and actual optimizer increments logged.
5. Require all 15 runs to reach 12,000 updates. Then evaluate synthetic-selected checkpoints on every scored frame of the selected three real sequences using one fixed adapter and metrics.
6. Produce per-seed/per-sequence tables and seed mean/sampleSD, plus whole-subset aggregate. Keep the selected-subset qualifier attached to tables and captions.

## Task 3 — Flow diagnosis and optimization

Owner: flow agent; local staging `outputs/flow_priority_20261108/`; g1 GPU3.

1. Keep all 15 completed formal training runs and ongoing four-sequence MVSEC matrix frozen.
2. Diagnose fixed synthetic validation errors by motion magnitude, event support and visibility, and check training loss/gradient behavior.
3. GT flow may diagnose and evaluate; do not silently replace the self-supervised optimizer objective with GT supervision.
4. Register one bounded paired intervention only if the diagnostic supports it. Equal initialization/budget/sampling across event generators; isolated outputs. No real-score-driven model changes.

## Task 4 — Pose stability repair

Owner: pose agent; local staging `outputs/pose_priority_20261108/`; g1 GPU4.

1. Preserve completed physical-light trials and their 14/30 failed adoption result.
2. Use equal-budget paired synthetic short training to distinguish light-sampling imbalance from the extra consistency loss. Protect normal/half-light depth and success while improving dark localization/depth.
3. Do not relax the original per-seed stability gates or import real E-POSE labels into optimization. Real scoring stays closed for a new candidate until its full synthetic gate passes.
4. Record failed branches, original teacher/control checkpoints and all paired sample hashes.

## Task 5 — Verification, evidence and handoff

1. Run meaningful offline rejection and CPU formula/recovery tests before deployment; then bounded device smoke checks before long work.
2. Verify source/checkpoint SHA chains, identity, registered scope and resource ownership. Check actual progress rather than equating launch or exit0 with complete scientific results.
3. Save reports/scripts/receipts in the shared WSL research repository; large weights and raw event banks remain on the servers.
4. Report completed/running/failed/blocked states separately, with concrete budgets and test scope. Publish only complete, correctly scoped aggregates.
