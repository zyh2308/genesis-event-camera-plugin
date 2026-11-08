# Visual-speed Catch It! Paired Evaluation Implementation Plan

> Execute the approved plan in this chat. The user has already authorized implementation and server experiments; request clarification only for a directional issue.

**Goal:** Make fast-motion difficulty observable in the actual control camera, then compare frozen EventGenesis, V2E and one-frame-only initialization under the same original catching policy.

**Architecture:** Retain the 7-degree camera offset, pretrained Catch It! checkpoint, estimator code, physics/scorer and frozen sparse EventGenesis core. Add task-only launch configuration and a causal streaming V2E input adapter. Pilot selection uses only state-reference feasibility and measured image motion, before looking at Ours/V2E performance.

**Tech Stack:** Genesis 1.2.2 CPU physics and EGL rendering on the server, the existing Torch 2.8 CUDA worker, archived SuperSloMo39 and V2E source, original policy at 25 Hz, 100 Hz camera endpoints, the original 10 ms simulated input availability delay.

## 1. Record and isolate

- Branch `exp/robot-catch-fast-paired-20261108`, based on `df51669`.
- Server root `/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/visual_fast_paired_20261108a`.
- Preserve every pilot and evaluation attempt, including initialization/infrastructure failures.
- Resource checks and atomic GPU reservations precede actual rendering or CUDA work. Reuse existing source/data/environments read-only. Previous g3 restriction has been explicitly revoked by the user; prefer currently accessible g1.

## 2. Pilot conditions without event-method selection

Create a task runtime snapshot from the last immutable source. The runner still calls the native random sampler first, then applies an explicit deterministic position/velocity override; this preserves the subsequent per-seed gravity, PID, release-time and noise draws. No launch parameters are passed to the estimator.

Pilot seeds are known development seeds 1702/1703. Compare ordinary and transverse variants at the same start position/height and vertical velocity. Start with lateral speeds 0.4, 1.5 and 2.2 m/s at y=1.9m, plus a nearer y=1.65m pair if needed. Keep sphere radius 0.04m and asset seed1702. Directions are mirrored across the two seeds.

The state-reference policy reads native noisy GT target observations; it is a feasibility reference, not an optimal oracle. Record 100 Hz scoring-only camera projections and visible target area. Measure both precontact image-speed distribution and a fixed early-flight window so earlier termination cannot artificially create a speed gain. Require a clear image-motion increase (target at least 1.5x median, or a documented substantially shorter observation window) while retaining useful GT catching feasibility and initial visibility. If no tested condition satisfies these together, report the directional tradeoff before changing robot, weights or camera.

## 3. Causal V2E adapter

Reuse archived official SuperSloMo39 and EventEmulator. Pin full source/checkpoint hashes. Use actual camera LDR grayscale for V2E and the original scene-linear HDR/geometry path for Ours; record native input differences. Match contrast threshold0.2 and ideal-noise settings for a controlled task comparison, explicitly labelled as such. Do not call EventEmulator alone and name that V2E.

The streaming implementation must preserve network preprocessing, interpolation equations, output quantization and timing conventions. Verify interpolation against the archived official file API on a short frame clip; verify event coordinates, polarity and timestamps through the same emulator sequence. Every interval is released only after its final endpoint exists, plus the common10ms simulated delay. The V2E worker never receives target pose/depth/segmentation as perception input. Any audit-only extra HDR capture is separately recorded.

## 4. Stop-event-update ablation

Use the existing `--initialization-only` path: one actual RGB-D reset frame, then no event packets, with the unchanged estimator and policy. Keep normal target physics and all failure rules. This must not freeze/teleport the ball, provide its launch velocity, or replace missing observations with GT.

## 5. Freeze a paired evaluation list

After pilot selection and adapter smoke, save exact condition and a new seed list before evaluation. Run ordinary/visual-fast/backlit as feasible, with Ours, V2E, initialization-only and same-seed state-reference. Start with six new seed blocks for bounded development evaluation; expand a separate untouched formal set only after the comparison is interpretable. Use the same RGB-D initializer, fixed estimator, policy weights, input-delay convention and original physical success criterion. Never exclude state-reference failures from the primary denominator.

## 6. Validate and deliver

- Verify before/after hashes for the frozen core and checkpoint and byte-identical estimator files.
- Pair initial states exactly; audit event availability and absence of GT actor inputs.
- Report complete catching and actual hand contact, all precontact localization/range/velocity errors and missing/fresh state coverage, measured image speed and visibility, and wall costs separately from simulated latency.
- Preserve all native videos and traces. Generate a full success/failure reel, same-seed method comparison and synchronized scene/hand/event demos.
- Hash complete server artifacts, retrieve a compact package, inspect rendered figures/videos and commit reports/code/portable evidence. No new training or core modification is authorized by this plan.
