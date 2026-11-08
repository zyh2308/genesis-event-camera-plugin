# Fair frontend selection, blind catching, and cost curves Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Execution note: this Codex chat implements the already user-authorized plan directly with the available execution workflow; no new approval gate is required.

**Goal:** Test whether the simulator advantage survives an equally tuned V2E frontend, a wider unseen throwing distribution, and measured generation cost.

**Architecture:** Reuse the current Catch It! Genesis runner, native event workers, original policy, existing raw caches, scoring, and atomic GPU launcher. Make only benchmark frontend parameters configurable, freeze both selected frontends before blind outcomes, and store every actual episode and failure under a new run ID. Old artifacts remain immutable.

**Tech Stack:** Python 3.11, Genesis 1.2.2, NumPy/SciPy, original PyTorch PPO checkpoint, frozen sparse event core, native V2E/SuperSloMo39, existing SSH relays and resource launcher.

---

## Fixed experimental contracts

- User authorization: execute frontend crossover and equal-budget tuning, lock configurations, wider new-seed blind testing, effect/cost curves, and demos. Ask only for consequential directional uncertainty.
- Camera remains 7 degrees; resolution640x480; known sphere radius0.04m; asset seed1702. Original policy weights, actions,25Hz control and frozen17-file event core remain unchanged.
- Shared calibrated contact profile remains hand-only friction1.5 and two physics substeps, with2ms outer step. No outcome-dependent physics changes.
- One actual rendered RGB-D initialization; subsequent control reads events and robot proprioception. No target GT, masks, crops or release labels reach the frontend/controller. GT is permitted only for scoring and the separately labeled noisy-state reference.
- Original parent availability is retained. Short historical windows are not early sensor arrivals or100Hz control.
- Native V2E is verified against the pinned upstream/checkpoint; source rates require corresponding1ms minimum-upsampling receipts.
- A catch requires task time completion, no failure termination, real hand contact, and real contact in the final40ms control step. This is not the former scene's continuous0.5s hold criterion. Supplement contact/hold summaries from traces without silently changing primary scoring.

## Task1: Seal lineage and prepare independent workspace

**Files:** new branch`exp/robot-catch-fair-frontend-20261108`, based on`d8802bf`; original branch and server snapshots read-only.

1. Read current AGENTS and actual state; user explicitly restored g3 earlier, overriding its stale ban.
2. Reuse Windows OpenSSH with trusted keys and new queues; inspect actual GPU/process/reservation, CPU/RAM/disk and immutable input paths before launch.
3. Reuse existing environments/data/models; do not install shared dependencies or clone datasets.

## Task2: Symmetric frontend candidate grid

**Modify:** `benchmark/mature_catch/event_sphere_tracker_rgbd_base.py`, `run_original.py`.
**Test:** `benchmark/mature_catch/test_event_confirmation_gap.py`; actual-cache default parity.

1. Expose motion association/confirmation gap50 or100ms, default50. Only the preflight association and two-observation confirmation share this parameter; leave post-contact hold logic unchanged.
2. The same eight candidates are available to both event sources: observation window0/10ms, causal flight fit0/1, confirmation gap50/100ms.
3. Record all three parameters in actual actor protocols; validate allowed values. No changes to threshold, pixel response, generator source, original weights or camera.
4. CPU checks must preserve default behavior and cover80ms-separated legacy observations. Candidate cache scoring runs on servers.

## Task3: Equal-budget cache screening and physical crossover

**Create:** independent replay/selection tools in the benchmark or current run folder. Reuse actual development caches for seeds31–46 only; these are all development now.

1. Use all eligible pre-contact polls, not only accepted/easy queries. Preserve original packet availability, event order and camera-history support.
2. Default native outputs must match the pre-attachment archived prefix; mark capture suspension and flight-only counterfactual limitations explicitly. Report full-window and exact-prefix diagnostics separately.
3. Predetermined screening score: episode mean of0.5*min(position_error/0.5m,1)+0.5*min(velocity_error/2m/s,1), with missing/nonmoving/unusable states penalized1, plus the pre-release false-moving query fraction. Average duplicate variants within seed/rate, then seeds and rates equally. GT never influences candidate inference.
4. For each source, choose one best candidate in each window family. Use deterministic score, coverage and minimal departure from default tie breaks. Take the union of both sources' shortlisted configurations, plus canonical10ms/fit1/gap50 if absent: at most five frontend configurations.
5. Cross every union configuration with both sources on the same four development seeds2026110841–44 and both25/12.5Hz rates. Each source gets the exact same physical validation budget. All actual outcomes are retained. Verified prior runs may be reused only after identical configuration/source/default-parity checks, and are labeled reused.
6. Choose one global frontend per source using equally weighted strict catch success across the two rates; deterministic predefined ties use availability/error and smaller default departure. Do not choose a different frontend per blind seed or source rate.
7. Save selected settings, selection inputs/results, hashes, and lock time before any blind outcome. Native/common frontend tables remain alongside the adapted comparison.

## Task4: Wider blind bank

**Create:** source-derived throwing bank and immutable registered protocol.

1. Thirty-two new seeds1371082001–1371082032; no reuse of1702 or prior20261108xx development episodes.
2. Reuse the original throwing sampler's marginal distributions with predefined spatial/height/speed strata. No narrow two-lane trajectory construction. Save generator code, upstream source SHA, actual position/velocity draws and bank SHA before blind execution.
3. Release time, gravity and other native randomization come from actual reset; do not fabricate them in the bank. Every unreachable, initialization failure, collision, drop and successful episode remains counted.
4. Execute both selected source/frontends at100,50,25,12.5Hz for every block:256 event episodes. Each block also has one initialization-only and one noisy-GT reference, reused explicitly across source rates:64 reference episodes,320 actual episodes total before any justified infrastructure recovery.
5. The independent sample unit is32 throwing blocks, not320 independent trials. FPS variants and reference reuse are paired repeated measures.
6. No parameter changes or seed exclusion after blind outcomes. Infrastructure failures retain receipts and recover into separate recorded outputs. If the task/input contract is fundamentally invalid, stop dependent work and report the concrete directional issue.

## Task5: Effect and matched-trajectory cost

**Create:** cost worker/wrapper using existing fixed archived trajectories and event workers.

1. Select fixed development trajectory/time intervals before cost results; identical physical states and length across methods/rates. Do not compare raw episode wall times when one method terminates earlier.
2. Measure source acquisition, required HDR/depth/segmentation/state costs for Ours, required LDR for V2E, generator/transfer, common frontend, and total measured replay pipeline wall. Audit-only extra capture is accounted separately, not charged as algorithm-required cost.
3. Rates100/50/25/12.5Hz; warm-up and repeated matched clips with explicit environment/hardware. Include all source-specific inputs. Keep measured wall separate from any additive/shared-physics estimate; no unmeasured GPU-time or real-time claim.
4. Plot catch success with uncertainty versus source rate and measured cost per simulated second. Show same-frame-budget results and actual cost tradeoffs; do not equate equal frame count with equal compute.

## Task6: Audit, visualization and delivery

1. Validate all planned cases, source/core/policy hashes, raw arrays, causal packet/windows and score definitions. Independent SHA verification of full server manifests and portable downloads.
2. Produce per-episode CSVs, full frontend crossover/selection, blind stratum and source-rate tables, effect/cost figures, and representative success/failure videos and looping GIFs. Retain V2E-favorable outcomes and prior negative results.
3. Archive small reviewable results/code/demos in Git; large raw data stay at shared immutable server paths with hashes. Do not rewrite old protocols to hide inherited metadata errors: new protocols contain correct sample units.
4. Copy the final delivery to a new Windows Desktop folder; commit/push this branch, verify clean state and all own GPU reservations released. No idle GPU holder.

## Completion criteria

The task is complete when the equal-budget selection and physical crossover have been audited, selected configurations are locked before the32-block blind runs, all scheduled blind outcomes and matched costs are retained/analyzed, and reports/figures/demos are delivered. An Ours win is not a completion requirement; unsupported superiority must not be inferred or manufactured.
