# Pose factor and illumination conflict diagnostic implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Identify which stage causes the normal/half/dark tradeoff, then run one evidence-targeted paired short study using the same downstream recipe for both event generators.

**Architecture:** First reuse immutable C0/C1/slow-feature completed synthetic results to decompose depth, independent center, and predicted-crop rotation effects. Then one frozen forward export of aligned synthetic normal/half/eighth windows supports CPU geometric factor substitutions and train-only branch/loss-gradient conflict probes. A new training treatment is registered only after this diagnostic; no hyperparameter sweep or real-score selection.

**Tech Stack:** Pinned Python3.11/PyTorch2.8, native CenterScale/rotation/crop/score, NumPy/SciPy, existing g1 GPU4 queue; immutable source bundles and SHA-indexed receipts.

## Constraints

- Frozen simulator core62da4f35ce3c5c96f04268bd1687a27657af9238 and physical bank/noise/occlusion unchanged.
- Protect g3 reconstruction training and g1 GPU3 Census / GPU2+5 MVSEC tasks. Use only owned GPU4 after157 exit0 with exact UUID; no competing task termination.
- New real E-POSE scoring remains disabled until synthetic quality/stability passes. Seen real tests never select this candidate.
- Preserve original30 illumination and8 confirmation checks. Report negative outcomes and all three seeds/methods. No per-light oracle switching in deployable predictions; GT substitutions are diagnostic only.
- Physical object anchor is keys[0]; anchor z is (R_gt@anchor+t_gt)[2], never origin_t_z unless equal. Center is normalized by346x260. Recompute translation from uv/z/R when substituting a factor.

## Task1: Reuse completed synthetic evidence, no inference or updates

Read slow_features final receipt+completion and six cell JSONs. Verify sources, controls, 12x288 updates, source/model hash links, gate18/30 and original8 pass. Extract macro-rollout per-light center/depth/log-size metrics and three native evaluation stages (depth_only, translation_only, full) across C0/C1/slow. Difference each stage against its C0 control; the telescoping differences are an ordered decomposition, not a unique causal/Shapley attribution. Record existing normal benchmark and relit sets use different trajectory/sample coverage; causal illumination comparisons need matched geometry and times.

## Task2: CPU factor attribution helper and meaningful tests

Create outputs/pose_factor_diagnostic_20261108/factor_geometry.py and test_factor_geometry.py. Validate camera rays, anchor offset, rotation validity, normalized coordinate units, all-GT zero-error ceiling and identity substitutions. Exported per-window predictions feed the helper; no hidden GT crop/box/mask in any deployable arm. Use native ADD-S threshold0.1*diameter and rollout-equal aggregation; ordinary rotation geodesic is label-diagnostic only because symmetry may apply.

## Task3: Bounded frozen forward and training-gradient diagnostic

Create a standalone pinned worker and contract. Load original C0/C1/slow center+depth branches and frozen rotation. Use all4 independent relit validation rollouts, exact same normal/half/eighth physical windows, all3seeds and both generators; retain old clean384/full validation parity as separate checks. Save uv,z,size,R,GT/K/anchor/sampleIDs and source hashes; no optimizer step. Include predicted native outputs, GT u/z/R single/factorial ceilings, and prediction-only C0/C1 factor crosses. Separately rerun rotation with GT/predicted center/size solely as crop diagnostics. Do not present oracle results as model scores.

Probe fixed TRAIN matched windows for each branch: normal/half/eighth physical supervision, normal teacher and log-size vs shape/depth components. Report full/group gradient norms/cosines and actual weighted mixture; distinguish local gradients from Adam steps. No validation/real data may set loss weights.

Admission: CPU helper tests, pinned-source/manifest/core assertions, owned GPU UUID/process admission, nonempty aligned samples/labels/units, bounded timer, zero updates, before/after model SHA, and native-result parity. Preserve failed attempt evidence; correct diagnostic code in a new version/root.

## Task4: Register one minimal treatment from evidence, then smoke and short study

Choose among retaining a reliable center branch while changing only depth, correcting a demonstrated inconsistent target/loss, or a shared input/loss robustness treatment when density/scale probes support it. Do not launch another LR/teacher-weight guess. Freeze the selected shared change, source/control lineage, 288-update budget (or justify a matched budget), sample/augmentation replay, and both-method/three-seed evaluation before training. Run one-update paired restoration smoke first, then bounded short training on GPU4. Full30+8 and predefined per-condition EG-vs-V2E gaps remain visible; no automatic long training or real scoring if checks fail.

## Task5: Report other lines on substantive change

Read Census final completion+155 done, all4x200 and528-window quality results when complete; scientific gate failure is not a process failure. Read g3 current jobs and MVSEC completed cell counts without new tuning from real scores. Publish immutable code/evidence and concise Chinese explanation of causes, tested remedy, limits and remaining work.
