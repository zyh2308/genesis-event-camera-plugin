# Stage 5 parallel diagnostics — 2026-10-19

This folder contains compact, machine-readable results from the fixed synthetic-only diagnostics. The raw JSON records exact windows, masks, inputs, model/checkpoint metadata, and per-window outputs. `diagnostic_summary.json` is a macro-average view for quick review.

## Flow input perturbation

`flow_input_perturbation_results.json` evaluates the two previously selected synthetic-best checkpoints on one fixed `dt1_000` window from each of eight registered validation rollouts. The input being perturbed is the stored five-bin tensor passed to STE-FlowNet, not the original unbinned event tuple stream. The variants are:

- normal five-bin event tensor;
- the same tensor set to zero before model encoding;
- the five temporal bins reversed independently for each polarity. This keeps total counts and each polarity's spatial support fixed.

Both models use the same labels, valid pixels, and one frozen event-active mask per rollout: the union of pixels active in either generator's normal input. Zeroing an input therefore does not shrink the evaluation area. The union covers 4,122 of 718,212 valid pixels across the eight windows (0.574%).

On the full valid region, the sample-macro mean EPE is 0.1087 for EventGenesis and 0.1060 for V2E, versus a shared zero-flow baseline of 0.1061 px per 30 ms. On the frozen active union, the means are 0.1621 and 0.0913, versus 0.0931 for zero flow. Setting events to zero changes the prediction, and reversing time changes it further for EventGenesis; V2E changes less. Those response changes do not establish that either network decoded the correct motion. The very small active support and the near-zero full-frame outputs limit interpretation.

The eight windows contain 17,916 EventGenesis and 1,037 V2E event counts in the stored tensors. This difference is recorded as an observation; no samples were selected or removed based on it.

## Flow GT-supervised diagnostic

The paired STE-FlowNet diagnostic is configured for 80 optimizer updates per generator. It uses 64 fixed training windows from four training rollouts and one held-out window from each of eight validation rollouts. It applies the pinned model's four-scale robust endpoint loss with conservative common-valid support. The shared physical flow labels supervise optimization, while model inputs remain event-only. This does not change the photometric/smoothness protocol or any official checkpoint selection rule.

The first attempt reached the NPZ adapter and exposed a real schema mismatch: rollout-level `label_sha256` is a scalar, but this pilot indexed every field as if it had a window dimension. `load_sample()` now preserves scalar metadata and indexes only window arrays. `py_compile` and a minimal NPZ fixture check pass. The failed attempt is retained below; the corrected pilot still needs a fresh run before results can be reported.

## Pose mechanism diagnostic

`pose_generic_mechanism_result.json` records a 31-epoch, paired initialization diagnostic on a generic Genesis box, not an E-POSE-matched object. It trained on eight windows from four training rollouts and evaluated four windows from two independent validation rollouts. BatchNorm statistics were recalibrated from the eight synthetic training inputs only, frozen during ordinary `eval()` validation, and included in the checkpoint. Each arm completed 128 confidence-supervised updates; PnP ran on all four validation windows and checkpoint restore hashes matched.

Ordinary-eval mean keypoint EPE remains poor: 34.65 px for EventGenesis and 176.82 px for V2E. PnP returned a solution for 4/4 windows, but mean rotation/translation errors were 55.5 degrees / 0.878 m and 115.5 degrees / 1.154 m respectively. The generic-object result demonstrates that the diagnostic path executes; it does not satisfy the E-POSE object/CAD/data contract or justify formal pose training.

## Failed launch records

The flow GT pilot had three zero-update failed invocations before a corrected rerun:

1. The initial attempt exited 1 after the output directory had been pre-created; no run log, checkpoint, or metrics were produced, so its exact Python exception is unavailable.
2. A retry exited 127 with `python: command not found` because the first wrapper looked for conda beneath the project directory. The actual environment is `/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM`.
3. A retry using the correct environment reached data loading, then exited 1 on the scalar `label_sha256` indexing bug described above.

The logs and exit files are retained under `launch_attempts/` and on g1. A later preflight-only command that pointed to a nonexistent interpreter was rejected before creating a tmux session; it performed no work. The corrected runner calls the actual environment's Python by absolute path, uses a fresh output directory, and records its exit code separately.

All tests here are synthetic. Real-data scoring remains disabled.
