# R1.5 shadow causal replay: episodes 004/005

## What this run tests

This is a cached-state, render-only shadow ablation. It restores the saved Panda/object states from episodes 004 and 005 and renders each exact state twice with shadows enabled/disabled. Physics is not advanced. MSAA, camera, HDR path, ideal DVS response, five causal bins, and the old Ours checkpoint are held fixed. The perception model receives only events; GT is joined after inference, and depth/segmentation are used only for audit.

The six query cutoffs per episode are exact rows from the saved trace: 0.5905, 0.5915, 0.5925, 0.5930, 0.5935, and 0.5945 s. They are not interpolated query times.

## Result

| Episode | Input | Mean XY error | Median XY error | State available |
|---|---|---:|---:|---:|
| 004, center | Archived 2 kHz Direct-HDR | 17.4 mm | 16.5 mm | 6/6 |
| 004, center | Matched direct render, shadows on | 14.8 mm | 14.5 mm | 6/6 |
| 004, center | Matched direct render, shadows off | 115.4 mm | 104.4 mm | 6/6 |
| 005, +55 mm | Archived 2 kHz Direct-HDR | 37.9 mm | 36.8 mm | 6/6 |
| 005, +55 mm | Matched direct render, shadows on | 15.4 mm | 15.2 mm | 6/6 |
| 005, +55 mm | Matched direct render, shadows off | 283.0 mm | 312.5 mm | 6/6 |

At these late cutoffs, shadow-on event tensors were similar to the archived reference (mean five-bin cosine 0.928 for episode 004 and 0.909 for episode 005); shadow-off cosine was 0.130 and 0.327. Therefore shadow removal strongly changes the event input and degrades this checkpoint on these clips. This does **not** show that shadows are the sole cause of the earlier +55 mm error spikes, nor that shadows generally help perception. The checkpoint is an old development probe, and the replay does not test a new closed-loop rollout.

The follow-up on the early interval found the same pattern: on episode 005, archived 2 kHz events give 58.7 mm mean error, current direct render with shadows on gives 16.2 mm, and shadows off gives 586.2 mm. E then matched the DVS start time to the archive; direct shadow-on error improved to 10.6 mm, but archived error stayed at 58.7 mm. Shadow removal is strongly harmful to this checkpoint, while shadow presence alone does not reproduce the archived error spikes. See the sibling D/E reports for query-level results and hashes.

E's matched initialization test establishes that DVS warm-up history is a contributor to the fresh replay but not the sole source of the archive mismatch. The next analysis uses evaluator-only regional event counterfactuals to test whether the failure depends on interactions between target, robot, and floor event structure.

## Cost and provenance

- Run: `r1_5_shadow_perception_causal_replay_20261029c`; g1, GPU0 for EGL rendering, Genesis CPU backend, DVS/model inference CPU; no physics stepping.
- Actual repository checkout during render: `855b4b0bc7de96170b4bfee8d7002b370782f346`.
- Render diagnostic source SHA256: `e99dcefde30110ef0b6c91a9b0267291ff2c63469c553b07b8ad642b7ed97071`.
- Scene config SHA256: `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`.
- Run config SHA256: `f705203d1ca768e232dd91da26faf2c309d8705385c117236190363a58a98c4b`.
- Old Ours checkpoint SHA256: `d391fec293e2c6c50cdb267e15f742ed09e77e7b28e56a79fdf700f633e804e9`.
- Render/DVS run wall time: 343.786 s; direct render capture itself was about 0.6–1.0 s per condition. Genesis scene-build cost dominated. See episode summaries for per-condition build, DVS and inference timings.
- Archived-event comparison is CPU post-analysis, not a rerender. Its exact event/trace/replay input hashes and per-query outputs are in `archived_reference_same_query_comparison.json` and `.csv`.
- Run stdout/stderr and resource allocation: `../runtime_logs/r1_5_shadow_perception_causal_replay_20261029c/`.

The original renderer-generated `summary.json` carried a stale hard-coded experiment ID (`...20261029a`) despite being launched with config C. The corrected copy records config C; the untouched original is preserved as `summary_uncorrected_generator_output.json`. This metadata correction does not change measurements. `SHA256SUMS` covers the archived result files in this directory.

## Reproduction

Run from the isolated g1 worktree with the preserved robot environment and the exact command recorded in the sibling runtime log metadata. The runner and config are `benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_shadow_perception_causal_replay.py` and `benchmark/robot_closed_loop/configs/r1_5_shadow_perception_causal_replay_20261029c.json`. The archived comparison can be recomputed on CPU with `analyze_r1_5_shadow_causal_replay_archived_reference.py`, pointing `--replay-root` to this directory and using the cache, archived-event, scene-config, and checkpoint paths described in the robot status entry.

This is a two-episode development diagnosis only. It does not alter the frozen simulator core, establish a general shadow policy, or constitute ranking data.
