# R1.5 same-trajectory intermediate-frame causal diagnosis

**Status:** completed as a two-clip development diagnosis on g1. This is enough to report a confirmed event-generation divergence and its strongest spatial contributor. It does not identify a single line-level defect or establish a physically correct event source.

## Question and method

The previous crossover found a large event-count difference between the native HDR plugin and high-frequency Direct-HDR on the same saved 0.4 m/s physical clips. This follow-up held the plugin's intermediate sample timestamps fixed and replaced the plugin's intermediate log-luminance images with pixelwise log-luminance interpolation between the already-rendered Direct-HDR states. The dense reference contains 59 exact state renders per episode at nonuniform 0.5–1.0 ms intervals. No state or image was rendered for this experiment; the substitute images are explicitly an offline approximation, not direct renders at the intermediate timestamps.

The two clips are episode 004 (center) and episode 005 (+55 mm), over 0.5505–0.6005 s. The plugin produces 35 and 38 intermediate frames. The nearest saved Direct-HDR state is at most 0.429 ms from a sample time, with a maximum bracketing interval of 1 ms. Each variant has its own continuous history in the same `DvsEmulator` class, initialized from the same exact 0.5505 s frame and evaluated with the same frozen checkpoints. Segmentation is used only offline to label/restrict image regions; it never enters the estimator.

DVS response settings are identical in every variant: positive/negative threshold 0.2, zero threshold noise, cutoff/leak/shot noise/refractory disabled, `physical_log`, linear-radiance input, gain 1, offset 0, floor `1e-6`, and seed 42. The exact helper code is hashed below.

Two additional spatial interventions replace only the visible robot pixels or only the non-robot pixels in the reconstructed frames. These mixed images use evaluator-only segmentation masks, are not valid sensor inputs, and serve only to estimate which region contributes more to the discrepancy.

## Findings

The frozen plugin was reproduced exactly from the archived HDR endpoints: event arrays match the saved arrays element for element on both episodes. With the native plugin's exact timestamps retained, replacing its intermediate images with the nearby Direct-HDR samples reduces event counts from 29,714 to 2,430 (center) and 28,948 to 4,439 (+55 mm), approaching the exact Direct-HDR totals of 3,539 and 5,739. This rules out the DVS implementation or timestamp schedule alone as the explanation for the count gap in these clips; the intermediate image values are the first confirmed divergence.

Only 1.05% and 1.04% of all pixel-time samples differ by more than 0.2 log-luminance. Of those affected samples, 67.4% and 76.0% are labeled as robot; 30.7% and 22.8% are other visible scene pixels; the target accounts for 1.9% and 1.3%. Within the robot region, 4.93% and 5.35% of pixel-time samples exceed the threshold. The largest discrepancies are 3.11 and 3.57 log units at robot-labeled pixels near the top image edge. `static_or_scene` is a segmentation grouping, not a shadow label.

| Episode | Native plugin events | Same-schedule Direct-HDR resample | Robot-only replacement | Non-robot-only replacement | Exact Direct-HDR events |
|---|---:|---:|---:|---:|---:|
| Center, 004 | 29,714 | 2,430 | 10,005 | 22,152 | 3,539 |
| +55 mm, 005 | 28,948 | 4,439 | 9,450 | 24,214 | 5,739 |

The robot-only intervention removes about two thirds of the native events, compared with a 16–25% reduction from the non-robot-only intervention. This makes articulated-robot pixels the strongest confirmed contributor, while non-robot regions still contribute. The earlier attribution that “DVS threshold history couples pixels over time” was inaccurate: this ideal DVS configuration keeps threshold/reference history per pixel. The regional intervention uses a segmentation mask that changes with time, so one pixel can switch which source supplies its intensity; this changes that same pixel's later threshold crossings. This is temporal state within a pixel combined with a time-varying intervention mask, not cross-pixel DVS coupling. A later 10 ms cached replay found 2 / 18 robot-mask pixel-time switches in episodes 004 / 005 and event-count nonadditivity of 4 / 50 events in that short window; the earlier full-window nonadditivity (13 / 277 events) has not been decomposed into mask-switch versus per-pixel-history contributions.

On the center clip, the Ours checkpoint mean planar XY error is 29.6 mm on native events, 76.8 mm on exact Direct-HDR, and 19.7 mm on the same-schedule resampled source. On +55 mm it is 27.5, 71.2, and 65.2 mm respectively. V2E remains accurate on exact and resampled Direct-HDR (center 5.2/5.8 mm; +55 mm 8.4/8.9 mm). Thus the frame-value substitution changes the Ours result substantially on the center clip but does not restore stable localization on +55 mm. The 120 paired saved five-bin count tensors have source cosine mean 0.191 (P05 0.144), showing a large input change, not a performance score.

## Interpretation and limits

- **Confirmed:** for these two clips, the first large source divergence is in the plugin-reconstructed intermediate luminance images. Keeping the same sample timestamps while replacing those image values removes most excess events.
- **Strongest spatial contributor:** discrepancies within robot-labeled pixels account for most large pixel differences and robot-only replacement removes most excess events. This points to moving/articulated geometry or visibility handling as the next code path to inspect.
- **Not yet established:** the precise interpolation failure mechanism. The residual non-robot discrepancy may include background, disocclusion, or moving shadows; this experiment does not label shadows and does not prove which mechanism causes them. The Direct-HDR substitute is time-interpolated saved imagery, not a new exact render at every plugin sample.
- **No fix or retraining yet:** frozen core, checkpoints, scene, controller, DVS response, and formal evaluation were unchanged. The old Ours checkpoint remains unstable on +55 mm.

The research-side conclusion is therefore: **report now that the plugin's intermediate image reconstruction is causally implicated, with robot-region reconstruction the leading contributor; keep the exact algorithmic cause open and do not switch the frozen core yet.** The next useful engineering check is to map these high-difference pixels to the interpolator's per-object warp, visibility/hold, and newly revealed-surface masks, then test a general correction on both moving robot and static-background controls.

## Provenance, hashes, and replay

- Crossover input/evaluator code commit: `5e0b918fe4d5e9244c6d5530d5a0e867fd70397`; original crossover report and full input hashes are in [the source-crossover result](../r1_5_same_trajectory_source_crossover_20261029c/README.md).
- Event-pattern analyzer commit: `9fb611922ace81ae1c4c3e3ccc41c65a2e951ba6`; analyzer SHA256 `aa2884ce8758f6c68bba493359c8c2e6f789309597eb08c406b18d553b27abdc`.
- Same-schedule counterfactual script commit: `72c9e9ff88d990e44fde5c06ec2384dc4322b80c`; script SHA256 `56a1a80c62854041eab2f923cd2d6bd0e5ce6f4a2a7d76041a8619cdc31c7523`. It ran with the isolated remote checkout at commit `15b36170b9f33dcbf80717c6d22c70c1a54eb9de`.
- Runtime core source hashes: `physics_plugin.py` `f01c6cb760153f09add80e0e18610a5710d07e0a7077e9552e0c7e7e009fb5cf`; `interpolator.py` `af9bd273341b7ea4ffb4d9e6d14ab13829bb2a397d09991a3dcd73f4a339f6a3`; `dvs_emulator.py` `914dc83525ff96dc2c1a1373819de74491cfbb03007b65e8c93609f6cd1cf638`; DVS response helper SHA256 `9c7a018d64c8d0809648a69b58d3be6175a2d85eeb464a1bb5758ec3049a669f`.
- Scene config SHA256 `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`. Ours/V2E checkpoint SHA256s: `6ecea6a4520ae09b8cc99ebfd8da7a6a324363d2e57dbf73790e27ee2dbec8d2` / `a7140b63f3b5042b0dc0ed7cc3aa2974c4e9859db919f318826358883f7ce8a0`.
- Compact event-pattern result: [`output/summary.json`](output/summary.json), [`output/input_source_cosine_by_query.csv`](output/input_source_cosine_by_query.csv); output manifest SHA256 `12de3d274a6131d0d60b9ef434fe65105ea1fa586ad217c28a069c3e094adce8`.
- Final same-schedule result: [`same_schedule_counterfactual_rerun6/summary.json`](same_schedule_counterfactual_rerun6/summary.json), per-query CSV and six raw event arrays in the same directory. Its `SHA256SUMS` file SHA256 is `495e74f49fcf8865795cece99a2be35bacdc9fc076a242c27dabb44fd1951642`; every listed file passed local verification. Full source HDR caches and exact Direct-HDR HDF5 remain on g1 at `/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-source-crossover-20261029a/benchmark/robot_closed_loop/results/r1_5_same_trajectory_source_crossover_20261029c/`.
- Repeatability: the all-pixel same-schedule event arrays in `same_schedule_counterfactual_rerun2/`, `rerun4/`, and `rerun6/` are byte-for-byte equal for both episodes. Rerun2 used script commit `e4561dd3a9504fdb208380991cb745f2b693c8e9`, SHA256 `2f0da9553e4f9856227acf51600fb7722666844d47b2ef28945fdc611b511ee0`; its output manifest is included. Rerun4 manifest SHA256 `8e8fb3a3079b810aba98b5a951ec155dafbdbb010fcae6c8151712e6a97a42dd`; its script was commit `1566abc`, SHA256 `1ab2e7b8a9802a678e48a0f791b3c6006aff735c0e228ca62c28450bf4796da9`.
- The final CPU-only command completed in 23.4 s; no renderer, GPU, physics, or training was run. It can be replayed from that g1 checkout using the command below and a new output directory:

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-source-crossover-20261029a
CUDA_VISIBLE_DEVICES= PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python \
  benchmark/robot_closed_loop/results/r1_5_same_trajectory_event_pattern_analysis_20261029a/run_r1_5_same_sample_schedule_dense_resample.py \
  --result-root benchmark/robot_closed_loop/results/r1_5_same_trajectory_source_crossover_20261029c \
  --cache-root benchmark/robot_closed_loop/results/r1_5_same_trajectory_source_crossover_20261029c/source_inputs/native_hdr_cutoff059 \
  --checkpoint-ours benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/ours_fit/spatial_event_state.pt \
  --checkpoint-v2e benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/v2e_fit/spatial_event_state.pt \
  --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --code-commit 72c9e9ff88d990e44fde5c06ec2384dc4322b80c \
  --output-dir benchmark/robot_closed_loop/results/r1_5_same_trajectory_event_pattern_analysis_20261029a/REPLAY_UNIQUE_ID
```
