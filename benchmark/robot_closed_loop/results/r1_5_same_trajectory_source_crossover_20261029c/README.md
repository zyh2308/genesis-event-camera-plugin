# R1.5 same-trajectory source crossover: completed development diagnosis

**Status:** completed on g1. This is a two-episode causal source crossover using frozen checkpoints, not a generator ranking, independent test, or retraining result.

## What was compared

The same saved 0.4 m/s physical state sequence was rendered as (1) the archived native scene-linear HDR endpoint stream consumed by the frozen Ours event plugin and (2) exact-state, high-frequency Direct-HDR frames rendered at every saved state row. Both frozen checkpoints read both event sources, producing the four paired groups. Each source had its own DVS history, initialized once at the shared 0.5505 s frame and advanced continuously; neither history was reset at query times. The existing five causal 5 ms bins, 64×64 input, `log1p` conversion, global soft-argmax decoder, fixed-plane backprojection, and DVS settings were kept. No physics was advanced.

The rendered segment contains 59 actual trace rows from 0.5505–0.6005 s, with nonuniform 0.5–1.0 ms spacing (median 1.0 ms). All 30 query times per episode are exact saved rows from 0.5755–0.6005 s. The two source caches match exactly at all six native 100 Hz endpoint checks: radiance max absolute error 0, depth finite-mask mismatch 0, and segmentation mismatch 0. Initial-frame luminance also matches exactly. State availability was 100% in all 240 group/query rows.

## Result

Planar XY errors are in millimetres; pixel errors are reported separately in `all_four_group_queries.csv`. Each cell below is mean / P95 / maximum over 30 adjacent query times. These time samples are correlated; they are not 30 independent episodes.

| Trajectory | Event source → checkpoint | XY error (mm) | Pixel error (px) | First query >50 mm |
|---|---|---:|---:|---:|
| Center, episode 004 | Native HDR → Ours | 29.6 / 78.1 / 312.3 | 8.4 / 22.7 / 70.2 | 0.5825 s |
| Center, episode 004 | Native HDR → V2E | 35.2 / 114.6 / 225.8 | 9.7 / 25.5 / 47.2 | 0.5825 s |
| Center, episode 004 | Direct-HDR → Ours | 76.9 / 318.3 / 374.5 | 20.4 / 82.2 / 100.8 | 0.5775 s |
| Center, episode 004 | Direct-HDR → V2E | 5.2 / 8.8 / 11.6 | 2.2 / 3.3 / 4.4 | None |
| +55 mm, episode 005 | Native HDR → Ours | 27.5 / 82.3 / 424.1 | 7.0 / 18.7 / 70.1 | 0.5855 s |
| +55 mm, episode 005 | Native HDR → V2E | 49.3 / 187.3 / 417.2 | 13.6 / 53.0 / 94.5 | 0.5855 s |
| +55 mm, episode 005 | Direct-HDR → Ours | 71.2 / 286.2 / 327.8 | 22.0 / 72.7 / 80.5 | 0.5755 s |
| +55 mm, episode 005 | Direct-HDR → V2E | 8.4 / 13.6 / 15.4 | 3.2 / 4.2 / 4.6 | None |

Changing only event source consistently worsened the Ours checkpoint on these two trajectories: paired Direct-minus-native error was positive at 24/30 center queries and 27/30 +55 queries, with mean deltas +47.2 and +43.7 mm. The V2E checkpoint showed the opposite interaction: Direct-HDR was better at 26/30 and 28/30 queries, with mean deltas −30.0 and −41.0 mm. This is direct evidence that checkpoint behavior depends strongly on event source for these clips. It does not show that either source is the more physically accurate one.

The event counts also differ substantially: native plugin 29,714 / 28,948 versus Direct-HDR 3,539 / 5,739 for center / +55. The response parameters and event history initialization were shared, so the count/temporal pattern difference remains a plausible part of the source interaction; this experiment does not isolate which event-generation step causes it.

### Where the errors go

All region labels below are computed after inference from the exact Direct-HDR frame segmentation and saved heatmap; they are never model inputs. Area-weighted heatmap mass is averaged over the 60 Direct-HDR→Ours queries. It falls on target 78.6%, robot 5.0%, and other nonzero scene segmentation 16.4%. The heatmap peak lies on the target for 54/60 queries, but the decoded prediction pixel falls on the robot for 10/60 and on other scene geometry for 8/60. Thus a target-local peak can coexist with an off-target soft-argmax location when substantial probability mass is spread elsewhere.

On the +55 mm trajectory, the first Direct-HDR→Ours error already exceeds 50 mm at 0.5755 s (118.9 mm). At that query the peak is still on the target, while heatmap mass is 36.9% target, 10.9% robot, 52.1% other scene, and the decoded prediction pixel lands on the robot. In the center clip, the first >50 mm Direct-HDR→Ours query is 0.5775 s (50.5 mm); both the peak and decoded pixel still overlap the target segmentation. A segmentation overlap is not equivalent to correct planar localization: the decoder backprojects to a fixed plane, while visible target pixels include the object's 3D surfaces. The `other_scene` category is not a shadow mask, so these numbers do not attribute the error specifically to shadows.

## Reproducibility and files

- Render/inference code commit: `5e0b918fe4d5e9244c6d5530d5a0e867fd70397e`; runner SHA256 `6ba1771848c6d24f15fcb58ff89a1eb7ec5c20ea8723a5c66e1705004d295ea2`.
- Post-inference region analysis commit: `15b36170b9f33dcbf80717c6d22c70c1a54eb9de`; analyzer SHA256 `33120259bba56e1be2291527d0d85fdf3a7d80352c7a059b6a03be4374ab4f9b`.
- Experiment config SHA256 `2fa77e82bb6e8de8ecfb7e09bee3e9b2926b6e5fc9a7c173d4c9c067a9cf4267`; scene config SHA256 `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`; shared-control config SHA256 `3d289c2bef1c562b26f9b713c2a826b813a286a1f7e21f9ead0a01c63c5e0d80`.
- Frozen Ours / V2E checkpoint SHA256s: `6ecea6a4520ae09b8cc99ebfd8da7a6a324363d2e57dbf73790e27ee2dbec8d2` / `a7140b63f3b5042b0dc0ed7cc3aa2974c4e9859db919f318826358883f7ce8a0`.
- Exact trace-window SHA256s: episode 004 `3f07e4e7a0846827be7a3d5decf7c6197d77ca0237bacff55f2c45a70a04349c`; episode 005 `f0f556842029cd54b3fc082ee0e2848dd194f3322a8a9c0769c2c889a6f6ae5d`. Native cache hashes: `0525c18c8d89f5831ef0cb71ac9e3b6b130d60579daf374f06d5cfcee5b879d7` / `91c1397642e317d6bb51150b4107c518b52e4fa9fcf5f1caf6a2a91315c7d434`.
- End-to-end render and inference time was 70.36 s for both episodes. Per episode, native event generation took 3.55/3.37 s, Direct-HDR DVS took 0.56/0.70 s, and four-group inference took 1.23/1.26 s. Renderer used g1 GPU0 UUID `GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23`; inference used CPU. No training or physics steps ran.
- Resource record: run ID `r1_5_same_trajectory_source_crossover_20261029c`, GPU0, PID `1400566`, start `2026-10-29T11:09:26Z`, finish `2026-10-29T11:12:14Z`, completed and released. The server-side record is `/home/zhaoyuhan/Eventbased_WAM/notes/robot_resource_allocations/records/r1_5_same_trajectory_source_crossover_20261029c.md`.
- Full output, including raw event arrays, all four inputs/heatmaps, exact-state frames, and source caches, remains on g1 at `/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-source-crossover-20261029a/benchmark/robot_closed_loop/results/r1_5_same_trajectory_source_crossover_20261029c/`. Its complete remote `SHA256SUMS` file has SHA256 `1711888ffdbb207de51f84d79c42ec2a79c26f4c510b43b9803c95f3a07f10a5`; the local `REMOTE_SHA256SUMS` is a copy. Nine compact local files matched that manifest after copying the updated overlays; 19 large source/output files remain remote-only.
- Compact local evidence: `summary.json`, `run_metadata.json`, `effective_config.json`, `run.log`, `all_four_group_queries.csv`, `post_inference_region_association.csv`, `postprocess_visualization_metadata.json`, two overlays, and `REMOTE_SHA256SUMS`. SHA256s for the new association outputs: CSV `6bec8b91fd53b85ee17f476593cf71970c4b80ac8c5fee00106c26f219500d1e`; postprocess metadata `ab08829ba52a611b2a844e8038fba8dfdfd784eeb872ae99567ed7453c6036a7`.

Re-run the analysis only (CPU, does not rerender or rerun inference) from the source-crossover worktree on g1:

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-source-crossover-20261029a
PYTHONPATH=. /home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python \
  -m benchmark.robot_closed_loop.diagnostics.render_r1_5_source_crossover_overlays \
  --result-root benchmark/robot_closed_loop/results/r1_5_same_trajectory_source_crossover_20261029c
```

The full source-crossover renderer command is defined by `run_r1_5_same_trajectory_source_crossover.py`; it uses the config, scene/control configs, exact staged caches and frozen checkpoints listed above. Use a new unique output ID when replaying because the runner refuses to overwrite existing results.

For a new renderer/inference replay, first obtain a fresh GPU0 reservation and recheck its current process/load state. Then use the command below with a new output name; it does not reuse or overwrite the completed run:

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-source-crossover-20261029a
PYTHONPATH=. CUDA_VISIBLE_DEVICES= EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl \
  /home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python \
  benchmark/robot_closed_loop/diagnostics/run_r1_5_same_trajectory_source_crossover.py \
  --config benchmark/robot_closed_loop/configs/r1_5_same_trajectory_source_crossover_20261029c.json \
  --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --control-config benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json \
  --source-cache-root benchmark/robot_closed_loop/results/r1_5_same_trajectory_source_crossover_20261029c/source_inputs/native_hdr_cutoff059 \
  --ours-checkpoint benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/ours_fit/spatial_event_state.pt \
  --v2e-checkpoint benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/v2e_fit/spatial_event_state.pt \
  --output benchmark/robot_closed_loop/results/r1_5_same_trajectory_source_crossover_REPLAY_UNIQUE_ID \
  --code-root . \
  --gpu-uuid GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23
```

## Scope limits and next decision

This is two saved clips at one speed and 30 adjacent queries per group. The full original episode traces were not staged: the diagnostic uses verified `trace_window.jsonl` rows and preserves the full-trace hashes declared in episode metadata, but cannot independently cross-check them against the absent full files. Both event histories start at 0.5505 s rather than the start of each episode. Only these two clips establish the observed source×checkpoint interaction. Follow-up raw-event characterization and same-timestamp intermediate-image substitution are now complete; see the [R1.5 intermediate-frame causal diagnosis](../r1_5_same_trajectory_event_pattern_analysis_20261029a/README.md). No claim about physical realism, retraining, or generator ranking follows from these development clips.
