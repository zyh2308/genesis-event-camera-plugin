# R1.5 continuous-surface candidate: cached replay

This is a diagnostic candidate run on the existing g1 cache. It does not modify the frozen simulator core and is not a closed-loop run.

## Reproduction and provenance

- Runtime code commit: `a92f25019982affb34427631e1c6921bf9ad4aa6`.
- Frozen core base: `e79e656ac00f7847b98db9e12bf43b093f3afff1`; `genesis_event_plugin/` is unchanged.
- Candidate config: [`r1_5_surface_raster_candidate_20261021a.json`](../../configs/r1_5_surface_raster_candidate_20261021a.json), SHA256 `eb7278f69c6a7dc586e3254356b22ee121cb90219244a3ad45862586811adc05`.
- Candidate implementation SHA256 `74b18a926f33a70740bbdc44a33d543b9c3ee9d7e99df4b2649e0bd3f04b8dd5`; replay script SHA256 `1eca8a4ed75f00efb5f4cc6d7d208223f9b206ccf7400e2201fe18d912571335`.
- Source episodes: cached `episode_004` (center control) and `episode_005` (+55 mm development failure) from `r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059`.
- Source trace and cache SHA256s are recorded in [`summary.json`](summary.json). Ours/V2E checkpoint SHA256s are `6ecea6a4520ae09b8cc99ebfd8da7a6a324363d2e57dbf73790e27ee2dbec8d2` and `a7140b63f3b5042b0dc0ed7cc3aa2974c4e9859db919f318826358883f7ce8a0`.
- Output manifest SHA256: `07a1009c6b7ab382f35a50fde57f5f7a24a4e0a90c8e899924f1ccf16f3c9730`.
- Result script: [`run_r1_5_surface_raster_cached_replay.py`](../../diagnostics/run_r1_5_surface_raster_cached_replay.py). The supervised g1 launcher used for the completed run is in the runtime log archive; the equivalent foreground command is:

  ```bash
  cd /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1_5_surface_20261021a
  PY=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python
  CACHE=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-diag-20261019/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059
  "$PY" benchmark/robot_closed_loop/diagnostics/run_r1_5_surface_raster_cached_replay.py \
    --cache-root "$CACHE" \
    --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
    --control-config benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json \
    --ours-checkpoint benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/ours_fit/spatial_event_state.pt \
    --v2e-checkpoint benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/v2e_fit/spatial_event_state.pt \
    --output benchmark/robot_closed_loop/results/r1_5_surface_raster_cached_replay_20261021b
  ```

## Result

The five general controls passed on g1. On both cached clips, the rasterized candidate itself covered only part of the image. Across 42 non-endpoint trace frames per clip, it requested full-frame rendering on every frame. The mean requested pixel fraction was 16.49% for episode 004 and 16.11% for episode 005; the composition had no surface support on 14.91% and 14.55% of pixels, respectively. Single-sided support accounted for another 1.59% and 1.56%; coplanar label conflict was zero. Because the renderer has no partial-frame API, any request triggered a full-frame direct render.

With that fallback enabled, the candidate radiance at every logged trace row and its raw event array are exactly equal to the exact logged-state Direct-HDR comparator: episode 004 has 2,807 events and episode 005 has 4,618. This eliminates the old warp artifact in the produced clips only by using a direct render at every non-endpoint state. It does **not** show that the rasterizer alone fixes the real clips.

Before fallback, supported pixels had low image error, while unsupported pixels remained explicitly unknown. Mean per-frame valid-pixel log-error P95 was `6.04e-5` (episode 004) and `4.37e-4` (episode 005). Mean target-region P95 was `0.00723` and `0.00535`; mean robot-region P95 was `0.00194` and `0.01706`. These are support-conditional values, not full-frame accuracy.

The candidate is currently too costly to use as a production reconstruction path. Candidate totals were 4.925 s and 4.230 s per 47-state clip, including fallback rendering, reconstruction and DVS. Exact direct rendering plus its DVS pass took 1.369 s and 1.552 s on the same cached traces. The candidate was therefore 3.60x and 2.72x slower, while invoking direct render for 42/42 intermediate states.

The old Ours checkpoint remains unstable on these exact direct-event inputs: across the 14 unique query times, its mean XY error is 64.2 mm; the old V2E checkpoint's is 8.9 mm. The candidate and direct-event NPZ arrays are bitwise identical, so these values do not measure a newly trained model.

## Decision

The controls establish a general continuous-surface rasterization fix for the reported tiny-motion failure, correct threshold timing for a smooth log-radiance ramp, and correct front-surface selection in the tested overlap. The cached replay does not establish a useful Ours generation path: unsupported scene regions cause full-frame fallback at every intermediate state, and the result costs more than direct rendering. Do not switch the frozen core or train on these fallback-generated events as if they validated the raster candidate. A new candidate must reduce genuine unknown coverage without weakening occlusion handling and must beat the measured direct-render cost before retraining or closed-loop work resumes.

`summary.json`, per-frame `frame_metrics.jsonl`, per-episode summaries, candidate/direct raw events, query tensors, visualizations and `SHA256SUMS` are the authoritative data. The summary preserves the actual nonuniform trace intervals (0.5–1.0 ms; effective mean about 1.15 kHz); no missing physical state was synthesized.
