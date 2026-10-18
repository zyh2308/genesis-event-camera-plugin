# R1 Ours failure-chain diagnosis

This is a small development diagnosis on g1. It is not a held-out test, formal comparison, or ranking. It preserves the `7fb72dc` batch and the existing nine-episode R1 inputs.

## Finding

The old Ours training route used cached display PNG grayscale values and the video `lin_log` transfer. It did not use the cached scene-linear HDR radiance. The separate failure replay now reads `trajectory_cache.h5/radiance` captured with `camera.render(radiance=True)` and sends it through the frozen plugin's `physical_log` path (`log(max(luminance, 1e-6))`, gain 1, uncalibrated 0.2 contrast threshold). This restores the intended HDR input for the diagnostic without changing the simulator core.

At the exact +55 mm failure query (`cutoff=0.5900 s`), the same old Ours checkpoint produced 185.8 mm XY error on the common Direct-HDR reference events, but 3.04 mm on the newly generated native-HDR Ours events and 1.38 mm on the legacy display-gray Ours events. These are transfer probes of a checkpoint trained on the legacy display-gray data; they do not estimate a model trained on native HDR. The target is visible in the source image. Ten-frame offline dense-render replay from the exact saved physical trace has mean log-radiance absolute error 0.00835 overall (0.00700 on the target region and 0.00218 on background); unknown/hold pixels average 0.0014%. Exact stored-state render snapshots match the original reference frames pixel-for-pixel.

Native Ours events cluster at the 100 Hz source-frame boundaries: in the +55 mm window, ±1 ms seam windows contain 31,116 events versus 1,501 in equal-width between-seam midpoint windows (20.73×). The center episode ratio is 24.80×. The unknown/hold fraction is negligible in these windows, so hold completion is not a plausible explanation for this particular burst. The dense render comparison checks pixel reconstruction, not event-camera ground truth. The frame-boundary concentration is established; its precise responsibility inside the frozen plugin is not.

The raw-event to network-input conversion was also checked at this cutoff. Raw records are finite `[seconds, x_px, y_px, ±1]`, all coordinates are inside the 320×240 frame, and both polarities occur. In the five `[cutoff−25 ms, cutoff)` bins, +55 mm Ours-HDR has 14,515 events (polarity counts 7,186/7,329; per-bin counts 1,976/5,095/1,221/5,335/888); the same common Direct-HDR window has 3,753 events (1,994/1,759; per-bin 801/768/812/685/687). The model applies `log1p` to counts with no per-window max normalization. Both checkpoints receive the same exact tensor byte hash for each source. This rules out a checkpoint-specific voxel, polarity, or scaling path in the diagnosis; the clear numerical mismatch is already present in the event timing/count distribution before inference.

Position is the first failure at the known query. With the same events and checkpoint, the Ours direct velocity head has 0.0060 m/s error while the 50 ms history OLS has 2.625 m/s error and the robust pairwise median has 2.167 m/s error. Replacing velocity cannot repair the 185.8 mm localization error; it does show that the history regression amplifies a position jump. This does not establish grasp success.

## Evidence and scope

- `native_hdr_cutoff059/native_hdr_replay_summary.json` records the exact query, HDR ranges, offline dense-render comparison, inference outputs, event timing counts, and runtime. Its `SHA256SUMS` covers the episode images, raw events, trace window, render evidence, and five-bin tensors.
- `native_hdr_cutoff059/episode_005/failure_chain_montage.png` shows the +55 mm failure source image and event/network stages; `episode_004/failure_chain_montage.png` is the center control.
- `native_hdr_timing/event_timing_summary.json` replays the cached raw arrays and confirms bitwise repeatability and the per-interval seam concentration. Its `SHA256SUMS` records the files.
- `velocity/per_query_velocity_comparison.csv` contains the full 58-poll, two-episode velocity strategy comparison; `velocity/localization_inputs_and_heatmaps.png` shows the fixed input/checkpoint comparisons.
- The common Direct-HDR evaluation reference is unchanged and remains development-only. All four event-source routes use the same causal five-bin tensor interface. GT is joined only after model inference.
- No frozen-core files were changed. No new physics trajectory was run for the dense image comparison; it replays exact stored state rows. No interpolation frame is represented as a direct-render observation.

## Small native-HDR retrain completed

One fixed-budget development retrain was launched separately to distinguish legacy checkpoint transfer from native-HDR training utility. It reuses the exact nine cached physical episodes, labels, timestamps, six/three episode split, original official V2E event archives, `heatmap_v2`, seed 606, 40 epoch/patience budget, and the same Direct-HDR reference. Only the Ours events are regenerated from native scene-linear HDR. This is intentionally not a response-matched sensor comparison: Ours uses the physical-log HDR path while V2E retains its official display-video + SuperSloMo path.

All nine episodes generated 12,323,507 native-HDR events on CPU. Per-episode event counts and generation wall times are `1,664,645 / 288.24 s`, `1,467,264 / 362.13 s`, `1,877,911 / 292.64 s`, `1,209,488 / 217.07 s`, `1,274,497 / 274.25 s`, `985,229 / 192.70 s`, `1,524,285 / 271.81 s`, `1,327,918 / 320.03 s`, and `992,270 / 234.60 s`. Summed plugin generation time was 2,453.48 s; the run summary phase including cache/source bookkeeping was 2,472.67 s.

Both fixed-budget fits and the common-reference evaluation completed with exit 0. Each model had 566 validation windows. On each model's own held-out event episodes, Ours native-HDR validation XY P95 was 38.8 mm and V2E was 51.6 mm. On the same previously inspected nine-episode Direct-HDR reference (581 common polls), however, Ours remained poor: position XY P95 444.9 mm and predicted-reachable coverage 315/581 (54.2%); V2E position XY P95 was 11.3 mm and reachable coverage 514/581 (88.5%). Ours +55 mm episodes remained poor at all three speeds. This does not establish a generator ranking: the two training event response/reconstruction paths are not matched and this is a development reference, not a held-out set. It does show that restoring HDR input and retraining alone did not restore Ours transfer to the common reference.

The run's internal phase costs were 2,472.67 s for Ours native-HDR generation, 1,434.39 s for the two CPU fits, and 13.74 s for common-reference evaluation; total was 3,920.80 s (65:23 GNU wall time, maximum RSS 5,114,596 KB). Cost excludes the existing physics rollout cache, V2E generation, and Direct-HDR reference capture. The g1 full-result SHA256 manifest passed all 110 entries. The committed small subset and both checkpoints are in [`r1_native_hdr_small9_retrain_20261019a`](../r1_native_hdr_small9_retrain_20261019a/README.md); raw event archives remain at the g1 run path described there.

Reproduction command on g1 (choose a new run ID for a fresh run; the launcher refuses to overwrite):

```bash
cd ~/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-nativehdr-20261019a
bash benchmark/robot_closed_loop/diagnostics/run_r1_native_hdr_development.sh r1_native_hdr_small9_retrain_<new_id>
```

The run uses CPU only. Input roots, official V2E archive provenance, and the Direct-HDR reference path are the defaults in the launcher; override with `R1_NATIVE_HDR_CACHE_ROOT`, `R1_NATIVE_HDR_V2E_SOURCE`, and `R1_NATIVE_HDR_REFERENCE_ROOT` only when reproducing from verified identical data.

## Provenance

- Branch: `exp/robot-closed-loop-v1`.
- Exact failure replay code: `59c756eb1e77f9e14f475fa3c923bcc6a9ee859e`; replay config SHA256 `242d6063bb136092b2c29bc12a7fb54db5e6233bf6b73506d163c8cfcbb8d96b`; frozen scene config SHA256 `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`; Direct-HDR reference manifest SHA256 `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`.
- Velocity strategy code: `e58ed7560bd02af07ba454fd406b29a999970877`; velocity diagnostic config SHA256 `c0391f951eb66b0f4d862d61d320415c14b0f3a20424a2bb9eae9c273355dd1d`; same scene and Direct-HDR reference hashes above.
- Cached event timing replay code: `cea174a6caf2db581e87ed47126b3fb647fe835e`; timing report verifies raw-array replay is bitwise identical to the exact failure replay.
- Every evidence subdirectory has a `SHA256SUMS` manifest. Verify from this folder with `sha256sum -c <subdirectory>/SHA256SUMS` after using that subdirectory as the working directory, or compare each listed file hash directly.
- Native-HDR retrain config SHA256: `351c09ae190f41e27ce2262cdf4bd5eaa932a96e685191ae9a309ffc3c1d5e1b`. Run code commit: `4c5f5f29d4c46dd50401c0aa9079874f97afcbc6`.
