# R1.5 early-window shadow replay: episodes 004/005

## Question and protocol

This cached-state, render-only replay tests the previously reported +55 mm localization spikes at exact trace-row cutoffs from 0.5855 to 0.5905 s. It compares archived 2 kHz Direct-HDR events with a fresh direct HDR replay using shadows on/off. The camera, scene, MSAA, ideal physical-log DVS response, five causal 5 ms bins, old Ours checkpoint, and query rows are fixed. Physics is not advanced. GT is joined after inference; depth and segmentation are audit-only.

The seven exact cutoff times per episode are 0.5855, 0.5860, 0.5865, 0.5875, 0.5885, 0.5895, and 0.5905 s. All three event sources returned a state at 7/7 cutoffs.

## Main result

| Episode | Input | Mean XY error | Median XY error | Maximum XY error |
|---|---|---:|---:|---:|
| 004, center | Archived 2 kHz Direct-HDR | 11.0 mm | 10.4 mm | 16.7 mm |
| 004, center | Fresh direct render, shadows on | 10.0 mm | 10.5 mm | 11.9 mm |
| 004, center | Fresh direct render, shadows off | 58.4 mm | 26.1 mm | 155.1 mm |
| 005, +55 mm | Archived 2 kHz Direct-HDR | 58.7 mm | 59.2 mm | 111.1 mm |
| 005, +55 mm | Fresh direct render, shadows on | 16.2 mm | 18.1 mm | 22.5 mm |
| 005, +55 mm | Fresh direct render, shadows off | 586.2 mm | 649.5 mm | 674.7 mm |

Shadow-on event tensors are similar to the archive (mean cosine 0.916 for episode 004 and 0.902 for episode 005), while shadow-off tensors have cosine 0.096 and 0.317. This reproduces the strong shadow sensitivity, but it shows the archived +55 mm spikes are not explained by simply turning shadows off: the fresh shadow-on renderer gives 16.2 mm mean error on the same seven states.

## Input mismatch probes

- At cutoff 0.5865 s in episode 005, the archive has 3,841 events in the 25 ms model window and shadow-on replay has 2,714. Per-bin cosines are about 0.89–0.91 and each bin has roughly 0.42–0.46 relative L1 difference. Across the seven cutoffs, uniform event-count matching does not fix the model (60.1 mm mean); per-bin matching is worse (72.5 mm); matching both time-bin and polarity totals helps only partially (48.8 mm).
- Replacing one archived 5 ms bin with the corresponding shadow-on bin can sharply change the prediction at a few cutoffs. This shows that the old checkpoint is sensitive to the temporal/spatial distribution inside the five-bin tensor; it does not identify one universal faulty bin.
- A timestamp sweep finds no stable global time offset: shifting archived event timestamps by +5 ms lowers the early +55 mm mean to 9.8 mm, but on the later 0.5905–0.5945 s interval the same shift raises error to 78.5 mm from 37.9 mm. The sweep is a counterfactual, not a proposed correction.
- Small input-plane translations can also lower the model error, but the shift selected by model error is not a valid camera correction. The unshifted tensor has the highest event-map cosine to the matched direct shadow-on replay in every query (about 0.89–0.93); translated maps are less similar. Thus the old checkpoint's improvement under a hand-selected translation is model sensitivity, not evidence of a measured camera-calibration offset.
- Repeated direct captures at every exact state were bit-identical in HDR (repeat MAE, p99 and max all zero), so within-run renderer jitter does not explain these differences.

## Follow-up: matched initialization history

Experiment `20261029e` kept the same queries and 25 ms model windows but initialized its DVS at the archived stream's exact start, 0.5505 s. On episode 005, fresh shadow-on mean error improved from 16.2 to 10.6 mm; the archived stream remained at 58.7 mm and its input cosine to fresh shadow-on remained about 0.90. Initialization history therefore affects this replay but does not explain the archived-event error by itself. Full E metrics, provenance, and hashes are in the sibling [`20261029e README`](../r1_5_shadow_perception_causal_replay_20261029e/README.md).

## Cost and provenance

- Render run `r1_5_shadow_perception_causal_replay_20261029d`, g1 GPU0 RTX 3090 UUID `GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23`; Genesis CPU backend, DVS/model CPU. Exit code 0; total wall `247.928 s`.
- Four condition builds took `50.97–60.45 s` each. Captured exact-state rendering itself took `0.60–1.01 s` per condition. No physics steps were taken.
- Render checkout commit `855b4b0bc7de96170b4bfee8d7002b370782f346`; diagnostic source commit `3c7101afe1d7360c956abb6a7ed0d123da527f5a`; render script SHA256 `b4f366ab798abc12926e863bedbe666f5c0de401ce4bff6adcf720e319ca58a1`.
- Config D SHA256 `ca83a66dec75a63b8cbc11c1bf02fee29b367bc3825a5872b98c7f9de26d923d`; scene config SHA256 `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`; checkpoint SHA256 `d391fec293e2c6c50cdb267e15f742ed09e77e7b28e56a79fdf700f633e804e9`.
- CPU post-analysis source commit: `8e806ac9d0236429905509074410a174067e5060`; query-by-query metrics, temporal/spatial counterfactuals, tensor/bin comparisons and source hashes are in `archived_reference_same_query_comparison.json` and `.csv`.
- Runtime stdout/stderr, launch/worker scripts and resource allocation record: `../runtime_logs/r1_5_shadow_perception_causal_replay_20261029d/`.
- The archive, trace and current direct replay are development references only. No retraining, ranking, closed-loop rollout, frozen-core edit, or formal test data is involved.
