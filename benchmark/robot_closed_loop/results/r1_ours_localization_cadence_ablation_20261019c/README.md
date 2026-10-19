# Ours localization cadence diagnostic: 2026-10-19c

## Question and result

This small, offline replay checks why the native-HDR Ours checkpoint can localize the cube on its own generated events but miss it on Direct-HDR events. It reuses two already inspected development episodes: episode 004 (center) and episode 005 (+55 mm failure), at the same cutoff, 0.5905 s. It is a diagnostic, not an independent test or a generator ranking.

The first divergence is at the learned image localization output. The Ours checkpoint is accurate on the sparse 100 Hz Ours stream (10.0/4.0 mm XY error) but shifts on both exact logged-state direct events (79.6/20.3 mm) and the archived 2 kHz Direct-HDR reference (54.1/14.8 mm). The V2E checkpoint receives the exact same tensor for each source and stays much closer in these two samples (8.0/5.9 mm sparse, 8.2/14.4 mm direct logged-state, 3.6/4.4 mm archived reference). This points to Ours checkpoint sensitivity to event-source distribution, rather than a shared camera projection or label error.

The 100 Hz Ours stream also has large endpoint bursts. In episode 005, the three seam windows at 0.5605/0.5705/0.5805 s contain 4,756/6,154/5,200 events within ±1 ms, while the equal-width between-seam windows contain 23/16/14. In the same 40 ms interval, the causal 25 ms input contains 15,278 events for this sparse stream, 2,888 for direct logged-state events, and 3,776 for the archived reference. Sparse-input cosine to the archived five-bin tensor is 0.217; direct logged-state cosine is 0.908. Episode 004 shows the same direction. This establishes a large event-distribution difference, but does not prove that the seam bursts alone cause the localization error.

## Measurements

| Episode | Input source | Events in causal 25 ms | Ours checkpoint XY error | V2E checkpoint XY error |
|---|---|---:|---:|---:|
| 004 center | Ours 100 Hz endpoints + frozen reconstruction | 14,779 | 10.0 mm | 8.0 mm |
| 004 center | Direct HDR at exact logged trace rows | 1,722 | 79.6 mm | 8.2 mm |
| 004 center | Archived Direct-HDR 2 kHz reference | 2,228 | 54.1 mm | 3.6 mm |
| 005 +55 mm | Ours 100 Hz endpoints + frozen reconstruction | 15,278 | 4.0 mm | 5.9 mm |
| 005 +55 mm | Direct HDR at exact logged trace rows | 2,888 | 20.3 mm | 14.4 mm |
| 005 +55 mm | Archived Direct-HDR 2 kHz reference | 3,776 | 14.8 mm | 4.4 mm |

The direct logged-state path renders 47 exact rows per episode from the saved trace. Their median interval is 1 ms, effective mean rate about 1.15 kHz, with a 0.5 ms minimum. Missing states were not synthesized, and this path is not described as a full 2 kHz replay. It passes those HDR frames through the same uncalibrated ideal physical-log response used for the sparse Ours path. The archived reference differs only in near-black log-floor convention (`log(luma + 1e-6)` versus `log(max(luma, 1e-6))`) among the stated response settings.

Both checkpoints were evaluated on the same event tensor at each episode/source pair; the input tensor hashes are recorded in each episode JSON. The models receive events only. Trace GT is joined after inference for scoring. No physics is advanced, no new rollout is created, and frozen simulator core files are unchanged. Direct rendering here is an offline cached-state diagnostic, not real event-camera truth.

## Interpretation and next fix

The scene projection, image coordinate conversion, and label join are unlikely to explain the large Ours-only errors in this sample: V2E predicts locations from the same tensors, and Ours is close on its own sparse input. The leading hypothesis is a domain mismatch caused by the Ours temporal reconstruction/event cadence and the resulting training distribution. The 100 Hz seam bursts are measured; their causal role is not yet isolated.

Do not change the predictor, camera, labels, or robot controller to compensate for these selected errors. The next bounded check is to compare event generation at several temporal cadences from the same cached HDR trajectory and measure whether Ours localization moves smoothly with the input distribution. If correcting the seam behavior requires changing `genesis_event_plugin`/the frozen interpolation path, bring this reproduction and a minimal patch proposal to the research review before editing core. A benchmark-layer cadence experiment or retraining on a newly justified Ours event path can follow after that boundary is resolved. No new training was done in this diagnostic.

## Reproduction and provenance

- Branch: `exp/robot-closed-loop-v1`
- Diagnostic code commit on g1: `458295f62cbe396ee861cc5ebbacc0cbbfff9cb0`
- Config: [`r1_ours_localization_cadence_ablation_20261019c.json`](../../configs/r1_ours_localization_cadence_ablation_20261019c.json), SHA256 `6845b34d6f22939dc5761c38cf82b0546ac70a907e78866131c05edf32171a1f`
- Scene SHA256: `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`
- Shared control SHA256: `3d289c2bef1c562b26f9b713c2a826b813a286a1f7e21f9ead0a01c63c5e0d80`
- Direct-HDR reference manifest SHA256: `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`
- Source state traces SHA256: episode 004 `caa5b7b7d8147c032590b8391112ef833bea47d34b07e1f533b92b4351194947`; episode 005 `310837ea07ab2f819f590ccce91b3952dd9e85ea5005537bc1f4d1a339bff28c`
- Checkpoints SHA256: Ours native HDR `6ecea6a4520ae09b8cc99ebfd8da7a6a324363d2e57dbf73790e27ee2dbec8d2`; V2E `a7140b63f3b5042b0dc0ed7cc3aa2974c4e9859db919f318826358883f7ce8a0`
- Summary SHA256: `c84b277be811bcc9ba083611608ea9fb10a89f8ed2e9302f5e0385c50c36c8b4`; CSV SHA256: `837b3b7de8113c8d029ea7150a2d5551948f2038b3b2b504fd34d8f7930fda00`; output `SHA256SUMS` SHA256: `4b95e03e93fcef4af78b535dfdc2884ce1de9ecf9ee7aad15cfa998168baeafe` (all listed files verify).
- Runtime log: [`r1loc_cadence_20261019c.log`](../runtime_logs/r1loc_cadence_20261019c.log), SHA256 `e1142c68792e0c79b1c9879445832274d9006aad5fd9d3c8ffcb36c5492ca24e`; exit 0. Script summary wall time 84.0 s; `/usr/bin/time` elapsed 91.13 s, peak RSS 1,724,560 KB. The 100 Hz plugin took 2.37/3.49 s per episode; exact-row HDR render took 0.99/1.25 s per episode, and direct event emulation 0.45/0.55 s.
- Portable server bundle SHA256: `fb0909b694314286e07a2ccfe1ac2d1bcec781dfe4096372a77583fe2e2c8d91`.

Run on g1 from the diagnostic worktree (the original R1 worktrees and source data are left untouched):

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-localization-cadence-20261019b
/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python \
  benchmark/robot_closed_loop/diagnostics/diagnose_r1_ours_localization_cadence.py \
  --config benchmark/robot_closed_loop/configs/r1_ours_localization_cadence_ablation_20261019c.json \
  --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --control-config benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json \
  --reference-root /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016 \
  --ours-checkpoint /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-nativehdr-20261019a/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/ours_fit/spatial_event_state.pt \
  --v2e-checkpoint /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-nativehdr-20261019a/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/v2e_fit/spatial_event_state.pt \
  --output benchmark/robot_closed_loop/results/r1_ours_localization_cadence_ablation_20261019c
```
