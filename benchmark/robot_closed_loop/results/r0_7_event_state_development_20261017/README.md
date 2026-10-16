# R0.7 Direct-HDR learned event-state development fit

This is a small episode-level development split, not a held-out scientific test, generator comparison, or model ranking. All nine R0.5 GT-shadow episodes remain development material.

## Provenance

- Training code commit: `64d5de94f4708333246195a5741fec489b57a6d0` on `exp/robot-closed-loop-v1`.
- Scene config SHA256: `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`.
- Source GT-shadow manifest/SHA256SUMS SHA256: `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`.
- Source episodes on g1: `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/`.
- Split: train episodes `0,2,3,4,7,8`; validation episodes `1,5,6`. Each split covers all three speeds and all three lateral offsets, with different speed-position combinations. Windows from one episode never cross the split.
- The source run directory stores the exact event/trace/episode CSV SHA256 for each used episode in `development_summary.json`. Raw events remain at the g1 source path above.

## Result

The 145,438-parameter spatial CNN was phase/episode reweighted and early-stopped at epoch 116. It ran on CPU with two threads; training took 254.76 s, full Python wall time was 264.91 s, and maximum RSS was 1,202,984 KB.

Validation has 417 windows: 195 pre-contact, 22 contact-to-lift, and 200 post-lift. XY position P95 is 49.2 mm overall, 59.4 mm pre-contact, 24.5 mm contact-to-lift, and 30.1 mm post-lift. XY velocity P95 is 0.235 m/s overall, 0.242 pre-contact, 0.125 contact-to-lift, and 0.0465 post-lift. The count/bin validity rule returned fresh for 417/417 windows, with 10 ms causal state age; this is event-window availability, not accuracy confidence. Under the same fixed predictor, pre-contact absolute arrival-time error P95 is 376.2 ms and lateral intercept error P95 is 11.6 mm.

The contact-to-lift location result improves over the earlier checkpoint’s six-trajectory diagnostic reported by the research session (94.7 mm), while pre-contact timing remains poor. This checkpoint is therefore a development candidate for failure-localization rollouts, not a pass or freeze candidate.

## Artifacts and hashes

- Checkpoint `spatial_event_state.pt`: `49b91d4d89d320d2267a332e0dd8b15de692a7fed37fd8828a2e67c0f79cb4dc`.
- Result `SHA256SUMS`: `5886dd065b6ee9a0b7b228b8845f5083fbe00c086ae694cb287a6b1f852dfc36`.
- `development_summary.json`, `run_manifest.json`, `train_predictions.csv`, `validation_predictions.csv`, and `training_history.json` are listed in `SHA256SUMS`.
- Training runtime log SHA256: `06db67970b7c76912528b24142c89b3168e1c0502994dd7dfff51d85dba9e3b5`; runtime SHA256: `d142e2b96076e5909329d5cdd18c9a706f241ba5e161be739761171d78cfe4a7`; exit file SHA256: `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
- A failed pre-fit launch due to the old CSV field names is retained as `results/runtime_logs/r0_7_event_state_development_20261017.attempt1.*`; it exited 1 before training. The corrected run exited 0.

Reproduction launcher: [`../runtime_logs/r0_7_event_state_development_20261017.launch.sh`](../runtime_logs/r0_7_event_state_development_20261017.launch.sh). It uses a new output directory for each run and the g1 source path above.
