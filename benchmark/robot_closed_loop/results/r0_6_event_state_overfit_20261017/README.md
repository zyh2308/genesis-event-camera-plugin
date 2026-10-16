# R0.6 tiny event-to-state overfit diagnosis

This run reuses three previously saved g1 GT-shadow Direct-HDR event trajectories: episode 0 (0.30 m/s, y=-55 mm), episode 4 (0.40 m/s, y=0), and episode 8 (0.50 m/s, y=+55 mm). No physics or HDR rendering was run for this diagnostic.

## What was fit

A 16,758-parameter 2D CNN receives only `log1p` of the causal 5×5 ms, 2-polarity, 64×64 event-count tensor at 10 ms controller cadence and 10 ms observation latency. It regresses world xyz position and velocity at the observation cutoff. Ground-truth object pose/velocity is used to build supervised labels and score predictions; it is not a feature. No RGB, GT segmentation/depth, or future state is fed to the model. The same three episodes were used for fitting and scoring, so these are strictly in-sample overfit numbers. There is no episode holdout, controller integration, or task-success result.

| Phase | Windows | XY position RMSE / P95 | XY velocity RMSE / P95 |
|---|---:|---:|---:|
| Pre-contact | 197 | 12.2 / 25.6 mm | 0.0126 / 0.0268 m/s |
| Contact to lift start | 21 | 18.9 / 32.9 mm | 0.0517 / 0.1001 m/s |
| Post-lift | 199 | 10.7 / 18.4 mm | 0.0181 / 0.0286 m/s |
| All phases | 417 | 11.9 / 22.4 mm | 0.0191 / 0.0327 m/s |

The fit establishes that this small network can encode the saved event trajectories under the current labels. Its 22.4 mm overall XY position P95 and 32.9 mm contact-to-lift P95 are above the current controller's useful tolerance, and the samples are reused for training; do not interpret them as heldout perception performance. The diagnostic network also has no calibrated confidence/status output and is not connected to the controller.

## Runtime and provenance

- Actual code commit: `1e2ddc98907138b21225306cdea3c4b5882111a2`.
- Config SHA256: `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`.
- Source raw-event `SHA256SUMS` SHA256: `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`.
- Selected raw event hashes: episode 0 `1427e38cea39dde2baa581dd8b2809abe7b5c6e99422ad38ab48c0ce2074ca54`; episode 4 `354b13c1b13c1ca9e085ae2768a53fecb606ad410d9cc5d9b529393ab93132a6`; episode 8 `e5207aa6d7d678bea2ad55d28c840956e4e40987bb6ef86287c96d53ffb02dca`.
- Source raw inputs were verified against the g1 manifest. Output `SHA256SUMS` covers the saved checkpoint, per-window predictions, and run manifest.
- CPU run used PyTorch 2.8.0+cpu, 4 threads, 120 epochs, batch size 64, seed 606. Training wall was 30.03 s; full Python process wall/user CPU/system CPU/max RSS was 38.12/114.38/9.34 s / 779,712 KB.
- Batch-1 model-only inference latency on CPU was mean/p50/p95 0.857/0.531/0.716 ms. It excludes event binning, rendering, DVS, physics, control, and I/O.
- g1 result path: `~/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_6-overfit/benchmark/robot_closed_loop/results/r0_6_event_state_overfit_20261017/`.
- `run_manifest.json` SHA256 `e1ffe514316cd6b8e6c81ded7212524fd22ebad1bea4a2e1eb745e67479a393d`; `tiny_event_state_overfit.pt` `972ec1b4359699263d097aaac203263ddf9a2d93dd94f5ff730e51b89d72561e`; `per_window_predictions.csv` `f82c090effc249a81b53076dd462d46e615e3e5b7478302fce54a6691c143791`; result `SHA256SUMS` `942661787926e14e44365256ca6120a21dc5d36deadc3f2d533d0e00b9884b1d`.

The saved predictions and model support later debugging only. They do not establish generalization, event-controller success, or Ours/V2E ranking.
