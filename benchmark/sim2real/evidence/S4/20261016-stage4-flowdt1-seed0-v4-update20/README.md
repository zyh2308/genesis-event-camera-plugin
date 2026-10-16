# Stage 4 paired flow seed-0 first checkpoint

This evidence bundle records the first delivery gate requested for Stage 4:
both STE-FlowNet dt1 arms completed at least 20 real optimizer updates and
saved checkpoints that the running process restored and checked. The full
45-epoch training continued after this snapshot. These results are not a
generator ranking or a Sim-to-Real claim.

## Run identity

- Server: `g1.pkuml.org`; project checkout: `/home/zhaoyuhan/Eventbased_WAM`.
- Run ID: `20261016-stage4-flowdt1-seed0-v4`.
- Started: `2026-10-16T16:34:32.177531Z` (EventGenesis metadata; V2E started in the same launch).
- Training source commit: `1fab7e17e201943c77703aa8a1c0c6eaefbb8f5a` on `exp/public-sim2real-v1`.
- Runtime status-monitor correction: commit `ea64262` (does not change training code or configuration).
- Config: `configs/stage4_training_launch_v1_torch_cuda.json`; SHA-256 `c0ab0d17f024b00106957c3e44d7a013e920f6e760f70266c43a9cf904634685`.
- Data adapter manifest: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/20261016-stage4-flowdt1-seed0-v4/steflow_30ms/training_bank_manifest.json`; SHA-256 `c9da4158d9775997f08dd75ce0ac2ccc33873d983422ef87c931e834f6da0624`.
- Shared initialization SHA-256: `acc6bbd11488a21868c5f147c7f397521480166c24914cc7988a6428d6ed71cf`.
- Objective: official multiscale Charbonnier photometric loss plus 10× second-order smoothness. GT flow was not used in the optimizer. Real data was not loaded.
- Budget: seed 0, 45 epochs × 198 optimizer batches = 8,910 updates, effective batch 8, Adam `4e-4`.

## Update-20 evidence

`eventgenesis_training_metrics_updates_1_20.jsonl` and
`v2e_training_metrics_updates_1_20.jsonl` are immutable snapshots of exactly
updates 1–20. Every record has batch 8, finite loss components and finite
gradients (70 gradient tensors).

| Arm | Physical GPU | Update-20 loss | Photometric | Smoothness | Gradient L2 | Peak allocated through update 20 |
|---|---:|---:|---:|---:|---:|---:|
| EventGenesis | 2 (V100S 32 GB) | 1422.0523681640625 | 1410.0911865234375 | 1.1961129903793335 | 1081.757617230945 | 24,354,208,256 bytes |
| V2E | 4 (V100S 32 GB) | 1403.4554443359375 | 1397.507080078125 | 0.5948320627212524 | 941.1672061242301 | 24,354,208,256 bytes |

The optimizer objective is `photo + 10 × smoothness`. These early loss values
are diagnostics, not comparable performance scores.

| Arm | Update-20 checkpoint (g1) | Size | SHA-256 | Restore probe max abs |
|---|---|---:|---|---:|
| EventGenesis | `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/flow_dt1_seed0/eventgenesis/update_000020.pt` | 580,038,432 bytes | `f7eb58f5dae4b914050a38c5c68596e23a64372f68584fa5c1ba00e3c14c63ec` | `7.152557373046875e-07` |
| V2E | `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/flow_dt1_seed0/v2e/update_000020.pt` | 580,038,432 bytes | `ce2b5b076b9ba6df9af525a6ca478c8f3d0b20a37e7783de433986619d7cc763` | `6.705522537231445e-08` |

Both restore probes loaded model, optimizer, and scheduler state from the saved
checkpoint; model-state hashes matched. The error is the maximum absolute
difference between live and restored synthetic validation probe predictions.

## Evidence hashes and live files

- EventGenesis updates 1–20 SHA-256: `df3a31f61cba5fcc7696f88e432ec6e6127028ac8fa310740c5efe10c3708bdb`.
- V2E updates 1–20 SHA-256: `fe9efd54a88ccd579551e0c307b1df13b02aa7226d5ffc2cbaaebbed9d731b02`.
- Status snapshot at 58 updates per arm SHA-256: `adb1ee0b2963f0f00ea68225410138d3f59256b42697acd7f4f5ef6f8e89969b`.
- EventGenesis run metadata SHA-256: `08fd4ab3a9f7b9e1f4658e0e025f14dafac02101435d13450d207beef039b879`.
- V2E run metadata SHA-256: `ffe5a99bc92ccbd3c3bb7c25dcdf56182078d7182949a9cfea0ab9592d67cc86`.
- Full mutable metrics and raw logs remain on g1 under
  `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/flow_dt1_seed0/{eventgenesis,v2e}/`.
- Current human-readable status: [`../../../TRAINING_LIVE_STATUS.md`](../../../TRAINING_LIVE_STATUS.md).
- Recovery details: [`../../../STAGE4_V4_RECOVERY.md`](../../../STAGE4_V4_RECOVERY.md).

The g1 tmux session is `stage4_flowdt1_seed0_shape_fix`; its wrapper and exact
paired launch commands are preserved in `scripts/start_stage4_training_after_shape_fix_tmux.sh`
and `scripts/resume_stage4_after_shape_fix_server.sh`.

The launch invocation was:

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real
bash benchmark/sim2real/scripts/start_stage4_training_after_shape_fix_tmux.sh
```
