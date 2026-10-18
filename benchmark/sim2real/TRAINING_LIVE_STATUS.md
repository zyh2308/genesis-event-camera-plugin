# Stage 4 paired flow training: live status

**Live snapshot:** 2026-10-18 01:59:27 UTC / 10:59:27 JST, read from g1 training logs while both jobs were active. Earlier status records in Git history are historical, not current observations.

## Current run

- Campaign: 20261017-stage4-flowdt1-seed0-corrected-v5; 45 epochs × 198 optimizer updates = 8,910 per generator.
- EventGenesis: update **4,871 / 8,910**; latest complete synthetic validation epoch 24 at update 4,752, EPE **0.2838378343** px/30ms; best **0.2838365008** at epoch 19.
- V2E: update **4,948 / 8,910**; latest complete synthetic validation epoch 24 at update 4,752, EPE **0.2831938486** px/30ms; best **0.2828732346** at epoch 20.
- Zero-flow full-validation EPE: **0.2842080520** px/30ms for both. Model selection remains synthetic-validation-only.
- Supervisor tmux stage4-flow-corrected-20261017-v7 PID 3590105; trainer PIDs 3594965 (EventGenesis, GPU 2) and 3594967 (V2E, GPU 4) were alive. GPU 2/4 utilization was 100%/100%; the isolated V2E resume supplement has completed and GPU 1 is idle again.
- Both groups remain on the registered data, model, configuration, optimizer budget, sampling, augmentation and checkpoint selection protocol. Synthetic-only training; real-data scoring disabled.

## Interpretation

The fixed synthetic diagnostic found near-zero predictions on most valid pixels and full-region EPE essentially equal to zero flow. It also found modest improvement on the common event-active union and the small independent-object foreground. No confirmed timing, label, unit, output-crop or paired-data contract error was found. A separate V2E update-198 to update-199 actual resume and forward-loss oracle also passed. Current learning is weak; this is not evidence that the run is invalid. Continue the registered budget without adjustment.

See [STAGE4_MOTION_DIAGNOSTIC_20261018.md](STAGE4_MOTION_DIAGNOSTIC_20261018.md) and [evidence/S4/20261018-motion-diagnostic/](evidence/S4/20261018-motion-diagnostic/) for fixed masks, per-stratum results, source/data/checkpoint hashes, commands and logs.

## External log snapshot

Full growing logs remain on g1 under /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/full_training_retry1/{eventgenesis,v2e}/training_metrics.jsonl. At 01:59:27 UTC, their sizes and hashes were:

- EventGenesis: 3,980,171 bytes; SHA256 a6f57f3f7cda4d2de184e91a3884137655b864bc666594eeca86599c7157d114.
- V2E: 4,035,035 bytes; SHA256 55d7312f4901737b3116591951512f84b912ec80d56944cac153cb06ca2c0e96.

These are append-only live snapshots and will change before the final report. The full training remains supervised by the existing detached tmux session.
