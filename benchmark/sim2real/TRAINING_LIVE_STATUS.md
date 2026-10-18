# Stage 4 paired flow training: live status

**Live snapshot:** 2026-10-18 01:52:44 UTC / 10:52:44 JST, read from g1 training logs while both jobs were active. Earlier status records in Git history are historical, not current observations.

## Current run

- Campaign: 20261017-stage4-flowdt1-seed0-corrected-v5; 45 epochs × 198 optimizer updates = 8,910 per generator.
- EventGenesis: update **4,761 / 8,910**; latest complete synthetic validation epoch 24 at update 4,752, EPE **0.2838378343** px/30ms; best **0.2838365008** at epoch 19.
- V2E: update **4,836 / 8,910**; latest complete synthetic validation epoch 24 at update 4,752, EPE **0.2831938486** px/30ms; best **0.2828732346** at epoch 20.
- Zero-flow full-validation EPE: **0.2842080520** px/30ms for both. Model selection remains synthetic-validation-only.
- Supervisor tmux stage4-flow-corrected-20261017-v7 PID 3590105; trainer PIDs 3594965 (EventGenesis, GPU 2) and 3594967 (V2E, GPU 4) were alive. GPU 2/4 utilization was 97%/98%; GPU 1 was idle immediately before the separate V2E resume supplement.
- Both groups remain on the registered data, model, configuration, optimizer budget, sampling, augmentation and checkpoint selection protocol. Synthetic-only training; real-data scoring disabled.

## Interpretation

The fixed synthetic diagnostic found near-zero predictions on most valid pixels and full-region EPE essentially equal to zero flow. It also found modest improvement on the common event-active union and the small independent-object foreground. No confirmed timing, label, unit, output-crop or paired-data contract error was found. Current learning is weak; this is not evidence that the run is invalid. Continue the registered budget without adjustment.

See [STAGE4_MOTION_DIAGNOSTIC_20261018.md](STAGE4_MOTION_DIAGNOSTIC_20261018.md) and [evidence/S4/20261018-motion-diagnostic/](evidence/S4/20261018-motion-diagnostic/) for fixed masks, per-stratum results, source/data/checkpoint hashes, commands and logs.

## External log snapshot

Full growing logs remain on g1 under /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/full_training_retry1/{eventgenesis,v2e}/training_metrics.jsonl. At 01:52:44 UTC, their sizes and hashes were:

- EventGenesis: 3,886,531 bytes; SHA256 4f283aa9b8f7168e3a8308cdb5a3146fa3d2d144c2e2015fd10d13f9eed90dc5.
- V2E: 3,939,208 bytes; SHA256 2654724064b9b90943352429e29b2fec58dac6eb4b9822fef2fda4b2df18e5c6.

These are append-only live snapshots and will change before the final report. The full training remains supervised by the existing detached tmux session.
