# R0.8.9 paired Ours/V2E R1 pipeline smoke

This is a two-episode pipeline contract check, not a ranking or generalization result. Code commit `e13b0b307c464e3f753cd3bacaeb092ee5e22586` ran on g1. Both generators consumed the same physical/robot rollout, frame timestamps, labels, and episode-level split. See `paired_pipeline_smoke.json` for full provenance and metrics.

The versioned subset includes both generator inputs, per-episode state/control traces, generated event outputs, small fit checkpoints, training/validation predictions, logs, and manifests. Full HDR/RGB/depth/segmentation frame caches and V2E runtime compatibility files are intentionally not copied into this subset; their hashes remain in `cache_sha256_manifest.json` and the original full `SHA256SUMS`. The complete run is retained on g1 at:

`/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_8-final-oracle-20261017/benchmark/robot_closed_loop/results/r0_8_paired_pipeline_smoke_20261017c/`

The g1 `SHA256SUMS` covers the complete 489-file result. In this checkout, `SHA256SUMS.portable` verifies the 64 files in the portable subset. The R2 evaluation uses a separate shared Direct-HDR ideal-DVS event source; the two training generators do not share a pixel-response model (see `paired_pipeline_smoke.json`).

The original launch command is preserved in `benchmark/robot_closed_loop/results/runtime_logs/r0_8_paired_pipeline_smoke_20261017c.launch.sh`; to repeat it, choose a fresh output directory and update `--output`.
