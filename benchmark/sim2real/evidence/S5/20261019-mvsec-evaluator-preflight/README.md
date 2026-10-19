# MVSEC evaluator checkpoint binding preflight

The candidate real-score entry remained closed. The `--self-test` command used only hand-built arrays and temporary checkpoint/artifact files. It confirmed:

- EPE/AEE = 2.5 px for hand-computed vector examples;
- outlier definition (EPE > 3 px and >5% GT magnitude) = 25%;
- top-190 crop includes row 189 and excludes row 190;
- an empty common event-active/GT region is reported undefined;
- artifacts are SHA-256 checked and each `checkpoint_sha256` is bound to the actual `manifest.checkpoint_binding` weight file;
- a stale checkpoint hash is rejected;
- real scoring remains closed without review-linked authorization.

Command: `python benchmark/sim2real/scripts/stage4_mvsec_dt1_metrics.py --self-test`.
