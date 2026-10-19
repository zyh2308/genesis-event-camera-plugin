# MVSEC evaluator checkpoint binding preflight

The candidate real-score entry remained closed. The `--self-test` command used only hand-built arrays and temporary checkpoint/artifact files. It confirmed:

- EPE/AEE = 2.5 px for hand-computed vector examples;
- outlier definition (EPE > 3 px and >5% GT magnitude) = 25%;
- native full-frame and legacy crop contracts are checked separately: legacy `main_steflow_dt1.py` center-crops native `y=[2,258), x=[45,301)` to 256×256, then `flow_error_dense(is_car=True)` keeps the first 190 cropped rows, equivalent to native `y=[2,192), x=[45,301)`;
- an empty common event-active/GT region is reported undefined;
- artifacts are SHA-256 checked and each `checkpoint_sha256` is bound to the actual `manifest.checkpoint_binding` weight file;
- a stale checkpoint hash is rejected;
- real scoring remains closed without review-linked authorization.

Command: `python benchmark/sim2real/scripts/stage4_mvsec_dt1_metrics.py --self-test`. The refreshed ROI-aware result is `synthetic_self_test_v2.json`; the earlier `synthetic_self_test.json` is retained unchanged.

The paper defines AEE over active pixels where GT and events are both present, but does not name a numeric rectangular ROI. The candidate primary recommendation is full native 346×260 for every sequence, matching native synthetic training and the real adapter. The exact legacy ROI is recorded for compatibility diagnosis; the difference is explicitly pending research freeze. No real score is authorized.
