# R1 small9 paired development run

This run completes the minimum R1 development chain: paired Ours/V2E event training with a matched ideal pixel response, followed by common evaluation on the same nine Direct-HDR reference episodes and query timestamps. It is development evidence, not formal held-out testing or a generator ranking.

## Protocol and provenance

- Code: `123369f4bd70a1fe039bf073a35333d0d30ead19` on `exp/robot-closed-loop-v1`.
- Experiment config: `r1_shared_ideal_small9_v1.json`, SHA256 `324363fe362c5ad3367e989d3839d4525ea87e148b1b1d3dc3244ead45426865`.
- Shared ideal response: `r1_shared_ideal_response_v1.json`, SHA256 `4b5bd964835b9b09559a12b127779bc05d371b68f1dc6a2e0d231d148c953a72`; V2E overlay SHA256 `8abe329a0013c5cd47dd3050576656cacfa9b15e487c6e8f8a1b9ff6d7ac1505`. Official V2E default YAML was not changed.
- Shared rollout episodes: nine; train IDs `[0,1,2,3,7,8]`, validation IDs `[4,5,6]`. Both fits used `heatmap_v2`, seed `606`, 40 epochs, batch 32, and the same 1203 training windows.
- Common reference: old nine-episode Direct-HDR development set, manifest SHA256 `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`; 2 kHz event stream and 100 Hz query cadence. Both models consumed all 581 identical contact-pre queries.
- Arrival/intercept references use the same fixed predictor applied to evaluation GT at each causal cutoff. They are not measured plane-crossing ground truth.
- Frozen simulator core and official V2E defaults are unchanged. Ours and V2E retain separate temporal reconstruction paths but use the same archived grayscale endpoints, timestamps, lin-log transfer, threshold, controllable noise/bandwidth settings, labels, split, and training budget.

## Results

Both models had fresh state at all 581 queries (10 ms age). All 581 evaluation-GT queries were classified reachable by the fixed predictor, so true-unreachable specificity has no denominator. Ours predicted reachable at 384/581 queries (66.1%); V2E at 514/581 (88.5%). Arrival/intercept errors are computed only for predicted-reachable queries.

| Model | XY position error mean / P95 | XY velocity error mean / P95 | Intercept XY error mean / P95 | Absolute predictor TTC error mean / P95 |
|---|---:|---:|---:|---:|
| Ours | 52.8 / 363.6 mm | 1.008 / 5.278 m/s | 29.6 / 91.7 mm | 198 / 604 ms |
| V2E | 4.8 / 11.3 mm | 0.127 / 0.297 m/s | 17.1 / 50.9 mm | 166 / 603 ms |

The per-speed and per-episode results are preserved in `common_reference_eval/r1_speed_metrics.csv`, `common_reference_eval/r1_episode_metrics.csv`, and the full evaluator JSON. Ours has a clear failure cluster at launch `y=+55 mm` at all three speeds: episode position-error means are 125–128 mm and reachable coverage is 41–51%. V2E does not show the same cluster in this small development set. This identifies a transfer/edge issue to diagnose; it is not a model or generator ranking.

## Cost and artifacts

Cached rollouts were reused in the successful run. Paired event generation from cache took `2930.69 s`; equal-budget fits took `1136.13 s`; measured wrapper wall time was `1:07:49`, with peak RSS `5,086,332 KB`. Common reference evaluation took `9.87 s`. The seven new rollouts had been captured in a separate preceding run in `183.7377 s`; that cost is not included in the paired-run timer.

The portable repository subset includes both checkpoints, full per-query CSV, per-episode/per-speed summaries, fit summaries, training histories, source hash manifests, runtime metadata, and the run log. The full training event NPZs and cached rollout HDF5/frames remain on g1 at:

`/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/`

The g1 `paired_r1/SHA256SUMS` manifest was checked with `sha256sum -c` and every listed artifact passed. Source cache hashes are in `paired_r1/cache_input_hashes.json`; training event hashes and source manifests are preserved under `paired_r1/ours_training/` and `paired_r1/v2e_training/`. The common reference lives at `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/`.

Verify the committed portable subset from the repository root with `sha256sum -c benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/PORTABLE_SHA256SUMS`.

## Reproduction

Use a new run ID to avoid overwriting this result. The existing immutable cache contains episodes 000–008:

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-ba137bd
tmux new-session -d -s r1_small9_repro_20261018 \
  'env R1_REUSE_CACHE_ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018a/trajectory_cache ROBOT_PYTHON=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python V2E_PYTHON=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python /bin/bash benchmark/robot_closed_loop/diagnostics/run_r1_small9_development.sh r1_shared_ideal_small9_20261018c'
```

The detailed stage record is in [`robot_status.md`](../../../../docs/coordination/robot_status.md#r1-small9-共同响应配对训练与共同-direct-hdr-开发评估g1-2026-10-18).
