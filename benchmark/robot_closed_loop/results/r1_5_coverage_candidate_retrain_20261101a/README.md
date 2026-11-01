# R1.5 coverage candidate: retrain and common-reference replay

## Result

The end-to-end candidate-event pipeline completed on g1 using CPU only. A new `heatmap_v2` model was trained from the coverage-splat candidate events, then replayed on the same nine Direct-HDR development trajectories and the same 581 query times as the previous Ours and V2E checkpoints.

The candidate model did **not** restore stable localization. Its position-error tail improved over the previous Ours model, but its typical position error, intercept prediction, arrival prediction, and reachable-decision coverage became worse. It remains far behind the fixed V2E model on this reference. This is a development comparison on previously inspected episodes, not a held-out test or a formal ranking.

| Model | XY position error, median / P95 (mm) | XY velocity error, median / P95 (m/s) | Intercept XY error, median / P95 (mm) | Absolute arrival error, median / P95 (ms) | Correct reachable decisions |
|---|---:|---:|---:|---:|---:|
| New Ours trained on coverage-candidate events | 63.6 / 338.9 | 1.254 / 4.789 | 41.6 / 216.2 (`n=223`) | 266 / 830 (`n=223`) | 223 / 581 (38.4%) |
| Previous native-HDR Ours checkpoint | 14.4 / 444.9 | 0.340 / 7.667 | 15.7 / 92.2 (`n=315`) | 177 / 746 (`n=315`) | 315 / 581 (54.2%) |
| Fixed V2E checkpoint | 4.1 / 11.3 | 0.108 / 0.297 | 11.1 / 50.9 (`n=514`) | 87.9 / 603 (`n=514`) | 514 / 581 (88.5%) |

All three estimators returned a fresh state for all 581 polls, with 10 ms state age. The large errors therefore come from incorrect state estimates and reachability decisions, not missing or stale outputs. Intercept and arrival errors are reported only where both the estimated state and the GT state were classified as reachable; the `n` values expose that reduced coverage. Arrival uses the same fixed predictor on GT at the causal cutoff, not measured plane-crossing time.

The candidate lowered overall position P95 from 444.9 mm to 338.9 mm, but raised the median from 14.4 mm to 63.6 mm. Across the three speed bins, candidate position P95 was 359.1 / 327.0 / 253.3 mm at 0.30 / 0.40 / 0.50 m/s; previous Ours was 428.3 / 408.9 / 460.7 mm, while V2E was 12.5 / 11.2 / 8.4 mm. The improvement in the Ours tail is not a stable recovery.

## Training and scope

- Training source: the diagnostic front-to-back coverage-splat candidate, generated from the existing nine cached Genesis trajectories. Candidate events total 5,396,955 across the nine episodes.
- Split and budget: train episodes `[0,1,2,3,7,8]`; validation episodes `[4,5,6]`; `heatmap_v2`; seed `606`; 40 epoch cap/patience; batch size `32`; two CPU threads. The best checkpoint was epoch 40. Validation position-error P95 was 50.5 mm across 566 windows.
- The candidate still uses the existing endpoint/hold photometry rule. This run does not test or repair missing intermediate brightness changes such as moving-shadow radiance.
- Geometry/GT is used only within event generation and as training/evaluation labels. Inference input remains causal five-bin events. The frozen simulator core was not changed.
- Generation took 1,508.5 s; training optimization took 548.0 s; common-reference evaluations took 18.2 s for the candidate and 13.4 s for the previous Ours checkpoint. No GPU was used because no GPU allocation was available.

The first orchestration attempt exited with code 1 after training because the evaluation script expected Git metadata inside the selective deployment snapshot. The checkpoint was intact. Commit `8e09ea1` added an explicit execution-commit fallback; both evaluations were then rerun into separate `retry1` directories and completed against the same reference manifest. The original failure output was retained.

## Reproduction and artifacts

The tracked generator/training entrypoint is [run_r1_5_coverage_candidate_retrain.py](../../diagnostics/run_r1_5_coverage_candidate_retrain.py); fixed settings are in [the experiment config](../../configs/r1_5_coverage_candidate_retrain_20261101a.json). The evaluation metadata fix is in [evaluate_r1_common_direct_hdr.py](../../diagnostics/evaluate_r1_common_direct_hdr.py).

On g1, the full output is at:

`/home/zhaoyuhan/Eventbased_WAM/runs/robot_r1_5_coverage_candidate_retrain_20261101a/`

This directory contains the trained checkpoint, training/validation predictions and history, all nine raw candidate event archives, per-episode provenance, both per-poll evaluation CSVs, and the final report. The final JSON summary is also copied here as [summary.json](summary.json).

Key SHA256 values:

- Training source commit: `6eb9dbf764edf93df0d968a7ded862b86cafdb49`
- Evaluation fix commit: `8e09ea19b7fd3da24b0a7c9473ed5b6ec0efc298`
- Experiment config: `3151d23bd18988b97172eee436a28e6d62b2ecfef34505d206e3adc5913bce00`
- Candidate event source manifest: `c640c5a61016b6ea532d976331372f6c55924d0688fab6f7f2f338a64afbcc44`
- Candidate checkpoint: `eb671a19c70bb9f004cc4c630764f497d6c2ed9b4361ef7904665fddce3cd796`
- Previous native-HDR Ours checkpoint: `6ecea6a4520ae09b8cc99ebfd8da7a6a324363d2e57dbf73790e27ee2dbec8d2`
- Fixed V2E checkpoint: `bfc0b233aaa91887811f57b493eb9152395ce1ed61263d4863ed2249e4860e45`
- Common Direct-HDR reference manifest: `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`
- Candidate poll CSV: `687961d67c57d5faa5dd35289cb6bd92101d830ce4a26266ec10be13a93a6038`
- Final remote summary JSON: `1e376b8e017837a554e35ab859e78b6d3106cbc8f226fcd5da5e3b220d96a167`
- This tracked summary JSON: `1e376b8e017837a554e35ab859e78b6d3106cbc8f226fcd5da5e3b220d96a167`

Full per-episode metrics and source hashes are in `summary.json`; per-query data and raw candidate events remain at the g1 path above.
