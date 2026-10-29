# R1.5 region-source factorial counterfactual

This CPU-only extension uses the same archived and fresh direct-render event
streams, old checkpoint, two episodes, and seven exact queries as
`r1_5_event_region_ablation_20261029b`. For each query it independently
selects target, robot, and floor events from either the archived stream or the
fresh shadow-on stream, evaluating all eight combinations. No simulation or
rendering is run.

## Result

| Episode 005 region source (target / robot / floor) | Mean XY error |
|---|---:|
| archived / archived / archived | 58.9 mm |
| fresh / archived / archived | 17.6 mm |
| archived / fresh / archived | 31.7 mm |
| archived / archived / fresh | 18.5 mm |
| fresh / fresh / archived | 11.9 mm |
| fresh / archived / fresh | 10.7 mm |
| archived / fresh / fresh | 14.7 mm |
| fresh / fresh / fresh | 10.6 mm |

All eight region combinations in the center episode remain within 10.2–11.3 mm.
The +55 mm failure therefore depends on source-specific event structure across
multiple regions in this old checkpoint. The factorial does not identify
which renderer property creates that structure: source, sample cadence, and
spatial/temporal event interactions remain confounded. Mask assignment uses
the nearest 10 ms segmentation frame and is diagnostic-only; masks never enter
the network.

## Reproduction and provenance

The command is identical to the companion ablation except for the output path:

```bash
python3 benchmark/robot_closed_loop/diagnostics/analyze_r1_5_archived_event_region_ablation.py \
  --reference-root benchmark/robot_closed_loop/results/r1_5_consistency_candidates_20261019b_realrender_g1 \
  --direct-root benchmark/robot_closed_loop/results/r1_5_shadow_perception_causal_replay_20261029e \
  --cache-root benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \
  --checkpoint benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/paired_r1/ours_fit/spatial_event_state.pt \
  --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --output benchmark/robot_closed_loop/results/r1_5_event_region_source_factorial_20261029a
```

- Analysis source commit: `917ea5daa23dc8d2e0b9aa661ffd7f58083a7d59`.
- Script SHA256: `fdfa8f34910c07e1dac0321aa6bba60de2cc177f20ae329f22a2492b5d538c4f`.
- Old Ours checkpoint SHA256: `d391fec293e2c6c50cdb267e15f742ed09e77e7b28e56a79fdf700f633e804e9`.
- The exact input hashes and per-query results are in `summary.json` and
  `per_query_region_ablation.csv`.
- CPU analysis time: 2.85 s.

Verify this directory with `sha256sum -c SHA256SUMS` from this directory.
