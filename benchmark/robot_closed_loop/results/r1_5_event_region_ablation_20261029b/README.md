# R1.5 archived-event regional counterfactuals

## Question and result

The archived 2 kHz event stream gives the old Ours checkpoint a mean XY error
of 58.9 mm on episode 005 (+55 mm), while the fresh direct-render, shadow-on
stream gives 10.6 mm at the same seven exact query cutoffs. This CPU-only
analysis asks which broad image regions contribute to that difference. No
physics or rendering ran.

Events were assigned to target, robot, floor, or unknown using the nearest
saved 10 ms segmentation mask. The moving target body ID was selected from
cached pose displacement. Segmentation was used only to filter events for
counterfactuals; the unchanged checkpoint still received only its five causal
5 ms event bins. These masks are evaluator-only and are not a proposed sensor
input.

| Episode 005 input | Mean XY error | Mean events per 25 ms query |
|---|---:|---:|
| Archived stream, all regions | 58.9 mm | 3,810 |
| Fresh direct stream, all regions | 10.6 mm | 2,875 |
| Archived stream with target-region events removed | 17.4 mm | 3,673 |
| Fresh direct stream with target-region events removed | 17.2 mm | 2,655 |
| Archived background plus fresh target events | 17.6 mm | 3,893 |
| Fresh background plus archived target events | 14.7 mm | 2,792 |

The counterfactuals show that the error is sensitive to how regions combine.
No one region can be named as the root cause from this test: replacing any
one region with its fresh-render counterpart reduces the +55 mm error, and
the full region-source factorial in the companion result further separates
target, robot, and floor contributions. In the center episode, the all-archive
error is 11.0 mm and the tested regional mixes stay about 10.2–17.7 mm. This
is an old-checkpoint development diagnosis, not a general performance claim.

## Reproduction and provenance

Run from the repository root:

```bash
python3 benchmark/robot_closed_loop/diagnostics/analyze_r1_5_archived_event_region_ablation.py \
  --reference-root benchmark/robot_closed_loop/results/r1_5_consistency_candidates_20261019b_realrender_g1 \
  --direct-root benchmark/robot_closed_loop/results/r1_5_shadow_perception_causal_replay_20261029e \
  --cache-root benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \
  --checkpoint benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/paired_r1/ours_fit/spatial_event_state.pt \
  --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --output benchmark/robot_closed_loop/results/r1_5_event_region_ablation_20261029b
```

- Analysis source commit: `2e59dc378b569a9e922e2e9606f8e7cc5b8bafb4`.
- Script SHA256: `4779b831080791e68c46dc93a3b4a9c6cbc70f5d49992716a06d284ef47f388b`.
- Old Ours checkpoint SHA256: `d391fec293e2c6c50cdb267e15f742ed09e77e7b28e56a79fdf700f633e804e9`.
- Query times: 0.5855, 0.5860, 0.5865, 0.5875, 0.5885, 0.5895, and 0.5905 s in each episode.
- `summary.json` contains source, cache, checkpoint, scene-config, and code-commit hashes; `per_query_region_ablation.csv` retains every query and counterfactual.
- CPU analysis time: 1.83 s. The segmentation labels are sampled at 10 ms, so region assignment around moving boundaries is approximate.

Verify this directory with `sha256sum -c SHA256SUMS` from this directory.
