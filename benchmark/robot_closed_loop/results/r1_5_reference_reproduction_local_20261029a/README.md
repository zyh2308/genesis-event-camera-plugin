# R1.5 exact-seed reproduction and localization comparison

## Finding

The archived episode 005 is reproducible byte-for-byte from the original R0.5
runner on g1. The seed-matched rerun used the same saved scene configuration,
frozen core, installed Genesis overlay, physics step, and 2 kHz event callback.
All 1,459 state rows match; the complete saved state trace and complete raw
event NPZ have exactly the same SHA256 values as the archive. Thus the archived
+55 mm failure is not explained by a seed change, physical trajectory drift,
corrupt archive, or a changed installed renderer overlay.

Scoring the old Ours checkpoint with exact float64 timestamps and the exact
seven E query times reproduces the archived-vs-cached-shadow result. On the
archived source, position error is 58.69 mm mean (p95 108.33 mm); on E's fresh
shadow-on event source it is 10.64 mm mean (p95 17.19 mm). The per-query voxel
hashes match E's saved comparison. The auxiliary velocity head behaves in the
opposite direction: 16.55 mm/s mean error on archive and 37.37 mm/s on fresh
shadow-on. The observed source sensitivity is therefore strongest in spatial
localization, not a uniform degradation of every output.

This still does not identify whether E's improvement comes from its sparse
render times, multisample state, or another cached-render-path detail. The
next experiment in the current branch is a matched same-state diagnostic:
render both the exact 81-row 2 kHz sequence and E's exact 47-row subsequence,
then compare GL multisampling on/off at 2 kHz. It advances no physics and uses
the same direct HDR renderer and ideal DVS helpers as E.

## Reproduction identity

- Host/resource: g1, GPU0 RTX 3090, UUID
  `GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23`; reserved by this run only.
- Run ID: `r1_5_reference_replay_episode005_20261029a`; tmux session
  `r15ref00520261029a`; Python PID `1294798`; exit `0`.
- Start/end: `2026-10-29T06:20:15Z` / `2026-10-29T06:25:27Z` (312 s).
- Actual R0.5 simulation code commit: `2100645b0473c294760789bc4475b644f261d9cf`.
- `run.py` SHA256:
  `0b5b9fe1d00c74689575015f0cb81c3bfc711958a60d3387d468c70456a7b2b6`.
- `event_state.py` SHA256:
  `f87599c46755a06b3fe7afb09c45d1d2ecae9527937e1ef39ff463576871d322`.
- Scene config SHA256:
  `dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd`.
- Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1`.
- Runtime: Python 3.11.15, Genesis 1.2.2, CPU physics; seed `2026103425`,
  speed `0.4 m/s`, launch y `+55 mm`, physics dt `0.5 ms`, event rate `2 kHz`.
- The eight installed Genesis renderer overlay hashes in the new manifest
  match the archived R0.5 run manifest exactly.

## Exact comparisons

| Quantity | Result |
|---|---:|
| Exact shared state timestamps | 1,459 / 1,459 |
| Target position/velocity error across shared rows | exactly 0 |
| Panda qpos/qvel and TCP error across shared rows | exactly 0 |
| Contact-record mismatches | 0 |
| Full raw event NPZ hash, archive and rerun | `b698817a…a7b2b6` in both |
| Full state trace hash, archive and rerun | `310837ea…bff28c` in both |
| Events matched in 0.5505–0.5905 s, same pixel/polarity, ±0.5 ms | 6,182 / 6,182 |
| Causal event rows in 0.5605–0.5905 s, exact array equality | 4,579 / 4,579 |

On the seven exact E query timestamps, archived and fresh-shadow voxel hashes
match the already saved E comparison file exactly. The position summaries are
58.692 mm mean / 108.332 mm p95 for archive and 10.641 mm / 17.185 mm for fresh
shadow-on; validity is 7/7 in both. Velocity errors are 16.546 mm/s and
37.365 mm/s respectively. This is a diagnostic comparison with the old Ours
checkpoint, not a new model validation or ranking.

## Inputs and outputs

- Full g1 rerun directory:
  `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r1_5_reference_replay_episode005_20261029a/`
- Full rerun raw event file:
  `episodes/episode_000/direct_hdr_events_1khz_2khz.npz` (11,937,381 bytes,
  SHA256 `b698817a8347ca48ad75015f0cb81c3bfc711958a60d3387d468c70456a7b2b6`).
- Full rerun state trace:
  `episodes/episode_000/state_control_trace.jsonl` (4,789,732 bytes,
  SHA256 `310837ea07ab2f819f590ccce91b3952dd9e85ea5005537bc1f4d1a339bff28c`).
- Server `run_manifest.json`, `SHA256SUMS`, `preflight.txt`, `run.log`, exit
  record, and resource allocation are copied into `remote_metadata/`.
- The compact float64 event/label extraction and exact-query localization CSV
  and JSON are also retained locally. `compact_replay_evidence_v3.json` SHA256:
  `579df42dccdf60a8ae183decd68103ffc4560aecde44338d454f7f40be53bfc3`.
- The exact archived scene configuration is in `inputs/` (SHA256 above).
- The E fresh shadow-on event file used for the model comparison is the already
  committed result `r1_5_shadow_perception_causal_replay_20261029e`, SHA256
  `c46f7bc28504ed9f43e76c334533534291d0de64b0c0f9bd429fa1a49eab58d8`.
- Old Ours checkpoint SHA256:
  `d391fec293e2c6c50cdb267e15f742ed09e77e7b28e56a79fdf700f633e804e9`.

## Commands

The exact g1 launch arguments are retained in the rerun's `run_manifest.json`.
Run the compact extraction on g1 with the repository diagnostic:

```bash
python benchmark/robot_closed_loop/diagnostics/extract_r1_5_seed_replay_compact.py \
  --rerun-root /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r1_5_reference_replay_episode005_20261029a \
  --archive-root /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016 \
  --output /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r1_5_reference_replay_episode005_20261029a/compact_replay_evidence_v3.json
```

Run the local CPU model comparison from the repository root:

```bash
python benchmark/robot_closed_loop/diagnostics/analyze_r1_5_seed_replay_localization.py \
  --compact-evidence benchmark/robot_closed_loop/results/r1_5_reference_reproduction_local_20261029a/remote_metadata/compact_replay_evidence_v3.json \
  --fresh-shadow-events benchmark/robot_closed_loop/results/r1_5_shadow_perception_causal_replay_20261029e/episode_005_shadow_on_raw_events.npz \
  --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --checkpoint benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/paired_r1/ours_fit/spatial_event_state.pt \
  --output /tmp/r1_5_exact_seed_localization
```

SHA256 manifest for this local evidence directory: `SHA256SUMS`.
