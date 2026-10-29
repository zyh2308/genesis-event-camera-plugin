# R1.5 shadow replay with matched DVS initialization history

## Question

Experiment D began its fresh DVS emulator at 0.5605 s, while the archived
Direct-HDR stream began at 0.5505 s. Experiment E initializes the fresh DVS at
the archived start time 0.5505 s, then gives the network only the same causal
25 ms history and exact query rows as before. This tests whether missing DVS
threshold-reference history explains the archived +55 mm localization spikes.

The renderer restores saved Panda and cube states without advancing physics.
It uses the old Ours checkpoint and the same fixed camera, 4-sample HDR
framebuffer, shadows on/off conditions, ideal physical-log DVS, and five
causal 5 ms input bins. GT is joined only after inference; segmentation and
depth are audit-only. This is development replay, not a new rollout or a
formal test.

## Result

| Episode | Event source | Mean XY error | Median | Maximum |
|---|---|---:|---:|---:|
| 004 center | archived 2 kHz Direct-HDR | 11.0 mm | 10.4 mm | 16.7 mm |
| 004 center | fresh direct, shadows on | 10.4 mm | 10.4 mm | 11.9 mm |
| 004 center | fresh direct, shadows off | 69.6 mm | 42.3 mm | 167.7 mm |
| 005 +55 mm | archived 2 kHz Direct-HDR | 58.7 mm | 59.2 mm | 111.1 mm |
| 005 +55 mm | fresh direct, shadows on | 10.6 mm | 9.5 mm | 19.3 mm |
| 005 +55 mm | fresh direct, shadows off | 217.1 mm | 159.8 mm | 400.2 mm |

Matching the earlier DVS start materially improves the fresh shadow-on result
for episode 005: D was 16.2 mm mean and E is 10.6 mm. It does not explain the
archived failure: the archived stream remains at 58.7 mm, and the five-bin
cosine between archived and fresh shadow-on inputs is still about 0.90. Thus
DVS initialization history contributes to the fresh replay but is not the
sole source of the archived-event mismatch. Shadows-off remains strongly
out-of-domain for this checkpoint.

Fresh shadow-on events average about 2,875 per 25 ms query for episode 005,
versus 3,810 in the archived stream. Episode 004 averages 1,749 versus 2,249.
The event difference is structured: cosine near 0.90 coexists with different
model outputs, so aggregate cosine alone is not a fidelity guarantee.

## Cost and provenance

- g1 GPU0: RTX 3090 UUID `GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23`; Genesis
  CPU backend, DVS and inference CPU; no physics advancement.
- Run ID `r1_5_shadow_perception_causal_replay_20261029e`; exit 0; wall time
  `238.969 s`. Per condition, scene construction took 48.8–63.2 s, exact-state
  captures 0.75–1.08 s, and DVS processing 0.29–0.59 CPU s.
- Render checkout commit `855b4b0bc7de96170b4bfee8d7002b370782f346`; diagnostic
  source commit `7f685817119d84b9df80120c3ebb6e85cc554c0a`.
- Render runner SHA256 `58e6a8c124c0be621b31fb6f757554bb7378c20894429620a47b9122e5df5112`;
  E config SHA256 `5fe4908707eac82df9413e4c7763066f75dbdc0f5676ebfb55dca04df2e5dcfe`;
  scene config SHA256 `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`;
  checkpoint SHA256 `d391fec293e2c6c50cdb267e15f742ed09e77e7b28e56a79fdf700f633e804e9`.
- CPU comparison analysis commit `7f685817119d84b9df80120c3ebb6e85cc554c0a`,
  script SHA256 `ecba7da4209c7531be9411b0be4ec38fed515542c8022886df97c0a2ae836b80`.
- Runtime scripts, log, resource allocation, PIDs, and exit status are in
  `../runtime_logs/r1_5_shadow_perception_causal_replay_20261029e/`.
- The archived event, fresh event, query voxel, cache, and trace hashes are in
  `summary.json`, each episode summary, and the CPU comparison JSON/CSV.

Reproduce with the command in the sibling runtime `launch.sh`, using the
isolated g1 worktree, preserved robot environment, E config, and the exact
cache/checkpoint paths recorded in `robot_status.md`. Recompute archived
comparisons on CPU with `analyze_r1_5_shadow_causal_replay_archived_reference.py`.

This run did not retrain, rank generators, alter the frozen core, or establish
closed-loop success. The archived stream and current exact-state rerender
remain different sources; the next experiment must isolate their source
cadence/region interactions rather than infer a fix from the old checkpoint.

Verify this directory with `sha256sum -c SHA256SUMS` from this directory.
