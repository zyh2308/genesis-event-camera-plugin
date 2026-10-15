# Stage-0 shared physics and paired event artifacts

Execution host: `g1` (Ubuntu 20.04.6), project `/home/zhaoyuhan/Eventbased_WAM`, conda `Eventbased_WAM`. These experiments ran on the server; this WSL checkout prepared code and records only.

## Evidence status

- **Actually run:** Genesis GPU produced train and validation fall-box rollouts; EventGenesis emitted raw events; pinned V2E processed the same endpoint image sets at 25 μs timestamp resolution.
- **Actually checked:** shared HDF5 schema/shapes and flow masks; event counts, timestamp monotonicity, polarity values, and coordinate bounds; V2E per-episode exit codes were zero.
- **Not captured:** the shared trajectory process shell exit code. Its full summary was printed and HDF5 contents were reopened and checked, but no exit code is claimed.
- **Excluded numerical attempt:** V2E at 1 ms completed with exit 0 but logged large IIR update eps=0.93 and a low-pass cutoff warning. Preserve its logs/output as failure evidence; do not use those event files for paired training.
- **Not done:** downstream task loader/training; real data evaluation or candidate ranking; radiometric/sensor calibration.

## Shared data

Server output root:
`/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage0/shared-trajectory-20261016-v1/`

It contains `shared_trajectory.h5`, `summary.json`, the run's config copy, and endpoint PNG inputs under `v2e_input/{train_00,validation_00}/`. Per-rollout split is shared by both generator branches. HDF5 includes radiance, depth, segmentation, camera/object poses, right-endpoint pose targets, full SE(3) forward flow, validity masks, and EventGenesis raw events.

- EventGenesis: 4,068 train events; 4,832 validation events; raw event columns are `t,x,y,p` with seconds and signed polarity.
- V2E accepted numerical follow-up: 2,084 train events; 2,571 validation events; raw columns are `t_microseconds,x,y,p` in uint32 with polarity 0/1.
- V2E 1 ms excluded trial: 2,107 train and 2,617 validation events.

V2E gets a single fixed mapping from the shared scene-linear RGB endpoints: Rec.709 luminance, exposure 1, Reinhard tone map, sRGB transfer, round-to-nearest uint8. No per-frame normalization/auto-exposure. This numerical mapping is not camera calibration. EventGenesis uses native linear radiance and its physics/pose interpolation; V2E uses pinned SuperSloMo between endpoints. See `configs/shared_trajectory_stage0.json` and `configs/v2e_stage0_eps005.json` for the exact limits and override.

## Raw logs and hashes

Raw logs, runner copies, exit records, preflight and artifacts remain on the server under:

- trajectory: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-shared-trajectory/`
- V2E 1 ms excluded run: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-v2e-pair-v1/`
- V2E 25 μs accepted Stage-0 pair: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-v2e-pair-eps005/`

`external_artifacts.json` in this directory records server paths and SHA-256s for the data, input images, copied configs, scripts, and raw logs. `samples/` contains a small checked copy of the main HDF5 and the four V2E event files for reproducible inspection.
