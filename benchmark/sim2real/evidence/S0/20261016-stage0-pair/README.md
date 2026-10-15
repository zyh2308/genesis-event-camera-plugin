# Stage-0 paired physics and event artifacts

These files document the common synthetic fixture produced on `g1`. All simulation and model runs were executed on the server; WSL was used for project preparation, evidence review, and Git handoff.

## Shared fixture

Server root:

`/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage0/shared-trajectory-20261016-v1/`

It contains the shared train/validation fall-box rollouts, 100 Hz endpoint renders, scene-linear radiance, depth, segmentation, camera/object poses, pose labels, ten forward SE(3) flow intervals per rollout, and EventGenesis raw events. The split is by rollout and is identical for EventGenesis and V2E.

- EventGenesis events: 4,068 train; 4,832 validation; columns `t,x,y,p`, seconds and signed polarity.
- V2E accepted events at 25 μs: 2,084 train; 2,571 validation; columns `t_microseconds,x,y,p`, uint32 and polarity 0/1.
- V2E 1 ms attempt: 2,107 train; 2,617 validation. It completed but emitted a large IIR update and coarse-step warning; these files are excluded from downstream training.
- The shared trajectory command exit code was not captured. Its printed summary and HDF5 structures/values were checked on g1; no exit code is claimed.
- EventGenesis consumes native linear radiance plus physics/pose interpolation. V2E consumes a fixed global Rec.709 → Reinhard → sRGB mapping of the same endpoints plus pinned SuperSloMo. This pilot pair checks feasibility; the generators are not claimed to be sensor-calibrated or an isolated causal comparison.

## Downstream smoke results

The same fixture also produced the paired inputs for E2VID+ and STE-FlowNet. One synthetic update and validation completed for each generator in both routes. Their model/loss results, adapter fixes, caveats, measured memory/time, and exact commands are in [`../20261016-stage1-pilot/README.md`](../20261016-stage1-pilot/README.md) and its archived run logs.

The checked small sample data and converted inputs are committed under [`../20261016-stage1-pilot/samples/`](../20261016-stage1-pilot/samples/). Logs, hashes, exit codes, and external checkpoint paths are in that evidence directory. Full datasets and checkpoints remain on g1. The exact artifact hashes and source paths are recorded in [`../20261016-stage1-pilot/external_artifacts.json`](../20261016-stage1-pilot/external_artifacts.json).
