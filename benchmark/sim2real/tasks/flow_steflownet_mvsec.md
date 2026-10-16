# Optical-flow route: STE-FlowNet → MVSEC

Status: Stage 2 paired synthetic training/validation completed for dt=1 and dt=4 on g1. MVSEC data and the real evaluation entry point are source-checked only; no MVSEC files were downloaded or scored.

## Executed

- Uses the same six shared physical rollouts as reconstruction: four complete train rollouts and two independent validation rollouts. Each generator pair uses seed 71, identical initial state SHA-256 `c4b49f1a152b32bc07e6fb61c7355be3a93f74eece65b8f4c490576834d8fee4`, and common physical labels and validity masks.
- The pinned STE-FlowNet model ran two epochs and 16 updates per generator for each dt. The official four-scale Charbonnier photometric loss and 10× second-order smoothness loss were finite; every update had finite gradients (at least 70 gradient tensors). Synthetic flow labels were used for validation EPE only and were never passed to the model or optimizer. All four checkpoints restored with matching state hashes.
- dt=1 uses adjacent 100 Hz endpoints; dt=4 spans four consecutive endpoint intervals. Train sampling selected interval endpoints from every train rollout; validation used 60 intervals for dt=1 and 54 for dt=4 across the two held-out physical rollouts.
- A memory-heavy `F.unfold` correlation implementation caused an OOM in a prior attempt. The successful run uses a benchmark-local pure PyTorch custom-autograd shift correlation for the pinned 1×1-kernel / 9×9-patch operation. Its float32 feature-sized numerical check against an explicit 81-shift reference had maximum forward error `7.63e-6` and gradient error `1.49e-8`; the check took about 0.034 s and allocated about 9.6 MB additional GPU memory. It is locked for this pilot and does not modify the upstream source.

## Synthetic-only diagnostics

| dt | Generator | Mean synthetic validation EPE (px / interval) | Mean train objective | Train wall | Peak allocated GPU |
|---|---|---:|---:|---:|---:|
| 1 | EventGenesis | 0.8301 | 115.06 | 16.33 s | 3.63 GB |
| 1 | V2E | 0.8082 | 109.80 | 15.21 s | 3.63 GB |
| 4 | EventGenesis | 2.8537 | 115.82 | 57.04 s | 12.56 GB |
| 4 | V2E | 2.8379 | 113.69 | 58.58 s | 12.56 GB |

These values describe a two-epoch synthetic pilot only. They do not establish a generator ranking, real performance, or a stable training curve. Peak dt=4 allocation was 12.56 GB on an RTX 3090 with 24 GB.

## MVSEC readiness and candidate protocol

- The pinned source provides the DAVIS HDF5 event/timestamp fields, `flow_gt` timestamp association helper and dt1/dt4 reference entry points. Candidate sequences are `indoor_flying1`, `indoor_flying2`, `indoor_flying3`, and `outdoor_day1`, at the native 346×260 orientation with supplied calibration.
- Candidate dt=1 uses adjacent DAVIS left APS endpoints; dt=4 spans four adjacent endpoint intervals. Flow GT timestamp alignment uses the pinned `estimate_corresponding_gt_flow` helper and NPZ keys `timestamps`, `x_flow_dist`, and `y_flow_dist`.
- The pinned `flow_error_dense` is retained as a reference reproduction. The primary valid-pixel support mask and treatment of zero-magnitude GT need research freeze and verification against downloaded data before real scoring. The manifest records the candidate common support as nonzero valid GT intersected with raw event support; this is not frozen protocol.
- The official page size sum is about 33.2 GB for the candidate bundles. This is a source-page estimate; no download or throughput measure was made.

## Failed attempts retained

- Run v1 stopped because eval mode returned a single flow tensor while the pinned photometric helper expected a scale list.
- Run v2 completed dt=1 but ran out of GPU memory at dt=4 with `F.unfold` (attempted 324 MB with 275 MB free at roughly 23.31 GiB allocated). It was not resumed by killing another user's process; the benchmark-local operator was implemented and a fresh run passed.

## Reproduction and evidence

- Candidate config: [`../configs/stage2_pilot.json`](../configs/stage2_pilot.json).
- Successful run: [`evidence/S2/20261016-stage2-pilot/runs/20261016-s2-steflow-pilot-v3/`](../evidence/S2/20261016-stage2-pilot/runs/20261016-s2-steflow-pilot-v3/); failed attempts v1/v2 are in neighboring run directories.
- Converted data hashes: [`evidence/S2/20261016-stage2-pilot/data_manifests/steflow_inputs_v2/conversion_manifest.json`](../evidence/S2/20261016-stage2-pilot/data_manifests/steflow_inputs_v2/conversion_manifest.json).
- External checkpoint paths, sizes, and hashes: [`evidence/S2/20261016-stage2-pilot/external_artifacts.json`](../evidence/S2/20261016-stage2-pilot/external_artifacts.json).

Real MVSEC scoring, real checkpoint selection, and generator adjustment were not performed.
