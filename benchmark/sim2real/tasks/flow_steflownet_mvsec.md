# Optical-flow route: STE-FlowNet → MVSEC

Status: Stage 3 corrected the synthetic checkpoint rescore and completed fixed-fragment event-only candidate inference on public `indoor_flying1` at dt=1 and dt=4. No aggregate real ranking or checkpoint selection was performed. See [`STAGE3_RESEARCH_HANDOFF.md`](../STAGE3_RESEARCH_HANDOFF.md).

## Stage 2 EPE correction

The Stage 2 EPE values below are **invalid**: the scorer extracted only horizontal flow `u`, then NumPy broadcast it into both GT components. The four Stage 2 checkpoints and their training evidence remain useful; Stage 3 recomputed full two-component EPE from those same checkpoints without retraining. Do not cite the old values or the earlier “nearly tied” statement.

## Synthetic-only Stage 2 training record

- Uses the same six shared physical rollouts as reconstruction: four complete train rollouts and two independent validation rollouts. Each generator pair uses seed 71, identical initial state SHA-256 `c4b49f1a152b32bc07e6fb61c7355be3a93f74eece65b8f4c490576834d8fee4`, and common physical labels and validity masks.
- The pinned STE-FlowNet model ran two epochs and 16 updates per generator for each dt. The official four-scale Charbonnier photometric loss and 10× second-order smoothness loss were finite; every update had finite gradients (at least 70 gradient tensors). Synthetic flow labels were used for validation only and were never passed to the model or optimizer. All four checkpoints restored with matching state hashes.
- dt=1 uses adjacent synthetic 100 Hz endpoints; dt=4 spans four consecutive endpoint intervals. Validation used 60 intervals for dt=1 and 54 for dt=4 across the two held-out physical rollouts.
- A memory-heavy `F.unfold` correlation implementation caused an OOM in a prior attempt. The successful run uses a benchmark-local pure PyTorch custom-autograd shift correlation for the pinned 1×1-kernel / 9×9-patch operation. Its float32 check against an explicit 81-shift reference had maximum forward error `7.63e-6` and gradient error `1.49e-8`; it does not modify the upstream source.

| dt | Generator | Historical Stage 2 EPE (invalid) | Mean train objective | Train wall | Peak allocated GPU |
|---|---|---:|---:|---:|---:|
| 1 | EventGenesis | 0.8301 px | 115.06 | 16.33 s | 3.63 GB |
| 1 | V2E | 0.8082 px | 109.80 | 15.21 s | 3.63 GB |
| 4 | EventGenesis | 2.8537 px | 115.82 | 57.04 s | 12.56 GB |
| 4 | V2E | 2.8379 px | 113.69 | 58.58 s | 12.56 GB |

The old EPE column must not be interpreted; training loss, time, memory, and saved checkpoints are retained as Stage 2 evidence.

## Stage 3 corrected synthetic rescore

The same four existing checkpoints were restored independently and scored with strict prediction/GT shape `[2,H,W]`, binary mask `[H,W]`, finite values, and `px_per_interval` units. Deterministic inference made repeated and separately restored full two-channel predictions bitwise equal. No retraining was performed.

| dt | Generator | Validation windows | Corrected mean EPE (px / interval) | Median EPE | Peak inference allocation |
|---|---|---:|---:|---:|---:|
| 1 | EventGenesis | 60 | 0.942304 | 0.982211 | 0.42 GiB |
| 1 | V2E | 60 | 0.824453 | 0.858964 | 0.42 GiB |
| 4 | EventGenesis | 54 | 2.963737 | 3.146995 | 0.55 GiB |
| 4 | V2E | 54 | 2.858816 | 3.014514 | 0.55 GiB |

These are corrected two-epoch synthetic-pilot metrics only. They establish neither a stable training comparison nor real generalization.

## Stage 3 MVSEC real-input smoke

- Downloaded the official `indoor_flying1` data/GT and left-camera calibration in WSL; full original HDF5s remain there. A preregistered one-second lossless slice and matched GT were transferred to g1 after resolving indices before loading model weights.
- Parsed slice: 7,294 events, 41 APS endpoints, 21 endpoint-flow maps, native raster 346×260. The first common APS endpoint is source index 9 at `1504645177.7358222 s`. The fixed windows cover 31 dt1 intervals and 28 dt4 intervals.
- Real dt=1 is APS endpoint `i→i+1` (about 31.86 ms in this sequence); dt=4 is `i→i+4` (about 127.44 ms). GT uses pinned STE `estimate_corresponding_gt_flow` timestamp propagation. The audit checked each entire `i→i+dt` interval and event volume.
- All four saved STE checkpoints ran event-only inference. Input representation is 5 equal-event-count ON/OFF slices per APS interval. Native 346×260 events pad to 352×352 with `(y=46,x=3)` offset; both predicted flow components crop back to 346×260. Per-window outputs, GT, masks, exact association metadata, and hashes are retained on g1; their manifest is versioned in Stage 3 evidence.
- The run emitted per-window EPE under a provisional inspection mask (`finite GT AND nonzero GT AND raw event support`). This mask and aggregation are **not frozen**. No aggregate score, model ranking, checkpoint selection, or generator adjustment was made from these values.

## Failed attempts retained

- Stage 2 v1 stopped because eval mode returned one flow tensor while the pinned photometric helper expected a list of scales.
- Stage 2 v2 stopped at dt=4 with OOM in `F.unfold`; the local correlation implementation avoided that training-time allocation.
- Stage 3 first rescore detected nondeterministic GPU outputs after independent restore. Deterministic cuDNN settings then gave bitwise-equal predictions; the final rescore wrapper recorded exit 0.
- The first two Stage 3 MVSEC data audits exposed full-source-versus-fixed-slice event-count handling and a manifest attribute name mismatch. The corrected CPU audit passed all 31 dt1 and 28 dt4 windows.

## Reproduction and evidence

- Stage 2 source, config, training logs, and all four checkpoints: [`evidence/S2/20261016-stage2-pilot/`](../evidence/S2/20261016-stage2-pilot/).
- Stage 3 commands, environment, all successful/failed run logs, data manifests, and candidate outputs: [`evidence/S3/20261016-stage3/`](../evidence/S3/20261016-stage3/).
- Candidate configuration: [`../configs/stage2_pilot.json`](../configs/stage2_pilot.json). Fixed real-sample registration: [`../protocol/stage3_real_pilot_registration.yaml`](../protocol/stage3_real_pilot_registration.yaml).

Formal MVSEC scoring and protocol freeze remain with the research review.
