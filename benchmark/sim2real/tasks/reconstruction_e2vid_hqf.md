# Reconstruction route: E2VID+ → HQF

Status: Stage 3 parsed the preregistered public HQF `bike_bay_hdr` bag and ran both candidate E2VID checkpoints across all 2,431 image/event endpoints. Outputs are event-only, have a single sequence-start state reset, and were not scored against HQF images. Formal HQF metrics and sequence aggregation remain for research freeze.

## Stage 3 real-file and candidate-inference smoke

- Source bag: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage3/real/HQF/bike_bay_hdr.bag`, 362,570,693 bytes, SHA-256 `02797ab6240587bf6ccc352f6efc8325f8b9a49d4f69e5dba7f8a8d9ec452935`. The actual ROS bag contains `/dvs/events` (`dvs_msgs/EventArray`, 2,972 messages) and `/dvs/image_raw` (`sensor_msgs/Image`, 2,431 messages); image encoding is mono8 and sensor size is 180×240.
- Converted according to pinned EVREAL MemMap conventions: 19,763,043 raw events, `uint8` polarity 0=OFF/1=ON, event/frame timestamps in seconds, monotone time, and image endpoints associated to the latest event timestamp not later than the image. The conversion manifest lists each output array size/hash. The full converted data remain on g1.
- EventGenesis checkpoint `eventgenesis_final.pt` SHA-256 `004e0baf4f4f95264cfa3642f69e7aa3a85ea2c501d8116d30f940676ad58828`; V2E checkpoint `v2e_final.pt` SHA-256 `5ba9199ea611207ee8e04607af9a262309699cec6c4348e0eabaf517d6363a19`. Both are the Stage 2 trained candidates; public EVREAL pretrained weights were not loaded.
- Each event-only `[1,5,180,240]` input was padded to `[1,5,184,240]` for the recurrent network and cropped back to `[1,1,180,240]`. Each complete chronological sequence processed 2,431 endpoints; recurrent state reset once at sequence start and persisted thereafter. Reference frame pixels were not supplied to the network and no real GT scoring was performed.
- Checkable reconstructions were saved at endpoint 0, every 100 frames, and endpoint 2,430. This verifies file reading, endpoint correspondence, model loading, recurrent-state handling and image sizing; it is not reconstruction-quality evidence. The first endpoint's window is empty due to the EVREAL first-reference convention and is retained as part of the sequence.
- The data, per-run manifest, full command/environment/exit record, and saved frames are in [`evidence/S3/20261016-stage3/`](../evidence/S3/20261016-stage3/). The checkpoint candidates and their synthetic-only selection remain from Stage 2.

Stage 2 synthetic-training facts below remain historical and valid. Its statement that HQF had not yet been downloaded is superseded by this Stage 3 section.

## Executed

- Six whole Genesis physical rollouts were shared between EventGenesis and V2E: four train and two independent validation. E2VID inputs use the pinned five-bin HDF5 interface, 31 frames per rollout, and forward synthetic flow labels from the shared bank.
- The pinned E2VID recurrent model and public synthetic recipe were run for two epochs per generator, sequence length 3, 112 sequences per epoch and 224 optimizer updates. Both sides used seed 71 and identical initial state SHA-256 `b07667dbbfe18c1432ee543501b2686fb026d0f02c3e69eb23ebd10e4a5cffb0`.
- The official LPIPS-VGG and temporal-consistency components were recorded separately. Temporal consistency ran 224 train calls and 56 validation calls per generator. All updates had finite nonzero gradients; six recurrent state tensors were present after a sequence. Both checkpoints restored and reproduced the saved model-state hash.
- R1 is repaired: three-frame sequences invoke temporal consistency during training. R2 is repaired: flow labels and the common geometry validity mask are loaded independently of event count and observed event span. Fixtures cover empty events, two sparse events, and three equal-time events; all three preserve the same flow-label hash and 64 valid pixels.
- The shared geometric fixture projects a +0.25 m x translation at 1 m depth and `fx=2` to +0.5 px. The E2VID warp fixture also records lower unmasked image-warp L1 for the forward label (0.00476) than for its reversal (0.05238). The upstream temporal loss uses a flow-based visibility mask, so its scalar alone is not used as the direction test; that caveat and source hashes are recorded in the Stage 2 report.

## Synthetic-only diagnostics

| Generator | Train loss components (perceptual / temporal) | Synthetic validation loss components | Train wall | Peak allocated GPU |
|---|---:|---:|---:|---:|
| EventGenesis | 0.3432 / 0.0370 | 0.2069 / 0.0153 | 14.08 s | 0.69 GB |
| V2E | 0.4429 / 0.0384 | 0.2401 / 0.0232 | 15.12 s | 1.08 GB |

These short-run values verify finite loss/gradient, sequence state, and checkpoint flow. They are not generator performance estimates or real-data evidence.

## HQF readiness and protocol candidate

- The pinned EVREAL source exposes `dataset.MemMapDataset`, an HQF config and `eval.py`; the manifest lists the 14 canonical sequences and expected event/frame timestamp files. This is source inspection only.
- The current candidate uses all 14 sequences for final test, continuous recurrent inference reset at sequence start, no RGB seed frame, LPIPS-Alex primary and SSIM/PSNR/MSE supporting metrics, aggregated frame → sequence → dataset with equal sequence weight. The candidate adapter and actual files still need verification and research freeze.
- HQF is the current candidate because it is a reconstruction-specific high-quality-frame benchmark and EVREAL has a native HQF route. HS-ERGB remains an extension: it originates in the TimeLens video-interpolation benchmark, so the reconstruction target timing/alignment needs a separate definition. This is a protocol rationale, not a claim that HQF is universally better.
- The public source estimate is 19.4 GB compressed / 94 GB uncompressed. These are estimates, not measured downloads. License and all 14 sequence manifests must be checked before testing.

## Reproduction and evidence

- Candidate config: [`../configs/stage2_pilot.json`](../configs/stage2_pilot.json).
- Run command and logs: [`evidence/S2/20261016-stage2-pilot/runs/20261016-s2-e2vid-pilot-v1/`](../evidence/S2/20261016-stage2-pilot/runs/20261016-s2-e2vid-pilot-v1/).
- Converted data hashes: [`evidence/S2/20261016-stage2-pilot/data_manifests/e2vid_inputs_v2/conversion_manifest.json`](../evidence/S2/20261016-stage2-pilot/data_manifests/e2vid_inputs_v2/conversion_manifest.json).
- External checkpoint paths, sizes, and hashes: [`evidence/S2/20261016-stage2-pilot/external_artifacts.json`](../evidence/S2/20261016-stage2-pilot/external_artifacts.json).

Real HQF scoring, checkpoint selection, and generator adjustment were not performed.
