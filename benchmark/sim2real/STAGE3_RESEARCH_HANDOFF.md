# Stage 3 research handoff — correctness fixes and real-input pipeline

Date: 2026-10-16<br>
Branch: exp/public-sim2real-v1<br>
Execution host: g1 (zhaoyuhan@g1.pkuml.org), repository /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real<br>
Frozen simulator core: e79e656ac00f7847b98db9e12bf43b093f3afff1 (unchanged)

This Stage 3 repaired two result-invalidating bugs, recomputed the affected synthetic flow metrics, completed fixed-sample HQF/MVSEC candidate inference, and rechecked the pose route. Real data was used only for preregistered parsing, event-only inference and evaluation-plumbing checks. No real aggregate ranking, model selection or generator tuning was performed.

## Findings at a glance

| Route | Stage 3 evidence | State for review |
|---|---|---|
| E2VID+ → HQF | Actual public bike_bay_hdr bag conversion; EventGenesis and V2E synthetic-only E2VID checkpoints each ran all 2,431 endpoints with recurrent state reset once at sequence start. Saved output frames have 180×240 dimensions and finite values. Reference image pixels never entered the model; no HQF score was computed. | The IO/inference path is connected. Freeze HQF final-test inventory, metrics and aggregation before formal scoring. HQF is the main reconstruction candidate; HS-ERGB remains a separate extension because its source task is video interpolation and needs an explicit reconstruction-target alignment. |
| STE-FlowNet → MVSEC | Corrected and independently checked full-vector EPE; all four saved checkpoints restored and rescored on all 60 dt1 / 54 dt4 synthetic validation intervals. Parsed public indoor_flying1; all four candidates ran on 31 dt1 / 28 dt4 fixed windows at actual sensor input size, with 2D outputs cropped to 346×260 and per-window artifacts saved on g1. | The synthetic score bug is repaired and the real input/evaluation path runs. The real support mask/aggregation are provisional and require research freeze. No real summary score is presented as a model comparison. |
| SingleShotPose → E-POSE | Corrected each sample's event endpoint to its label time; an independent raw-timestamp recount checked two intervals and boundary events. Ran a corrected paired synthetic pilot, a one-sample fit diagnostic, and source-checked the pinned official decoder. | The corrected synthetic behavior is still not ready for E-POSE. Confidence-cell selection and batch-one BatchNorm state create unstable eval inference. The official E-POSE page/API were WAF-blocked; no raw sample or matching CAD was obtained. Request a sample/CAD and research direction review before scaling. |

## Error 1 — optical-flow EPE dropped v

**Root cause.** The Stage 2 output adapter returned only output[0,0] from [B,2,H,W]. That retained horizontal flow u; NumPy then broadcast it to the GT's two channels. The recorded Stage 2 EPE table and “nearly tied” statement are invalid. The training losses, gradients, model weights and checkpoint hashes are unaffected.

**Fix and validation.** The adapter now retains [u,v] as [2,H,W]. The EPE entry point requires prediction and GT to both be finite numeric [2,H,W], a finite binary [H,W] mask, and explicit px_per_interval units; it rejects shape broadcasting and unit mismatches. Independent hand-computable tests cover a perfect (3,4) vector, different u/v, vertical-only error, a masked region and empty validity. Empty validity is reported as no score, not a fabricated zero. The g1 controlled check ran 19 tests successfully.

Checkpoint restoration initially showed small output differences despite equal parameter hashes. The differences disappeared with deterministic cuDNN/algorithm settings: an independently restored checkpoint produced bitwise-identical two-channel output (max_abs_difference=0) and the same EPE. A final clean run wrapper recorded exit 0.

**Corrected synthetic validation, no retraining.** The four existing Stage 2 STE checkpoints were restored and evaluated on every held-out synthetic interval:

| dt | Generator | Rollout intervals | Mean EPE (px / interval) | Median EPE | Valid pixels across intervals |
|---|---|---:|---:|---:|---:|
| 1 | EventGenesis | 60 | 0.942304 | 0.982211 | 1,205,069 |
| 1 | V2E | 60 | 0.824453 | 0.858964 | 1,205,069 |
| 4 | EventGenesis | 54 | 2.963737 | 3.146995 | 1,041,639 |
| 4 | V2E | 54 | 2.858816 | 3.014514 | 1,041,639 |

These are corrected two-epoch synthetic-pilot diagnostics, not a stable training comparison or real generalization result. Each full output, checkpoint hash, interval metric and restore probe is in evidence/S3/20261016-stage3/server_runs/20261016-stage3-flow-rescore-final/corrected_flow_rescore.json.

## Error 2 — pose event windows ended at the wrong time

**Root cause.** collect_samples computed a label_time_index while scanning all visible intervals, then reused that loop variable when building selected samples. The event window could therefore end at the rollout tail rather than the corresponding pose label time. Stage 2 pose EPE values 158,581 / 59,141 px are invalid and are retained only as historical output.

**Fix and independent check.** Every selected interval now recomputes label_time_index = interval_index + 1; the event window is (times[i], times[i+1]], and event_end_s == label_time_s is asserted. Every persisted sample includes sample/rollout ID, interval, event start/end, label time, event count and source schema. A separate fixture implementation recounts raw timestamps without calling the collector's selector. It checked (0.0,0.1] and (0.1,0.2]: EventGenesis counts 3,3; V2E counts 4,3. Exact endpoint events are included once, the maximum selected timestamp equals the label endpoint, and no event is later than the label.

**Corrected pilot.** Both generators ran 16 epochs / 128 updates with eight train windows from four training rollouts and four validation windows from two held-out rollouts. Labels, IDs, splits and paired initialization hash a47ecb230e60d205f5973b981e959c2226cc650d821ac1c4f126a1cf1fde4796 match. Checkpoints restored to identical state hashes. Mean standard eval-mode keypoint EPE was 35.79 px for EventGenesis and 245.00 px for V2E. This is a small generic-cuboid synthetic diagnostic, not a generator result and not an E-POSE score.

The pinned official get_region_boxes implementation was inspected: with this one-class/one-anchor configuration it selects the global maximum objectness cell, sigmoids the centroid offset, leaves the corner offsets linear and normalizes by grid size. The adapter's decode follows that behavior, so its global confidence-cell selection is not a custom alternative. A synthetic-only GT-cell decode (diagnostic only; not an allowed inference input) was substantially lower than confidence-selected decode, isolating confidence-cell selection as one problem.

A one-sample 16-update fit reduced coordinate loss 10.58→0.19. On that same sample, train-mode GT-cell decoding had 14.85 px EPE, while ordinary eval-mode GT-cell decoding had 10,650 px EPE and coordinates outside the unit square; after a train-mode BN-stat update, the next eval-mode result was still 1,262 px. Confidence-selected eval output was 913,856 px. The corrected four-window pilot is less extreme but still shows confidence-selected errors of 29.86–42.74 px for EventGenesis and 182.56–384.03 px for V2E. This is direct evidence of an inference-state/BatchNorm running-stat mismatch in addition to weak confidence-cell selection; it cannot be dismissed as training budget alone. We did not switch inference to train mode or use a GT-selected cell.

The generic object is not matched to an E-POSE class or CAD. DataCite resolved the public DOI 10.6084/m9.figshare.28106717 and a registered archive size of 29,759,153,475 bytes. On 2026-10-16 the official Springer Nature Figshare landing page returned HTTP 202 with an AWS WAF challenge; its article API returned HTTP 403. The request bodies and metadata are preserved in evidence/S3/20261016-stage3/epose_access/. No raw events, matching CAD, class, K, pose label or E-POSE inference was obtained. Minimum next step: provide an authorized sample route or author mirror containing one event/pose sample and its exact CAD, then audit time, coordinates, units, K, synchronization and transform direction. The research session must separately decide whether the pinned model's confidence/BatchNorm inference protocol can be kept or a mature event-compatible full-frame pose model is required.

## Public reconstruction data and inference

The registered HQF sample is bike_bay_hdr, the first alphabetic sequence in the public 14-sequence inventory. Its actual ROS bag is 362,570,693 bytes, SHA-256 02797ab6240587bf6ccc352f6efc8325f8b9a49d4f69e5dba7f8a8d9ec452935. It contains /dvs/events (dvs_msgs/EventArray, 2,972 messages) and /dvs/image_raw (sensor_msgs/Image, 2,431 messages); images are mono8 at 180×240.

The converter produced 19,763,043 raw events and EVREAL MemMap arrays with seconds timestamps and polarity 0=OFF/1=ON. Image endpoints are associated to the latest event timestamp not later than each image, following the pinned EVREAL conversion contract. The full per-file sizes and SHA-256 values are in external_artifacts_stage3.json and hqf_real_data/conversion_manifest.json.

Both synthetic-only E2VID checkpoints ran chronologically across all 2,431 event endpoints. Each used [1,5,180,240] event tensors, padded to [1,5,184,240], and emitted cropped [1,1,180,240] outputs. Recurrent state was reset once at sequence start and carried through the sequence. Public pretrained EVREAL weights were not loaded. Reference image pixels were checked for endpoint correspondence/shape only and were not model inputs. Finite raw output ranges were [-0.494,1.167] for EventGenesis and [-1.395,1.341] for V2E; saved PNG previews clip to 8-bit for inspection. The run did not compute HQF metrics or tune normalization from image targets.

Outputs at indices 0, 100, …, 2,400 and 2,430 are committed as small, inspectable PNG evidence. The Stage 2 synthetic E2VID training and R1/R2 results were reused without retraining. HQF's all-sequence metric and aggregation protocol still needs research freeze. HQF remains the main reconstruction candidate because it has reconstruction-specific targets and a native EVREAL path; HS-ERGB is an extension because its video-interpolation source needs a separately defined reconstruction time/target mapping.

## Public optical-flow data and inference

Full official indoor_flying1 data and GT HDF5s plus calibration were downloaded to WSL. The exact fixed slice was resolved before candidate weights were loaded and then copied to g1. The g1 slice contains 7,294 events, 41 APS endpoints, 21 GT flow maps, and native 346×260 raster. Its full source hashes and subset hashes are in the access manifests and external_artifacts_stage3.json.

The first common APS endpoint is index 9 at 1504645177.7358222 s. dt1 is APS endpoint i→i+1 (mean 31.859 ms in this sequence); dt4 is i→i+4 (mean 127.437 ms). These are endpoint intervals, not assumed 10 ms durations. For each window, raw event timestamp boundaries, the official image_raw_event_inds endpoint indices, complete event support, and pinned STE timestamp-propagated endpoint GT were checked. The calibration YAML's left-camera intrinsics K and projection matrix differ; both are preserved without guessing which protocol to standardize.

All four STE checkpoints ran event-only inference over all preregistered windows. Native event tensors were padded from 346×260 to 352×352 at offset (y=46,x=3) and both flow channels were cropped back to [2,260,346]. All predictions and GT were finite with matching shapes. Max allocated GPU memory was 1.12–1.42 GiB at the target input size. The entry point emitted one per-window EPE under a provisional inspection support (finite GT AND nonzero GT AND event support) and saved per-window input/prediction/GT/mask NPZs on g1. This support and aggregation are not frozen; no aggregate comparison or checkpoint choice was made. Per-window hashes, times, indices and mask counts are in each candidate's manifest.json and external_artifacts_stage3.json.

## Environment, resources and provenance

- g1: Ubuntu 20.04.6; Python 3.11.15; PyTorch 2.5.1+cu121; h5py 3.14.0; NumPy 2.4.6; OpenCV 5.0.0.93; rosbags 0.11.7; PyYAML 6.0.3.
- Flow rescoring ran on RTX 3090 GPU0. HQF inferences used RTX 3090 GPU1. MVSEC used RTX 3090 GPU0/GPU1 and V100S GPU4. Pose diagnostic and paired pilot used TITAN RTX GPU5. GPU3 was occupied by another user's CARLA processes and was not used, stopped or modified.
- The WSL download of the four registered source files (HQF bag, MVSEC data, MVSEC GT, calibration) took 06:41:05–06:50:29 UTC, 9 min 24 s. Full original data remain in WSL; only the preregistered ~11.8 MB MVSEC data/GT/calibration slice was placed on g1. HQF converted arrays use about 441 MB. MVSEC per-window NPZ outputs use about 206 MB on g1. Large datasets and checkpoints remain outside Git.
- Wall times: final four-checkpoint synthetic rescore 07:17:43–07:21:49 UTC; HQF candidate passes 38.7 s / 31.3 s model time; pose paired training about 3.8 s per generator; MVSEC candidate model time 19.0 s / 14.3 s for dt1 and 41.2 s / 62.9 s for dt4. These are fixed-sample interface costs, not full-corpus estimates.
- Pinned external source commits are in manifests/downstream_source_bundle.json; archive copies without .git are verified by file and tree hashes. No frozen-core source was changed.

## Reproduction and evidence

- Run commands, Python/package environment, physical-GPU preflight, start/end time and exit code: evidence/S3/20261016-stage3/server_runs/. Successful and failed attempts are preserved.
- Core source/config: scripts/run_steflow_stage2_pilot.py, scripts/rescore_steflow_checkpoints.py, scripts/audit_mvsec_real_fragment.py, scripts/run_mvsec_candidate_inference.py, scripts/infer_hqf_e2vid_candidate.py, scripts/run_pose_stage2_pilot.py, scripts/run_pose_overfit_diagnostic.py, scripts/audit_pose_event_windows.py.
- Controlled checks: bash benchmark/sim2real/scripts/run_stage3_controlled_checks.sh (19 tests, server exit 0).
- Fixed real-data preregistration: protocol/stage3_real_pilot_registration.yaml.
- External full-data/checkpoint/prediction paths and per-file hashes: evidence/S3/20261016-stage3/external_artifacts_stage3.json.
- Evidence index: evidence/S3/20261016-stage3/artifact_index.json; repository-wide index: artifact_index.json.

## Research decisions to freeze next

1. **HQF:** confirm the 14-sequence final test scope, target/image clipping and LPIPS-Alex versus supporting metrics, aggregation across frames/sequences, and the relationship to optional HS-ERGB. Do not generalize the single bike_bay_hdr IO smoke to final performance.
2. **MVSEC:** freeze dt1/dt4 endpoint interpretation, GT support, event-support use, zero-flow/invalid-pixel treatment, empty-window handling and sequence aggregation against the official reference. The current mask is only a plumbing check.
3. **Pose:** obtain one raw E-POSE sample and matched CAD; then decide whether to retain SingleShotPose with a synthetic-only-validated BN/confidence protocol or approve a mature event-only full-frame alternative. No GT crop, mask, box or segmented events may enter inference.

Training data, checkpoint selection, and generator settings remain synthetic-only. These Stage 3 results establish interfaces and expose correctness limits; they do not establish that EventGenesis outperforms V2E on public real events.
