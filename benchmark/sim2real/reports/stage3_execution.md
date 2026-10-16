# Stage 3 — correctness fixes and end-to-end real-input pipeline

Date: 2026-10-16 (UTC run logs; local workspace Asia/Tokyo)<br>
Branch: exp/public-sim2real-v1<br>
Local repository: /home/科研/Eventbased_WAM/repositories/gec-sim2real<br>
g1 repository: /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real<br>
Frozen simulator core: e79e656ac00f7847b98db9e12bf43b093f3afff1, unchanged

## Stage scope and outcome

Stage 3 repaired two errors found by independent Stage 2 review, proved the corrected model/data/metric/checkpoint-restoration chain on synthetic validation, and connected candidates to fixed public real data. The run scope was preregistered in protocol/stage3_real_pilot_registration.yaml before candidate weights were loaded.

No retraining was done for the corrected optical-flow EPE. The accepted E2VID Stage 2 synthetic training and R1/R2 closure were reused. A corrected small pose pilot was run because the Stage 2 event window was misaligned. Real samples were not used for training, checkpoint selection, generator changes or aggregate ranking. Event-only inference was maintained.

## Correction 1 — two-component optical-flow EPE

Stage 2 extracted only output[0,0] from a [B,2,H,W] prediction, retaining u. NumPy broadcast that plane against the two-channel GT, so the historical EPE table and “nearly tied” conclusion are invalid. The Stage 2 checkpoints and training histories remain intact.

The scorer now converts [1,2,H,W] to [2,H,W] without dropping either component. It rejects any prediction/GT/mask shape mismatch, wrong flow units, non-finite prediction values, non-binary masks and accidental broadcasting. Independent controlled examples verify:
- exact predicted (u,v)=(3,4): EPE 0;
- vertical-only error of +2: EPE 2;
- u error +3 and v error +2: EPE sqrt(13);
- a masked invalid region does not contribute;
- an empty mask returns explicit no-score.

Checkpoint restore initially differed slightly in output despite equal model-state hashes. Diagnostic isolation showed non-deterministic GPU kernels. With deterministic cuDNN and deterministic algorithms enabled, repeated and independently restored predictions became bitwise equal, max absolute difference 0. The clean final wrapper run records exit 0.

| dt | Generator | Synthetic validation intervals | Corrected mean EPE px/interval | Median | Total valid pixels |
|---|---|---:|---:|---:|---:|
| 1 | EventGenesis | 60 | 0.942304 | 0.982211 | 1,205,069 |
| 1 | V2E | 60 | 0.824453 | 0.858964 | 1,205,069 |
| 4 | EventGenesis | 54 | 2.963737 | 3.146995 | 1,041,639 |
| 4 | V2E | 54 | 2.858816 | 3.014514 | 1,041,639 |

These numbers are a corrected re-score of the same two-epoch synthetic pilot. They do not establish a stable generator comparison or real generalization. Full interval details, checkpoint hashes and restore probes are in evidence/S3/20261016-stage3/server_runs/20261016-stage3-flow-rescore-final/corrected_flow_rescore.json.

## Correction 2 — pose event-window endpoint

The Stage 2 collector reused label_time_index from the final visibility-scan iteration while constructing selected sample windows. As a result, event windows could end at the rollout tail instead of the pose label time. The old Stage 2 pose EPE values are invalid and retained only for history.

For every selected interval, the collector now sets label_time_index = interval_index + 1 and asserts event_end_s == label_time_s. It records sample ID, rollout ID, interval, start/end, label timestamp, event count and source event schema. An independent fixture recounts the raw timestamps with a separate comprehension:
- EventGenesis (0.0,0.1] has 3 events and (0.1,0.2] has 3.
- V2E has 4 and 3 events in those intervals.
Exact right-boundary events are included once; no selected event is after the label time.

The corrected synthetic paired pilot used 16 epochs, 8 train samples and 4 independent validation samples per generator, 128 updates each, and identical paired initialization SHA-256 a47ecb230e60d205f5973b981e959c2226cc650d821ac1c4f126a1cf1fde4796. Checkpoints saved/restored with exact state hashes. Eval-mode synthetic mean keypoint EPE was 35.79 px (EventGenesis) and 245.00 px (V2E); no result is an E-POSE score or generator conclusion.

The official pinned SingleShotPose get_region_boxes source was checked. The adapter selects the same single maximum-objectness cell and applies the same centroid sigmoid, linear corner offsets and grid normalization. A GT-cell-only diagnostic is not an inference path; it shows that confidence-cell selection remains a source of error. The small fit reduced coordinate loss 10.58 to 0.19, but eval-mode GT-cell decoding still reached 10,650 px in a one-sample fixture while train-mode BN statistics gave 14.85 px; after updating BN in train mode, eval decoding was 1,262 px. Confidence-selected eval output was 913,856 px. The results expose an inference-state/BatchNorm mismatch in addition to weak confidence learning. Train-mode inference and GT-cell decoding were not adopted.

## Route A — HQF real file and E2VID candidates

The preregistered public sequence bike_bay_hdr was read from the actual 362,570,693-byte bag (SHA-256 02797ab6240587bf6ccc352f6efc8325f8b9a49d4f69e5dba7f8a8d9ec452935). The bag has 2,972 /dvs/events messages and 2,431 /dvs/image_raw messages; images are mono8, 180×240. Conversion produced 19,763,043 events and the pinned EVREAL MemMap schema with seconds timestamps and 0=OFF / 1=ON polarity. All converted arrays have sizes and SHA-256 in hqf_real_data/conversion_manifest.json.

Both Stage 2 synthetic-only E2VID candidate checkpoints ran all 2,431 endpoints in chronological order. Inputs are [1,5,180,240], padded to [1,5,184,240], outputs cropped to [1,1,180,240]. Each model resets state once at sequence start, then carries recurrence forward. The Stage 2 candidate weights were loaded strictly; public EVREAL pretrained weights were not used. Reference image pixels were never model input and no real GT score was computed. Output ranges are finite but can extend outside [0,1]; saved PNG preview images clip to 8-bit for inspection. This is an IO/recurrent inference check only.

## Route B — MVSEC file, alignment, and candidates

The full public indoor_flying1 data/GT and calibration zip were downloaded to WSL with recorded full-file sizes/hashes. The fixed fragment was selected before checkpoint loading and transferred to g1: 7,294 events, 41 APS endpoints and 21 endpoint-flow maps at 346×260. Source and slice hashes are listed in access_and_download/ and external_artifacts_stage3.json.

The first common APS endpoint is source index 9 at 1504645177.7358222 seconds. dt1 means APS endpoint i to i+1 (mean 31.859 ms in this sequence); dt4 means i to i+4 (mean 127.437 ms). The audit checks whole dt windows, raw event boundaries and the official endpoint event-index array. GT endpoint displacement follows the pinned STE-FlowNet timestamp-propagation semantics. Calibration intrinsics K and projection matrix differ; both are recorded without selecting an unapproved convention.

All four synthetic-only STE checkpoints ran event-only inference on every preregistered window: 31 windows at dt1 and 28 at dt4 per checkpoint. The model input is the pinned equal-event-count ON/OFF representation with five slices per APS interval. The 346×260 sensor raster pads to 352×352 with offset y=46, x=3; the complete two-channel output is cropped back to [2,260,346]. All predictions and GT are finite and shape-matched.

The evaluation adapter emits per-window EPE with a provisional inspection mask: finite GT, nonzero GT vector and event support at the same pixels. The mask and aggregation are explicitly not frozen. No aggregate real result, ranking or checkpoint selection was used. Exact per-window GT association, event/prediction/GT arrays, support counts, output hashes and resource usage are saved in the four g1 run manifests. Raw .npz outputs remain on g1 and are indexed in external_artifacts_stage3.json.

## Route C — E-POSE access and direction blocker

DataCite resolved DOI 10.6084/m9.figshare.28106717 and publisher metadata listing 29,759,153,475 bytes. At 2026-10-16T06:53:22Z, the official Springer Nature Figshare page returned HTTP 202 with an AWS WAF challenge and empty body; the public article API returned HTTP 403. No file-level manifest, raw sample, matched CAD, class, K or pose label was obtained. This establishes a current access blocker in this environment, not global unavailability. The generic cuboid is not represented as an E-POSE match. Request an authorized sample and matching CAD; then audit units, time synchronization, axes, K and transform direction before choosing a model path.

## Environment, resources, and costs

g1 environment: Ubuntu 20.04.6, Python 3.11.15, PyTorch 2.5.1+cu121, h5py 3.14.0, NumPy 2.4.6, OpenCV 5.0.0.93, rosbags 0.11.7, PyYAML 6.0.3.

- Data download: 06:41:05–06:50:29 UTC (9 min 24 s) for four registered files. Full originals remain on WSL; only the preregistered MVSEC slice plus calibration (~11.8 MB) was transferred to g1.
- Four-checkpoint synthetic flow rescore: wrapper wall 07:17:43–07:21:49 UTC; accumulated model time about 223 s. Peak inference allocation was 0.42 GiB for dt1 and 0.55 GiB for dt4.
- HQF E2VID candidates: 38.7 s EventGenesis and 31.3 s V2E model time on RTX 3090 GPU1, 2,431 sequence endpoints each.
- MVSEC candidates: 19.0 / 14.3 s model time for dt1 and 41.2 / 62.9 s for dt4, on GPUs 4/0/1/4. Peak target-size inference allocation ranged from 1.12 to 1.42 GiB.
- Pose paired pilot used about 783 MiB and 3.8 s train time per generator on TITAN RTX GPU5. The one-sample diagnostic took about 1.9 s training. Full pose checkpoints occupy about 809 MB and remain outside Git.
- HQF converted arrays occupy about 441 MB; MVSEC per-window .npz prediction bundles occupy about 206 MB. Original datasets and all large weights/checkpoints stay outside Git.
- GPU3 was occupied by another user's CARLA job and was untouched. No process belonging to another user was stopped or modified.

## Run records, versions, and failures

Each run record stores command, environment, UTC start/end, exit code, physical-GPU preflight where applicable, and stdout under evidence/S3/20261016-stage3/server_runs/. Controlled server checks ran 19 tests and exited 0.

Retained failures and corrected retries:
- First flow rescore found non-deterministic outputs after independent checkpoint restore. Deterministic mode fixed it. A result JSON was written on this exploratory run, but replacing the wrapper script during its execution prevented the wrapper from recording exit status; the final immutable-wrapper run completed with exit 0.
- MVSEC audit attempt 1 compared a 7,294-row fixed slice to the 14,071,304-event full-sequence count. Attempt 2 used an incorrect data-manifest attribute name. Corrected audit passed all 31 dt1 and 28 dt4 windows.
- First pose diagnostic looked for source manifest at the repository root rather than benchmark/sim2real/manifests. Corrected path rerun passed.
- First g1 controlled check could not import the pinned GT helper because the test assumed a WSL-only external-source path. The source resolver now recognizes both WSL and g1 layouts; final g1 suite passed.

## Paths and reproducibility

- Compact research handoff: ../STAGE3_RESEARCH_HANDOFF.md.
- Current status: ../sim2real_status.md.
- Fixed real-sample registration: ../protocol/stage3_real_pilot_registration.yaml.
- Candidate commands and exact indices: run command files in server_runs/20261016-stage3-*.
- Corrected-flow implementation/rescore: ../scripts/run_steflow_stage2_pilot.py and ../scripts/rescore_steflow_checkpoints.py.
- Pose event-window audit/pilot: ../scripts/audit_pose_event_windows.py and ../scripts/run_pose_stage2_pilot.py.
- HQF converter/inference: ../scripts/convert_hqf_bag_rosbags.py and ../scripts/infer_hqf_e2vid_candidate.py.
- MVSEC audit/inference: ../scripts/audit_mvsec_real_fragment.py and ../scripts/run_mvsec_candidate_inference.py.
- External data, checkpoints, and every g1 per-window output path/size/hash: ../evidence/S3/20261016-stage3/external_artifacts_stage3.json.
- Stage 3 evidence hash index: ../evidence/S3/20261016-stage3/artifact_index.json.

## Stage 2 report corrections

The old Stage 2 optical-flow EPE figures are marked invalid because v was dropped. The old Stage 2 pose EPE figures are marked invalid because the event window endpoint did not match the pose label time. Stage 2 training losses/checkpoints and unrelated accepted E2VID R1/R2 results remain preserved. The full independent Stage 2 review is retained at ../reviews/stage2_2834540/REVIEW.md.

## Research decisions to freeze

1. HQF: confirm the 14-sequence final-test scope, target clipping, LPIPS-Alex and supporting metrics, and aggregation; review the HQF versus HS-ERGB role.
2. MVSEC: freeze dt and GT-time semantics, support mask, zero/non-finite GT, event-support rule, empty windows and sequence aggregation against the official reference.
3. E-POSE: obtain one original sample and matched CAD; then decide whether SingleShotPose can be retained with a synthetic-only-validated BN/confidence policy or whether a mature event-only full-frame pose model is needed.

No frozen-core change was made. Stage 3 establishes correctness and inference interfaces; it does not establish sim-to-real superiority.
