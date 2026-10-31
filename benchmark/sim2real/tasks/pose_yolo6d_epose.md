# Object-pose route: SingleShotPose / YOLO-6D → E-POSE

Status (updated 2026-11-01): the pose-window bug is fixed, the matched E-POSE block MAT/raw HDF5/CAD are present on g1, and a fresh paired 31-epoch generic-cuboid short training completed on CPU with normal eval, train-only BN calibration, PnP attempts, and checkpoint restoration. The remaining synthetic bank still renders a generic cube, so it is not an E-POSE CAD-matched result. Real pose scoring remains closed.

## Model and event-only input contract

- Pinned sources: SingleShotPose commit `2c282619101f9f31666dd18b42393375bea1600f`; E-POSE annotation tooling commit `b96ae783fcb6de84e1e9c602e91bce941cab0234`.
- The adapted model is the pinned SingleShotPose 416×416 full-frame Darknet/RegionLoss model. Its input is only three event-derived channels (ON log-count, OFF log-count, total log-count); no RGB, GT crop, GT mask, GT box, segmented events, depth, segmentation, or pose enters inference.
- Synthetic supervision uses shared Genesis pose, camera pose, actual K, and the configured 0.16 m generic cuboid to create normalized centroid plus eight 3D box-corner projections. This label is only a generic cuboid, with no E-POSE/YCB CAD/category match.
- The output decoder was compared with pinned `utils.get_region_boxes`: for one class/one anchor, both select the globally highest objectness cell; both apply sigmoid to centroid offsets and raw offsets to the eight corners, then normalize by grid width/height. The adapter follows that behavior. The pinned validation code also consumes this single highest-confidence box.

## Corrected event window and timestamp audit

Stage 2 had a stale loop-variable bug: selected samples reused the timestamp from the last visibility scan, so their event windows could extend to the rollout tail after the associated pose label. This made the Stage 2 pose EPE values invalid. The fix derives `event_end_s = label_time_s = times[interval_index + 1]` for every sample and persists sample ID, rollout ID, interval, event start/end, label time, and event count.

An independent fixture recounts the raw timestamps without using the adapter selector. For two distinct windows `(0.0,0.1]` and `(0.1,0.2]`, it verifies counts 3/3 for EventGenesis and 4/3 for V2E, including exact boundary events; the latest included event is at the label time and none is later.

## Corrected paired synthetic pilot and diagnosis

- Config: 16 epochs, 8 train windows from four whole physical rollouts, 4 validation windows from two held-out rollouts, 128 updates per generator, batch size 1. EventGenesis/V2E use the same labels, sample IDs, split and initial-state hash `a47ecb230e60d205f5973b981e959c2226cc650d821ac1c4f126a1cf1fde4796`.
- The original confidence term is disabled for epochs 1–15 and enabled for only the final 8 updates. Both checkpoints saved and restored with exact state hashes. Training loss and gradients were finite; loss means were 4.95 (EventGenesis) and 30.32 (V2E), with gradient-norm maxima 5,454 and 7,249. Peak allocated GPU memory was about 783 MiB and training took about 3.8 s per generator on a TITAN RTX.
- Standard eval-mode confidence decode gave synthetic validation mean keypoint EPE 35.79 px (EventGenesis) and 245.00 px (V2E). These are corrected-window pilot diagnostics only; they are not an E-POSE result or generator conclusion.
- A separate GT-cell-only decode (not an allowed inference input) gave 12.77–26.44 px across EventGenesis validation windows and 7.40–18.09 px for V2E, while confidence-selected decoding was worse. This isolates confidence-cell selection as one source of error; the confidence head had only eight updates after its 15-epoch pretraining phase.
- A one-sample 16-update overfit diagnostic reduced the coordinate loss from 10.58 to 0.19. In train mode, GT-cell decode was 14.85 px, but ordinary eval-mode GT-cell decode was 10,650 px with keypoints outside the unit square; after one train-mode BN update the next eval decode was still 1,262 px. The same diagnostic's confidence-selected eval output was 913,856 px. This is direct evidence of a severe batch-one BatchNorm running-stat/inference gap, in addition to the confidence-cell issue. It is not evidence that longer training alone will fix the route. Train-mode or GT-cell decoding is not substituted for real inference.
- The corrected four-window pilot's PnP solve returned success on all windows, but this does not repair the projection-error and BN behavior. No real pose metric was computed.

## E-POSE sample and CAD access

The 2026-10-16 web-access failures below are historical; they no longer describe the current local/g1 data state. The raw sample and matching CAD were later transferred and verified. No new E-POSE download probe was made for the 2026-11-01 follow-up.

- Verified MAT: `/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1/extracted/block/block/r_01_gl/0001_data.mat`, 6,017 bytes, SHA256 `bd369f867e671a194b067ba03cf11664adb6b389b1762203aba3b2ed1b6edb0c`.
- Verified raw event HDF5: `/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1/extracted/original_data/original_data/block/r_01_gl/data.h5`, 46,199,720 bytes, SHA256 `a589e204be51970f128a39315954bdb814157945be90d6c63f56d739566f863a`.
- Verified matched PLY: `/home/zhaoyuhan/Eventbased_WAM/external/sim2real_assets/epose_wood_block_20261024/nontextured.ply`, 983,765 bytes, SHA256 `8c653731426cf73b5b5d4ffbfa3bdd4ebb14cb82d413da82559bce1b5ee3ab37`. Registered CAD extent is approximately `[0.103917, 0.103464, 0.206003]` m; see [`epose_wood_block_cad_asset_manifest_20261024.json`](../manifests/epose_wood_block_cad_asset_manifest_20261024.json).
- A corrected projection-domain sidecar exists for `block/r_01_gl/0001_data.mat`; the original MAT remains unchanged. A forward/inverse distortion round trip on that sample was about `6.36e-14 px`. This is a one-sample geometry check only.
- Earlier inspection confirmed the MAT `events` field matches raw event frames; undistorted event images are stored separately. Remaining audit scope includes sequence-specific time calibration, object pose fields, full CAD object frame/origin, and symmetry-aware scoring. No GT crop/mask/box may be passed to inference.

Historical access attempt: at `2026-10-16T06:53:22Z`, the official landing page returned HTTP 202 with an AWS WAF challenge and an empty body; the public Figshare article API returned HTTP 403. Those responses explain the initial delay, not the present data state.

## Current engineering blockers

- The available shared synthetic bank is the original `0.16 m` generic cube. It does not render the verified wood-block PLY. Therefore the completed short training exercises model plumbing and paired event inputs, but not E-POSE-object appearance or CAD-frame keypoints.
- The new CPU run completed 31 epochs / 248 updates per arm, with 128 confidence-supervised updates and train-only BN calibration. Standard eval keypoint EPE over four held-out generic windows was `31.58 px` for EventGenesis and `23.18 px` for V2E. Confidence argmax selected the GT centroid cell on `0/4` and `2/4` windows respectively. PnP returned success on all windows, but that is not an accuracy result.
- A CAD-specific short pilot now needs: a Genesis bank rendered from the exact PLY, keypoints computed in the CAD's verified object frame, multiple held-out synthetic rollouts, ordinary confidence-argmax inference, PnP, and a frozen symmetry-aware metric. Real-data score remains disabled.

The old confidence/BN concerns must be kept distinct. Train-only BN calibration is included in the current pilot, so ordinary eval no longer depends on train-mode inference. Confidence-cell selection is still visibly unreliable and needs an audit of the pinned RegionLoss confidence targets before anyone changes the model or objective.

## Evidence and reproduction

- Stage 3 paired pilot and one-sample diagnostic commands, environments, outputs, checkpoint hashes and failures: [`evidence/S3/20261016-stage3/`](../evidence/S3/20261016-stage3/).
- Independent raw-window recount and confidence decoder fixture: `evidence/S3/20261016-stage3/controlled_checks/pose_window_audit_latest.json`.
- Config: [`../configs/pose_stage3_paired_16epoch.json`](../configs/pose_stage3_paired_16epoch.json); candidate program: [`../scripts/run_pose_stage2_pilot.py`](../scripts/run_pose_stage2_pilot.py).
- Latest short-run report and evidence: [`METRIC_AND_POSE_PROGRESS_20261101.md`](../reviews/stage8_20261101/METRIC_AND_POSE_PROGRESS_20261101.md), with compact logs, exact hashes, and remote-only checkpoint paths under [`evidence/20261101-stage8-pose-short/`](../evidence/20261101-stage8-pose-short/).
- Stage 2 keypoint values `158,581 / 59,141 px` are historical and invalid because the selected event windows were time-misaligned. The Stage 2 training/checkpoint logs remain archived.

No E-POSE sample, matched CAD, real inference, real score, or real-data tuning is claimed.
