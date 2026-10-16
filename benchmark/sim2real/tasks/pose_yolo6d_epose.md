# Object-pose route: SingleShotPose / YOLO-6D → E-POSE

Status: the pose-window bug is fixed and a corrected, paired event-only synthetic pilot completed. The E-POSE sample/CAD contract remains blocked by the publisher's current WAF responses. The generic Genesis cuboid is not an E-POSE class match.

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

- DataCite resolved DOI `10.6084/m9.figshare.28106717` to the official [Springer Nature Figshare record](https://springernature.figshare.com/articles/dataset/E-POSE_A_Large_Scale_Event_Camera_Dataset_for_Object_Pose_Estimation/28106717), with a listed total of 29,759,153,475 bytes and CC-BY 4.0 metadata.
- At `2026-10-16T06:53:22Z`, the official landing page returned HTTP 202 with an AWS WAF challenge and an empty body. The public Figshare article API returned HTTP 403. DataCite did not provide file-level IDs. No raw sample, matching CAD, class record, K, or pose annotation was obtained; therefore no E-POSE inference ran.
- Raw timestamp unit/order/polarity, camera K convention, transform direction/axes/units, event/pose synchronization, CAD scale and object-class mapping remain unverified. Do not substitute a generic cuboid for the missing matched sample.

## Research review gate

Minimum data request: obtain one original E-POSE event/pose sample and its matched CAD through an authorized publisher route, author mirror, or direct sample handoff. Then run the raw event/K/pose/CAD contract audit before integration. Separately, the research session should decide whether the pinned SingleShotPose confidence/BatchNorm inference behavior can be addressed within the frozen model/training definition or whether a mature event-compatible full-frame pose model is required. Do not scale the current generic pilot; this gate does not block reconstruction or optical flow.

## Evidence and reproduction

- Stage 3 paired pilot and one-sample diagnostic commands, environments, outputs, checkpoint hashes and failures: [`evidence/S3/20261016-stage3/`](../evidence/S3/20261016-stage3/).
- Independent raw-window recount and confidence decoder fixture: `evidence/S3/20261016-stage3/controlled_checks/pose_window_audit_latest.json`.
- Config: [`../configs/pose_stage3_paired_16epoch.json`](../configs/pose_stage3_paired_16epoch.json); candidate program: [`../scripts/run_pose_stage2_pilot.py`](../scripts/run_pose_stage2_pilot.py).
- Stage 2 keypoint values `158,581 / 59,141 px` are historical and invalid because the selected event windows were time-misaligned. The Stage 2 training/checkpoint logs remain archived.

No E-POSE sample, matched CAD, real inference, real score, or real-data tuning is claimed.
