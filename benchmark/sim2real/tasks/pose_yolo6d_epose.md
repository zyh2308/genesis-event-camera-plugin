# Object-pose route: SingleShotPose / YOLO-6D → E-POSE

Status: full-frame event-only synthetic closure was executed on a generic shared Genesis cuboid. The E-POSE-specific sample/CAD contract and class-matched task remain blocked pending one accessible raw sample and matching CAD; this is not a claim that E-POSE is unavailable.

## Model/input audit

- Pinned candidates are Microsoft SingleShotPose commit `2c282619101f9f31666dd18b42393375bea1600f` and E-POSE annotation source commit `b96ae783fcb6de84e1e9c602e91bce941cab0234`. The E-POSE repository is annotation tooling, not a ready downstream trainer. The pinned SingleShotPose implementation uses an RGB/LINEMOD loader and real-data validation path, so neither can be used for this protocol.
- The benchmark pilot adapts the pinned SingleShotPose 416×416 full-frame Darknet input and RegionLoss target format. Inference input is only three raw-event-derived channels: ON log-count, OFF log-count and total log-count, with a fixed count scale of 8 over each 10 ms window. Supervision is shared Genesis pose/K projected centroid and eight cuboid corners.
- No RGB, GT crop, GT mask, GT box, segmented events, depth, segmentation or pose enters inference. The generic cuboid class has no matched E-POSE/YCB CAD and does not establish compatibility with the 13 published object classes.

## Executed pilot and result

- Both generators used the same six whole rollout bank, seed 71 paired initialization (initial state hash `a47ecb230e60d205f5973b981e959c2226cc650d821ac1c4f126a1cf1fde4796`), two epochs, 16 updates, eight train windows and four independent synthetic validation windows. Checkpoint save/restore state hashes matched. Peak allocation was about 782 MiB; the second pilot's train wall time was 1.05 s for EventGenesis and 0.46 s for V2E.
- Pilot v1 selected too few useful visible windows and produced extremely large validation keypoint error. The selector was corrected to choose evenly spaced fully visible windows from each train rollout; pilot v2 completed with more windows. Its synthetic validation keypoint mean EPE remained `158,581 px` for EventGenesis and `59,141 px` for V2E. These enormous values indicate the generic short pilot has not produced a usable pose predictor. They are not a generator comparison and cannot be treated as an E-POSE score.
- The interface/optimizer/gradient/checkpoint route executes, but task closure is incomplete: labels need a real E-POSE class and CAD match, and the current output behavior is poor. Do not scale this generic route or report the apparent difference as a performance result.

## E-POSE sample contract and access evidence

- The paper reports 306 sequences, 13 single-object classes plus four multi-object scenes, raw H5 events and per-window annotations including `T_dvs^obj`, `IntrinsicMatrix`, `T_base^dvs`, time, projected keypoints and class index. It describes 10 ms windows. These are publication facts only; the raw sample has not been parsed.
- The publisher record lists a 27.72 GB full archive. On g1, the read-only Figshare API metadata route returned HTTP 403 and the publisher landing-page request returned HTTP 202 with an empty body. An earlier local API request also returned HTTP 403. These responses blocked sample retrieval in the current environment; they do not establish global data unavailability. No automated retries or bulk archive download were run.
- The raw event timestamp unit/polarity/order, exact K matrix convention, object/camera transform direction, coordinate axes and units, timestamp synchronization, CAD mesh/model scale and class mapping remain to be checked against one released raw sample and its corresponding CAD.
- Candidate real metric is ADD/ADD-S success at `0.1d` with supporting reprojection/translation/rotation metrics. That candidate is not frozen, and no real evaluation entry point was exercised.

## Recommended next gate

Obtain one original E-POSE raw H5 plus its matching CAD through an accessible route, then write a sample-level contract audit for raw events, timestamps, K, transforms, axes, units and CAD scale. Preserve the event-only full-frame inference rule. After that audit, decide whether a class-matched synthetic bank and the mature SingleShotPose output/loss are appropriate. If the paper's GT labels or CAD format do not match that output contract, ask the research session to approve a concrete model or task change before implementing it. Reconstruction and optical flow do not depend on this gate.

## Reproduction and evidence

- Candidate config: [`../configs/pose_stage2_pilot.json`](../configs/pose_stage2_pilot.json).
- Pilot v1 and v2 commands, logs, selected windows and result JSONs: [`evidence/S2/20261016-stage2-pilot/runs/20261016-s2-pose-pilot-v1/`](../evidence/S2/20261016-stage2-pilot/runs/20261016-s2-pose-pilot-v1/) and [`evidence/S2/20261016-stage2-pilot/runs/20261016-s2-pose-pilot-v2/`](../evidence/S2/20261016-stage2-pilot/runs/20261016-s2-pose-pilot-v2/).
- External checkpoint paths, sizes and hashes: [`evidence/S2/20261016-stage2-pilot/external_artifacts.json`](../evidence/S2/20261016-stage2-pilot/external_artifacts.json).

No E-POSE sample, real model score, checkpoint selection, or real-data tuning was performed.
