# Object-pose route: YOLO-6D → E-POSE

Status: source and public data page checked; no E-POSE sample, loader, or model run yet. Integration remains unresolved.

- Pinned candidates: `microsoft/singleshotpose@2c282619101f9f31666dd18b42393375bea1600f`; E-POSE annotation source `oussamaabdulhay/E_Pose@b96ae783fcb6de84e1e9c602e91bce941cab0234`.
- Full pinned source snapshots are present on g1. E-POSE repository provides annotation tooling rather than a complete downstream trainer; SingleShotPose's path is LINEMOD/RGB/mask/crop based.
- Required adapter: full-frame, unsegmented event representation plus shared pose/keypoint labels; fixed-seed paired init; synthetic-only validation. Avoid LINEMOD evaluation/weight init, GT mask crops, segmented events, and RGB compositing.
- The publisher Figshare record reports a 27.72 GB public archive. It was not downloaded in this pilot; the shared fall-box object also does not establish YCB CAD, camera intrinsics, or pose-frame mapping.
- Next, inspect one raw event/pose/CAD/K sample and design a full-frame unsegmented event input. GT masks, crops, segmented events, LINEMOD validation, or real-data checkpoint selection cannot become model inputs or selection signals.
- This route is unresolved at leakage-safe adapter/integration, not confirmed unavailable. Continue the two passing routes while the research review decides whether the existing mature pose source fits the locked task definition.
