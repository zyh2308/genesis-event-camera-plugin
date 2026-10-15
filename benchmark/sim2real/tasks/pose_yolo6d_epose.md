# Object-pose route: YOLO-6D → E-POSE

Status: source audit only; real data availability remains unresolved.

- Pinned candidates: `microsoft/singleshotpose@2c282619101f9f31666dd18b42393375bea1600f`; E-POSE annotation source `oussamaabdulhay/E_Pose@b96ae783fcb6de84e1e9c602e91bce941cab0234`.
- The available local code was an incomplete source snapshot, not a runnable checkout. E-POSE source adapter is for ROS bags and a hard-coded three-object case; SingleShotPose loader is LINEMOD/RGB/mask based.
- Required adapter: full-frame, unsegmented event representation plus shared pose/keypoint labels; fixed-seed paired init; synthetic-only validation. Avoid LINEMOD evaluation/weight init, GT mask crops, segmented events, and RGB compositing.
- Verify E-POSE official data availability, CAD units/origin/K, pose direction, and the three-channel accumulator semantics before claiming even a loader smoke.
- If source/data remains unavailable, keep this route optional and report exact blocker; continue L1/L2.
