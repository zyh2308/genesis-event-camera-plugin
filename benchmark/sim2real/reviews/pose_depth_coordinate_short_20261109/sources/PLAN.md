# Pose depth coordinate short trial

The C1 depth branch predicts log projected scale `s` and log shape `h`. It decodes depth as `log z = h - s + log f`. In the fixed darkest cohort, EventGenesis has a mean log-shape bias of −0.05337 and a log-scale bias of +0.00930. The biases combine in log depth. PoseCNN uses a directly predicted object depth with the projected center and camera intrinsics. This experiment adapts that separation to the existing focal-normalized geometry. [PoseCNN, section III-B](https://arxiv.org/pdf/1711.00199)

The single candidate changes the final affine layer's trainable coordinates to `(s, q)` with `q = h - s`. Initialize its second row by subtracting the native scale row from the shape row. Reconstruct the native output `(s, h=s+q)` before the original C1 loss, teacher terms and native evaluator. The parameter count and function at initialization are preserved. The optimizer step can differ because AdamW operates on the new coordinates. This tests the parameterization, distinct from the earlier teacher-gradient projection and center-mask supervision.

1. Lock the actual RAW128 caches, old C1 depth source states and all six 288-step data/label tapes. Perform strict GPU4 queue admission.
2. Verify source-to-basis and basis-to-native output parity and run one real candidate update per method. Independently rebuild the transformed model and Adam state. Compare the first scalar to the original C1 log.
3. After the SHA-bound smoke, fit the six depth branches for 288 updates each from the original source initialization. Reuse C1 center and frozen rotation. Export the trained native-layout state after conversion parity.
4. Use the frozen crop, scoring and original 30+8 checks. Include complete clean384 and original clean_full, matched-normal64, half64 and eighth64, with every rollout including 031 and 063. The experiment reads the established synthetic development cohort and leaves E-POSE and the independent 32-rollout replication closed.
