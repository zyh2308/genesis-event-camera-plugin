# Focal-normalized depth coordinate short trial

The candidate changes the depth branch's final affine coordinates from `(s,h)` to `(s,q=h-s)`. The native projected scale is `s`, and the native shape output is reconstructed as `s+q`. The initial affine rows and biases transfer by subtraction and convert back by addition. All original losses, normal teacher, physical sampling, augmentations and optimizer hyperparameters follow C1. The original shape loss still couples the two outputs. AdamW operates in the new coordinates, so its effective step direction may differ.

The fitted coordinate checkpoint keeps the optimizer state in its training coordinates. A separately exported native-layout checkpoint loads into the original `CenterScale` and serves the unchanged scoring path. Initial original-to-coordinate, initial coordinate-to-native, and final coordinate-to-native parity cover logits, UV, log size, log shape, metric depth and scale with the fixed 2e-6 bound. The checkpoint check rebuilds the transformed model and Adam state independently. Both phases start from their registered source initialization.

The two method caches are the actual hash-bound RAW128 arrays prepared by the earlier admission, with original NPZ labels and relit geometry. Only RAW images, original targets and source records are used. All six original C1 depth sample and label tapes are replayed before any update. Smoke performs one actual update per method from the full original 1,529-window tape and verifies the native scoring contract on both methods. The study performs six 288-update fits. The existing independent C1 center and original rotation weights serve both arms. The predicted scale continues to drive the native crop.

Evaluation retains clean384, matched normal64, half64, eighth64 and all original clean confirmation windows. Per-rollout results retain 031 and 063 as part of the complete inventory. The frozen coverage module computes the original 30 checks and eight confirmation comparisons. They remain the adoption criteria. The darkest 031 crop diagnosis showed a separate center problem, which this depth-coordinate intervention does not target. The established synthetic development cohort supplies all results. E-POSE and the independent 32-rollout replication remain separate.

The existing GPU4 guard is supplemented by the exact parent PID, start ticks, command line and script SHA. The verified queue command ends in file descriptor 6. The parent submits the job through the owned queue after the current job releases it.

## Commands

```sh
python -B -m unittest -v test_depth_coordinates
python -B run_depth_coordinate_short.py --contract ACTUAL_CONTRACT --contract-sha256 ACTUAL_SHA --source-manifest-sha256 SOURCE_SHA --phase smoke --out FRESH_SMOKE_DIR
python -B run_depth_coordinate_short.py --contract ACTUAL_CONTRACT --contract-sha256 ACTUAL_SHA --source-manifest-sha256 SOURCE_SHA --phase study --out FRESH_STUDY_DIR --smoke-receipt FRESH_SMOKE_DIR/completion.json --smoke-receipt-sha256 ACTUAL_SMOKE_SHA
```

`DEPTH_COORDINATE_CONTRACT_TEMPLATE.json` documents the parent-verified server inputs. The actual contract digest is supplied separately. Local CPU evidence includes five tests, including the actual original and transformed first-step fixture and a failed tape mutation. The full server data and GPU run are verified during actual admission.
