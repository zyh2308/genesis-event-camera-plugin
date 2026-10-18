# Pose line: E-POSE access and normal-inference audit

Checked from WSL on 2026-10-19 01:34–01:37 JST (2026-10-18 16:34–16:37 UTC). This is a pose-only update; it did not use g1/SSH, real sample scores, or any other task's state.

## E-POSE data and CAD access

The official Figshare landing page was requested again. It returned HTTP 202 with an empty body; the public Figshare article API returned HTTP 403 with a 118-byte response. The prior captured request on 2026-10-16 recorded the landing response's AWS WAF challenge header. This confirms that this WSL route still cannot retrieve the archive; it does not establish that E-POSE is unavailable worldwide. The DataCite record previously reports 29,759,153,475 bytes and CC-BY-4.0 but no file IDs.

The authors' public repository is available locally at `/home/科研/Eventbased_WAM/external_sources/E_Pose`, clean at commit `b96ae783fcb6de84e1e9c602e91bce941cab0234`. It contains meshes for the paper's named object categories, including `wood_block`. I downloaded and hash-checked the author repository's `wood_block/google_16k/textured.obj` asset: 1,430,303 bytes, SHA-256 `a1cccc1d4be3fa5ef3b12b780edf15d4f8978bfae93ecfb0265314ce1d7c4e0c`, and Git blob `43b117a6b9b621d0f871d96123bce8d551f5a3c5`, matching the pinned author tree. The OBJ has 8,194 vertices and raw-coordinate extents `[0.103917, 0.103464, 0.206003]`; its coordinate unit is not specified, so I do not assume metres. The category-named model is available, but no E-POSE record was obtained to prove that it is the CAD paired with a particular event/pose sample.

The paper states that the original data is H5 and that per-sample MAT records include event packets, `Pose`, `IntrinsicMatrix`, `cameraPose`, `time`, projected keypoints, and class index. The authors' README says its MATLAB processing scripts currently read ROS bag files and that direct processing of the published raw H5 files was future work. Consequently, event timestamps/polarity, K, time alignment, transform convention, exact mesh scale, object correspondence, and symmetry remain unaudited against any actual record. No real inference ran. The generic simulated cuboid remains only the synthetic diagnostic fixture.

Sources: [E-POSE paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11822054/), [authors' E-POSE code/CAD repository](https://github.com/oussamaabdulhay/E_Pose), [official Figshare record](https://springernature.figshare.com/articles/dataset/E-POSE_A_Large_Scale_Event_Camera_Dataset_for_Object_Pose_Estimation/28106717).

## SingleShotPose BN and confidence findings

The previous paired pilot config uses 16 epochs and 8 training windows. Pinned `region_loss.py` defaults to `pretrain_num_epochs=15` (lines 80–93), and includes the confidence term only when `epoch > pretrain_num_epochs` (lines 146–161). Therefore only the eight updates in epoch 16 supervised confidence. This is a direct, verified explanation for why ordinary confidence argmax cannot be trusted at that pilot budget. It is not a reason to select the target cell at inference.

I added `recalibrate_batchnorm_from_synthetic_train` to the pose training adapter. It resets and cumulatively estimates BN running buffers using only the fixed synthetic training event tensors, without reading labels, validation inputs, or real data; it changes no model parameters and returns the model to eval mode. The pose adapter now records calibration IDs/hash and BN state, exposes the confidence warmup length, and can fail before model construction if a configured minimum number of confidence-supervised updates is not met. The prior 16-epoch config is preserved; the new seed-0 candidate config requests 31 epochs, retains the pinned 15-epoch warmup, and requires at least 128 confidence updates for the eight-window training split.

A new CUDA fixture audit was run against the clean pinned SingleShotPose commit on the local RTX 2050. It uses two fixed event windows from one tiny synthetic training rollout, reports predictions only after `model.eval()` using the ordinary global confidence-argmax decoder, and never uses GT-cell selection or crop/mask/box inputs. Results:

- After the first 16 updates, eval-mode mean keypoint EPE on the same tiny training fixture was 23,431.24 px; predictions had not learned the confidence-selected cell and were numerically extreme.
- Re-estimating BN buffers from the two synthetic training inputs, with weights frozen, reduced the eval-mode mean EPE to 28.43 px. The confidence argmax was still one cell away, so BN recalibration alone does not fix the confidence head.
- After 64 additional confidence-supervised synthetic updates, the regular eval-mode confidence argmax selected the center cell on both fixture windows. A final train-only BN calibration yielded mean EPE 1.42 px (1.88 / 0.97 px per window).
- Saving and restoring the calibrated model preserved the complete state hash and ordinary eval outputs. BN calibration changed no trainable parameter. The exact report and runtime evidence are in `eval_contract_fixture.json`, `exit_code.txt`, `started_utc.txt`, `finished_utc.txt`, and `environment.txt`.

This only demonstrates that the two training-state issues can be corrected on a deliberately tiny fixture. Both windows are from the same rollout and are training data; the result is an overfit/mechanism check, not held-out-rollout validation or evidence of E-POSE performance. The actual fixed physical bank must still pass normal eval and checkpoint restore on independent synthetic validation rollouts before the pose line can start its paired seed-0 training.

## Reproduction

```bash
python benchmark/sim2real/scripts/audit_pose_eval_contract_fixture.py \
  --pose-source /home/科研/Eventbased_WAM/external_sources/SingleShotPose
python benchmark/sim2real/scripts/audit_pose_event_windows.py
```

The first command completed with exit code `0` in 12 seconds on Python 3.14.6, PyTorch 2.13.0+cu130, CUDA 13.0, NVIDIA GeForce RTX 2050. The pinned source was clean at `2c282619101f9f31666dd18b42393375bea1600f`. The second command independently verifies window endpoints and deterministic confidence-cell decoding; its raw output is retained in `pose_window_fixture.json`.

The WSL evidence paths and hashes are indexed in `evidence_index.json`. The author CAD asset remains outside Git under `/home/科研/Eventbased_WAM/external_sources/E_Pose/`; the E-POSE raw sample remains unavailable through the checked publisher route.
