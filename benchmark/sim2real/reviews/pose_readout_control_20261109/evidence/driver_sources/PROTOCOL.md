# Fixed predicted-peak readout diagnostic

Zero optimizer updates and no real data. C1 RAW control uses the original Torch global expectation unchanged. The sole candidate multiplies its original 16x16 probability by the registered predicted-peak Gaussian, sigma 1/16, and computes its normalized expectation. Original probability, FP32 Torch control, and original FP64 expectation remain separately recorded.

Depth and size use the unchanged independent C1 depth model. Both arms actually rerun the original predicted crop, ray frame, frozen rotation head, and native seven-metric scorer. The control must match all original matched-64 reference predictions and native scores before candidate pose inference. Baseline and full use identical code; baseline uses seed0, full uses all three seeds and requires the SHA-bound successful baseline.

All four rollouts and all 64 windows remain. Baseline has 384 paired cases and 12 scores; full has 1152 paired cases and 36 scores. `readout_maps.npz` retains every original FP32 attention and focused FP64 weight in JSONL order, explicitly indexed by `readout_map_index.json` and each case's `map_index`. Vertices and all nine object keypoints are retained for independent CPU checks. Model, input, contract, source, and reference hashes are checked; assigned GPU visibility is preserved.

This is a fixed synthetic mechanism comparison. Matched normal64 is distinct from clean384. The original formal30 and formal8 are not retested and cannot be claimed passed by this probe. A separate registered mechanism criterion may assess these results; it does not authorize real scoring or checkpoint selection. No GT enters the candidate readout, crop, depth, or rotation forward pass.
