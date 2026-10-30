# Partial forensic run: r1_5_cuda_support_forensics_20261031b

This run used source commit 71593e29ea8327d02ad5c5576de516fd0f4e7f25, g1 GPU1 (RTX 3090, UUID GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460), Torch 2.5.1+cu121, and the unchanged 0.995 coverage threshold. Both input cache hashes passed preflight. The computation took 3.40 s, but output generation exited 1 because event_metrics.csv combined rows with different columns. The summary and all 424 per-target rows were written; the raw array bundle, complete event table, and layer-contributor JSONL were not.

## What the partial output establishes

- With identical float32 projection arrays supplied to both compositors, the CPU and CUDA paths had zero final-mask disagreements for either projector on both episodes. Component support, surface consistency, and surface IDs also matched exactly. Coverage accumulation differed by at most 1.20e-7; supported log-luminance differed by at most 4.77e-7.
- Replacing only the projector reproduced the full prior mask disagreements: 79 pixels in episode 004 and 27 in episode 005. The independent projections differ by at most 6.11e-5 px and 2.39e-7 m in projected depth. Every originally XORed target endpoint decision matched the prior arrays.
- On those original XOR targets, endpoint-1 coverage rejection was the most common reported reason (004: 27/79 with NumPy projection, 41/79 with CUDA projection; 005: 8/27 and 6/27). Other targets included cross-endpoint depth-match and surface-ID/consistency rejections. This is consistent with one-float32-ULP projection differences being amplified at strict coverage and depth-match boundaries; it does not support changing the 0.995 threshold.
- The in-memory summary reported equal CPU/CUDA DVS event counts and zero matched position, polarity, and time differences for the tested paths, plus passing isolated threshold probes. Since raw event arrays were not saved, these are preliminary only.

The archived acceptance_result.all_gates_pass in this partial file was calculated from same-projection compositor comparisons and therefore does not apply the frozen end-to-end NumPy-projector/NumPy-compositor versus CUDA-projector/CUDA-compositor mask gate. The native end-to-end result remains failed at 79/27 mask pixels. The summary run_id is also a logging defect. No acceptance or freeze decision should use this partial run.

See the complete traceback and resource record in ../runtime_logs/r1_5_cuda_support_forensics_20261031b/. The per-target evidence is preserved in point_components.csv; summary.json includes projection, compositor, DVS, and control metrics. A new run will correct the event-table schema, report the native end-to-end gate explicitly, and save the event arrays and layer traces.
