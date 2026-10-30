# r1_5_core_cuda_parity_20261030c

Bounded NumPy versus Torch/CUDA parity on cached robot states, plus noiseless DVS threshold-adjacent controls. This does not validate Genesis rendering, photometric sampling fidelity, real-time cost, or closed-loop recovery.

- Source commit: `6e8edc57b4109c85b5566ba9b7f5496e46fc2e66`
- GPU: `NVIDIA GeForce RTX 3090`; Torch `2.5.1+cu121`; CUDA runtime `12.1`
- Acceptance: `{"threshold_dvs_cpu_cuda_exact":true,"episodes":{"004":{"support_mask_exact":false,"supported_log_luma_max_within_limit":true,"cached_event_count_relative_delta_within_limit":true},"005":{"support_mask_exact":false,"supported_log_luma_max_within_limit":true,"cached_event_count_relative_delta_within_limit":true}},"all_gates_pass":false,"note":"backend parity only; this is not a renderer-fidelity or closed-loop acceptance decision"}`
- Inputs: immutable cached episodes 004 and 005, interval 0.5805–0.5905 s.
- Detailed measurements: `summary.json`, `cached_episode_metrics.csv`, `threshold_dvs.json`; arrays: `cuda_parity_arrays.npz`.
