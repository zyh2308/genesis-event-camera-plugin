# Optical-flow route: STE-FlowNet → MVSEC

Status: source audit only; no code/data execution yet.

- Pinned source: `ruizhao26/STE-FlowNet@eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6`.
- Model remains the official event-only STE-FlowNet with its native polarity-separated Gaussian-weighted counts; grayscale supervision is synthetic only.
- Must fix in the benchmark adapter, not frozen core: undefined generic `data_split`, empty/duplicate timestamp slices, uint8 count overflow, training-time real-MVSEC validation, and invalid `outdoor_day1` loader range.
- Environment mismatch: server PyTorch/CUDA differs from the README-era PyTorch 1.10.2/cu113 and correlation sampler stack; compile/install only after a small feasibility check.
- Required smoke: identical paired trajectories/events/gray frames/flow labels; loader edge cases; forward/backward; short synthetic train and synthetic validation for both generators.
- MVSEC is test-only; no model score or sequence selection from real results in this phase.
