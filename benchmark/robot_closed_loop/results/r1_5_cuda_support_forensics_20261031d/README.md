# R1.5 CUDA support forensics run D (incomplete output)

This run used g1 GPU1 (RTX 3090, UUID `GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460`) and code commit `1b5767a332fa4ecbf333a69fca1d40419e8bec8d`. Cache SHA checks and CUDA setup passed. The public `PhysicsInterpolator` multi-sample CPU/CUDA and four DVS combinations ran through the numerical phase, but the process exited `1` while serializing the final JSON summary: the summary object still contained NumPy event arrays. Consequently the full arrays, CSV, and summary were not written; this is **not an accepted parity result**. The captured run log records the traceback.

The defect was in the diagnostic output shape, not in projection or event code. Run E removes raw arrays from the JSON object while preserving them in `forensic_arrays.npz`; it uses a new run ID and will rerun all computations.

- Runtime logs: `../runtime_logs/r1_5_cuda_support_forensics_20261031d/`.
- Remote run root: `/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_cuda_support_forensics_20261031d/`.
- Original source/config snapshot and partial output metadata are retained in this directory.
- No input cache or other process was changed; the run released its own GPU reservation on exit.
