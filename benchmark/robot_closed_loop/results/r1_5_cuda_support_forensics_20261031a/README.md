# Failed run record: `r1_5_cuda_support_forensics_20261031a`

The run launched on g1 GPU1 (RTX 3090, UUID `GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460`) with source commit `aeeb832c3d94a514775aa5861c48a85a0a3ba9b5` and exited with code 1. The cached episode files passed SHA256 preflight; the failure occurred while serializing the first per-point diagnostic row: Python's JSON encoder rejected a NumPy `float32` value.

No `summary.json`, point CSV, event table, or diagnostic NPZ was produced. The run therefore provides no evidence about the 106 support discrepancies. The complete traceback and resource logs are in [`runtime_logs/r1_5_cuda_support_forensics_20261031a`](../runtime_logs/r1_5_cuda_support_forensics_20261031a/). The GPU reservation was released by the worker.

The runner now converts NumPy scalar and array values to JSON-safe Python values. The retry uses a new output directory and run ID `r1_5_cuda_support_forensics_20261031b`; it does not overwrite this failed attempt.
