# R1.5 CUDA support forensics run E (incomplete output)

The run used g1 GPU1 (RTX 3090, UUID `GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460`) and source commit `7fac4b16281aeab4b70109860092bacebefbc975`. Both cached input hashes and the CUDA environment passed preflight. The public CPU/CUDA interpolation and DVS work completed, but the process exited `1` before writing the summary because the isolated remote source bundle did not include a historical cost-summary JSON that the report assembly tried to read. No complete numerical arrays or summary were emitted; this run is **not accepted**.

The failure is in result packaging. Run F will make the cost reference self-contained and keep the per-run computation limited to files present in the staged source and the verified read-only state cache.

- Runtime traceback and resource record: `../runtime_logs/r1_5_cuda_support_forensics_20261031e/`.
- Remote run root: `/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_cuda_support_forensics_20261031e/`.
- The only files in this output directory are the original config and run manifest. No cache or other user's process was changed; this run released its own reservation.
