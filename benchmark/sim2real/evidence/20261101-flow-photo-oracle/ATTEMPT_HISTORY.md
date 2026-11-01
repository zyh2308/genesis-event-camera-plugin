# Flow photometric oracle attempts

All six diagnostic runs used isolated g1 run directories. Runs 01–05 exited before producing a scientific result; their logs are retained under `remote_runs/`. No optimizer update, checkpoint, real-data score, or persistent GPU reservation was produced by those failed attempts.

| Run | Exit | Failure point | Correction carried forward |
|---|---:|---|---|
| `20261101-stage8-flow-photo-oracle-v1` | 1 | The external STE-FlowNet source directory was verified but not placed on `sys.path`; `import models` failed. Earlier setup/preflight attempts in this directory also record a missing helper copy and a Git lookup against a copy-only snapshot. | Add the verified model bundle root to the import path and assert import origins. |
| `20261101-stage8-flow-photo-oracle-run02-v1` | 1 | The pinned model's optional correlation extension was not installed, so its symbol was undefined during the first forward. | Inject the already used, benchmark-local PyTorch equivalent and run its target-shape forward/backward comparison. |
| `20261101-stage8-flow-photo-oracle-run03-v1` | 1 | The post-loss per-scale decomposition removed the singleton batch axis from a `[B,H,W]` grayscale tensor; OpenCV received a 2-D array where the pinned loss expected a batch. | Preserve `[B,H,W]` and assert the adapter shape. |
| `20261101-stage8-flow-photo-oracle-run04-v1` | 1 | A descriptive full-resolution warp passed the coarsest 44×44 pyramid flow to 352×352 images. The pinned loss's own multi-scale warps had already run. | Use the shared full-resolution physical flow for the full-resolution regional metric. |
| `20261101-stage8-flow-photo-oracle-run05-v1` | 1 | The shared flow pyramid was deleted after the EventGenesis arm and then reused for V2E. | Keep it alive until both paired event arms complete. |
| `20261101-stage8-flow-photo-oracle-run06-v1` | 0 | Completed. | Final result is `remote_runs/20261101-stage8-flow-photo-oracle-run06-v1/output/result.json`. |

One local tar packaging attempt before run02 also used a relative second `-C` directory and failed before a Python process was started; the transfer was retried with absolute paths. All five remote runtime failures are harness defects, not evidence against the data, simulator, or model. The final run recomputed the complete registered diagnostic from the unchanged bank.
