# Stage 4 v4 recovery record

Run ID: `20261016-stage4-flowdt1-seed0-v4`  
Run root on g1: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4`

## Preserved completed work

- The registered Torch CUDA shared bank completed all 32 rollouts with campaign stage exit code `0`.
- Shared HDF5 SHA-256: `e9e57131c84059047bddd0d34ac3271f23f5e4ff0e4fa5e0b9f8579772ee337a`.
- The four pinned V2E workers each completed their eight assigned rollouts; all 32 episode rows and four worker exit codes are `0`.
- Worker outputs remain under `v2e_events/gpu{1,2,4,5}/`; none were deleted or regenerated.

## Failed finalization and root cause

The first finalizer invocation exited `1` before writing a manifest. Its traceback is preserved at:

`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/v2e_finalize.stderr.log`

The finalizer assumed flat paths such as `v2e_events/s0_00.h5`, while the shard runner writes `v2e_events/gpu1/s0_00.h5` (and similarly for GPUs 2, 4, and 5). This was an interface mismatch between the sharded runner and finalizer, not a V2E generation failure. At the time of this record, no adapter or optimizer update had run.

## Recovery

`scripts/finalize_stage4_v2e.py` now locates exactly one matching HDF5/status pair across the four registered shard directories (with flat layout retained for prior single-directory runs), verifies that the status record names the same path and rollout, and checks the recorded SHA-256. The preserved outputs are finalized in place.

`scripts/resume_stage4_after_v2e_server.sh` verifies all shard completion records, runs finalization and adapter preparation into their original target paths, then launches the paired seed-0 training on GPUs 2 and 4. `scripts/start_stage4_training_after_v2e_tmux.sh` runs this recovery path in a detached tmux session. Both scripts preserve the initial failed `exit_code.txt`; recovery has separate stage exit records.

The corrected finalizer rerun completed with exit code `0` and wrote
`v2e_events/v2e_complete_manifest.json` (SHA-256
`5d79bad2f9fe81e1fad8af0676b9048aefaf73c0b6edd8db7f39f31ef82beb2f`).

## Adapter path mismatch

The first adapter attempt then exited `1`; its traceback is preserved at:

`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/steflow_adapter.stderr.log`

The adapter also assumed flat V2E paths and failed when opening
`v2e_events/s0_00.h5`. Before that failure it had created all 32
EventGenesis rollout NPZ files (24 train, 8 validation) in the partial adapter
directory. The retry runner preserved those files as
`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/steflow_30ms_partial_adapter_failure1`
before the retry.

The shared `stage4_v2e_artifacts.py` resolver is now used by both finalizer
and adapter. The adapter also cross-checks each shard status and HDF5 hash
against the complete V2E manifest. A dedicated retry runner archives the
partial adapter, rebuilds the canonical adapter path, then launches both
training arms. The corrected adapter completed with exit code `0`. Its
training bank manifest is 71,504 bytes with SHA-256
`c9da4158d9775997f08dd75ce0ac2ccc33873d983422ef87c931e834f6da0624`.

## Upstream photometric-loss device mismatch

Both seed-0 training arms then exited `1` at the first photometric-loss call,
before any optimizer update. Their `training_metrics.jsonl` files were empty
and no checkpoint was written. The paired stderr logs are under each arm's
`flow_dt1_seed0/{eventgenesis,v2e}/train.stderr.log`.

The pinned `STE-FlowNet/multiscaleloss.py` converts endpoint image tensors to
NumPy for its OpenCV resize, while Stage 4 had passed those tensors on CUDA.
The objective adapter now detaches only the grayscale endpoint inputs to CPU
before this resize; predicted flow remains on CUDA with autograd enabled. The
Charbonnier, warp, scale aggregation, and smoothness formulas are unchanged.
The same device adaptation was applied to the dt4 objective helper.

The retry runner archives the two no-update attempts and restarts both arms
from the same seed-0 initialization. The first retry exited `1` before any
optimizer update; its failure and the next shape correction are recorded below.

## Pinned loss grayscale-shape mismatch

The detached retry (`training_retry2`) reached the first loss call but both
arms exited `1` before an optimizer update. In addition to detaching the gray
frames to CPU, the adapter had retained a singleton channel dimension:
`[B,1,H,W]`. The pinned loss indexes each batch item as `[H,W]` before calling
OpenCV; with the extra channel, OpenCV 5 received `[1,H,W]` and raised
`(-215:Assertion failed) m->dims <= 2`. The exact paired traceback is preserved
in each arm's `train.stderr.log` under the run root.

`_gray_batch_for_pinned_loss()` now explicitly accepts `[B,H,W]` or `[B,1,H,W]`,
validates nonempty finite images, removes only the singleton channel, and
returns contiguous detached CPU tensors. Both dt1 and dt4 objectives use this
adapter; the GPU flow output and its autograd graph are unchanged. The corrected
rerun is launched by `scripts/start_stage4_training_after_shape_fix_tmux.sh`,
which first archives both no-update attempts under
`flow_dt1_seed0/failed_attempt02_pinned_loss_channel_shape/`. The result remains
subject to the paired update-20 evidence below.

## Paired seed-0 update-20 delivery

The corrected run started on g1 at `2026-10-16T16:34:32Z` under tmux session
`stage4_flowdt1_seed0_shape_fix`, with the Stage 4 Torch CUDA config
(`c0ab0d17f024b00106957c3e44d7a013e920f6e760f70266c43a9cf904634685`). It uses
the pinned STE-FlowNet source, seed `0`, the same initial state for both arms
(`acc6bbd11488a21868c5f147c7f397521480166c24914cc7988a6428d6ed71cf`),
effective batch `8`, Adam `4e-4`, and the registered `45 × 198 = 8910` update
budget. EventGenesis ran on physical GPU 2; V2E ran on GPU 4. No real data or
GT flow entered training.

Both arms completed at least 20 actual `optimizer.step()` calls. Their immutable
first-20 metric snapshots are committed under
`evidence/S4/20261016-stage4-flowdt1-seed0-v4-update20/`; each has exactly 20
contiguous update records, effective batch 8, finite loss components, finite
gradient norm, and 70 finite gradient tensors per update. The `update_000020.pt`
checkpoints are external on g1, 580,038,432 bytes each:

- EventGenesis: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/flow_dt1_seed0/eventgenesis/update_000020.pt`, SHA-256 `f7eb58f5dae4b914050a38c5c68596e23a64372f68584fa5c1ba00e3c14c63ec`.
- V2E: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/flow_dt1_seed0/v2e/update_000020.pt`, SHA-256 `ce2b5b076b9ba6df9af525a6ca478c8f3d0b20a37e7783de433986619d7cc763`.

For each checkpoint the training process restored model, optimizer, and
scheduler state and compared a synthetic validation probe. The model-state
hash matched and prediction max absolute error was `7.152557373046875e-07`
for EventGenesis and `6.705522537231445e-08` for V2E. This is a checkpoint
integrity check, not a real-data score. Peak allocated GPU memory through
update 20 was 24,354,208,256 bytes per arm.

The `TRAINING_LIVE_STATUS.md` snapshot in the evidence directory was captured
at 58 updates per arm; the full run continued after the first delivery point.
The mutable current JSONL, stdout/stderr, PID and exit records remain under the
g1 run root. The live-status reader was also corrected to read per-update JSONL
records between checkpoint writes; commit `ea64262` contains only this monitor
correction, while the actual training code was launched from `1fab7e1`.

Reproduction/inspection entrypoints:

- Launch/recovery: `scripts/start_stage4_training_after_shape_fix_tmux.sh` and
  `scripts/resume_stage4_after_shape_fix_server.sh`.
- Full metrics: `.../flow_dt1_seed0/{eventgenesis,v2e}/training_metrics.jsonl`.
- Human-readable live status: `TRAINING_LIVE_STATUS.md`.
- Immutable first-20 metric records and SHA-256 values:
  `evidence/S4/20261016-stage4-flowdt1-seed0-v4-update20/README.md`.

At the time this update-20 snapshot was written, the registered epochs and
synthetic-validation selection had not completed. The subsequent full run did
complete; final metrics, checkpoints and exit records are reported separately
in [`STAGE4_SEED0_TRAINING_REPORT.md`](STAGE4_SEED0_TRAINING_REPORT.md). These
training loss values remain optimization diagnostics only, and no real-data
ranking has been run.
