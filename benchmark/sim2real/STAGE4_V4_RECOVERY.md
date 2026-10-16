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

The rerun of finalization, adapter preparation, and both training arms is pending until this patch is synchronized to g1.
