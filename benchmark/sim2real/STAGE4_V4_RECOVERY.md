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
from the same seed-0 initialization. Its execution is pending synchronization
to g1.
