# g1 GPU0/1 read-only synthetic-flow parent

This runtime is preparation only. Nothing is deployed, connected, reserved, trained or
evaluated by creating these files or running the CPU tests. The parent imports standard
Python only and creates no CUDA context or dummy process.

Fixed resources: g1 RTX3090 GPU0 `GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23`
for the seed0/1 shard, GPU1 `GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460` for
seed2. Both require no compute application and >=22,000MiB free at actual admission.

Install the runtime under `ROOT/runtime/`; the scientific worker and its flat
`filename -> SHA256` manifest live in `ROOT/sources/`. ROOT is an independent
`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-flow-full-synthetic-readonly-*`
directory. Its existing source files are never changed by this parent.

After the research parent has checked the source/CPU evidence and decided to publish
ready, its **new** `ROOT/DEPLOYMENT_READY.json` must contain:

```json
{
  "schema": "flow_full_synthetic_readonly_ready_v1",
  "passed": true,
  "task_root": "ABSOLUTE_ROOT",
  "source_manifest_sha256": "ACTUAL_MANIFEST_DIGEST",
  "worker_contract_sha256": "ACTUAL_CONTRACT_DIGEST",
  "read_only_model": true,
  "optimizer_updates": 0,
  "real_scores_computed": false
}
```

`build_command.py` will not publish ready or allocate a resource. It only derives an
exclusive new command file from the actual bound files, verifies the contract and emits
the SHA to pass explicitly on the parent's command line:

```sh
CUDA_VISIBLE_DEVICES='' python -B ROOT/runtime/build_command.py \
  --task-root ROOT --gpu-index 0 --out ROOT/commands/gpu0.json
python -B ROOT/runtime/lease_flow_eval.py \
  --task-root ROOT --gpu-index 0 --contract-sha256 PRINTED_EXACT_SHA \
  < ROOT/commands/gpu0.json
```

The scientific worker CLI is exactly:
`python -B -u ROOT/sources/evaluate_frozen_synthetic.py --contract
ROOT/sources/contract.json --seeds 0 1 --out ROOT/seed01`, or the seed2 shard on GPU1.
Arbitrary shell commands, inline Python, extra flags, alternate interpreters, altered
seeds, symlink/path traversal, changed sources and existing task/output directories are
rejected. The stdin bytes themselves must match their explicit SHA256. The full source
manifest, entrypoint, scientific contract, parent source and root-ready receipt are bound
and verified again immediately before spawning the actual worker.

Admission holds the shared global reservation flock while acquiring the target UUID
flock, reading **every** registry JSON and checking hardware. Matching UUID records must
follow the existing `released*` naming convention and have `current_job=null` (or that
field absent). Existing legacy release suffixes are preserved. `complete`, `preparing`,
unknown nonrelease states and malformed records do not qualify. Busy/locked devices are
refused; there is no takeover, queue, retry or occupancy reservation before the real job.

The parent creates its own child session/process group and records leader PID, start
ticks, session, UID and owning parent PID/start ticks. The only possible termination is
SIGTERM then SIGKILL to that **new own group**, after verifying these boundaries. It never
signals an existing GPU or another task. A 60-minute worker limit, signals and failure all
pass through the same own-group cleanup. Failed cleanup keeps a `cleanup_failed` registry
status and never claims release. Completion or failure writes logs, sealed argv, worker
exit, live state, parent completion/failure and `resource_release.json`; no automatic retry.
The parent reports a successful worker exit, not an independently audited scientific result.

CPU safety tests use fake hardware strings, temporary source/ready fixtures, actual
nonblocking file locks and newly created tiny CPU child groups. They verify that a separate
unrelated child stays alive during cleanup. They never invoke `nvidia-smi`, SSH or CUDA.

```sh
CUDA_VISIBLE_DEVICES='' python -B runtime/test_lease_flow_eval.py
```
