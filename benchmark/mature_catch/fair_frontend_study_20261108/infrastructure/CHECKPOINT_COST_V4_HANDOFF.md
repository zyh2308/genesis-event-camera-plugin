# Block-boundary GPU handoff

Deploy only `checkpoint_blind_for_cost_v4.py` to a fresh server tools path and
run it once with the verified Genesis Python, without command arguments. It uses
the fixed actual fair-study root and old manager names supplied by research.
It never connects SSH, kills a process, changes GUI state, or rewrites the
320-actor protocol/source/parameters/bank ordering.

```sh
/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python -B -u R/tools/checkpoint_blind_for_cost_v4.py
```

`R` means the existing fair-study root. Before deployment, root verifies the
two old actual managers still have their supplied names. The helper refuses
already-sealed data or a stage where all blocks have already been claimed.

It atomically creates `deferred_for_cost_checkpoint` claim placeholders only
where no worker has claimed a block. Running blocks finish normally. It waits
for both original jobs to report real completed/exit0, their queue finished and
progress receipts, and their own central reservations to be released. A genuine
old worker failure stops this helper and remains a failure.

Own unexecuted placeholders are moved intact to
`blind32/scheduling_checkpoints/cost_v4/held_claims`. Old job/result/claim/progress
files are retained unchanged. The immutable scheduling checkpoint records their
hashes and the exact unchanged protocol SHA. Both future blind jobs are then
registered in the original blind directory with new worker IDs and the same
`scheduling_checkpoint_sha256` before either resumes; the existing collector
therefore waits for both registered jobs.

GPU1 resumes its original queue through the original atomic launcher. GPU0 then
uses that launcher for the separate actual cost worker:

```text
verifiedPython -B -u R/tools/run_cost_v4_study.py --root R/cost_v4_matched
```

Cost job metadata/logs live in fresh `R/cost_v4_gpu0_job`; this helper does not
create `cost_v4_matched`. The job inherits the entire unchanged original job
environment, uses a 14,400-second cost timeout, and adds no duplicate thread or
EGL settings. GPU1 continues blind work during the cost job. After the actual
cost manager exits and its reservation release is confirmed, GPU0 resumes the
same queue even if cost failed. An unconfirmed release stops the GPU0 handoff.
New blind managers use the existing 86,400-second timeout.

The helper narrows its CPU affinity to8/9, restores the full host affinity before
each resource-manager spawn, and immediately returns to8/9. It never starts two
tasks on the same GPU. The existing collector still requires all320 scientific
actors and all four registered blind workers to finish normally; no exemption
is added. Cost is outside the blind actor budget.

Independent root receipts are `BLIND_COST_V4_CHECKPOINT_STATUS.json`,
`BLIND_COST_V4_CHECKPOINT_COST_RECEIPT.json`, and
`BLIND_COST_V4_CHECKPOINT_RECEIPT.json`. Immutable request/checkpoint records live
in the blind directory and become part of its final manifest.

One CPU-only end-to-end handoff test passed: it retains completed original
blocks, archives30 unexecuted placeholders, checks the protocol and original
receipts never change, checks both new jobs carry the checkpoint SHA and are
preregistered, simulates a failed cost job with confirmed release, and verifies
GPU1/cost/GPU0 launch order and full-affinity manager spawning. No real process,
physics, renderer, GPU or SSH is invoked by that test.
