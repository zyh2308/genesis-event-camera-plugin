# Independent v3 coordinator

Goal: validate the isolated cost-v3 path before consuming both GPUs with the
full blind study, while preserving every existing task, parameter, source file,
development outcome, and blinded bank.

The only new executable is `advance_fair_study_v3.py`. It uses the fixed actual
server root already used by v2, sealed source `source_cae1ab9e7f48`, and commit
`4931807d1cb229a55a96961b5c4d99b054ef4fb0`. It never edits the actor or simulator.
The independent five-test CPU suite starts no real job, renderer, physics or SSH.

Execution order:

1. Complete the existing development audit, or verify and reuse its existing
   completion marker/result/audit/manifest hashes. An existing selection lock is
   verified against all three development hashes and reused; it is not reranked.
2. Generate the unchanged 320-actor, 32-block blind protocol after the lock.
   An existing blind directory is refused to prevent duplicate execution.
3. Start the GPU1 blind worker through the original atomic launcher.
4. On GPU0, use that same atomic launcher for one cost-only child sequence:
   record the new public physics reference, strict Ours25 legacy validation,
   required-only validation, then exact ordered-event parity. All outputs are
   in the fresh `cost_v3_preblind_smoke` directory. Inputs are the locked Ours
   frontend and predeclared seed31 development reference. No formal cost repeat
   or CSV is generated here.
5. Wait for the cost launcher process to finish, record its actual result and
   sequence hashes, and confirm its own reservation is released. Normal cost
   errors stop the remaining smoke steps, retain all logs/receipts, and do not
   prevent launching the GPU0 blind worker. An unconfirmed resource release is
   the sole safety stop before that GPU0 launch.
6. Audit all 320 blind actors using the same collector. Missing/failed actors
   cannot become partial results. Blind waiting permits 86,400 seconds.

The coordinator narrows itself to CPU8/9. Before starting each resource manager
it restores the actual full host affinity (128 CPUs on g1); the manager inherits
that affinity and performs its original CPU/GPU allocation. The coordinator
immediately narrows itself again. It never launches two workers on one GPU.

Deploy only the new driver to a fresh server path, for example
`R/tools/advance_fair_study_v3.py`, and start it with the existing Genesis Python:

```sh
/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python -B -u R/tools/advance_fair_study_v3.py
```

`R` above means the fixed fair-study root, not a literal shell variable. Its
`cost_tools_v3`, `analysis_v1`, current runtime metadata and existing atomic
launcher must already be deployed. The driver's cost child invokes the driver's
actual deployed path, so no second helper deployment is needed.

New records are `ADVANCE_V3_COORDINATOR.json` (PID, process-start ticks, script
hash and affinity), `ADVANCE_V3_STATUS.json`, `ADVANCE_V3_COST_RECEIPT.json`,
`v3_*.log`, and the fresh cost-stage artifacts. Existing v2 state/logs remain.
The new PID receipt is created exclusively: starting v3 twice is refused.

Safe replacement of the old CPU coordinator is a separate root action:

- Confirm its current phase is still `audit_development`, no blind directory or
  blind manager has been launched, and the exact coordinator PID/start ticks and
  command line match the owned v2 process.
- Identify only its waiting collector child by parent PID and command line
  (`collect_fair_stage.py --root .../frontend_validation --wait`). Stop those two
  exact CPU PIDs and confirm they exited before starting v3. Do not kill a process
  group, a GPU resource manager, a workqueue worker, a shard runner, an event
  worker, or a Genesis actor.
- If the old coordinator has already advanced beyond that phase, inspect its
  created blind directory/jobs before any replacement. v3 deliberately refuses
  to overwrite that directory rather than duplicating actors.

The CPU contract tests cover immutable completed-state/lock reuse, exact cost
command/reference binding, truthful cost failure with skipped later steps,
reservation-release checks, and an end-to-end mocked coordinator in which cost
fails but GPU1/cost/GPU0 launch order and complete blind auditing remain correct.
