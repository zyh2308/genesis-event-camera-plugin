# Fair frontend stage collection

This directory closes actual actor artifacts. It launches no renderer, physics,
training, SSH connection, or event generator. Deploy these four Python files
together: `collect_fair_stage.py`, `audit_fair_raw.py`,
`audit_contact_observation_windows.py`, and `select_frontend.py`.
The runtime needs NumPy, SciPy, and h5py, matching the existing artifact auditor.

After all development jobs have been submitted:

```sh
python analysis/collect_fair_stage.py --root DEVELOPMENT_ROOT --stage validation --wait
python analysis/select_frontend.py --root DEVELOPMENT_ROOT
```

The default wait timeout is 43,200 seconds; change `--timeout-s` if necessary.
The collector discovers all registered shard IDs from the actual protocol. In
the finite workqueue mode it verifies the queue plan, all registered top-level
worker jobs, their normal-exit results, completed atomic shard claims, each
worker's progress/finished receipt, and each shard's complete/progress pair.
Worker counts may differ from shard counts. In the legacy nonqueue mode it
expects one normal top-level resource-launcher `*.result.json` per shard.
Failed infrastructure is reported; it is never relabeled as a task failure.
For legacy approved replacement jobs, repeat `--job-result-file NAME` for
exactly the current normal-exit receipts. In queue mode this option must still
name all registered workers; it cannot discard a failed worker. The completion
receipt lists the worker/job/claim hashes.

An explicitly preregistered OOM recovery can be audited without pretending the
old worker succeeded:

```sh
python analysis/collect_fair_stage.py --root DEVELOPMENT_ROOT --stage validation --wait --recovery-plan INFRASTRUCTURE_RECOVERY.json
```

The recovery manifest's schema is `fair_workqueue_infra_recovery_v1`; it records
`registered_utc`, unchanged `protocol_sha256`, reason `cuda_out_of_memory`,
`historical_failed_workers`, and `replacement_worker_ids`. Each historical worker
declares `worker_id`, `job_file/job_sha256`, `result_file/result_sha256`,
`progress_file/progress_sha256`, and `archived_failed_claims`. Each archived claim
declares `shard`, `claim_file/claim_sha256`, and `archived_attempt_root`. All paths
are under the stage root; failed claims and attempt directories remain under
`infrastructure_failures`. An attempt directory preserves `shardN_progress.json`,
`cases/<case_id>/<method>/`, and original logs. The collector checks the original
actor prefix, seed, source, endpoint rate, and frontend against the unchanged
registered protocol. Replacement job JSON must include
`infrastructure_recovery_sha256` binding it to that manifest.

All final shard claims, shard progress/completion and final actor exits must
still pass. A historical failed worker's successful claims must match its actual
progress and remain in the final data; its failed claims must match the archive.
Its overall `finished PASS` is not fabricated. A completion marker with recovery
therefore says `all_actual_jobs_normal_exit=false`,
`all_final_actor_attempts_normal_exit=true`, and
`historical_infrastructure_failures_retained=true`. Archived infrastructure
attempts are hashed and retained, not added as new scientific episodes. Without
the explicit recovery argument, the original failed jobs still block completion.

The development protocol currently declares 48 actual actors: three shared
frontend candidates, two sources, two rates, and four physical seed blocks.
The default blind study declares 320 actual actors: 32 blocks, four rates and
two event sources (256), plus one initialization-only and one noisy-GT reference
per block (64). Sample counts come from `cases`, not inherited old metadata.

`FRONTEND_SELECTION.json` contains `frontend_by_method`, immediately usable in
the blind protocol:

```json
{
  "frontend_by_method": {
    "ours": {"CATCHIT_EVENT_OBSERVATION_WINDOW_MS": "selected value", "CATCHIT_EVENT_FLIGHT_FIT": "selected value", "CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS": "selected value"},
    "v2e": {"CATCHIT_EVENT_OBSERVATION_WINDOW_MS": "selected value", "CATCHIT_EVENT_FLIGHT_FIT": "selected value", "CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS": "selected value"}
  }
}
```

Those strings above are a schema example, not selected values. Selection requires
complete actual validation, both sources' equal candidate union and all four
development blocks at 25/12.5 Hz. It chooses one global configuration per source.
Prespecified ranking: exact macro catch rate, actual state availability,
normalized position/velocity loss, parameter differences from canonical, then
numeric window/fit/gap order. Missing states and zero-query initialization
failures receive the declared maximum error; static and false-grasp-propagated
states remain scored. The manifest records every candidate and tie. It refuses
blind data and refuses to overwrite an existing lock. Its status is `LOCKED`
and `locked_utc` records the actual UTC lock time. The validation root and
result/completion/protocol hashes bind the lock to its complete development data.

After all blind jobs:

```sh
python analysis/collect_fair_stage.py --root BLIND_ROOT --stage blind --wait
```

The collector verifies runtime/frozen17/policy/profile/native-V2E hashes, actual
loaded module hashes, each source's input protocol, first-RGB-D budgets, contact
profile, raw events, packet timing and controller causality. Window10 reuses the
unchanged established subwindow auditor; window0 uses the established raw audit.
The raw auditor has only three documented collection adaptations (expected count,
dynamic shards, transient supervisor-log exclusion); see `LEGACY_AUDIT_REUSE.json`.

Every registered actor is retained, including initial invisibility, invalid
RGB-D initialization, physical failure, and noisy-GT failure. The new runtime's
initialization failure is a normal completed zero-step task failure with an empty
trace/raw stream and no native worker or HDR file. It remains in the denominator.
An infrastructure failure prevents a final completed stage from being scored.

Outputs are `FAIR_EPISODES.csv`, `公平前端逐集结果.csv`,
`FAIR_STAGE_RESULTS.json`, backward-compatible `PAIRED_PORTABLE.json`,
`DATA_COMPLETION_RECEIPT.json`, `FINAL_AUDIT.json`, `SHA256SUMS`, and finally
`FAIR_COLLECTION_COMPLETE.json`. Only this last marker seals completion after
the raw audit passes. It includes the result/manifest/audit hashes. A collector
rerun refuses to overwrite an already sealed stage.

CSV fields include `physical_case_id`, `seed`, `method`, `endpoint_hz`,
`frontend_id`, `verified_catch`, `failure_class`, `initialization_valid`,
`hand_contact_steps`, `final_contact`, `sim_time_s`, `wall_s`, raw-event counts,
trace counts and output path. JSON adds per-episode perception tie metrics and
full frontend overrides. References have `endpoint_hz=null`; reuse is not
expanded into new rate-specific samples. Metadata identifies validation/blind.
The explicit physical-case/seed block is the statistical unit; repeated rates
and candidates are correlated conditions.

The catch score keeps the registered evaluator semantics: time-limit reached,
no failure termination, real hand contact, and actual contact in the final
40 ms control step. Consecutive contact-containing steps are only a supplementary
40 ms-resolution measure, not proof of continuous 0.5 s contact. Noisy-GT is a
reference policy condition rather than a guaranteed upper bound. Actor wall time
includes initialization and logging; it is not a dedicated throughput benchmark.

Local verification used nine artifact-only contract tests, including a complete
temporary zero-step failure through the collector and inherited raw auditor.
Finite-queue tests cover two workers completing three shards, incomplete/failed
workers, and unregistered claims; selection tests check the real UTC lock and
refusal to overwrite it.
The OOM-recovery test retains a failed worker with a prior successful shard,
audits its archived failed attempt, accepts a bound replacement, and rejects
missing final actors or a changed archived seed.
Synthetic fixtures are deleted by TemporaryDirectory and never become study
episodes. The actual registered development protocol was checked without reading
its outcomes: 48 actors, 4 physical blocks, shard IDs 0..23, and the same three
frontends for both sources. No blind outcome was read by these tools' authors.
