# Core / Architecture migration status

**Migration date:** 2026-10-08
**Source repository:** `/home/科研/Eventbased_WAM/repositories/genesis-event-camera-plugin`
**New worktree:** `/home/科研/Eventbased_WAM/repositories/gec-core`
**New branch:** `work/core-maintenance`

## Current repository state

- Original worktree branch: `main`
- Original worktree HEAD at audit: `568ebb5480b487340172230fa9490b017141d339`
- New branch HEAD at worktree creation: `568ebb5480b487340172230fa9490b017141d339`
- Original worktree status at audit: clean; `main` matched `origin/main`.
- No merge, rebase, cherry-pick, revert, or bisect was in progress.
- No ordinary untracked files, untracked datasets, checkpoints, credentials, or
  private-key-like files were found. Existing ignored files were Python
  `__pycache__` artifacts and were deliberately not migrated.

## What was migrated

The existing Core implementation was already committed in the source history,
so there were no uncommitted Core diffs to replay. The new worktree therefore
retains the established Core directories without modifying the original:

- `genesis_event_plugin/` physics/radiance v3 implementation;
- Core simulator tests;
- Genesis HDR renderer overlay;
- calibration schema;
- existing Core reports and reproducibility documentation.

This migration adds only the coordination contract and this status record. It
does not copy benchmark outputs, robot-task experiments, generated data, or
checkpoints, and it does not change event-generation code or runtime defaults.

## MIGRATION_PENDING

`MIGRATION_PENDING`: none for the audited working tree. The audit found no
uncommitted mixed files requiring classification. Benchmark and quality assets
already tracked on `origin/main` remain outside the Core workflow by policy;
they were not modified or reclassified by this migration.

## Frozen simulator and development boundary

The formal frozen simulator remains:

```text
e79e656ac00f7847b98db9e12bf43b093f3afff1
tag: v3.1-quality-benchmark
```

The maintenance branch may host architecture work, but any output-affecting
change is **development only; NOT adopted as formal frozen core** until the
project-control approval and three matched-experiment upgrade rule in
`CORE_CONTRACT.md` is satisfied. The present coordination-only change has no
formal experiment impact.

## Validation status

Read-only audit and preservation checks completed:

- original path, branch, HEAD, remote, status, log, diffs, worktrees,
  ahead/behind state, submodules, ignored names, large-file and secret-name
  checks recorded;
- external backup created and verified at
  `/home/科研/Eventbased_WAM/repositories/_reorg_backups/core_20261008_003047/`;
- `git fetch origin` completed with local `main` still equal to `origin/main`;
- new sibling worktree created without altering the original worktree.

Lightweight validation completed before the coordination commit:

- `python -m compileall -q genesis_event_plugin tests
  integrations/genesis_renderer_hdr_v2_overlay/tests`: passed;
- direct dependency-light harness for `tests/test_physics_core.py` (10 tests):
  passed;
- direct dependency-light harness for `tests/test_v3_mainline.py` (14 tests):
  passed;
- `git diff --exit-code e79e656 -- genesis_event_plugin/`: empty.

`pytest` was not installed in the active environment, so the two test modules
were invoked directly rather than through pytest. No HDR benchmark, formal
quality comparison, robot training, or server job was started as part of the
migration.

## Next steps

1. Review this contract with project control and keep formal comparisons pinned
   to `v3.1-quality-benchmark`.
2. Use `work/core-maintenance` for isolated Core changes only; keep benchmark
   and robot-task workflows separate.
3. For any output-affecting candidate, run the unified three-experiment
   protocol, record provenance and diagnostics, and seek approval before a new
   frozen tag.
4. If a future audit finds a mixed or ambiguous file, preserve it in place and
   mark it `MIGRATION_PENDING` rather than splitting it automatically.
