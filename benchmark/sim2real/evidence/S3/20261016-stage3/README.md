# Stage 3 evidence bundle

This directory contains the auditable Stage 3 run records and small visual
samples. The concise research handoff is
[`../../../../STAGE3_RESEARCH_HANDOFF.md`](../../../../STAGE3_RESEARCH_HANDOFF.md);
the full execution report is
[`../../../../reports/stage3_execution.md`](../../../../reports/stage3_execution.md).

## Main successful runs

| Work | Run record | Result |
|---|---|---|
| Corrected STE synthetic validation, four restored checkpoints | `server_runs/20261016-stage3-flow-rescore-final/` | dt1: 60 intervals per generator; dt4: 54. Correct two-component EPE and exact restore parity. No retraining. |
| HQF EventGenesis candidate inference | `server_runs/20261016-stage3-hqf-eg-run/` and `server_runs/20261016-stage3-hqf-eg-output/` | 2,431 endpoints, recurrent reset once, saved preview frames. |
| HQF V2E candidate inference | `server_runs/20261016-stage3-hqf-v2e-run/` and `server_runs/20261016-stage3-hqf-v2e-output/` | Same fixed sequence and endpoint count. |
| MVSEC candidate inference | `server_runs/20261016-stage3-mvsec-{eg,v2e}-dt{1,4}-run/` and matching `*-output/` | Four event-only runs; output manifests include per-window paths/hashes for external NPZ files. |
| Corrected pose paired pilot | `server_runs/20261016-stage3-pose-paired-run/` | 16 epochs per generator; corrected windows and held-out synthetic windows. |
| Pose one-sample fit diagnosis | `server_runs/20261016-stage3-pose-overfit-diag-fixed/` | Exposes eval BatchNorm/confidence-cell issues; not a valid inference workaround. |
| Controlled checks | `server_runs/20261016-stage3-controlled-checks-final/` | 19 tests passed; command and environment are recorded. |

Failed and superseded attempts are intentionally retained in sibling folders.
The final handoff/report identifies each failure and the corrected run. The
initial Stage 3 MVSEC audit failure and first controlled-check failure remain
under `mvsec_audit_initial_failed/` and `server_runs/20261016-stage3-controlled-checks/`.

## Data and output locations

`external_artifacts_stage3.json` records full paths, sizes and hashes for public
source files, derived data and candidate checkpoints. The originals are in WSL
under `/home/科研/Eventbased_WAM/data/sim2real/stage3/real/`; candidate
checkpoints, converted HQF arrays, and MVSEC prediction NPZs are on g1 under
`/home/zhaoyuhan/Eventbased_WAM/`. The 118 MVSEC NPZs are deliberately external
to Git (about 206 MB); each is individually indexed in the external manifest.

`hqf_real_data/` includes the topic inventory and conversion manifest.
`access_and_download/` records original-file downloads and the preregistered
MVSEC fragment selection. `epose_access/` records DOI metadata and the dated
HTTP access responses; no E-POSE sample or matched CAD was retrieved.

## Reproduction entry points

From the repository root:

```bash
bash benchmark/sim2real/scripts/run_stage3_controlled_checks.sh
```

The exact g1 commands, environment snapshots, UTC start/end times, exit codes,
GPU preflights and logs are in each `server_runs/<run>/` directory. The fixed
real sample registration is
[`../../../protocol/stage3_real_pilot_registration.yaml`](../../../protocol/stage3_real_pilot_registration.yaml).

## Integrity index

`artifact_index.json` hashes this bundle and the Stage 3 report, handoff,
protocol, relevant task/status docs, code/tests, and preserved Stage 2 review.
The repository-wide `benchmark/sim2real/artifact_index.json` also hashes all
non-ignored Sim-to-Real files except itself.
