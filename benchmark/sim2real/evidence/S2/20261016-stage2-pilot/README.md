# Stage 2 evidence package

Execution host: g1 (`/home/zhaoyuhan/Eventbased_WAM`), conda `Eventbased_WAM`, physical GPU 1 RTX 3090. The indexed logs record the per-run package versions, source hashes, exact command, preflight, start/end UTC, exit status and stdout. External checkpoints and generated datasets are not copied into Git; see [`external_artifacts.json`](external_artifacts.json) for every server path, byte size and SHA-256.

## Run map

| Run | Purpose | Exit | Key result |
|---|---|---:|---|
| `runs/20261016-s2-pilot-data-v1/` | Six-rollout physics bank and six V2E conversions; first E2VID adapter-count assertion | 1 overall; trajectory and all six V2E steps 0 | Failure retained. Exact timestamp endpoint ties were audited in the adapter retry. |
| `runs/20261016-s2-pilot-adapters-v2/` | Rebuild E2VID and STE adapters from the same unchanged physical bank | 0 | Both manifests pass; paired labels/masks match and endpoint ties are accounted for. |
| `runs/20261016-s2-e2vid-pilot-v1/` | R1/R2, event boundary fixtures, 2-epoch paired E2VID training/validation | 0 | 224 updates and 56 val sequences per generator; loss/gradient/state/checkpoint evidence saved. |
| `runs/20261016-s2-steflow-pilot-v1/` | Initial dt1/dt4 flow interface | 1 | Eval helper expected a list but received one eval tensor. |
| `runs/20261016-s2-steflow-pilot-v2/` | Retry with dt1 and memory-heavy correlation at dt4 | 1 | dt1 completed; dt4 `F.unfold` ran out of memory. |
| `runs/20261016-s2-steflow-pilot-v3/` | Memory-efficient correlation; paired dt1/dt4 training/validation | 0 | Four checkpoints restored; finite losses/gradients and synthetic EPE saved. |
| `runs/20261016-s2-pose-pilot-v1/` | Generic cuboid event-only closure, first window selection | 0 | Closure passed, but few selected samples and extremely poor synthetic EPE exposed sampling issue. |
| `runs/20261016-s2-pose-pilot-v2/` | Corrected visible-window selection and rerun | 0 | 8 train/4 validation samples per generator; checkpoint restore passed; synthetic EPE still unusable. |

## Data manifests

- `data_manifests/shared/config.json`: Stage 2 physical bank and generator settings.
- `data_manifests/shared/summary.json`: per-rollout count, actual K, coordinate/time convention, event and validity summaries.
- `data_manifests/e2vid_inputs_v2/conversion_manifest.json`: 12 converted HDF5s, flow label hashes and endpoint tie audit.
- `data_manifests/steflow_inputs_v2/conversion_manifest.json`: 12 NPZs with dt1/dt4 representation shapes, labels and common validity counts.
- `external_artifacts.json`: g1 data/checkpoint paths, sizes and hashes, source revisions and data availability boundary.
- `artifact_index.json`: SHA-256 and byte-size index for this package, Stage 2 reports/status/task configs, and the preserved Stage 1 review bundle.

## Reproduction entry points

From the committed repository checkout on g1, use the versioned wrapper scripts under `benchmark/sim2real/scripts/`. They refuse to overwrite existing run/data paths. The completed pilot's exact command lines are also preserved as `*.command.txt`; inspect them with the associated config before any new run.

```bash
bash benchmark/sim2real/scripts/run_stage2_data_server.sh <new-run-id>
bash benchmark/sim2real/scripts/run_stage2_adapters_server.sh <new-run-id>
bash benchmark/sim2real/scripts/run_stage2_e2vid_server.sh <new-run-id>
bash benchmark/sim2real/scripts/run_stage2_steflow_server.sh <new-run-id>
bash benchmark/sim2real/scripts/run_stage2_pose_server.sh <new-run-id>
```

The data wrapper has a fixed data root in the checked-in pilot script and will refuse an existing root. For a new trajectory bank, version the config and root together; preserve the committed Stage 2 data and logs. Check GPU ownership and available resources before launching.
