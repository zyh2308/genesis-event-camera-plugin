# Stage-1 pilot evidence index

Execution host: `g1`, `/home/zhaoyuhan/Eventbased_WAM`, conda environment `Eventbased_WAM`. The recorded runs used physical GPU 1 (RTX 3090). This WSL directory contains a small checked copy of synthetic artifacts and the text logs needed for review; full run outputs/checkpoints remain on the server.

## Contents

- `samples/data/sim2real/stage0/`: shared trajectory HDF5, summary/config, both V2E attempts, original endpoint PNGs, E2VID-ready HDF5, and STE-FlowNet-ready NPZ.
- `raw_logs/runs/sim2real/`: commands, environment/preflight, stdout, exit codes, result JSON, and retained failed attempts. Checkpoint files and generated media were intentionally omitted from this log archive.
- [`external_artifacts.json`](external_artifacts.json): canonical g1 paths, data and checkpoint hashes, environment, dataset availability status, and caveats.
- [`artifact_index.json`](artifact_index.json): SHA-256/byte-size index over the committed samples, archived logs, and this README/manifest.

## Key runs

| Purpose | Server run directory | Status |
|---|---|---|
| Shared Genesis trajectory | `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-shared-trajectory-v1/` | Summary/HDF5 checked; shell exit code not captured |
| V2E 1 ms attempt | `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-v2e-pair-v1/` | Completed with IIR/coarse-step warnings; excluded |
| V2E 25 μs pair | `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-v2e-pair-eps005/` | Train and validation exit 0 |
| E2VID+ final run | `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-e2vid-stage0-v3/` | One synthetic update and validation per generator; exit 0 |
| STE-FlowNet final run | `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-steflow-stage0-v2/` | One synthetic update and validation per generator; exit 0 |

## Important interpretation

The pilot used a tiny synthetic-only fixture and one update. E2VID losses and STE synthetic EPE values are not a generator ranking. No public real dataset was used in a model run. E-POSE data, loader, and model training were not run. See [`../../../reports/stage1_feasibility.md`](../../../reports/stage1_feasibility.md) for the route findings, results, costs, limitations, and next gates.
