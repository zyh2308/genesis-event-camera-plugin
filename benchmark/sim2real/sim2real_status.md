# Sim-to-Real status — Stage 1 feasibility and pilot

Updated: 2026-10-16 (Asia/Tokyo)

## Repository and execution location

- Upstream: `https://github.com/zyh2308/genesis-event-camera-plugin`
- Local review/preparation worktree: `/home/科研/Eventbased_WAM/repositories/gec-sim2real`
- Dedicated branch: `exp/public-sim2real-v1`; current code is recorded in Git history and the delivery commit is reported at handoff.
- Required server project: `/home/zhaoyuhan/Eventbased_WAM`
- Local WSL distro: Ubuntu 22.04.5 LTS (`Ubuntu-22.04`). Server distro banner: Ubuntu 20.04.6 LTS.
- Server plugin checkout observed at `repositories/genesis-event-camera-plugin`, `main`, clean and one commit behind its recorded origin/main; HEAD `15479d97154274796043d5e5cc42a336f57b2dd9`.
- Dedicated server worktree exists at `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real`. The server completed a moving two-rollout physics fixture, EventGenesis raw event generation, and a paired V2E conversion at a numerically stable 25 μs timestamp resolution. Downstream loaders and training cycles remain the next gate.
- Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1`, present locally and an ancestor of the dedicated branch. No core files changed in this work.

## Route status

| Route | Source checked | Real data checked | Shared Genesis/V2E corpus | Loader + synthetic train/val | Current blocker / next action |
|---|---|---|---|---|---|
| E2VID → HQF | Pinned source audit completed; full upstream checkout now bundled for server inspection | HQF was not found in the inspected project data roots; official global availability not disproven | Paired Stage-0 EventGenesis/V2E inputs exist; E2VID data conversion/loader and short train/val not yet run | Exact source interfaces and required narrow adapter changes are listed in the report; next run is on server after source transfer |
| STE-FlowNet → MVSEC | Pinned source audit completed; full upstream checkout bundled for server inspection | MVSEC was not found in inspected project data roots; official download page exists | Paired events plus synthetic flow/endpoint labels exist; native loader/forward/train-val not yet run | Correlation extension and stock encoder edge cases remain unverified; next run is on server, keeping official photometric objective and synthetic-only validation |
| YOLO-6D → E-POSE | SingleShotPose and E-POSE source audited and bundled; E-POSE repo is annotation tooling, not a complete training reproduction | Sample data not found in inspected roots; one Figshare API path returned 403, which does not establish unavailability | Shared pose labels exist in the fixture; event-only data/model adapter and short train-val not yet run | Official data route and pose/CAD conventions remain unresolved; only proceed with full-frame unsegmented events and no privileged masks/crops |

Stage-0 shared data is an engineering fixture, not a benchmark result. The server ran two independent fall-box rollouts (train and validation) from the same config and generated 4,068 / 4,832 EventGenesis events. Pinned V2E produced 2,084 / 2,571 events at 25 μs. A 1 ms V2E trial is preserved as failure evidence because its 300 Hz IIR update emitted a large-step warning. The first V2E trial is excluded from paired training. No downstream candidate model has yet completed a loader/optimizer/validation cycle, and no real model ranking has been performed. The Stage-0 process exit status was not captured, but its full summary and all HDF5 groups were read and validated on the server.

## Server resources observed

- Distro/host: server hostname `g1`; project under `/home/zhaoyuhan/Eventbased_WAM`.
- Environment `Eventbased_WAM`: Python 3.11.15, PyTorch 2.5.1+cu121, CUDA available, runtime CUDA 12.1, Genesis 1.2.2. Import emitted Genesis warning that Torch `<2.8.0` is unsupported.
- GPUs: 2×RTX 3090, 3×V100S-32GB, 1×TITAN RTX. At audit time GPU 0 had a `chekw` process; GPU 3 had `cmy` processes. GPUs 1, 2, 4, 5 had no compute processes. Recheck ownership/utilization immediately before any launch; avoid GPUs 0 and 3.
- `/home` NFS: 12 TB available. This is filesystem availability, not a per-user quota check.
- Existing user process `gpu_watch.py` remains untouched.
- One renderer-only run selected idle GPU 1 after an immediate preflight; it exited with code 1 at `OffscreenRenderer.render(..., radiance=True)`. No model or physics task was launched. The run emitted a shared Quadrants cache-lock warning; the shared cache was left untouched.
- Corrected full-overlay renderer smoke passed on GPU 1. The next adapter smoke also passed after project-scoping `XDG_CACHE_HOME`, `GS_CACHE_FILE_PATH`, and `TI_OFFLINE_CACHE_FILE_PATH`; it exercised plugin attach and two capture calls. Runtime copy occupies 277 MB, the two successful run directories occupy approximately 10 MB combined, and no checkpoint was produced.
- One adapter attempt used a misspelled `EventBased_WAM` path, loaded the original conda package, and failed before event capture; the failed log is retained and indexed separately.

## Controls

- Synthetic train and synthetic validation only for all pilots.
- Exact shared physical rollouts, endpoint renders, split, labels, and event windows. Generator-specific temporal interpolation and input transfer (native linear radiance for EventGenesis; fixed global radiance-to-display mapping plus pinned SuperSloMo for V2E) remain known numerical differences and are recorded in the config; this is an adapter feasibility pair, not a calibrated sensor comparison.
- Same downstream initialization, seed, data order, preprocessing, optimizer and update count between generators.
- Real data is reserved for final testing; no checkpoint selection or simulator tuning from real results.
- E-POSE inputs must be full-frame, unsegmented event representations. GT masks/crops/segmented events remain labels or audit artifacts only.
- No experiment may modify the frozen core.

## Evidence

- Server audit and renderer smoke metadata: [`evidence/S0/20261016-server-audit/`](evidence/S0/20261016-server-audit/).
- Moving shared trajectory and both V2E runs: [`evidence/S0/20261016-stage0-pair/`](evidence/S0/20261016-stage0-pair/); small sample artifacts and raw logs are indexed there and remain under `/home/zhaoyuhan/Eventbased_WAM/`.
- Imported baselines and source audit: [`research/input_2026-10-16/`](research/input_2026-10-16/)
- Full feasibility report: [`reports/stage1_feasibility.md`](reports/stage1_feasibility.md)

## Next actions

1. Use the small trajectory to implement and validate task-specific endpoint HDF5 adapters, beginning with E2VID and STE-FlowNet.
2. Run matching short synthetic training/validation updates on g1 after fresh GPU/process preflight. Keep the 1 ms V2E artifacts excluded.
3. Check E-POSE's official download route and CAD/pose conventions; continue the pose route only if full-frame event inputs can be built without privileged masks or crops.
4. Keep all real sets test-only; no real candidate ranking in this feasibility stage.
