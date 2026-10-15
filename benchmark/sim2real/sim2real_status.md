# Sim-to-Real status — Stage 1 feasibility and pilot

Updated: 2026-10-16 (Asia/Tokyo)

## Repository and execution location

- Upstream: `https://github.com/zyh2308/genesis-event-camera-plugin`
- Local review/preparation worktree: `/home/科研/Eventbased_WAM/repositories/gec-sim2real`
- Dedicated branch: `exp/public-sim2real-v1`; current pushed commit `798df82c805629b4943e53e23982dace33394853`
- Required server project: `/home/zhaoyuhan/Eventbased_WAM`
- Local WSL distro: Ubuntu 22.04.5 LTS (`Ubuntu-22.04`). Server distro banner: Ubuntu 20.04.6 LTS.
- Server plugin checkout observed at `repositories/genesis-event-camera-plugin`, `main`, clean and one commit behind its recorded origin/main; HEAD `15479d97154274796043d5e5cc42a336f57b2dd9`.
- Dedicated server worktree exists at `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real` at the current branch commit. Isolated HDR renderer and stationary-scene EventGenesis adapter smokes passed. No moving trajectory, paired generator corpus, task loader, or training experiment has run.
- Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1`, present locally and an ancestor of the dedicated branch. No core files changed in this work.

## Route status

| Route | Source checked | Real data checked | Shared Genesis/V2E corpus | Loader + synthetic train/val | Current blocker / next action |
|---|---|---|---|---|---|
| E2VID → HQF | Static audit of pinned E2VID+ training files and EVREAL adapter; exact route details in report | HQF not present in the inspected server project data tree; not checked outside it | Not run | Not run | HDR adapter boundary now passes; still need one moving shared EventGenesis/V2E corpus, complete pinned trainer/EVREAL source, the documented class-name/CSV/5-channel/validation-index fixes, then synthetic-only train/val |
| STE-FlowNet → MVSEC | Static audit at `eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6` | MVSEC not present in the inspected server project data tree; not checked outside it | Not run | Not run | HDR adapter boundary now passes; still need paired event/flow labels, pinned dependencies, encoder edge-case fixes and synthetic-only train/val |
| YOLO-6D → E-POSE | Static audit of SingleShotPose and E-POSE snapshots | E-POSE sample not present in inspected server project data tree; Figshare API endpoint returned 403, while the paper describes public Figshare data | Not run | Not run | HDR adapter boundary now passes; still need paired event/pose labels, official data path/assets, and a pure full-frame event adapter; never feed GT mask/crop/segmented events |

All three task rows mean **task-level experiments not run**. The corrected isolated renderer smoke and plugin adapter contract smoke passed. The latter used a stationary scene and did not require a nonzero event count, so it is not the requested moving shared corpus. No real model ranking has been performed.

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
- Exact shared trajectory IDs and labels per route; generator is the only intended paired difference.
- Same downstream initialization, seed, data order, preprocessing, optimizer and update count between generators.
- Real data is reserved for final testing; no checkpoint selection or simulator tuning from real results.
- E-POSE inputs must be full-frame, unsegmented event representations. GT masks/crops/segmented events remain labels or audit artifacts only.
- No experiment may modify the frozen core.

## Evidence

- Server audit and smoke metadata: [`evidence/S0/20261016-server-audit/`](evidence/S0/20261016-server-audit/); raw server logs and runtime manifest are indexed in `external_artifacts.json` and remain under `/home/zhaoyuhan/Eventbased_WAM/`.
- Imported baselines and source audit: [`research/input_2026-10-16/`](research/input_2026-10-16/)
- Full feasibility report: [`reports/stage1_feasibility.md`](reports/stage1_feasibility.md)

## Next actions

1. Use the already verified isolated HDR runtime and write a deterministic moving Genesis trajectory with camera/object pose and common labels.
2. Export that exact frame/label manifest through EventGenesis and pinned V2E, preserving raw events and generator provenance.
3. Pin or transfer complete third-party sources and check the native loaders; do not treat partial audit snapshots as executable.
4. In parallel, run one loader/forward/short synthetic training-validation cycle per candidate route, failing closed on interface, data, or fairness issues.
