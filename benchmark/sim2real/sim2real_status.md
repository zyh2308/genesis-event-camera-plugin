# Sim-to-Real status — Stage 1 feasibility and pilot

Updated: 2026-10-16 (Asia/Tokyo)

## Repository and execution location

- Upstream: `https://github.com/zyh2308/genesis-event-camera-plugin`
- Local review/preparation worktree: `/home/科研/Eventbased_WAM/repositories/gec-sim2real`
- Dedicated branch: `exp/public-sim2real-v1` (pre-existing branch; base `cfb16c665216fda0f591bee877ed0b9635da20d0`)
- Required server project: `/home/zhaoyuhan/Eventbased_WAM`
- Server plugin checkout observed at `repositories/genesis-event-camera-plugin`, `main`, clean and one commit behind its recorded origin/main; HEAD `15479d97154274796043d5e5cc42a336f57b2dd9`.
- The dedicated worktree `repositories/gec-sim2real` does not yet exist on the server. Server-side experiment work has not started.
- Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1`, present locally and an ancestor of the dedicated branch. No core files changed in this work.

## Route status

| Route | Source checked | Real data checked | Shared Genesis/V2E corpus | Loader + synthetic train/val | Current blocker / next action |
|---|---|---|---|---|---|
| E2VID → HQF | Static audit of pinned E2VID+ training files and EVREAL adapter; exact route details in report | HQF not found in local WSL data; not yet checked on server | Not run | Not run | Need server data scan and complete pinned trainer/EVREAL source; patch absent `HDF5Dataset` class name, CSV file-list expectation, 5 vs 10 channels, and empty L=200 validation; then run tiny paired synthetic smoke |
| STE-FlowNet → MVSEC | Static audit at `eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6` | MVSEC not found in finite local WSL scan; not yet checked on server | Not run | Not run | Missing upstream dependencies and exact encoder edge bugs; use pinned full source + narrow adapter, synthetic-only validation |
| YOLO-6D → E-POSE | Static audit of SingleShotPose and E-POSE snapshots | Data record says Figshare API returned 403; no local E-POSE sample; server availability not yet checked | Not run | Not run | Obtain/check official data and assets; train loader is LINEMOD/RGB-mask based and needs a pure-event adapter; do not use GT mask/crop/segmented events as inputs |

All three rows currently mean **not run**. Static source checks do not establish a working experiment. No real model ranking has been performed.

## Server resources observed

- Distro/host: server hostname `g1`; project under `/home/zhaoyuhan/Eventbased_WAM`.
- Environment `Eventbased_WAM`: Python 3.11.15, PyTorch 2.5.1+cu121, CUDA available, runtime CUDA 12.1, Genesis 1.2.2. Import emitted Genesis warning that Torch `<2.8.0` is unsupported.
- GPUs: 2×RTX 3090, 3×V100S-32GB, 1×TITAN RTX. At audit time GPU 0 had a `chekw` process; GPU 3 had `cmy` processes. GPUs 1, 2, 4, 5 had no compute processes. Recheck ownership/utilization immediately before any launch; avoid GPUs 0 and 3.
- `/home` NFS: 12 TB available. This is filesystem availability, not a per-user quota check.
- Existing user process `gpu_watch.py` remains untouched.

## Controls

- Synthetic train and synthetic validation only for all pilots.
- Exact shared trajectory IDs and labels per route; generator is the only intended paired difference.
- Same downstream initialization, seed, data order, preprocessing, optimizer and update count between generators.
- Real data is reserved for final testing; no checkpoint selection or simulator tuning from real results.
- E-POSE inputs must be full-frame, unsegmented event representations. GT masks/crops/segmented events remain labels or audit artifacts only.
- No experiment may modify the frozen core.

## Evidence

- Server read-only audit: [`evidence/S0/20261016-server-audit/`](evidence/S0/20261016-server-audit/)
- Imported baselines and source audit: [`research/input_2026-10-16/`](research/input_2026-10-16/)
- Full feasibility report: [`reports/stage1_feasibility.md`](reports/stage1_feasibility.md)

## Next actions

1. Publish this dedicated branch with imported research inputs and S0 evidence; create a clean server worktree at the same branch/commit.
2. Read server project rules and inventory exact benchmark data locations and quotas without broad scans.
3. Pin or fetch full third-party source repositories to `external/`; no partial source snapshot is executable code.
4. Select a currently unoccupied owned GPU only after an immediate resource check; run a tiny Genesis physical-trajectory export shared by EventGenesis and V2E.
5. In parallel, run one loader/forward/short synthetic training-validation cycle per candidate route, failing closed on interface or data issues.
