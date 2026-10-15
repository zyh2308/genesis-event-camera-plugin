# Sim-to-Real status — Stage 1 feasibility and pilot

Updated: 2026-10-16 (Asia/Tokyo)

## Repository and execution location

- Upstream: `https://github.com/zyh2308/genesis-event-camera-plugin`
- Local review/preparation worktree: `/home/科研/Eventbased_WAM/repositories/gec-sim2real`
- Dedicated branch: `exp/public-sim2real-v1`; pushed server/local baseline commit `75217b0cba3c655aa8491ae0a381af79a72844b9` (this continuation has pending local edits)
- Required server project: `/home/zhaoyuhan/Eventbased_WAM`
- Local WSL distro: Ubuntu 22.04.5 LTS (`Ubuntu-22.04`). Server distro banner: Ubuntu 20.04.6 LTS.
- Server plugin checkout observed at `repositories/genesis-event-camera-plugin`, `main`, clean and one commit behind its recorded origin/main; HEAD `15479d97154274796043d5e5cc42a336f57b2dd9`.
- Dedicated server worktree exists at `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real`; first server-side renderer smoke was attempted and failed during runtime assembly validation. No task data/training experiment has run.
- Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1`, present locally and an ancestor of the dedicated branch. No core files changed in this work.

## Route status

| Route | Source checked | Real data checked | Shared Genesis/V2E corpus | Loader + synthetic train/val | Current blocker / next action |
|---|---|---|---|---|---|
| E2VID → HQF | Static audit of pinned E2VID+ training files and EVREAL adapter; exact route details in report | HQF not present in the inspected server project data tree; not checked outside it | Not run | Not run | Common EventGenesis/V2E corpus is gated on successful full HDR runtime smoke; then retrieve complete pinned trainer/EVREAL source, patch class-name/CSV/5-channel/validation-index interfaces, and run tiny paired synthetic smoke |
| STE-FlowNet → MVSEC | Static audit at `eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6` | MVSEC not present in the inspected server project data tree; not checked outside it | Not run | Not run | Common EventGenesis/V2E corpus is gated on HDR runtime; then resolve pinned dependencies and encoder edge cases and run synthetic-only validation |
| YOLO-6D → E-POSE | Static audit of SingleShotPose and E-POSE snapshots | E-POSE sample not present in inspected server project data tree; Figshare API endpoint returned 403, while the paper describes public Figshare data | Not run | Not run | Common EventGenesis/V2E corpus is gated on HDR runtime; verify official download path and object assets, then implement pure full-frame event adapter; never feed GT mask/crop/segmented events |

All three task rows mean **not run**. A renderer-only smoke was attempted and failed because the first runtime copy omitted six overlay files under `genesis/ext/pyrender/`; the complete overlay runtime builder is now in the local worktree and needs a server rerun. Static source checks do not establish a working task experiment. No real model ranking has been performed.

## Server resources observed

- Distro/host: server hostname `g1`; project under `/home/zhaoyuhan/Eventbased_WAM`.
- Environment `Eventbased_WAM`: Python 3.11.15, PyTorch 2.5.1+cu121, CUDA available, runtime CUDA 12.1, Genesis 1.2.2. Import emitted Genesis warning that Torch `<2.8.0` is unsupported.
- GPUs: 2×RTX 3090, 3×V100S-32GB, 1×TITAN RTX. At audit time GPU 0 had a `chekw` process; GPU 3 had `cmy` processes. GPUs 1, 2, 4, 5 had no compute processes. Recheck ownership/utilization immediately before any launch; avoid GPUs 0 and 3.
- `/home` NFS: 12 TB available. This is filesystem availability, not a per-user quota check.
- Existing user process `gpu_watch.py` remains untouched.
- One renderer-only run selected idle GPU 1 after an immediate preflight; it exited with code 1 at `OffscreenRenderer.render(..., radiance=True)`. No model or physics task was launched. The run emitted a shared Quadrants cache-lock warning; the shared cache was left untouched.

## Controls

- Synthetic train and synthetic validation only for all pilots.
- Exact shared trajectory IDs and labels per route; generator is the only intended paired difference.
- Same downstream initialization, seed, data order, preprocessing, optimizer and update count between generators.
- Real data is reserved for final testing; no checkpoint selection or simulator tuning from real results.
- E-POSE inputs must be full-frame, unsegmented event representations. GT masks/crops/segmented events remain labels or audit artifacts only.
- No experiment may modify the frozen core.

## Evidence

- Server read-only audit and failed HDR smoke metadata: [`evidence/S0/20261016-server-audit/`](evidence/S0/20261016-server-audit/); raw smoke log is `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-hdr-render-smoke/stdout.log`.
- Imported baselines and source audit: [`research/input_2026-10-16/`](research/input_2026-10-16/)
- Full feasibility report: [`reports/stage1_feasibility.md`](reports/stage1_feasibility.md)

## Next actions

1. Update the server worktree with the complete overlay runtime builder, then rerun the renderer smoke with project-scoped cache paths and a fresh GPU ownership check.
2. Generate one small shared Genesis physical trajectory on the server; export EventGenesis and V2E from the same frame/label manifest.
3. Pin or fetch complete third-party sources; do not treat partial audit snapshots as executable.
4. In parallel, run one loader/forward/short synthetic training-validation cycle per candidate route, failing closed on interface or data issues.
