# Stage 1 feasibility report

Date: 2026-10-16 (Asia/Tokyo)

Status: source/server audit plus one failed HDR renderer smoke. No simulator trajectory, event pair, downstream loader/training cycle, or real scoring has run.

## Scope and evidence labels

This report records static source review, server environment/repository checks, and one failed renderer-only smoke on the server. `Source checked` means code/config metadata was inspected at a pinned revision or in the provided audited snapshot. The renderer smoke did not exercise a simulator trajectory, event export, model forward, optimizer step, or real-set scoring. Cost estimates are labeled as estimates.

## Repository and simulator baseline

- The existing dedicated branch is `exp/public-sim2real-v1`, commit `cfb16c665216fda0f591bee877ed0b9635da20d0`, based on upstream main `568ebb5480b487340172230fa9490b017141d339`.
- Frozen core `e79e656ac00f7847b98db9e12bf43b093f3afff1` resolves locally and is an ancestor of the dedicated branch. This work has not modified core files.
- Server checkout: `/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin`, branch `main`, commit `15479d97154274796043d5e5cc42a336f57b2dd9`, clean and reported one commit behind `origin/main`. The dedicated server worktree is now `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real`, branch `exp/public-sim2real-v1`, commit `75217b0cba3c655aa8491ae0a381af79a72844b9`.
- The frozen core's physics-first plugin requires linear radiance and explicit camera/object pose providers. The repository overlay contains eight Genesis source/shader files. The conda-installed Genesis 1.2.2 has no `radiance` argument. The first renderer smoke used an incomplete runtime copy with only the two `genesis/vis` files applied and failed with `TypeError: OffscreenRenderer.render() got an unexpected keyword argument 'radiance'`; the remaining six overlay files are under `genesis/ext/pyrender/`. A benchmark-local builder now applies the complete overlay to an isolated package copy. The corrected copy still needs a rerun before EventGenesis export.
- The local WSL checkout is preparation/review only. The user has directed all experiments to the remote research server.

## Server environment and resource observation

Measured during a read-only remote inspection:

- Host `g1`, project path `/home/zhaoyuhan/Eventbased_WAM`.
- Conda `Eventbased_WAM`: Python `3.11.15`; PyTorch `2.5.1+cu121`; `torch.cuda.is_available() == True`; runtime CUDA `12.1`; Genesis `1.2.2`.
- Genesis import emitted its compatibility warning that installed Torch `<2.8.0` is unsupported. This warning is recorded; no compatibility conclusion has been assumed.
- GPU inventory: 2× RTX 3090 (24 GB each), 3× V100S (32 GB each), 1× TITAN RTX (24 GB). During audit, GPU 0 and GPU 3 were occupied by other users' processes. GPU 1, 2, 4, and 5 showed no compute processes. Recheck immediately before launch; do not use GPU 0 or 3.
- `/home` NFS had 12 TB available; per-user quota was not checked.
- An existing user-level `gpu_watch.py` process was observed and left untouched.
- Immediately before the renderer smoke, GPU 1 (RTX 3090) was idle and selected; no other compute process was assigned to it. GPU 3 had two other-user CARLA processes and was not used. The smoke exited with code 1. A shared Quadrants cache-lock warning appeared at `/home/zhaoyuhan/.cache/quadrants/qdcache/kernel_compilation_manager/ticache.lock`; that cache was not cleaned or modified. Raw log: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-hdr-render-smoke/stdout.log`.

## Route findings

### E2VID recurrent reconstruction → HQF

**Source checked:** E2VID+ public training snapshot at `TimoStoff/event_cnn_minimal@78aa471a66593f55e349303563997297fbc816e8`; EVREAL at `ercanburak/EVREAL@398f03551b7f5d150ae10aa89e4e731557fce240`. The local research package is an extracted subset, not complete checkouts.

**Interface findings:** public reconstruction config has a 10-bin model field while loader uses 5 bins; training and validation must both be explicitly configured for 5 signed channels. `SequenceDataset` dynamically evaluates the dataset class name, but the pinned config says `HDF5Dataset` while the source defines `DynamicH5Dataset`, so the stock config raises `NameError`. `HDF5DataLoader` expects a CSV manifest of HDF5 paths, not one HDF5 file or a directory. Also, `SequenceDataset` returns zero length when `L >= dataset.length`; for the candidate between-frame validation clip (`L=200`, 200 intervals from 201 images), this yields an empty validation dataset. Stored polarity must be OFF=`0`, ON=`1` because the loader maps it with `2p-1`; writing ±1 directly gives an incorrect OFF magnitude. The event-window/index endpoint mapping needs a known-timestamp fixture to catch off-by-one windows. The candidate also has a train/validation image normalization mismatch (`normalize_image=false` for train and `true` for validation), which must be explicitly resolved. A narrow, method-independent loader/indexing correction plus per-split file-list CSVs is required. EVREAL has distinct E2VID and E2VID+ normalization paths; use the pinned E2VID+ route consistently.

**Data:** HQF was not found in the finite local WSL project/data scan. The authors' public page and EVREAL preparation tools expose a dataset route, but no files were downloaded or hashed. EVREAL expects converted memmap files (`events_ts.npy`, `events_xy.npy`, `events_p.npy`, `images.npy`, `images_ts.npy`, `image_event_indices.npy`), not a raw bag directly. HQF server-side presence/download has not yet been checked.

**Actually run:** no loader, forward pass, training, or real reconstruction evaluation.

**Next:** create server worktree; retrieve complete pinned E2VID+ and EVREAL repositories; inspect server HQF data; fix the dataset class string, file-list CSV interface, 5-channel setup, and validation-length indexing in a benchmark-local adapter; make paired short synthetic clips and run loader plus short synthetic-only training/validation for both generators.

### STE-FlowNet → MVSEC

**Source checked:** `ruizhao26/STE-FlowNet@eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6`. The supplied files are a partial source snapshot, not a runnable checkout.

**Interface findings:** native representation is polarity-separated Gaussian-weighted event counts, not a generic signed voxel. The encoder leaves `data_split` undefined in the generic five-slice case, fails on empty/tiny or duplicate-timestamp slices, discards remainder events, and writes counts to `uint8` (possible wrap). The official training entry performs real-MVSEC validation during training; the pilot needs synthetic-only validation. Some `outdoor_day1` loader range logic disagrees with `validate()` and must not define real-test samples. Official code targets Python 3.9 / PyTorch 1.10.2+cu113 / correlation sampler 0.3.0, unlike the server environment; correlation extension feasibility remains untested.

**Data:** MVSEC was not found in a finite local WSL search. Protocol data are downloadable in principle; the four-sequence size estimate is about 33.2 GB (14.4 GB raw plus 18.8 GB GT, source-page sizes added), not a measured download. Server-side dataset presence is unchecked.

**Actually run:** no dependency installation, loader, model pass, training, or real evaluator.

**Next:** retrieve full pinned source; check/install only the narrow correlation dependency on a currently free GPU environment; implement benchmark-local encoder/loader edge fixes; build paired synthetic inputs; run edge-case loader checks, forward/backward, and a short synthetic-only train/val cycle. Keep MVSEC test-only.

### YOLO-6D / SingleShotPose → E-POSE

**Source checked:** `microsoft/singleshotpose@2c282619101f9f31666dd18b42393375bea1600f`; E-POSE annotation code `oussamaabdulhay/E_Pose@b96ae783fcb6de84e1e9c602e91bce941cab0234`. The supplied files are partial snapshots, not executable checkouts.

**Data:** no local E-POSE bag/HDF5 sample or object assets were found. The recorded Figshare API request for article `28106717` returned HTTP 403 on 2026-10-16; this establishes current access trouble, not permanent unavailability. Server-side files and official access must be checked before any claim that the dataset cannot be obtained.

**Interface and leakage findings:** E-POSE scripts are ROS-bag-oriented and the multi-object script is hard-coded for three objects. SingleShotPose's trainer/loader is RGB/LINEMOD-oriented and depends on masks, object crops, image compositing, pretrained weights, and periodic LINEMOD evaluation; it cannot be directly used as the event-only synthetic-to-real pipeline. It also does not reliably enforce the proposed update cap. A pure-event adapter and synthetic-only checkpoint selection are required. Full-frame unsegmented event tensors are the only permitted model inputs; GT masks, crops, and segmented events cannot be inputs or checkpoint-selection data. The event-map accumulator semantics and CAD/pose/projection coordinates require data-backed validation before progressing.

**Actually run:** no source install, data conversion, loader, model pass, or training.

**Next:** check server data and official access; retrieve complete pinned source; verify CAD/K/pose/frame conventions; if access and a small class subset are viable, implement a small full-frame event adapter and short synthetic-only train/val smoke. Otherwise keep this route explicitly optional and continue L1/L2.

## Shared constraints and current blockers

- No shared physical trajectory has been generated; therefore no EventGenesis/V2E event pair, shared label artifact, loader example, short training, or synthetic validation exists yet. The frozen EventGenesis physics plugin requires a scene-linear HDR renderer plus explicit camera/object pose providers; the first HDR smoke failed due to an incomplete runtime copy, so valid EventGenesis export remains gated on rerun.
- No real public dataset has been opened for model evaluation. The candidate models have not been scored or ranked on any real data.
- Main engineering gates are rerunning the full HDR overlay, then checking source/data availability. Next is a small common Genesis-rendered physical trajectory, exported once and consumed by both EventGenesis and V2E.
- The research candidate's formal corpus sizes and update budgets are not adopted as Stage 1 pilot requirements; Stage 1 uses a few trajectories and a handful of synthetic-only updates.
- No cost is estimated as a measured runtime. The 33.2 GB MVSEC figure is a download-size estimate only.

## Recommendation

Proceed with L1 and L2 paired synthetic smoke in parallel as soon as the server worktree and sources are available. Check E-POSE official data access and local artifacts in parallel; begin a small L3 smoke only if its data/assets and event-map semantics can be established without leaking GT masks/crops. Do not change model or dataset based on a real-test score. Do not modify the frozen simulator core.
