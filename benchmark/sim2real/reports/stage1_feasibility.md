# Stage 1 feasibility report

Date: 2026-10-16 (Asia/Tokyo)

Status: source/server audit, HDR renderer and adapter-contract smoke pass. No moving shared trajectory, EventGenesis/V2E pair, downstream loader/training cycle, or real scoring has run.

## Scope and evidence labels

This report records static source review, server environment/repository checks, a renderer-only smoke, and a stationary-scene plugin adapter smoke on the server. `Source checked` means code/config metadata was inspected at a pinned revision or in the provided audited snapshot. The successful smokes establish the HDR and plugin call interfaces; they did not exercise a moving trajectory, paired event exports, downstream model forward, optimizer step, or real-set scoring. Cost estimates are labeled as estimates.

## Repository and simulator baseline

- Dedicated branch is `exp/public-sim2real-v1`, current commit `798df82c805629b4943e53e23982dace33394853` (upstream `origin/exp/public-sim2real-v1`).
- Frozen core `e79e656ac00f7847b98db9e12bf43b093f3afff1` resolves locally and is an ancestor of the dedicated branch. This work has not modified core files.
- Server checkout: `/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin`, branch `main`, commit `15479d97154274796043d5e5cc42a336f57b2dd9`, clean and reported one commit behind `origin/main`. Dedicated server worktree is `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real`, branch `exp/public-sim2real-v1`, current commit `798df82c805629b4943e53e23982dace33394853`.
- The frozen core's physics-first plugin requires linear radiance and explicit camera/object pose providers. The repository overlay contains eight Genesis source/shader files. The conda-installed Genesis 1.2.2 has no `radiance` argument. An initial isolated copy applied only the two `genesis/vis` files and failed with `TypeError: OffscreenRenderer.render() got an unexpected keyword argument 'radiance'`; the remaining six overlay files are under `genesis/ext/pyrender/`. The benchmark-local builder now applies the full overlay to a project-local package copy. The corrected renderer and adapter smokes passed without changing the conda package or frozen core.
- The local WSL checkout is preparation/review only. The user has directed all experiments to the remote research server.

## Server environment and resource observation

Measured during a read-only remote inspection:

- Host `g1`, project path `/home/zhaoyuhan/Eventbased_WAM`.
- Conda `Eventbased_WAM`: Python `3.11.15`; PyTorch `2.5.1+cu121`; `torch.cuda.is_available() == True`; runtime CUDA `12.1`; Genesis `1.2.2`.
- Genesis import emitted its compatibility warning that installed Torch `<2.8.0` is unsupported. This warning is recorded; no compatibility conclusion has been assumed.
- GPU inventory: 2× RTX 3090 (24 GB each), 3× V100S (32 GB each), 1× TITAN RTX (24 GB). During audit, GPU 0 and GPU 3 were occupied by other users' processes. GPU 1, 2, 4, and 5 showed no compute processes. Recheck immediately before launch; do not use GPU 0 or 3.
- `/home` NFS had 12 TB available; per-user quota was not checked.
- An existing user-level `gpu_watch.py` process was observed and left untouched.
- Immediately before each successful smoke, GPU 1 (RTX 3090) was idle and selected; GPU 3 had two other-user CARLA processes and was not used. The first renderer smoke failed due to incomplete runtime assembly. The corrected renderer smoke passed: legacy RGB was `uint8` (48×64×3), HDR radiance was finite `float32` (48×64×3, max 10.60, 295 pixels >1), and depth was `float32` (48×64). Measured call times were 25.1 s for the first legacy RGB render (including cold compilation) and 34.6 ms for the following HDR RGBD render. A shared Quadrants cache-lock warning appeared in that first successful renderer smoke; the cache was not cleaned or modified. The adapter smoke passed with run-scoped Genesis, Taichi, and Quadrants caches and called plugin attach/capture twice on a stationary scene. It checks the Nx4 event-array interface but does not require a nonzero event count. Raw logs and hashes are indexed in `evidence/S0/20261016-server-audit/external_artifacts.json`; files remain under `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/`.

## Route findings

### E2VID recurrent reconstruction → HQF

**Source checked:** E2VID+ public training snapshot at `TimoStoff/event_cnn_minimal@78aa471a66593f55e349303563997297fbc816e8`; EVREAL at `ercanburak/EVREAL@398f03551b7f5d150ae10aa89e4e731557fce240`. The local research package is an extracted subset, not complete checkouts.

**Interface findings:** public reconstruction config has a 10-bin model field while loader uses 5 bins; training and validation must both be explicitly configured for 5 signed channels. `SequenceDataset` dynamically evaluates the dataset class name, but the pinned config says `HDF5Dataset` while the source defines `DynamicH5Dataset`, so the stock config raises `NameError`. `HDF5DataLoader` expects a CSV manifest of HDF5 paths, not one HDF5 file or a directory. Also, `SequenceDataset` returns zero length when `L >= dataset.length`; for the candidate between-frame validation clip (`L=200`, 200 intervals from 201 images), this yields an empty validation dataset. Stored polarity must be OFF=`0`, ON=`1` because the loader maps it with `2p-1`; writing ±1 directly gives an incorrect OFF magnitude. The event-window/index endpoint mapping needs a known-timestamp fixture to catch off-by-one windows. The candidate also has a train/validation image normalization mismatch (`normalize_image=false` for train and `true` for validation), which must be explicitly resolved. A narrow, method-independent loader/indexing correction plus per-split file-list CSVs is required. EVREAL has distinct E2VID and E2VID+ normalization paths; use the pinned E2VID+ route consistently.

**Data:** HQF was not found in the finite local WSL project/data scan or targeted server scan of `~/Eventbased_WAM/data` (directory-name scan to depth 4); this is not a global search and does not establish unavailability. The authors' public page and EVREAL preparation tools expose a dataset route, but no files were downloaded or hashed. EVREAL expects converted memmap files (`events_ts.npy`, `events_xy.npy`, `events_p.npy`, `images.npy`, `images_ts.npy`, `image_event_indices.npy`), not a raw bag directly.

**Actually run:** no loader, forward pass, training, or real reconstruction evaluation.

**Next:** retrieve complete pinned E2VID+ and EVREAL repositories; verify the HQF route from the author source and server data root; fix the dataset class string, file-list CSV interface, 5-channel setup, and validation-length indexing in a benchmark-local adapter; make paired short synthetic clips and run loader plus short synthetic-only training/validation for both generators.

### STE-FlowNet → MVSEC

**Source checked:** `ruizhao26/STE-FlowNet@eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6`. The supplied files are a partial source snapshot, not a runnable checkout.

**Interface findings:** native representation is polarity-separated Gaussian-weighted event counts, not a generic signed voxel. The encoder leaves `data_split` undefined in the generic five-slice case, fails on empty/tiny or duplicate-timestamp slices, discards remainder events, and writes counts to `uint8` (possible wrap). The official training entry performs real-MVSEC validation during training; the pilot needs synthetic-only validation. Some `outdoor_day1` loader range logic disagrees with `validate()` and must not define real-test samples. Official code targets Python 3.9 / PyTorch 1.10.2+cu113 / correlation sampler 0.3.0, unlike the server environment; correlation extension feasibility remains untested.

**Data:** MVSEC was not found in the finite local WSL search or targeted server scan of `~/Eventbased_WAM/data` to depth 4; this is not a global search. The official download page lists the HDF5 recordings and flow ground truth. The four-sequence size estimate is about 33.2 GB (14.4 GB raw plus 18.8 GB GT, source-page sizes added), not a measured download.

**Actually run:** no dependency installation, loader, model pass, training, or real evaluator.

**Next:** retrieve full pinned source; check/install only the narrow correlation dependency on a currently free GPU environment; implement benchmark-local encoder/loader edge fixes; build paired synthetic inputs; run edge-case loader checks, forward/backward, and a short synthetic-only train/val cycle. Keep MVSEC test-only.

### YOLO-6D / SingleShotPose → E-POSE

**Source checked:** `microsoft/singleshotpose@2c282619101f9f31666dd18b42393375bea1600f`; E-POSE annotation code `oussamaabdulhay/E_Pose@b96ae783fcb6de84e1e9c602e91bce941cab0234`. The supplied files are partial snapshots, not executable checkouts.

**Data:** no local WSL or targeted server project-data-tree E-POSE bag/HDF5 sample or object assets were found. The Scientific Data paper describes the dataset as public via Figshare; one recorded API endpoint for article `28106717` returned HTTP 403 on 2026-10-16. This establishes a failed API route, not dataset unavailability; the official download path still needs checking.

**Interface and leakage findings:** E-POSE scripts are ROS-bag-oriented and the multi-object script is hard-coded for three objects. SingleShotPose's trainer/loader is RGB/LINEMOD-oriented and depends on masks, object crops, image compositing, pretrained weights, and periodic LINEMOD evaluation; it cannot be directly used as the event-only synthetic-to-real pipeline. It also does not reliably enforce the proposed update cap. A pure-event adapter and synthetic-only checkpoint selection are required. Full-frame unsegmented event tensors are the only permitted model inputs; GT masks, crops, and segmented events cannot be inputs or checkpoint-selection data. The event-map accumulator semantics and CAD/pose/projection coordinates require data-backed validation before progressing.

**Actually run:** no source install, data conversion, loader, model pass, or training.

**Next:** verify the official download path and retrieve complete pinned source; check CAD/K/pose/frame conventions; if assets and a small class subset are viable, implement a small full-frame event adapter and short synthetic-only train/val smoke. Otherwise keep this route explicitly optional and continue L1/L2.

## Shared constraints and current blockers

- No moving shared physical trajectory has been generated; therefore no paired EventGenesis/V2E event output, shared label artifact, task loader, short training, or synthetic validation exists yet. The HDR and plugin call interfaces passed small static-scene smokes, but valid paired export still needs a motion/pose fixture and shared frame/label manifest.
- No real public dataset has been opened for model evaluation. The candidate models have not been scored or ranked on any real data.
- Main engineering gates are the moving trajectory/pose provider, V2E input adapter, and full third-party model source compatibility. Next is a small common Genesis-rendered physical trajectory, exported once and consumed by both EventGenesis and V2E.
- The research candidate's formal corpus sizes and update budgets are not adopted as Stage 1 pilot requirements; Stage 1 uses a few trajectories and a handful of synthetic-only updates.
- Measured renderer cost: the corrected 64×48 HDR smoke took about 45 s wall-clock on an idle RTX 3090; the first RGB call took 25.1 s including cold compilation and the second HDR RGBD call 34.6 ms. The stationary adapter smoke passed, but its GPU memory peak was not sampled. The corrected renderer run directory was 788 KB and the adapter run directory 9.6 MB including isolated caches. No checkpoint or dataset was created. The 33.2 GB MVSEC figure remains a download-size estimate only; model-training cost is not measured.

## Recommendation

Proceed with L1 and L2 paired synthetic smoke in parallel as soon as the server worktree and sources are available. Check E-POSE official data access and local artifacts in parallel; begin a small L3 smoke only if its data/assets and event-map semantics can be established without leaking GT masks/crops. Do not change model or dataset based on a real-test score. Do not modify the frozen simulator core.
