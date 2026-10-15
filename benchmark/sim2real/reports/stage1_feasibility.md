# Public real-data Sim-to-Real — Stage 1 feasibility report

Date: 2026-10-16 (Asia/Tokyo)  
Execution host: `g1`, `/home/zhaoyuhan/Eventbased_WAM`, conda `Eventbased_WAM`  
Repository branch: `exp/public-sim2real-v1`

## 1. Outcome

The experiment ran on the remote research server. WSL was used to prepare and review source/configuration, then retrieve small artifacts and commit the handoff. Two routes reached the requested minimum synthetic loop:

1. Shared Genesis physics rollouts → EventGenesis/V2E events → E2VID+ data loader → one synthetic optimizer update → synthetic validation.
2. The same rollouts/events/labels → STE-FlowNet representation/loader → forward and backward → one synthetic optimizer update → synthetic validation.

The E-POSE route remains at source and data availability review. Its public dataset archive was not downloaded, and a leakage-safe full-frame event adapter for the pinned RGB pose implementation is unresolved. This is an explicit feasibility blocker, not a successful model run.

No real dataset was used for model training, validation, checkpoint selection, simulator tuning, or ranking. There are no real-test scores in this report. One update per branch on two tiny rollouts only confirms interfaces, finite losses, and resource footprint; it is not evidence of model quality or relative generator performance.

Evidence terms used below:

- **Source checked:** pinned implementation/configuration inspected; no claim that it ran.
- **Actually run:** command ran on `g1`; stdout/result/exit status were retained where captured.
- **Estimated:** derived from page metadata, a rough wall-clock observation, or extrapolation and not measured as a benchmark.

## 2. Repository, versions, and frozen-core boundary

- Remote: `https://github.com/zyh2308/genesis-event-camera-plugin.git`.
- Work branch: `exp/public-sim2real-v1`; server checkout `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real`.
- Frozen simulator baseline: `e79e656ac00f7847b98db9e12bf43b093f3afff1`. The branch retains it as an ancestor. No frozen-core files were changed.
- Actual server environment: Ubuntu 20.04.6; Python 3.11.15; PyTorch 2.5.1+cu121; runtime CUDA 12.1; Genesis 1.2.2; OpenCV 5.0.0; h5py 3.14.0; NumPy 2.4.6. Genesis printed a compatibility warning that Torch below 2.8 is unsupported; despite that warning, the recorded renderer, event generation, and model runs completed.
- GPU inventory: 2×RTX 3090, 3×V100S-32GB, 1×TITAN RTX. The successful runs selected physical GPU 1 (RTX 3090) after a fresh idle check. GPU 0 and GPU 3 were occupied by other users and were left alone.
- The full pinned source bundle was copied into the server project and verified against its file manifest: 121 `event_cnn_minimal`, 480 EVREAL, 14 STE-FlowNet, 56 SingleShotPose, and 15 E-POSE annotation files (686 total). V2E is separately pinned at `SensorsINI/v2e@cac638444557a224ba6ee5c29260916f152c00bd`.
- Task source pins: E2VID training `TimoStoff/event_cnn_minimal@78aa471a66593f55e349303563997297fbc816e8`; EVREAL `ercanburak/EVREAL@398f03551b7f5d150ae10aa89e4e731557fce240`; STE-FlowNet `ruizhao26/STE-FlowNet@eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6`; SingleShotPose `microsoft/singleshotpose@2c282619101f9f31666dd18b42393375bea1600f`; E-POSE annotation tools `oussamaabdulhay/E_Pose@b96ae783fcb6de84e1e9c602e91bce941cab0234`.
- Third-party source was not edited. Compatibility code and adapters live under `benchmark/sim2real/scripts/` and are invoked at runtime.

## 3. Shared physics and event-generation fixture — actually run

### Shared trajectory

The server generated two independent Genesis GPU rollouts of a falling box: `train_00` and `validation_00`. Each rollout used a fixed 160×128 camera, 1 ms physics steps, 100 Hz endpoint renders, and a 0.1 s window. The box moved from z=0.5 m to about z=0.450459 m. The train/validation split was fixed before either event generator ran.

The HDF5 stores 11 shared scene-linear RGB radiance, depth, and segmentation endpoints; camera/object poses; right-endpoint pose labels; ten forward SE(3) flow intervals; validity masks; timestamps; fixed display-luma endpoints; and EventGenesis raw events. The flow validity fraction was 1.0 in each rollout. The trajectory-generation shell exit code was not captured: its summary was printed and the resulting HDF5 was reopened for schema, shape, and value checks, but this report does not claim an exit status.

### EventGenesis and V2E

- EventGenesis consumed native scene-linear radiance and the plugin's physics/pose interpolation. It emitted 4,068 events for train and 4,832 for validation. Its stored event columns are `t,x,y,p` with seconds and signed polarity.
- V2E consumed the same endpoint PNGs and shared labels. A single global luminance conversion was used: Rec.709 luminance, exposure 1, Reinhard tone mapping, sRGB transfer, then round-to-nearest uint8. No per-frame normalization or auto-exposure was used. Pinned SuperSloMo interpolated endpoint frames. The conversion is a fixed engineering mapping, not a calibrated camera response.
- At 1 ms, V2E returned exit code 0 and emitted 2,107/2,617 train/validation events, but the logs reported IIR update `eps=0.93` and a cutoff warning for 300 Hz. Those files are retained and explicitly excluded from paired model inputs.
- The accepted follow-up used 25 μs, set by the configured 300 Hz time constant and the V2E `sensor_eps_target=.05` numerical target. It emitted 2,084/2,571 train/validation events; both V2E episodes recorded exit code 0 and no large-step or coarse-low-pass warning. The HDF5 arrays were sorted, in bounds, and had polarity 0/1. This is a numerical stability setting, not a real sensor calibration.
- Approximate wall time for shared trajectory generation was four minutes by observation (not a benchmark). V2E logs measured 21.64 s and 22.08 s for the accepted train and validation episodes; the excluded 1 ms episodes measured 8.81 s and 6.61 s.

The common physics, endpoint frames, labels, windows, and split are shared. Temporal interpolation and display/radiance input differ by design: EventGenesis sees native linear radiance and physics pose, while V2E sees globally mapped display-luma endpoints and SuperSloMo intermediates. Results below are adapter feasibility checks, not a calibrated generator-only comparison.

## 4. Route 1 — E2VID+ recurrent reconstruction → HQF

### Source/data findings

**Source checked:** the pinned E2VID and EVREAL trees are available on g1 and their exact file set is hashed. The source config names `HDF5Dataset`, while the pinned source defines `DynamicH5Dataset`; it expects CSV HDF5 manifests. It has a small-dataset guard where sequence length at least the data length produces an empty validation set. The stock path also needs the stored polarity convention OFF=0/ON=1 (the loader maps it to signed channels). An endpoint timestamp tie in its binary search can select an arbitrary tied index without a right-sided search.

**Data availability checked:** the `tudelft/event_flow` source links the public HQF dataset route and reports a packaged HDF5 path. HQF was not found in the inspected project data roots and no HQF files were downloaded or evaluated. This local scan does not establish that HQF is unavailable. HS-ERGB remains an optional extension and was not acquired.

### Adapter and actual run

Benchmark-local conversion writes paired HDF5 clips at:

`/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage0/e2vid_loader_20261016-v1/{eventgenesis,v2e}/{train,validation}.h5`

The adapter preserves event windows, uses five combined bins, maps polarity to OFF=0/ON=1, builds right-endpoint targets, and exposes the shared synthetic flow only to the official training loss. It applies `numpy.searchsorted(..., side="right")` for tied event timestamps, uses the actual dataset class, creates the required CSV manifests, shortens the tiny pilot sequence to length 2, and sets the same image normalization for train/validation. The stored flow is scaled by observed event dt so the stock loader restores the shared interval displacement. Runtime-only compatibility aliases cover the removed `skimage.measure.compare_ssim` symbol and an unused IPython debug import; upstream source remains untouched.

The final server run used official `E2VIDRecurrent`, seed 71, same initial state, Adam with AMSGrad at 1e-4, LPIPS-VGG weight 1 and temporal-consistency weight 2 (`L0=2`), five bins, one update, and one synthetic validation sequence per generator. Both branches reported exit code 0 and `real_data_accessed=false`.

| Event source | train loss | synthetic validation loss | one update (s) | peak allocated GPU memory | checkpoint SHA-256 |
|---|---:|---:|---:|---:|---|
| EventGenesis | 0.805037 | 0.806664 | 5.9919 | 398,907,904 B | `c46985327ee727bb07ab4ff713f0ac585fd56c1a63e218f4b15587469bf2730e` |
| V2E | 0.812223 | 0.814109 | 0.0555 | 417,556,480 B | `6f55982c4c079b6c63703bd5a15db09d008d6991ca5e7e18e2c6db116ff69673` |

Initial-state SHA-256 matched at `d2d016f8b7589e0764cf7257a629dac84f1c6b6c70392964a185035b4acee008`. The one-update timings differ sharply and include first-call/warm-up effects; they are retained as raw observations and should not be compared as stable throughput. The loss values are not a generator ranking.

Earlier failures are preserved: v1 stopped at an absent IPython module; v2 initialized the model/loader but failed because the runner passed a config dictionary with an extra nesting level. The final v3 run passed. A 528 MiB-class VGG16 loss-network cache was downloaded by LPIPS at `/home/zhaoyuhan/.cache/torch/hub/checkpoints/vgg16-397923af.pth`; its exact hash and path are recorded in the external artifact index. This was a fixed perceptual loss network, not a model initialization or real-data checkpoint.

### Next gate

For a formal experiment, acquire the intended HQF sequences on g1, record file hashes and camera/event timing metadata, lock the exact EVREAL preprocessing/evaluation path, and expand the synthetic corpus and update budget before enabling a real-only final test. No HQF real score is reported here.

## 5. Route 2 — STE-FlowNet → MVSEC

### Source/data findings

**Source checked:** pinned STE-FlowNet is now on g1. Its native input is polarity-separated Gaussian-weighted equal-event-count slices. Inspection found generic-slice indexing failures, empty/tied timestamp edge cases, remainder-event loss, potential uint8 count overflow, and a stock training entry that evaluates MVSEC during training. The pilot uses benchmark-local input conversion and a synthetic-only runner, not the stock real-data validation loop.

**Data availability checked:** the official MVSEC page lists sequence data/calibration and provides code and artifacts for dense optical-flow ground truth generated from pose/depth. No MVSEC recording or ground truth was downloaded. The source-page estimate for the four candidate sequences plus ground truth is about 33.2 GB (estimate only).

### Adapter and actual run

Paired NPZs are under:

`/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage0/steflow_inputs_20261016-v1/{eventgenesis,v2e}/{train,validation}.npz`

The representation is `[10,2,256,256,5]` per rollout: polarity separated, five Gaussian-weighted event-count slices, 160×128 inputs center-padded by 48 px horizontally and 64 px vertically. The adapter retains remainder events, guards empty/tied/constant timestamp slices, and fails closed on uint8 overflow. The shared mapped grayscale endpoints and synthetic forward flow labels are carried alongside it.

The optional `spatial_correlation_sampler` extension was not installed. The benchmark runner supplied a pure PyTorch `F.unfold` equivalent for the pinned model's `(kernel_size=1, patch_size=9)` correlation call; the official photometric objective (four-scale Charbonnier plus 10× second-order smoothness) was retained. The source's hard `.cuda()` call worked with `CUDA_VISIBLE_DEVICES=1`, which exposes physical GPU 1 as logical device 0. Synthetic flow was only used for validation diagnostics, not passed to the model or optimizer.

Both branches used seed 71, identical initial state, official STE-FlowNet, and one update. Each passed forward/backward and synthetic validation with exit code 0 and `real_data_accessed=false`.

| Event source | train objective | synthetic validation objective | validation EPE (px/interval) | one update (s) | peak allocated GPU memory | checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| EventGenesis | 140.4264 | 336.4695 | 0.708796 | 1.8461 | 12,739,819,008 B | `9b19db44e66cc9394c3ccfab6d1ed1b174d1c3941ddca2f9af43f65245c84c0c` |
| V2E | 132.1949 | 286.0927 | 0.654945 | 0.5528 | 13,060,759,552 B | `936136dd9334fc937308f0de995060221a505059849714ac5a4f3443417ec3d4` |

Initial-state SHA-256 matched at `c4b49f1a152b32bc07e6fb61c7355be3a93f74eece65b8f4c490576834d8fee4`. Both validation figures use one interval after one update; the unequal event counts are generator outputs. They are diagnostics only and not a performance ranking.

The first STE attempt failed on an extra array dimension sent to OpenCV resize. The corrected v2 run passed. Both attempts are preserved. No extension was installed into the shared conda environment.

### Next gate

Before formal training, either test/install the exact pinned correlation extension in a project-isolated environment or retain and independently review the narrow PyTorch equivalent. Then scale synthetic trajectories, pin the eligible MVSEC sequences/calibration/flow masks, and keep all MVSEC records out of training, validation, and checkpoint selection.

## 6. Route 3 — YOLO-6D / SingleShotPose → E-POSE

### Source/data findings

**Source checked:** full pinned SingleShotPose and E-POSE annotation source snapshots are available on g1. The E-POSE annotation repository is tooling, not a complete downstream training reproduction. SingleShotPose's pinned data path is RGB/LINEMOD-oriented and assumes object crops/masks, object-specific pretrained initialization, and periodic LINEMOD evaluation; it does not supply the needed full-frame event-only synthetic train/validation path.

**Public data availability checked:** the Scientific Data paper links a Figshare record for the E-POSE dataset. The publisher's Figshare page was accessible during the source/data audit and lists the full archive as 27.72 GB. The paper describes event, pose, camera-intrinsic, CAD/keypoint, mask, crop-related, and segmented-event records. No data were downloaded in this pilot; there are therefore no sample hashes, and this is not an assertion that the dataset is unavailable.

### Integrity constraint and blocker

The task requires that real inference inputs be computable from raw, full-frame events alone. GT mask, GT crop, and segmented events may be labels/audit material but cannot enter the model input or checkpoint-selection loop. The paper's provided mask and segmented-event products are explicitly not used here. In addition, the current fall-box fixture is not an E-POSE YCB CAD object and does not establish the E-POSE camera intrinsics, object coordinate convention, or pose-frame mapping.

The data, labels, mature implementation, and annotation conventions have not yet been brought together into a runnable leakage-safe adapter. No E-POSE loader, model forward pass, optimizer update, or validation was run. This route remains unresolved at integration, not blocked by confirmed data unavailability or compute cost.

### Next gate

Inspect the Figshare file manifest and a single raw event/pose/CAD/K sample on g1. Confirm coordinate frames, units, timestamps, and the raw full-frame event source. Then check whether the pinned mature pose model can train/validate from synthetic full-frame event representations without mask/crop/segmented-event inputs or real-data checkpoint selection. If not, keep the pose candidate explicitly parked and ask the research review session before changing the main model or task definition.

## 7. Resource accounting and limits

| Item | Observed / estimated cost | Evidence status |
|---|---|---|
| Shared Genesis fixture | ~4 min wall time; one pair of tiny 160×128 rollouts | Wall-clock estimate; not benchmark timing |
| V2E accepted 25 μs pair | 21.64 s train / 22.08 s validation | V2E logs |
| E2VID one-step training | 5.99 s / 0.056 s, peak 399 / 418 MB | Run result; first-step timings not comparable |
| STE-FlowNet one-step training | 1.85 s / 0.55 s, peak 12.74 / 13.06 GB | Run result |
| Checkpoints | Two E2VID checkpoints ≈171 MB each; two STE checkpoints ≈580 MB each | Server files; path/hash recorded; excluded from Git |
| LPIPS VGG16 loss model | Downloaded into user cache, ≈528 MiB | Actual presence; exact hash recorded |
| MVSEC candidate set + GT | ≈33.2 GB | Source-page size estimate; not downloaded |
| Full E-POSE dataset | 27.72 GB | Publisher Figshare page metadata; not downloaded |
| Full formal training / test | Not estimated from a one-step pilot | Not run |

Peak GPU memory above is PyTorch allocated memory, not total device occupancy. Dataset sizes are acquisition estimates, not server disk-use measurements. The `/home` volume had been observed with available space during initial audit, but no per-user quota was checked.

## 8. Reproduction entry points and evidence

Commands, environment captures, preflight snapshots, stdout, result JSON, exit codes, and the first failed attempts are retained in `evidence/S0/20261016-stage1-pilot/raw_logs/`. The model runs used detached tmux launchers with recorded PIDs and per-run log directories. The matching configs and scripts are tracked under `configs/` and `scripts/`.

Key server inputs:

- Shared data: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage0/shared-trajectory-20261016-v1/`.
- V2E accepted events: `.../v2e_output_eps005/{train_00,validation_00}.h5`.
- E2VID loader data: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage0/e2vid_loader_20261016-v1/`.
- STE-FlowNet inputs: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage0/steflow_inputs_20261016-v1/`.
- E2VID successful run: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-e2vid-stage0-v3/`.
- STE-FlowNet successful run: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-s0-steflow-stage0-v2/`.
- Full data/checkpoint paths and SHA-256 values: [`evidence/S0/20261016-stage1-pilot/external_artifacts.json`](../evidence/S0/20261016-stage1-pilot/external_artifacts.json).
- Small sample files and a committed artifact index: [`evidence/S0/20261016-stage1-pilot/`](../evidence/S0/20261016-stage1-pilot/).
- Stage-0 renderer/source environment audit: [`evidence/S0/20261016-server-audit/`](../evidence/S0/20261016-server-audit/).
- Imported formal baseline and candidate protocol: [`research/input_2026-10-16/`](../research/input_2026-10-16/).

Dataset access references checked for this report: [HQF route in the event-flow source](https://github.com/tudelft/event_flow#download-datasets), [official MVSEC data and flow page](https://daniilidis-group.github.io/mvsec/download/), [E-POSE Figshare record](https://springernature.figshare.com/articles/dataset/E-POSE_A_Large_Scale_Event_Camera_Dataset_for_Object_Pose_Estimation/28106717), and [E-POSE dataset paper](https://www.nature.com/articles/s41597-025-04536-5).
