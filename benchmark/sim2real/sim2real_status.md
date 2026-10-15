# Sim-to-Real status — Stage 1 feasibility pilot

Updated: 2026-10-16 (Asia/Tokyo)

## Repository and execution location

- Upstream remote: `https://github.com/zyh2308/genesis-event-camera-plugin.git`.
- Dedicated branch: `exp/public-sim2real-v1`.
- Experiments ran on server `g1`, Ubuntu 20.04.6, project `/home/zhaoyuhan/Eventbased_WAM`, conda environment `Eventbased_WAM`. WSL (`Ubuntu-22.04`, `/home/科研`) was used for source preparation, evidence review, and repository work only.
- The server checkout `repositories/gec-sim2real` is a dedicated worktree. The source bundle includes pinned EventGenesis/V2E/E2VID/EVREAL/STE-FlowNet/SingleShotPose/E-POSE annotation sources; its 686-file SHA-256 manifest was checked on the server.
- Frozen simulator core `e79e656ac00f7847b98db9e12bf43b093f3afff1` remains unchanged and is an ancestor of this branch.
- No candidate model was evaluated or ranked on real data. HQF, MVSEC, and E-POSE were not opened for model evaluation or used for training, checkpoint selection, or simulator tuning.

## Route status

| Route | Source and data availability | Shared synthetic pair | Downstream loader/train/validation | Current status / next gate |
|---|---|---|---|---|
| E2VID recurrent → HQF | Pinned E2VID and EVREAL source checked. Public HQF route is listed by the event-flow source; no HQF data were downloaded to the project. | Same two physics rollouts; EventGenesis/V2E paired event inputs converted to the E2VID five-bin HDF5 interface. | **Actually run on g1:** loader, one official E2VID+ optimizer update per generator, and one synthetic validation sequence per generator. Exit 0. | Feasibility loop passes. Before formal training: obtain/verify all 14 HQF sequences and license, lock EVREAL preprocessing and the official evaluation script, then scale synthetic trajectories. |
| STE-FlowNet → MVSEC | Pinned STE-FlowNet source checked. The official MVSEC page lists event data, calibration, and dense-flow ground-truth generation; no MVSEC data were downloaded. | Same two physics rollouts, endpoint frames, synthetic forward SE(3) flow labels, and splits. Native STE-FlowNet five-slice representation generated for each event source. | **Actually run on g1:** loader, forward/backward, one official photometric optimizer update per generator, and one synthetic validation interval per generator. Exit 0. | Feasibility loop passes. Correlation extension was unavailable; the runner used a benchmark-local PyTorch equivalent for the official 1×1-kernel/9×9-patch operation. Before formal training, install/validate the pinned extension or review this adapter and then lock the official MVSEC test protocol. |
| YOLO-6D / SingleShotPose → E-POSE | Pinned SingleShotPose and E-POSE annotation sources checked. The paper reports the public Figshare dataset; its full archive is 27.72 GB. No sample or full E-POSE archive was downloaded. | Shared fixture has pose labels but uses a generic fall-box object, not an E-POSE/YCB object with matched CAD, calibration, and class annotations. | **Not run:** no E-POSE-compatible full-frame event loader or synthetic-only pose training/validation. | The remaining gate is the event-only input/annotation adapter and CAD/K/pose convention check. SingleShotPose's RGB/LINEMOD loader depends on crops/masks and real LINEMOD validation; those paths cannot be reused as model inputs or checkpoint-selection data. Keep GT masks, crops, and segmented events out of inference. First inspect the E-POSE file manifest and raw event/pose sample, then decide whether a leakage-safe full-frame adapter can use the pinned mature model. |

## Actual results and evidence

- Two independent Genesis GPU fall-box rollouts were generated, one train and one validation. Both branches share the same endpoints, labels, time windows, split, and fixed luminance mapping. The main HDF5 has scene-linear radiance, depth, segmentation, camera/object poses, pose labels, and ten forward SE(3) flow intervals per rollout. EventGenesis emitted 4,068 train and 4,832 validation events.
- Pinned V2E (`SensorsINI/v2e@cac638444557a224ba6ee5c29260916f152c00bd`) processed those same endpoints at 25 μs: 2,084 train and 2,571 validation events. A prior 1 ms run is retained but excluded: V2E completed yet warned of IIR update `eps=0.93` and an overly coarse step for 300 Hz cutoff.
- E2VID+ used five bins, seed 71, identical initial-state SHA-256 `d2d016f8b7589e0764cf7257a629dac84f1c6b6c70392964a185035b4acee008`, Adam 1e-4 with AMSGrad, LPIPS-VGG plus the pinned temporal-consistency loss, and one update. EventGenesis train/validation losses were `0.805037 / 0.806664`; V2E `0.812223 / 0.814109`. This is an interface smoke with one update, not a comparative result. Peak allocated GPU memory was 399 / 418 MB. The measured update wall times were 5.99 / 0.056 s; this single-update timing includes process/warm-up effects and is not a stable throughput comparison.
- STE-FlowNet used the same seed 71 and initial-state SHA-256 `c4b49f1a152b32bc07e6fb61c7355be3a93f74eece65b8f4c490576834d8fee4`. Each branch performed one update with the official four-scale Charbonnier photometric objective plus 10× second-order smoothness; flow labels were validation-only. Synthetic validation objective/EPE was `336.469 / 0.709 px per interval` for EventGenesis and `286.093 / 0.655 px per interval` for V2E. These one-update diagnostics are not a ranking. Peak allocated GPU memory was 12.74 / 13.06 GB; measured update time was 1.85 / 0.55 s.
- No real data were passed to either trainer or validation loader. No real model ranking, checkpoint selection, or simulator adjustment took place.
- Paired raw sample files, converted E2VID/STE inputs, command records, preflight, stdout, results, exit codes, hashes, and caveats are indexed under [`evidence/S0/20261016-stage1-pilot/`](evidence/S0/20261016-stage1-pilot/). Full datasets and model checkpoints stay on g1; their paths and hashes are in that evidence directory's `external_artifacts.json`.

## Server resource and cost notes

- Observed environment: Python 3.11.15, PyTorch 2.5.1+cu121, runtime CUDA 12.1, Genesis 1.2.2, h5py 3.14.0, NumPy 2.4.6. A run-scoped LPIPS VGG16 cache file (SHA-256 in the evidence index) was fetched by the pinned loss implementation.
- Model runs used physical GPU 1 (RTX 3090) after a fresh idle check. GPU 0 and GPU 3 had other users' jobs and were not used. Their processes were left untouched.
- Shared trajectory generation took roughly four minutes by wall-clock observation (estimate, not a benchmark). V2E measured 21.64/22.08 s for train/validation at 25 μs. E2VID and STE-FlowNet memory and one-update times above are measured from run artifacts.
- The source-page estimate for selected MVSEC data plus flow ground truth is about 33.2 GB (download estimate, not downloaded or measured here). E-POSE's page lists a 27.72 GB full archive. Disk quota was not measured. Formal corpus-size and full-training costs are not measured.

## Controls and known limits

- Within each task, both generator branches use the same physics samples, window labels, split, seed, initial model state, architecture, optimizer, update budget, and validation inputs/labels. Generator-specific event interpolation and V2E's fixed global Rec.709 → Reinhard → sRGB mapping remain numerical differences; neither simulator nor sensor was calibrated on real data. This pilot checks interface feasibility rather than a generator-only causal comparison.
- The 1 ms V2E output is excluded from all paired downstream inputs. The V2E 25 μs override follows the pinned sensor cutoff/timestep stability guidance; it is not claimed as a physical EVK/DAVIS calibration.
- The trajectory-generation process's shell exit code was not captured. Its printed summary was reviewed and its HDF5 reopened and shape/range checked on g1; no process exit status is claimed.
- An earlier E2VID attempt failed on a missing unused IPython import; a second attempt loaded the model but stopped on a wrapper config nesting error. The final run passed. An earlier STE run failed on an extra grayscale array dimension passed to OpenCV; its corrected run passed. These failed logs are retained.
- No frozen-core changes were made. Compatibility shims and adapters are benchmark-local; pinned third-party source trees remain unmodified.

## Recommended next action

The Stage-1 feasibility gate is complete for reconstruction and optical flow, incomplete for pose. Research review should first approve the limits of the one-update pilots. Then scale the two passing synthetic pipelines and define exact HQF/MVSEC split and preprocessing manifests before acquiring real test data. For E-POSE, inspect a raw event/pose/CAD/K sample and write down a full-frame unsegmented input contract; stop the route if the mature implementation cannot train without privileged crops, masks, segmented events, or real-data checkpoint selection. Only after that gate should any real-only test be unlocked.

## Pointers

- Full report: [`reports/stage1_feasibility.md`](reports/stage1_feasibility.md).
- Candidate tasks and adapter contracts: [`tasks/`](tasks/).
- Imported research protocol and source audit: [`research/input_2026-10-16/`](research/input_2026-10-16/).
- Public dataset records: [HQF in the event-flow source](https://github.com/tudelft/event_flow#download-datasets), [official MVSEC download and flow page](https://daniilidis-group.github.io/mvsec/download/), [E-POSE Figshare record](https://springernature.figshare.com/articles/dataset/E-POSE_A_Large_Scale_Event_Camera_Dataset_for_Object_Pose_Estimation/28106717), and [E-POSE paper](https://www.nature.com/articles/s41597-025-04536-5).
