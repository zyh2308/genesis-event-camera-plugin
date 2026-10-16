# Stage 2 — full synthetic training-chain validation and real-evaluation preparation

> **Correctness corrections added in Stage 3:** (1) the Stage 2 STE-FlowNet EPE values retained below are invalid because the scorer extracted only horizontal flow `u` and NumPy broadcast it across both GT channels; the same four checkpoints were re-scored correctly without retraining. (2) the Stage 2 pose pilot reused the rollout-tail timestamp as the event-window end, misaligning selected windows and labels; its reported keypoint EPE is not a valid model diagnostic and was replaced by the corrected Stage 3 paired pilot. See [`STAGE3_RESEARCH_HANDOFF.md`](../STAGE3_RESEARCH_HANDOFF.md).

Date: 2026-10-16 (Asia/Tokyo)
Execution host: g1, Ubuntu 20.04.6, conda `Eventbased_WAM`, physical GPU 1 (RTX 3090, 24 GB).
Repository branch: `exp/public-sim2real-v1`.
Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1` (unchanged).

## Scope and result

Stage 2 checked whether shared physical trajectories can pass through both event generators, the mature task-specific data/model interface, short synthetic-only training, and independent synthetic validation. It did not run formal training or rank models on real data.

| Route | Source checked | Actually executed | Real data/evaluator status | Stage 2 result |
|---|---|---|---|---|
| E2VID+ → HQF | E2VID, EVREAL, HQF loader and evaluator entry points | Six-rollout paired conversion; 224 train updates and 56 validation sequences per generator; R1/R2 and edge fixtures | HQF manifest and candidate protocol only; no files downloaded or scored | Synthetic chain passes. HQF file/license and candidate adapter remain to be verified. |
| STE-FlowNet → MVSEC | Pinned STE-FlowNet encoder, losses, GT association helper and dt1/dt4 references | Six-rollout paired conversion; dt1/dt4, 16 updates and synthetic validation per generator/task; checkpoint save/restore and correlation operator | Stage 2 had no real files; Stage 3 parsed a fixed public MVSEC slice and ran event-only candidate inference, with no formal ranking | Training/checkpoint interface passes. **Stage 2 EPE is invalid**; corrected rescore and real-input smoke are in Stage 3. |
| SingleShotPose → E-POSE | Pinned SingleShotPose and E-POSE annotation source/paper | Generic-cuboid event-only full-frame pilot, two paired short runs; window-label timing was later found wrong | Publisher metadata/sample retrieval attempted; no raw sample or CAD parsed; no real evaluation | Stage 2 pose metrics are invalid due to time-misaligned event windows. Stage 3 reran corrected windows and documented remaining confidence/BatchNorm inference issues. E-POSE class/CAD contract remains blocked. |

The reconstruction validation losses below are diagnostics for code/data-path verification only. They do not estimate real generalization. The Stage 2 optical-flow EPE table is invalid due to the component-extraction bug described above; corrected figures are recorded only in Stage 3.

## Shared bank and fairness controls

- Shared trajectory and labels: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage2/20261016-s2-pilot-v1/shared/shared_trajectory.h5` (26,526,701 bytes, SHA-256 `68011b77d292ca5600fb10b3464af26c958c785d2258e8a8cfea0db35b15763d`). It contains six complete 0.3 s rollouts at 100 Hz, 31 frames / 30 intervals each; four whole train rollouts and two whole validation rollouts. Both generators consume the same rollout frames, times, labels and splits.
- Camera matrix in each episode: `[[137.24844291261175,0,80],[0,137.24844291261175,64],[0,0,1]]` in pixel coordinates at 160×128. Time is float64 seconds from rollout start. World coordinates are Genesis z-up, meters; camera coordinates follow OpenCV x-right/y-down/z-forward. Pose labels use the right endpoint. Flow labels are forward source-to-target pixel displacement over one endpoint for dt1 and four endpoints for dt4. Shared validity uses source depth, positive target depth and target-image bounds; no target-occlusion z-buffer is implemented in this small bank.
- Direction fixture: a +0.25 m world-x translation at 1 m depth and `fx=2` projects to +0.5 px. Flow labels are created once from physical geometry and copied to each generator adapter; pair label hashes match. E2VID boundary fixtures show the same label hash and 64 valid pixels for empty events, two sparse events and tied timestamps.
- Each model pair uses the same seed 71 and same serialized initial weights, architecture, optimizer, update count, rollout split and synthetic validation. Model inference stays event-only. The pose pilot uses event-derived channels only; Genesis pose/K are supervised targets. No depth, segmentation, GT pose, masks, crops, boxes, RGB or segmented events enter inference.
- The generator configurations are not calibrated to a real sensor. V2E receives the same scene-linear renders through a fixed Rec.709 luminance → Reinhard → sRGB uint8 mapping; EventGenesis consumes linear radiance. This is retained as a documented engineering input difference, not presented as a calibrated, generator-only causal comparison. There was no real-data feedback into simulator settings.

## Route A — reconstruction and Stage 1 R1/R2

### Executed evidence

- Pinned E2VID recurrent architecture, five bins, Adam at `1e-4` with AMSGrad, official LPIPS-VGG and temporal-consistency components; sequence length 3; two epochs; 112 train sequences per epoch; 224 updates per generator. Four physical train rollouts and two independent validation rollouts were used. The initialization hash matched across EventGenesis/V2E: `b07667dbbfe18c1432ee543501b2686fb026d0f02c3e69eb23ebd10e4a5cffb0`.
- R1: temporal consistency is no longer silently skipped by a two-frame sample. It made 224 valid training calls and 56 validation calls on each generator. Perceptual and temporal components were saved separately. Every update had finite nonzero gradients (minimum 30 finite gradient tensors); recurrent state contained six tensors after a sequence. Both checkpoints restored with an identical model-state hash.
- R2: the dataset adapter bypasses the legacy event-span scaling and uses physical endpoint displacement plus the common geometry mask. Empty, sparse-two-event and same-timestamp fixtures all preserve flow hash `5d08c7b0296c57168a294d3c56971406c919892896a69cfd87421cac20ac71d5`; each has 64 valid pixels. The sparse and tied cases have observed event span 6 ms and 0 s respectively, yet labels do not change.
- Synthetic diagnostics: EventGenesis train perceptual/temporal means `0.3432 / 0.0370`, validation `0.2069 / 0.0153`, 14.08 s train wall, 0.69 GB peak allocation. V2E train `0.4429 / 0.0384`, validation `0.2401 / 0.0232`, 15.12 s, 1.08 GB. These numbers are not a performance conclusion.
- Direction check: the shared geometric projection gives positive forward x flow. A one-pixel image translation yields unmasked E2VID warp L1 `0.004762` for forward flow and `0.052381` for reversed flow. The pinned E2VID temporal objective estimates a flow-based visibility mask; its raw scalar is near zero for the reversed fixture because invalid correspondences can be masked. Consequently the unmasked warp/alignment and geometry fixtures, rather than the masked scalar alone, establish direction. The exact pinned loss code is `event_cnn_minimal@78aa471a66593f55e349303563997297fbc816e8`, `model/loss.py` SHA-256 `1b28728a1fb8645230b67f5d3c0598b2836a5cf0bf3828dde8a81dfb6e08df48` and `utils/loss.py` SHA-256 `3d94483e3bc05582560efd9fa2d50f751ecff52321ec812ae04429b36c88aeb4`. This visibility-mask behavior is retained as a caveat, not hidden as a successful loss-direction metric.

### HQF preparation (source checked, not run on data)

The pinned EVREAL commit is `398f03551b7f5d150ae10aa89e4e731557fce240`; the source bundle has native `MemMapDataset`, HQF config and `eval.py`. The candidate is all 14 HQF sequences, continuous recurrent evaluation reset at sequence start, no RGB seed frame, LPIPS-Alex primary with SSIM/PSNR/MSE support metrics, equal sequence weight. These are a candidate protocol, not frozen. HQF is chosen over the earlier HS-ERGB extension because HQF is reconstruction-specific and has a native EVREAL route; HS-ERGB originates from video interpolation and requires a separately defined reconstruction target time/alignment. The study should freeze that rationale before any real inference. No real files, license, or candidate E2VID wrapper have yet been checked. Public source estimates are 19.4 GB compressed / 94 GB uncompressed; estimates only.

## Route B — optical flow and dt1/dt4

### Executed evidence

- Pinned STE-FlowNet commit `eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6`. dt1 and dt4 each ran two epochs and 16 updates per generator on the same four train/two validation rollouts and paired initialization `c4b49f1a152b32bc07e6fb61c7355be3a93f74eece65b8f4c490576834d8fee4`.
- The official four-scale Charbonnier photometric objective and 10× second-order smoothness term were finite. At least 70 gradient tensors were finite on every update. Synthetic labels were used for validation EPE only. All four checkpoints were saved, restored and hash-checked.
- **INVALID historical EPE** — Stage 2 dt1 numbers were EventGenesis `0.8301 px/interval` and V2E `0.8082`; dt4 numbers were `2.8537` and `2.8379`. The output conversion retained only `u`, then broadcasting duplicated it as `v`. Keep these values only to understand the earlier run; do not use them for comparison. The corrected values from the same checkpoints appear in Stage 3. Stage 2 train walls and peak training allocation remain valid: dt1 `16.33 / 15.21 s`, `3.63 GB`; dt4 `57.04 / 58.58 s`, `12.56 GB` on an RTX 3090.
- The pinned optional correlation extension was not used. A benchmark-local pure-PyTorch custom-autograd 9×9 shift correlation was checked against an explicit 81-shift reference in float32 at input `(1,64,128,128)` / output `(1,9,9,128,128)`: maximum forward error `7.63e-6`, gradient error `1.49e-8`, 0.034 s forward and 9,568,256 additional allocated bytes. Small CPU checks also matched. This alternative is recorded and locked for the pilot; it does not alter the frozen core or upstream STE source.

### MVSEC preparation (historical Stage 2 state)

At the end of Stage 2 no MVSEC file, calibration, or GT mask had been parsed. Stage 3 subsequently downloaded and parsed a fixed one-second `indoor_flying1` fragment and ran candidate inference; see [`STAGE3_RESEARCH_HANDOFF.md`](../STAGE3_RESEARCH_HANDOFF.md). The real support rule remains a protocol-freeze item; the Stage 3 per-window mask is an inspection candidate, not the official metric.

### Failed implementation attempts retained

- STE run v1 stopped at evaluation because the model returned one tensor while the pinned photometric helper expected a list of scales. The adapter now explicitly evaluates the final flow at endpoint resolution.
- STE run v2 completed dt1, then dt4 exhausted GPU memory using `F.unfold` (324 MB requested with 275 MB available at about 23.31 GiB allocated). The previous process exited normally; no other user's job was interrupted. The memory-efficient correlation operator was then verified and the fresh v3 run completed.

## Route C — pose and E-POSE

### Executed closure and limitation

- The benchmark-local runner uses the pinned SingleShotPose 416×416 Darknet/RegionLoss interface with three fixed-scale full-frame event channels (ON, OFF, total counts). Only raw event-derived tensors enter inference. Shared Genesis pose and actual camera K produce centroid/cuboid-keypoint training targets.
- Pilot v1 completed its optimizer/checkpoint interface but selected only one sample per rollout; validation keypoint EPE was 531,107 px (EventGenesis) and 3,106,779 px (V2E). The window selector was fixed to select evenly spaced fully visible windows from all train rollouts, and pilot v2 completed with 8 train / 4 validation windows per generator, 16 updates, same paired initialization, successful checkpoint restoration and 782 MiB peak allocation.
- **INVALID historical metric:** pilot v2 printed 158,581 px (EventGenesis) and 59,141 px (V2E), but the selected sample's event window ended at the rollout tail rather than its own pose-label time. These values do not diagnose a correctly aligned model and are retained only as history. Stage 3 records the corrected pilot and its independent decoder/BatchNorm diagnostics. No E-POSE score or generator comparison is inferred.
- The pilot object is a generic box, not one of the 13 E-POSE/YCB classes and has no class-matched CAD. The paper lists raw event, pose, K, camera transform, time and projected keypoint fields; these facts have not been confirmed against a downloaded raw sample. The stock model's RGB/LINEMOD loader and validation path were bypassed; no GT crop/mask/box, segmented events or RGB entered the benchmark runner.

### Directional blocker and access record

The Figshare API on g1 returned HTTP 403. A read-only publisher landing-page request returned HTTP 202 with an empty body; an earlier local API request also returned 403. The publisher record lists a 27.72 GB archive, but no archive was downloaded. This is an environment access/sample-contract blocker, not evidence the public dataset is globally unavailable. A single raw H5 plus the matching CAD is required to check timestamp unit/order/polarity, K convention, pose axes/transform direction/units, synchronization and mesh scale. The E-POSE paper candidate is ADD/ADD-S success at `0.1d` with supporting 2D reprojection, translation and rotation errors; this metric and aggregation are not frozen. Do not scale or replace the route until the research session reviews a sample-level contract and the extremely poor pilot behavior.

## Real-data isolation and evaluator readiness

No HQF, MVSEC, or E-POSE event/image/label file was passed to training, synthetic validation, checkpoint restoration, model selection, parameter tuning or simulator adjustment. No candidate real-test ranking exists. The committed manifest records source links, expected schemas, candidate evaluation functions and unresolved protocol choices, but it is a source audit rather than an executed data/evaluator check.

Before real scoring, research review should freeze:

1. **HQF:** complete 14-sequence inventory, license, frame/event timing and EVREAL preprocessing; candidate E2VID adapter; frame/sequence aggregation and metrics.
2. **MVSEC:** candidate sequence inventory, timestamp tie/association behavior, mask/support definition, units/resolution/orientation, primary and secondary metrics, and empty-frame handling for both dt values.
3. **E-POSE:** raw sample and matched CAD contract, object classes and synthetic model matching, class-aware event-only output/labels, plus ADD/ADD-S implementation and aggregation. If SingleShotPose cannot be trained/evaluated on this event-only full-frame contract, the model/task change must be explicitly approved.
4. **Shared protocol:** production sensor profile and V2E mapping, rollout-bank size/splits, initialization pairing, total training budget, seeds, synthetic-only validation/checkpoint selection and data licenses. Current values are an engineering pilot, not a calibrated sensor benchmark.

## Compute, data size and cost

- g1 environment recorded per run: Python `3.11.15`, PyTorch `2.5.1+cu121`, runtime CUDA `12.1`, Genesis `1.2.2`, h5py `3.14.0`, NumPy `2.4.6`, OpenCV `5.0.0.93`, pandas `2.2.3`.
- GPU 1 was checked idle before each model run and used sequentially. GPU 3 showed another user's CARLA job and was left untouched. No other process was stopped, reprioritized or modified.
- Shared bank build + six sequential V2E conversions occupied `03:36:13–04:10:11 UTC` (34 min wall, including an E2VID adapter-count assertion failure). The successful adapter retry took 27 s; E2VID run 65 s wall; STE v3 6 min 46 s wall (about 147 s summed training wall); pose v2 18 s wall. V2E individual conversions were roughly 50 s each. These are this pilot's observed times, not scaling estimates.
- The shared HDF5 is 26.5 MB; raw V2E outputs total 557,294 bytes; converted E2VID inputs total 25,219,610 bytes; converted STE inputs total 34,159,379 bytes. The eight successful checkpoints occupy 3.47 GB on g1; including the two v1 pose discovery checkpoints, the ten Stage 2 checkpoints indexed here occupy 4.28 GB. They remain outside Git. Per-file paths, sizes and hashes are in the external artifact manifest.
- Public download estimates: HQF 19.4 GB compressed / 94 GB uncompressed; MVSEC candidate bundle 33.2 GB; E-POSE full archive 27.72 GB. These are quoted source metadata estimates only. No full download, full-corpus throughput or quota measurement was made.
- No full training resource projection is claimed. For planning only, the measured upper pilot point is dt4 STE batch-one at 12.56 GB allocated and about 59 s per generator for two epochs on selected tiny data. A formal corpus-scale estimate requires target rollout count, epochs, storage format and training budget to be frozen.

## Source versions and evidence locations

- Frozen simulator core: `e79e656ac00f7847b98db9e12bf43b093f3afff1`; Stage 2 code did not alter it.
- EventGenesis/plugin worktree code commits used by run groups: shared bank `a654ae023f8bbcf13a383a0757bddc5211ff3836`; adapter retry `86f336d4c3d034196ed0594c13a7dde00fcac5d1`; E2VID trainer `f6325a3d2ac51ce6e44122179def0a6e72714395`; STE successful runner/operator `6a4c1458e7034d9c65ec288835d99d784a28f4d0`; pose v1 `54d5c7cb1bf787f0505a7df29d22eb6faea22a6b`; pose v2 selector `b36f2126f3fe6295556b46b7b8faa66501e5075b`.
- Pinned task sources: V2E `cac638444557a224ba6ee5c29260916f152c00bd`; E2VID training `78aa471a66593f55e349303563997297fbc816e8`; EVREAL `398f03551b7f5d150ae10aa89e4e731557fce240`; STE-FlowNet `eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6`; SingleShotPose `2c282619101f9f31666dd18b42393375bea1600f`; E-POSE annotation source `b96ae783fcb6de84e1e9c602e91bce941cab0234`.
- Exact run commands, package versions, source hashes, start/end times, exit codes, preflight and stdout are under [`evidence/S2/20261016-stage2-pilot/runs/`](../evidence/S2/20261016-stage2-pilot/runs/). Successful and failed attempts are both retained. Shared/config/conversion manifests are in [`evidence/S2/20261016-stage2-pilot/data_manifests/`](../evidence/S2/20261016-stage2-pilot/data_manifests/). External g1 artifact paths/sizes/hashes are in [`evidence/S2/20261016-stage2-pilot/external_artifacts.json`](../evidence/S2/20261016-stage2-pilot/external_artifacts.json). The SHA index is `evidence/S2/20261016-stage2-pilot/artifact_index.json`.
- Stage 1 independent review and its audit artifacts were preserved unchanged in [`reviews/stage1_88b0d14/`](../reviews/stage1_88b0d14/); `REVIEW.md` SHA-256 remains `d8f739cf006b1ac2ca7924760868c85feb3267401174baf591468a2c6e974e78`.

## Reproduction entry points

All scripts refuse to overwrite an existing run/data path. Start from the committed branch on g1, select new run IDs and a new data-root/config before a fresh trajectory generation, and check the exact GPU and disk paths first. The commands and arguments for this completed pilot are preserved verbatim in each run directory; use these as the reference:

1. `benchmark/sim2real/scripts/run_stage2_data_server.sh` — shared bank and six V2E conversions. The original first E2VID conversion assertion failure is preserved; that step was rerun through `benchmark/sim2real/scripts/run_stage2_adapters_server.sh` with the original data unchanged.
2. `benchmark/sim2real/scripts/run_stage2_adapters_server.sh` — successful E2VID and STE adapters.
3. `benchmark/sim2real/scripts/run_stage2_e2vid_server.sh 20261016-s2-e2vid-pilot-v1`.
4. `benchmark/sim2real/scripts/run_stage2_steflow_server.sh 20261016-s2-steflow-pilot-v3`.
5. `benchmark/sim2real/scripts/run_stage2_pose_server.sh 20261016-s2-pose-pilot-v2`.

The recorded data runner uses a fixed `DATA_ROOT` and deliberately fails if it exists. For a new bank, update the checked-in config and the run script's root as a new versioned protocol change; do not remove or overwrite the archived Stage 2 bank. Each exact Python invocation and its outcome is captured in the associated `*.command.txt` and `*.exit_code.txt`.

## Recommended next stage

The two core scientific routes have passed a useful synthetic-only engineering gate. The next work should first freeze HQF and MVSEC data/evaluation manifests and scale only those two synthetic pipelines within an approved training budget. Pose should pause at the sample/CAD contract gate until an original E-POSE sample can be inspected and the model's poor generic pilot is addressed. Research review should approve the candidate real-evaluation protocol before any real score or checkpoint comparison is run.
