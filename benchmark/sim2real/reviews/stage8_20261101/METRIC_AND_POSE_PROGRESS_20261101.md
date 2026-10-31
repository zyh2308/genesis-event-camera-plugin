# Stage 8 progress: metrics and pose training

**Status time:** 2026-11-01 02:33:19 JST (2026-10-31 17:33:19 UTC), g1 paired-input hash check.  
**Repository:** `exp/public-sim2real-v1`, source base `de55334b824c504a1a722b880edce4dec993fa16`.  
**Frozen simulator core:** `e79e656ac00f7847b98db9e12bf43b093f3afff1`, unchanged.

## Main message

The current flow evidence does not justify another long self-supervised run. On the fixed ReplicaCAD validation windows, both models react to event input, but their normal predictions do not beat zero flow in the jointly event-supported moving region. The next useful step is to test whether the **actual training photometric loss** prefers the correct synthetic flow over zero and reversed flow.

The pose data-access blocker has changed: g1 already contains an E-POSE block sample, its original raw-event HDF5, the MAT record, and a sample-matched wood-block CAD. The remaining gap is that the available paired Genesis training bank still renders a generic cube. It cannot establish an E-POSE-object result. A full short generic-cuboid synthetic training/validation run has now completed on CPU as an engineering check.

## Flow: what the scores say and how to improve them

The Stage 7 100-update self-supervised diagnostic used the same fixed synthetic validation samples and shared event-active masks for both generators:

| Generator / validation path | Active-region EPE | Zero-flow EPE | Predicted / GT mean motion | Direction cosine |
|---|---:|---:|---:|---:|
| EventGenesis, sc2-X | 4.113 px | 2.674 px | 5.733 / 2.674 px | 0.788 |
| EventGenesis, sc2-Y | 3.310 px | 1.666 px | 4.364 / 1.666 px | 0.794 |
| V2E, sc2-X | 3.034 px | 2.674 px | 4.078 / 2.674 px | 0.766 |
| V2E, sc2-Y | 2.873 px | 1.666 px | 3.646 / 1.666 px | 0.709 |

The arrows are usually in a broadly compatible direction, but the predicted movement is too large. On the same fixed mask, zero-event inputs often scored better than normal event inputs. This is evidence of event sensitivity, not evidence of useful flow recovery. It does not identify the generator as the cause.

Recommended order:

1. On the fixed textured ReplicaCAD windows, evaluate the real multi-scale photometric objective at the correct GT flow, zero flow, and reversed flow. Keep the same pre-fixed visible/active mask. If GT does not lower photometric error, inspect radiance, endpoint timing, flow direction, warp interpolation, and occlusion handling before training.
2. If the GT warp is clearly best but model predictions remain poor, use a bounded synthetic GT-supervised STE-FlowNet diagnostic on the same physical bank and independent rollout split. That separates an optimization/input problem from a weak self-supervised target. It is diagnostic only; do not silently replace the formal self-supervised objective.
3. Only after those checks, expand the registered synthetic scene and motion coverage. Keep all changes based on synthetic geometry and signal coverage, not MVSEC scores. Keep full-valid EPE as a diagnostic and use the already authorized fixed event-union-active region for new versions.

For reconstruction, the existing HQF metrics trade off: EventGenesis wins LPIPS-Alex on 13/14 sequences while V2E wins PSNR/MSE on 13/14; both have visible block/grid artifacts. Do not tune against those real-test results. Use fixed synthetic visual fixtures and synthetic validation to isolate input/output range, recurrence, and training-scene diversity before any new real run.

## Pose: blockers and current action

### Evidence already available

- g1 raw MAT: `/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1/extracted/block/block/r_01_gl/0001_data.mat`, 6,017 bytes, SHA256 `bd369f867e671a194b067ba03cf11664adb6b389b1762203aba3b2ed1b6edb0c`.
- g1 raw-event HDF5: `/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1/extracted/original_data/original_data/block/r_01_gl/data.h5`, 46,199,720 bytes, SHA256 `a589e204be51970f128a39315954bdb814157945be90d6c63f56d739566f863a`.
- Matched CAD PLY: `/home/zhaoyuhan/Eventbased_WAM/external/sim2real_assets/epose_wood_block_20261024/nontextured.ply`, 983,765 bytes, SHA256 `8c653731426cf73b5b5d4ffbfa3bdd4ebb14cb82d413da82559bce1b5ee3ab37`; registered extent is about `[0.103917, 0.103464, 0.206003]` m. These files were verified as present on g1 on 2026-11-01; no download probe was made.
- These three source hashes and sizes were recomputed on g1 at `2026-10-31T17:32:49Z`; the receipt is [`epose_source_sha256_20261031T173249Z.txt`](../../evidence/20261101-stage8-pose-short/epose_source_sha256_20261031T173249Z.txt).
- The generic synthetic shared-bank config and all six paired V2E event files were also rehashed on g1 after the run; see [`paired_data_sha256_20261031T173319Z.txt`](../../evidence/20261101-stage8-pose-short/paired_data_sha256_20261031T173319Z.txt). The shared trajectory hash was recorded at launch in `remote_run02/critical_sha256_at_start.txt`.
- A corrected projection-domain sidecar exists for the checked sample; its parent MAT remains unchanged. One-sample Brown–Conrady round-trip error was about `6.36e-14 px`. It is a geometry check, not a training result.

### Functional issues shown by prior short training

The Stage 5 31-epoch generic-cube training completed with 248 updates per generator and 128 confidence-supervised updates. Train-only BN recalibration and ordinary `model.eval()` were used. On four held-out generic-cube windows, the confidence-argmax cell was different from the GT centroid cell on all four windows for both generators. Mean ordinary keypoint EPE was `34.65 px` (EventGenesis) and `176.82 px` (V2E); forcing the GT cell only as an offline diagnostic gave `18.84 px` and `10.30 px`, respectively. GT-cell decoding is not an allowed inference shortcut. This localizes a major remaining issue to confidence-based cell selection / its training signal, although it does not yet prove the exact mechanism.

The current data mismatch is separate: those windows render a `0.16 m` generic cube, not the matched E-POSE CAD. Therefore the existing generic pilot cannot validate CAD-specific event appearance, object-frame keypoints, or the E-POSE inference contract.

### Current CPU-only engineering run

- Run: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-stage8-pose-cpu-short-v2`.
- Isolated execution snapshot: `/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage8-pose-cpu-short-20261101`.
- Code snapshot hash for `run_pose_stage2_pilot.py`: `e187c4200c59eb05c121cc384bdf92e8e282a1114f35811dc2d09af8307e253e`.
- Protocol: pinned SingleShotPose, seed 0, 31 epochs, 8 train windows from 4 physical rollouts, 4 validation windows from 2 held-out rollouts, batch 1, 248 optimizer updates per arm, including 128 confidence-supervised updates. Input remains full-frame event-derived only. The existing train-only BN calibration and ordinary `eval` validation are retained.
- g1 had all six GPUs actively occupied by other users; g3 SSH timed out twice, so this run was deliberately assigned one CPU thread with CUDA hidden. The measured CPU update probe was about 1.69 s. The run was detached in tmux as `sim2real_pose_stage8_cpu_20261101_run02`.
- Both paired arms completed in `11m12s`; launcher exit code is `0`, and tmux had exited normally. Each arm ran 248 optimizer updates with 128 confidence-supervised updates. Training wall times were `329.39 s` EventGenesis and `311.88 s` V2E. The initial parameter hashes are identical (`4d752c…359c9`).
- Train-only BN calibration used all 8 fixed training event windows for 22 BN layers. Both ordinary `model.eval()` validation and checkpoint restore checks passed; state hashes match and the script's restored-output equality assertion did not fail. The retained weights are 404,531,771 bytes (EventGenesis, SHA256 `c875eceb26d85311527efa9f1b545c27310b841dedc0a4174ec49aefe7bd6be2`) and 404,518,443 bytes (V2E, SHA256 `34a9355a9ce4de44fc8f0dc8d0087416d24840a067176169cbeafcf6c8eaeaa4`). They remain on g1.
- On four held-out generic-cube windows, mean ordinary eval keypoint EPE was `31.58 px` for EventGenesis and `23.18 px` for V2E. PnP returned success on all eight windows, which only shows the solver accepted the predicted points; it does not establish accurate pose. The confidence-argmax cell matched the GT centroid cell on `0/4` EventGenesis and `2/4` V2E windows. Offline GT-centroid-cell decoding averaged `14.10 px` and `17.70 px`, respectively; that decoder uses label information and is not a valid inference path. The mismatched confidence cell remains an active correctness concern.
- No real data entered training or checkpoint selection. These metrics describe a tiny generic-cuboid synthetic pilot, not E-POSE generalization or a generator ranking. At start and completion checks, g1's six GPUs remained occupied by other jobs; CUDA was hidden and the run used one CPU thread. g3 was not used after two SSH timeouts.
- First launch attempt `run-v1` failed before model initialization with exit 127 because the launcher pointed to the wrong conda path. It performed zero optimizer updates and created no checkpoint. The exact failure is retained in `evidence/20261101-stage8-pose-short/failed_attempt_v1.json`; the remote stderr and earlier event-window audit are archived under `evidence/20261101-stage8-pose-short/remote_run01_preflight_failed/`. `run-v2` uses the verified interpreter path and completed successfully.

## What remains before claiming an E-POSE pose pilot

1. Build a small Genesis bank that renders the exact checked CAD rather than reusing the generic cube. Share those physical poses, labels, split, and initialization across EventGenesis/V2E.
2. Use CAD-frame cuboid/keypoints derived from the matched PLY vertices and the sample's audited object frame. Keep E-POSE real annotations out of training and model selection.
3. On multiple held-out synthetic rollouts, check confidence-argmax keypoint output, PnP, symmetry-aware pose error, and checkpoint restoration in ordinary eval mode.
4. Freeze a symmetry-aware metric and full-frame event-only real inference contract before any E-POSE real score. No real score has been run in this work.

## Reproduction and evidence

- Existing window endpoint/independent event recount check passed before run-v2; log: `evidence/20261101-stage8-pose-short/pose_event_window_audit.json` after remote artifacts are synchronized.
- New CPU support is an explicit optional `--device cpu --cpu-threads 1`; default remains CUDA. The training config, model, optimizer, seed, split, and loss are unchanged.
- Launcher: [`start_pose_stage8_cpu_short_g1.sh`](../../evidence/20261101-stage8-pose-short/start_pose_stage8_cpu_short_g1.sh).
- Registration: [`run_registration_20261101_v2.json`](../../evidence/20261101-stage8-pose-short/run_registration_20261101_v2.json).
- GPU and g3 observations are point-in-time checks only. Refresh before assigning any new GPU work.
