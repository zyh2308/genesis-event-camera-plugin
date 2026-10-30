# Stage 7 engineering progress — 2026-10-30

## Summary

This is a partial execution report for the SSH-verified Stage 7 taskbook. It is suitable for research-side progress review, but it does **not** establish the Stage 7 experimental acceptance criteria. No mature ReplicaCAD scene has yet been imported into Genesis, rendered, or used for paired flow training. No new HQF scoring was performed. Real scoring remains closed.

Follow-up work after the initial report added a fresh CPU audit of both reconstruction checkpoints and a corrected E-POSE projection-domain sidecar. The pose correction is now verified on one original MAT sample; it is not yet a trained pose pipeline.

## Verified actions

### Flow checkpoint continuation probe

The earlier verifier restored the models and ran evaluation only. A new bounded probe now loaded the two seed-0 fixed-sample checkpoints in a fresh process and performed one real supervised optimizer step for each arm, from Adam step 200 to 201.

- Both probes passed and wrote new `resume_update_201.pt` files on g1.
- The model state changed; all 70 gradient tensors were finite and nonzero in aggregate.
- Both Adam parameter groups contained 41 parameter tensors; every restored Adam counter advanced 200 → 201.
- Both probes bound to the same sample (`s5q_02_train/dt1_000`), config SHA256 `ce7c7e6c7670f2514e75db821eb7ef93939ff6794bcb5af8bc90e024b4597665`, and paired manifest SHA256 `02f8eab65559a9b8442fe04b46ae202fe23a15116ae62fe7b3385e7e6c3cf50d`.
- Paired physical flow labels were bytewise equal. Source checkpoint hashes were unchanged before and after the probe.
- The EG and V2E losses differ because their event tensors differ; this one-step diagnostic is not a performance comparison.
- This ran on CPU with a process-local `Tensor.cuda()` identity shim for the pinned STE-FlowNet implementation. It verifies checkpoint/optimizer continuation through one CPU update, **not CUDA kernel correctness, CPU/GPU parity, or strict full-trajectory bitwise equivalence**. The original 200-update runs were not repeated.
- A first bounded attempt failed before an optimizer update because the probe expected the wrong textual manifest flow-unit schema. That failure is retained; the schema check was corrected and the subsequent run passed.

The execution checkout on g1 was isolated at commit `d0e36796f04826a14a0931878405abed943a49a3`; the exact probe script SHA256 was `80c35a1d2963fab7d8d998b4d63d45022fe11f3d79bcbb51a60c58aab037697b` (tracked in this repository at commit `0f738c4`). The source model checkpoints remain unchanged at their original Stage 5 paths. New 201-update probe checkpoints are retained on g1 and indexed below.

### ReplicaCAD asset acquisition

The official ReplicaCAD v1.5 archive was downloaded, integrity-checked, and copied to the shared g1/g3 `/home` filesystem. Local and remote SHA256 values match, and the local ZIP CRC test passed.

- Official archive URL: `https://dl.fbaipublicfiles.com/habitat/ReplicaCAD/ReplicaCAD_dataset_v1.5.zip`
- Size: 138,078,834 bytes
- SHA256: `d571c2c1315601c7faed35452e54a15c56646393a4c330235f20fb6fc8d8edfd`
- WSL staging path: `/home/科研/Eventbased_WAM/external_sources/ReplicaCAD_dataset_v1.5.zip`
- g1/g3 shared path: `/home/zhaoyuhan/Eventbased_WAM/external/sim2real_assets/replicacad_interactive_v1_5/ReplicaCAD_dataset_v1.5.zip`
- Archive inventory: 627 members, 157,483,899 uncompressed bytes; 5 stage GLBs and 12 URDFs.
- The archive's bundled license says CC BY-NC 4.0. The current dataset webpage describes a different license; this report conservatively follows the archive's stricter license until resolved.

Update at `2026-10-30 09:46:53 UTC`: the archive was extracted to `/home/zhaoyuhan/Eventbased_WAM/external/sim2real_assets/replicacad_interactive_v1_5/extracted_v1_5` (606 files; 251 MiB on disk). A reproducible static archive audit of `apt_1` passed: 120 object instances, 86 unique object templates, 87 render GLBs including the stage, all referenced configs/render/collision assets present, 87/87 GLBs contain PBR materials and 83/87 embed images. This establishes package completeness and that the assets contain PBR/material/image data, **not** Genesis import or rendered appearance. Scene conversion compatibility, renderer signal provenance, native float/HDR availability, camera path, GT flow, event generation, and motion/texture coverage remain unverified.

### Reconstruction loader parity on a shared synthetic input

The training HDF5 loader and pinned EVREAL evaluation loader were compared on the same `s0_00` train rollout for both generators. This checks loader tensors and labels, not checkpoint predictions.

- Run: g1 `20261030-stage7-recon-loader-parity-run03`, 2026-10-30 09:58:54–09:59:10 UTC, exit 0; Python 3.11.15, PyTorch 2.5.1+cu121, NumPy 2.4.6; two CPU threads, CUDA disabled.
- For each arm, all 200 10 ms windows were compared at full 260×346 resolution. Training index `i` matched EVREAL `between_frames` index `i+1`; EVREAL index 0 is the unobserved sequence boundary. The audit applies the same `dataset.length = dataset.num_frames` extension as the registered HQF inference adapter.
- Event endpoint offsets matched independently computed `numpy.searchsorted(..., side="right")`. Event voxel tensors matched exactly for 200/200 windows per arm: zero mismatches, zero nonfinite windows, maximum absolute difference 0.0. Target images and timestamps were exact, and both arms shared the same concatenated target hash.
- Event count range: EventGenesis 40–1,091 (median 115.5); V2E 8–93 (median 40). No interval had fewer than three events or a zero time span.
- Source HDF5 SHA256: EventGenesis `471960cf7c33e204dcaa3e3328bfa13fcaf8995092406937f6ebc50ada78749e`; V2E `e9f0a4f34c48828a15efd1850b6a74617437e730d4d1e702771a7ba85aa38654`.

This rules out a basic event-bin, event-window, polarity, target-intensity, or target-timestamp mismatch for this fixed synthetic rollout. It does not test model predictions, crop/augmentation effects, padding/unpadding, long recurrent state, output mapping, or explain the HQF artifacts. Static source review also found two unexercised edge paths: the training loader returns zero for windows with fewer than three events while EVREAL bins any nonempty window; the training voxel helper has a zero-duration fallback while EVREAL divides by the event-time span. This rollout contains no such windows, so there is no evidence those edge cases affected this run.

### Candidate reconstruction forward, recurrent reset, and padding audit

After the loader parity check, both registered Stage-5 synthetic-best checkpoints were loaded in ordinary `eval()` mode on g1 CPU and run on two consecutive event-only windows from the same fixed synthetic `s5q_02_train` fixture. No reference pixels were passed to the model and no image metric was calculated.

- Run: g1 `20261030-stage7-recon-state-pad-cpu-run04`, 2026-10-30 10:26:15–10:26:34 UTC, exit 0; Python 3.11.15 / PyTorch 2.5.1+cu121 / NumPy 2.4.6; two CPU threads, GPUs not used. The live launch snapshot recorded load 6.47 on 128 CPUs and about 467 GiB available RAM; all six GPUs remained occupied by other users' jobs.
- Checkpoints matched the frozen HQF v3 bindings: EventGenesis `506e81bd558f1bcca2b67d34d4c1004836fc7cce65393084eb52a378dfd554bc`; V2E `598b1e7837b4b02b50a7f390889c7737ed797840043a3f23121f54dc99d050d8`. Model arguments came from the registered corrected-v2 config, SHA256 `6fc6d28673689868fa18b76be4c1f4af1d84da0cad287cbf596801cb7ca6610d`, because these historical checkpoints do not embed `model_args`.
- Both checkpoints produced finite `[1,1,260,346]` outputs. After the first and second windows, six recurrent state tensors were present; `reset_states()` cleared them all. Replaying the first window after reset matched the original output exactly: max absolute difference 0.0 for both padded and native outputs, for both generators.
- An actual coordinate-coded five-channel raster passed the pinned model `CropParameters.pad()` and `.crop()` unchanged at both synthetic-bank resolution 260×346 (padded to 264×352) and HQF resolution 180×240 (padded to 184×240): exact max absolute difference 0.0.
- Run02 is preserved as a failed attempt: it stopped before model forward because checkpoint `model_args` must be supplied from the registered config. An earlier launcher-only failure also stopped before Python; both failures and the corrected run are retained.

This verifies basic candidate loading, finite event-only forward, recurrent reset/replay, and pixel-grid crop geometry. It does **not** establish that predictions are accurate, identify the HQF grid/block artifacts, test the 200-frame recurrent trajectory, or prove CUDA equivalence. Together with the loader parity result above, the checks make a basic one-window event-bin/time mismatch or a simple native-size crop offset less likely as the cause of the HQF appearance, while training distribution, long-sequence dynamics, and learned reconstruction quality remain open.

### Full synthetic-rollout recurrent-state trace

To check whether recurrence itself becomes numerically unstable over the registered sequence, both same Stage-5 checkpoints processed all 200 EVREAL event windows from the fixed `s5q_02_train` synthetic rollout in ordinary `eval()` mode on g1 CPU. The target image returned by the loader was discarded; no reconstruction metric or real image was used.

- Run: g1 `20261030-stage7-recon-long-state-cpu-run03`, 2026-10-30 10:43:48–10:46:57 UTC, overall and per-arm exit code 0; Python 3.11.15 / PyTorch 2.5.1+cu121 / NumPy 2.4.6; two CPU threads. Each arm took about 95.6 s (EventGenesis) and 74.1 s (V2E). At launch the CPU load was 7.74/128 and available memory about 461 GiB; all six GPUs remained occupied by other users.
- All 200 outputs per arm had the required native shape and finite values; six recurrent state tensors were present after the final window and finite. After resetting, replay of the first event window reproduced the original first output exactly (maximum absolute difference 0.0) for both arms.
- On this fixture, the mean output level ranged `0.2984–0.3028` for EventGenesis and `0.3535–0.3653` for V2E. These are numerical traces only, not image-quality or target-agreement scores. The fixture's event voxel nonzero-count medians were 95 (EventGenesis) and 154 (V2E); these counts are recorded descriptively and were not used to select or filter samples.
- Failed runs are preserved: run01 omitted the registered EVREAL final-endpoint length extension and indexed past the last sample; run02 used the corrected endpoint mapping but its audit-only state collector failed to traverse the model's nested `model.states` dictionary. Both were audit implementation errors; run03 reused the already established endpoint mapping and nested state traversal. No model weights or training records changed.

This reduces the likelihood that a basic long-sequence recurrent-state explosion or failure to reset explains the HQF artifacts on this synthetic fixture. It still does not establish accurate reconstruction, test a complete HQF sequence's state behavior, or identify whether the artifacts come from training coverage or another inference/training difference.

### E-POSE keypoint projection domain correction

The E-POSE raw-event/label coordinate mismatch was followed through on the already acquired `block/r_01_gl/0001_data.mat` sample without changing the source MAT or running a real pose score.

- Successful run: g1 `20261030-stage7-epose-projection-domain-run02`, 2026-10-30 10:31:25–10:31:27 UTC, exit 0; Python 3.11.15, CPU, one thread. The source MAT is 6,017 bytes, SHA256 `bd369f867e671a194b067ba03cf11664adb6b389b1762203aba3b2ed1b6edb0c`; official `CreateSaveDataset.m` is pinned to commit `b96ae783fcb6de84e1e9c602e91bce941cab0234`, SHA256 `62c5668be502b42aee8e612e491a2f9e4b1acb19f984e03eb190f9556f09a438`.
- Source inspection confirms the MAT `events` come from the raw event frames, while undistorted event images are stored separately. The selected adapter contract is therefore full-frame raw event pixels with distortion-aware keypoint labels `cv2.projectPoints(camera_xyz, rvec=0, tvec=0, K, D)`. No GT crop, mask, or box is involved.
- An independent Brown–Conrady forward implementation agrees with OpenCV `projectPoints` to at most `5.86e-14 px`. Analytic-Jacobian Newton inversion recovers ideal pinhole coordinates after distortion with maximum round-trip error `6.36e-14 px` in five iterations. The default OpenCV point inverse has a separately recorded maximum `0.00642 px`; the first attempt failed the predeclared `0.001 px` gate and is preserved.
- On the eight audited keypoints, old MAT UV labels differ from the corrected ideal pinhole projection by mean `16.4501 px` / max `32.7438 px`. The corrected raw-distorted labels are emitted only in a sidecar (SHA256 `44103ae29fc8b12ccdff0f4175d96db45110c3833376199ac6a0a305b44f1de7`); original MAT labels remain untouched.

This verifies projection arithmetic and the selected raw-event coordinate domain on one sample. It does **not** validate all object timestamps, CAD symmetry, all camera pose fields, multi-rollout training, or PnP accuracy. The pose line has a corrected label candidate, not a closed train/validation/real-test pipeline.

### Isolated Genesis/PyTorch environment update and fresh resource check

After the user asked why work had not continued, engineering resumed with an isolated environment update and a fresh check of both nodes. The shared `Eventbased_WAM` environment was not modified. The cloned `/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28` now imports PyTorch `2.8.0+cu126`, torchvision `0.23.0+cu126`, torchaudio `2.8.0+cu126`, and Genesis `1.2.2` on CPU; Genesis no longer emits its unsupported-PyTorch warning. The checked wheels and their SHA256 values are recorded in `evidence/20261030-stage7-followup/genesis_env_torch28/environment_setup_index.json`. Installation and CPU imports completed with exit code 0 in g1 run03, `20261030-stage7-genesis-torch28-setup-run03`, from `2026-10-30T11:51:24Z` to `12:06:35Z`.

This is environment readiness only: the check created no CUDA context, did not initialize a renderer, and did not run training, scene import, or real-data inference. Optional `orjson`, `egl_probe`, and `tensorboardX` requirements were reported missing by other packages but were not needed for the successful Genesis/pyrender import. Earlier slow-transfer attempts were stopped and recorded; the final wheels were fetched on g3 over HTTPS, hash-checked, then installed on g1 into the clone. The setup README lists each run path and exact reproduction launchers.

A later sequential refresh at g1 `2026-10-30T12:13:18Z` and g3 `12:13:21Z` again found other-user processes on all six g1 GPUs and all seven g3 RTX 4090s. g1 load was 6.69/7.50/7.71 on 128 cores with 435 GiB available RAM; g3 load was 573.98/511.01/510.23 with 296 GiB available. The exact GPU UUID, utilization, memory, process, and reservation checks are saved in `evidence/20261030-stage7-followup/genesis_env_torch28/resource_refresh_20261030T121318Z.txt`. No renderer or training job was started. The current blocker is verified GPU availability, not an unfinished environment setup; refresh again immediately before any future GPU allocation.

At `2026-10-30T12:16:14Z`, a bounded CPU-only inspection of the cloned Genesis 1.2.2 package also passed (g1 run `20261030-stage7-genesis-api-inspection-run01`, exit 0). It confirms the installed `gs.morphs.Mesh` API exposes GLB/Z-up, file coordinate, transform, visualization, and collision controls. CUDA was hidden and no scene or asset was constructed. This narrows the importer implementation surface but is not evidence that ReplicaCAD imports or renders; the exact command and remote log hashes are in `evidence/20261030-stage7-followup/genesis_env_torch28/genesis_api_inspection_20261030T121600Z.txt`.

### Earlier live compute observation (historical snapshot)

At `2026-10-30 10:48:00 UTC`, g1 had CPU load averages `7.06/7.87/7.34` on 128 CPUs and about 460 GiB available RAM, but all six GPUs were occupied by other-user processes with 69–89% utilization. At `10:48:08 UTC`, g3 had load averages `560/584/596`, about 302 GiB available RAM, and other-user processes on all seven RTX 4090s (roughly 2.5–10.4 GiB allocated per card); low utilization on several cards did not make them free. No GPU was claimed. The CPU audit completed before this refresh and released its two threads. No other user's work was changed. These are timestamped observations, not continuing availability guarantees.

## Work still required before the main Stage 7 delivery

| Line | Current evidence | Not yet established | Next bounded action |
|---|---|---|---|
| Flow | Stage 5 model/optimizer state can perform one real CPU update; ReplicaCAD `apt_1` package references and PBR assets pass a reproducible static audit. The isolated Genesis/PyTorch clone imports successfully on CPU. | No Genesis scene import/preview, no trajectory/coverage audit, no new multi-window paired GT-supervised pilot, no CUDA resume probe. The latest check found no GPU available on either g1 or g3. | Continue bounded CPU-side importer contract work; produce the declared RGB/depth/flow/events preview when a GPU can be safely allocated, then start the pre-registered paired pilot. |
| Reconstruction | One fixed synthetic rollout passes exact train/eval loader parity for both generators (200 windows/arm). Both candidates pass two-window forward/reset/padding checks and a new full 200-window CPU trace; all outputs and final states are finite, and post-sequence reset/replay is exact. | Candidate output quality and the HQF grid/block artifacts remain unexplained; no rich-scene training has started. The long-state trace is one synthetic rollout, not a quality metric. | Diagnose candidate outputs against registered synthetic references without using HQF feedback; begin shared rich-scene paired training when a GPU can be safely allocated. |
| Pose | Existing source/CAD checks are preserved. A corrected raw-distorted projection sidecar now passes an independent forward/inverse round trip on the already acquired sample; official MAT remains unchanged. No new E-POSE access probe was made. | No paired synthetic training, independent rollout eval, or PnP/checkpoint recovery validation with corrected labels. | Use the corrected sidecar in a synthetic-only adapter pilot and validate ordinary eval/PnP on independent rollouts; do not use this one-sample geometry check as model evidence. |

No real-dataset score was used for selecting models, assets, or configuration. No frozen simulator core file was changed.

## Reproduction and evidence

Probe, loader-parity, checkpoint-state/padding, long-state, pose projection, and asset-audit evidence and hashes are in [`evidence/20261030-stage7-followup/`](../../evidence/20261030-stage7-followup/). Run JSON records exact input/config/data bindings and limitations. Loader parity is reproducible with `benchmark/sim2real/scripts/audit_e2vid_train_eval_loader_parity.py`; its driver and raw logs are under `evidence/20261030-stage7-followup/reconstruction_loader_parity/`. The checkpoint forward/state/padding audit is `benchmark/sim2real/scripts/audit_e2vid_checkpoint_state_padding.py`; its driver and failed attempts are under `evidence/20261030-stage7-followup/reconstruction_state_padding/`. The 200-window trace is `benchmark/sim2real/scripts/audit_e2vid_long_sequence_state.py`; its launch/driver scripts, preserved failed attempts, per-window JSON, and logs are under `evidence/20261030-stage7-followup/reconstruction_long_state/`. The E-POSE sidecar audit is `benchmark/sim2real/scripts/audit_epose_projection_domains.py`; its original sample/source bindings, failed inverse attempt, and corrected sidecar are under `evidence/20261030-stage7-followup/pose_projection_domain/`. The archive static check is reproducible with `python benchmark/sim2real/scripts/audit_replicad_archive_contract.py /home/科研/Eventbased_WAM/external_sources/ReplicaCAD_dataset_v1.5.zip --scene configs/scenes/apt_1.scene_instance.json`. Large weights and the ReplicaCAD archive stay outside Git at the g1 paths above.

The successful probe command is recorded in the per-run `driver.log` and `start` script. The Python entry point is `benchmark/sim2real/scripts/resume_stage5_flow_fit_one_update.py`; the failure preceding the successful run is retained in the g1 run directory `20261030-stage7-flow-resume-cpu-200to201-g1-run01` (exit 1, manifest unit mismatch). The successful run directory is `20261030-stage7-flow-resume-cpu-200to201-g1-run02` (exit 0).

## Recommendation to research session

The newly completed work is recorded for review, and engineering continues. It is not enough to claim a mature-scene pilot, useful motion learning, or readiness for formal training. Keep the next engineering gate at: actual Genesis scene import + pre-registered geometry/texture/motion coverage evidence, followed by a paired multi-window training pilot. In parallel, use the corrected pose projection sidecar in synthetic-only adapter work and examine reconstruction outputs against synthetic references. The HQF figures remain report-only and are not to be reused as feedback for configuration changes.
