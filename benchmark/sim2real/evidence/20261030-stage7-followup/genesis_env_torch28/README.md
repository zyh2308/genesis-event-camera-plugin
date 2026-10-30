# Isolated Genesis/PyTorch Environment Update

This folder records an isolated Genesis/PyTorch environment setup, CPU renderer probes, and ReplicaCAD `apt_1` camera-motion pilots. One fixed short camera trajectory has now passed flow-geometry checks and both EventGenesis and pinned V2E have processed the same radiance frames. This is still a scene/coverage pilot: it does not establish a useful training bank, paired flow training, or real-dataset scoring.

## Result

- The shared `Eventbased_WAM` conda environment was left unchanged at its prior PyTorch version.
- Only the cloned environment `/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28` was updated.
- g1 run03 completed at `2026-10-30T12:06:35Z` with exit code 0. CPU import checks reported PyTorch `2.8.0+cu126` / CUDA build `12.6`, torchvision `0.23.0+cu126`, torchaudio `2.8.0+cu126`, Genesis `1.2.2`, and its bundled pyrender path. Genesis import no longer emitted the earlier unsupported-PyTorch warning.
- No CUDA context, renderer, or real data was used by the import check.
- pip reported pre-existing missing optional requirements for `tensordict` (`orjson`) and `robomimic` (`egl_probe`, `tensorboardX`). Genesis and its bundled pyrender imported successfully; these packages were not added because they are not needed for the current scene-import probe.

## Artifact provenance

All large wheels remain outside Git under:

`/home/zhaoyuhan/Eventbased_WAM/external_sources/pytorch_wheels/stage7-20261030/`

| Wheel | SHA256 |
|---|---|
| torch 2.8.0+cu126, CPython 3.11 | `542cdbf042aaf5d6ddbed43cb8cd8c4df1e586bebb1338f5dcba14fa52830d3c` |
| torchvision 0.23.0+cu126, CPython 3.11 | `b697f5ca72399822d6454929cfb8835298527881e0ba58982f80ff0401331a92` |
| torchaudio 2.8.0+cu126, CPython 3.11 | `77859ec6c6a92ca3b666a3f02363a5631e9676c027fb4dce99a461a089c9962c` |
| triton 3.4.0, CPython 3.11 | `7b70f5e6a41e52e48cfc087436c8a28c17ff98db369447bcaff3b887a3ab4467` |

Torch/vision/audio came from the official [PyTorch CUDA 12.6 wheel index](https://download.pytorch.org/whl/cu126/); the Triton wheel and hash came from [PyPI Triton 3.4.0](https://pypi.org/project/triton/3.4.0/). Each file was hash-checked before installation. The CUDA 12.6 wheel set matches the official [PyTorch previous-version installation table](https://docs.pytorch.org/get-started/previous-versions/).

## Runs and transfer findings

- g1 `20261030-stage7-genesis-torch28-setup-run01`: direct PyTorch CDN install attempt, stopped by this session at exit 143 after observing roughly 58 KB/s for the 822 MB torch wheel.
- g1 `20261030-stage7-genesis-torch28-setup-run02`: local verified torch wheel was used, but PyTorch-index torchvision/torchaudio/Triton transfers measured roughly 26–30 KB/s. This attempt was stopped at exit 143 before pip reached installation; the cloned environment still reported PyTorch 2.5.1 afterward.
- Attempts to send the wheel from WSL to g1 over Windows SSH/SCP reached only about 0.35 MB/s. Their partial sizes and paths are recorded in run02 `transfer_attempts.log`; only those task-owned partial files were removed.
- g3 direct HTTPS downloads were substantially faster: the 821,911,765-byte torch wheel completed and matched its expected hash; the three remaining wheels were then downloaded from direct official links and checked against the PyTorch index/PyPI SHA256 values. Download logs and hashes are in the g3 run directories below.
- g1 `20261030-stage7-genesis-torch28-setup-run03`: installed the four checked wheels plus CUDA runtime dependencies in the cloned environment; installation and CPU imports exited 0. Full logs, package snapshots, timestamps, and exit codes are at the g1 run path.

## Resource boundary

A later sequential refresh at `2026-10-30T12:13:18Z` on g1 and `12:13:21Z` on g3 found other-user processes on all GPUs. The g1 static scene render then used only CPU, with no GPU reservation. Its fresh launch snapshot at `12:33:45Z` again showed all six g1 GPUs occupied; the g3 snapshot remains the earlier `12:13:21Z` observation and was not refreshed because this run requested no GPU. Any GPU-dependent flow training still needs a new g1/g3 resource check and reservation.

## Reproduction

From the repository root, the setup launchers used were:

```bash
bash benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/start_g3_download.sh
bash benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/start_g3_slow_wheels.sh
bash benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/start_run03.sh
```

The environment installation driver rechecks CPU/memory, confirms no process is using the cloned environment, validates wheel hashes, records GPU ownership without allocating a GPU, and performs CPU-only imports. The corresponding remote command output is retained at the run paths below.

## CPU-only importer API check

At `2026-10-30T12:16:14Z`, the cloned environment exposed Genesis `gs.morphs.Mesh` controls for GLB/Z-up parsing, coordinate convention, transforms, visualization, and collision. CUDA was hidden. This was package API introspection only; it did not load a ReplicaCAD file or initialize a renderer. The command, limitations, and remote log hashes are recorded in `genesis_api_inspection_20261030T121600Z.txt`.

## CPU renderer and ReplicaCAD static preview

- A 96×72 plane-only camera render passed on g1 CPU in run `20261030-stage7-genesis-cpu-render-probe-run02` at `12:23:44Z–12:24:51Z`, exit 0. Its raw files are under `remote_cpu_render_run02/`.
- The first static ReplicaCAD `apt_1` import/render passed in g1 run `20261030-stage7-replicad-apt1-cpu-preview-run04` at `12:33:45Z–12:35:16Z`, exit 0. It used PyTorch 2.8.0+cu126 / Genesis 1.2.2 with CUDA hidden and software rasterization requested. It added the stage, 120 rigid instances and six articulated furniture URDFs, then rendered 320×240 RGB, depth and segmentation. The preview montage is `replicad_apt1_cpu_render_run04/apt_1_rgb_depth_seg_preview.png`; raw PNG/NPY outputs, machine result, resource snapshot and logs are in that folder and its `remote_run_records/` subfolder.
- The stock cloned Genesis package does not accept `radiance=True`. The already isolated Stage-4 overlay at `/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2` does: a CPU probe returned finite scene-linear float32 RGB-like radiance while the ordinary API remained uint8. Re-running static `apt_1` through the overlay in `20261030-stage7-replicad-apt1-cpu-preview-run05` at `12:44:30Z–12:45:27Z` passed, with separate radiance, ordinary RGB, and depth/segmentation passes. The run output and manifest are under `replicad_apt1_cpu_radiance_run05/`; ordinary RGB matched run04 byte-for-byte.
- Genesis bundled `urdfpy` initially rejected ReplicaCAD's comma-separated URDF mesh scales and one malformed box-size delimiter. The successful importer copies only the six relevant URDF directories into the unique run output, normalizes numeric delimiters without changing values, validates all six with the bundled parser, and records original/patched hashes. The source archive and shared extracted dataset were left unchanged. The exact attempt ledger is `replicad_apt1_cpu_render_run04/static_scene_preview_manifest.json`; failed run02/run03 logs remain preserved.
- At that point these were static layout previews. The overlay returns scene-linear float32 radiance, but exposure, physical radiometric units and camera calibration remain unverified. The RGB/radiance montage applies a fixed display-only tone mapping; it is not fed to either event generator. Raw renderer segmentation IDs have not yet been reconciled to physical link IDs. Motion and event generation were added in the later run07/run09/run10/run11 records below; no training has run. Genesis also emitted non-watertight-mesh and inertial-property warnings; do not infer collision/physics correctness from the renders.

The recorded launcher's paths point to the completed run04 and refuse to overwrite it. To reproduce, copy the launcher and all companion scripts, use a new unique remote run directory/session/cache, and keep the source-asset hashes unchanged.

## Camera-motion pilot and paired event generation

- g1 CPU run07 rendered three 10 ms intervals for a registered 3 cm camera translation. The 30 ms flow agrees with the independent pinhole relation to p99 `3.82e-6 px`; EventGenesis produced 36,960 events.
- Pinned V2E run09 processed the exact same four scene-linear radiance frames and produced 26,753 events. Its source revision, frame hash, conversion, event bounds, timestamps, command, environment, and logs are retained under `v2e_camera_motion_run09/`.
- The predeclared shared event-support audit found that this view is unsuitable for training coverage: textured pixels with GT motion over 1 px occupy only `0.80–1.30%` of the image, despite shared event support covering `53.5–78.0%` of that small region. This is a geometry/coverage diagnostic, not a generator ranking.
- A second geometry-only camera placement at Habitat x=1.5 m completed on g1 CPU run10. Its pinhole flow check passed. Pinned V2E run11 processed the exact same radiance frames and emitted 16,826 events. In the registered audit, shared event support covers 22.0–45.7% of the texture∩GT-motion (>1 px) region, but that region is only 3.40–3.48% of the frame; the shared support overlap within it is 0.75–1.59% of the full raster.
- A faster camera-motion layer was registered before g1 CPU run12: same scene and x=1.5 m view, 0.06 m over 30 ms (2 m/s), with all sensor and generator settings fixed. The rationale is the run10 geometry-only displacement distribution, not either generator's event count. Run12 completed at `2026-10-30T13:28:38Z`, exit 0; its fixed geometry-only audit shows 16.82–17.21% texture∩GT-motion area and median flow around 1.16 px/10 ms. Pinned V2E run13 processed the exact same four radiance frames on CPU and completed at `2026-10-30T13:36:33Z`, exit 0 in 377.059 s. The HDF5 contains 36,155 events (SHA256 `fd791c46b3e01c38c77246f39cabb6f509b38d99e704c79eb849ded3a9831b10`). The fixed shared EventGenesis∪V2E support covers 30.60–44.76% of texture∩GT-motion per 10 ms and 10.12–16.00% of valid raster pixels. Full inputs, HDF5, command, and logs are under `v2e_camera_motion_fast_run13/`; the audit is `camera_motion_coverage_audit_fast_run12.json`. This remains a one-path coverage diagnostic, not a training result. The 75 MB temporary URDF compatibility copies remain off shared NFS.
- Three additional ReplicaCAD `Stage_v3_sc*` staging-00 layouts (sc0/sc1/sc2) pass static archive closure with PBR assets and no missing references. Their pilot train/validation candidate roles are registered in `../replicad_layout_candidate_registration_v1.json`. These are layout families from one ReplicaCAD collection, not independent real homes.
- g1 CPU run14 imported/rendered `v3_sc0_staging_00` and run15 did the same for `v3_sc1_staging_00`; both passed the independent pinhole-flow check at p99 `9.54e-7 px`. Run14's EventGenesis radiance/events were then consumed by pinned V2E run17. The paired sc0 support audit gives EventGenesis∪V2E support of `9.61–17.22%` over texture∩GT-motion and `4.70–8.65%` over valid pixels across the three 10 ms windows. This is a weak-support diagnostic and does not qualify sc0 alone for training.
- g1 CPU V2E run20 consumed sc1 run15's exact radiance frames and completed at `2026-10-30T16:38:35Z`, exit 0, in `379 s`. Its H5 SHA256 is `f7a764c5fa99d36a4c53b711b8daa6739b38ca519c6824117ea53eced572f8da`; timestamp-recounted event counts are `6,413 / 10,475 / 10,733` across the three left-open/right-closed 10 ms windows. The fixed paired coverage audit reports shared EventGenesis∪V2E support over texture∩GT-motion of `12.53% / 19.32% / 20.98%`, and over valid raster of `6.78% / 9.45% / 10.81%`. Evidence is under `replicad_layout_runs/v3_sc1_staging_00/v2e_run20/remote_run20/`; the audit is `camera_motion_coverage_audit_v3_sc1_run15_run20.json`.
- The registered sc2 validation candidate Genesis CPU run16 completed at `2026-10-30T16:34:15Z`, exit 0, in `115.5 s`. Its independent camera-only pinhole check compared 74,018 pixels and had p99 flow error `4.77e-7 px`; geometry-only texture∩GT-motion covers `22.48–22.73%` of the raster. Evidence is under `replicad_layout_runs/v3_sc2_staging_00/remote_run16/`; geometry audit is `camera_motion_geometry_texture_audit_v3_sc0_sc1_sc2_run14_run15_run16.json`.
- At `2026-10-30T16:40:26Z`, all six GPUs were at 0% with 1–2 MiB allocated and no compute-app rows; base PyTorch `2.5.1+cu121` initialized CUDA on RTX 3090 UUID `GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23`. Pinned V2E run21 consumed the exact sc2 radiance frames on that GPU, with an atomic project reservation. It completed `16:42:45Z`, exit 0, in `52.998 s`; the V2E process allocated about 594 MiB before finishing. Its H5 SHA256 is `e5d94d9fed3826769673b93e0cccb96ea91dcac0b6eedee05f87e844d68691aa`, with `12,904` events. The paired support audit covers only `4.92% / 9.12% / 9.96%` of texture∩GT-motion and `4.62% / 6.50% / 7.28%` of valid pixels across the three windows. This was the pre-registered validation candidate, not a sample selected by its count. Audit: `camera_motion_coverage_audit_v3_sc2_run16_run21.json`; visual montage: `replicad_layout_runs/v3_sc2_staging_00/paired_flow_event_montage_sc2_run16_run21.png`.
- No STE-FlowNet training has been launched. Genesis rendering in this pilot used CPU software rasterization; model fitting will use a confirmed GPU when the training input/diagnostic is ready. Radiometric exposure/physical units and raw renderer segmentation-to-physical-link mapping remain unverified; static mesh/inertia warnings also mean these camera-motion probes do not establish collision dynamics. Resource snapshots are not continuing availability guarantees.

Reproduction scripts for run07/run09, run10/run11, run12/run13, and each registered layout import/event conversion are stored alongside their run records. The fixed coverage definition and audits are in `camera_motion_coverage_criteria_v1.json`, `camera_motion_coverage_audit_run07.json`, `camera_motion_coverage_audit_viewx1p5_run10.json`, `camera_motion_coverage_audit_fast_run12.json`, and the `camera_motion_coverage_audit_v3_sc*_run*_run*.json` files; geometry-only comparisons including sc0/sc1/sc2 are in the `camera_motion_geometry_texture_audit_*.json` files. Run21's reservation was released at normal process exit; the live follow-up check found the six GPUs idle again. No STE-FlowNet training has run yet.
