# Stage 7 engineering progress — 2026-10-30

## Summary

This is a partial execution report for the SSH-verified Stage 7 taskbook. It is suitable for research-side progress review, but it does **not** establish the Stage 7 experimental acceptance criteria. No mature ReplicaCAD scene has yet been imported into Genesis, rendered, or used for paired flow training. No new HQF scoring was performed. Real scoring remains closed.

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

### Live compute observation

At `2026-10-30 09:43:08 UTC` (`18:43:08 JST`), all six g1 GPUs showed 2.3–2.6 GiB allocated and 70–86% utilization; the listed compute processes belonged to `/home/lzx/.../r2dreamer`. A later g1 check at `09:46:21 UTC` showed CPU load average 7.22/6.88/5.94 and 478 GiB available memory, but the GPUs remained occupied. g3 was checked at `09:45:44 UTC`: load average 582/602/609 and all seven GPUs had other-user processes and 2.0–10.4 GiB allocated, even though GPU utilization was 0–2%. Neither node had a safe GPU allocation. No other user's process was modified. These are timestamped observations, not continuing availability guarantees.

## Work still required before the main Stage 7 delivery

| Line | Current evidence | Not yet established | Next bounded action |
|---|---|---|---|
| Flow | Stage 5 model/optimizer state can perform one real CPU update; ReplicaCAD `apt_1` package references and PBR assets pass a reproducible static audit. | No Genesis scene import/preview, no trajectory/coverage audit, no new multi-window paired GT-supervised pilot, no CUDA resume probe. | Use g1's available CPU for bounded scene-loader/API investigation; produce the declared RGB/depth/flow/events preview only when a GPU can be safely allocated, then start the pre-registered paired pilot. |
| Reconstruction | The Stage 7 research review's HQF outputs and scores are independently verified; quality degradation remains a valid concern. | No fresh training-vs-evaluation loader array parity audit has been completed in this follow-up. | On a fixed synthetic fixture compare event bins, target image domain, crop/pad, recurrent reset, and candidate output through both paths. |
| Pose | Prior E-POSE source/CAD checks remain preserved; no new real-data access probe was made. | No corrected projection-domain adapter or synthetic paired run is delivered here. | Implement and round-trip-check one consistent distorted/ideal projection domain on the already verified object, then continue synthetic-only multi-window validation. |

No real-dataset score was used for selecting models, assets, or configuration. No frozen simulator core file was changed.

## Reproduction and evidence

Probe and asset-audit evidence and hashes are in [`evidence/20261030-stage7-followup/`](../../evidence/20261030-stage7-followup/). Run JSON records the exact input/config/data bindings and CPU-shim limitation. The archive static check is reproducible with `python benchmark/sim2real/scripts/audit_replicad_archive_contract.py /home/科研/Eventbased_WAM/external_sources/ReplicaCAD_dataset_v1.5.zip --scene configs/scenes/apt_1.scene_instance.json`. Large weights and the ReplicaCAD archive stay outside Git at the g1 paths above.

The successful probe command is recorded in the per-run `driver.log` and `start` script. The Python entry point is `benchmark/sim2real/scripts/resume_stage5_flow_fit_one_update.py`; the failure preceding the successful run is retained in the g1 run directory `20261030-stage7-flow-resume-cpu-200to201-g1-run01` (exit 1, manifest unit mismatch). The successful run directory is `20261030-stage7-flow-resume-cpu-200to201-g1-run02` (exit 0).

## Recommendation to research session

This is enough to report the newly verified checkpoint-resume fact and asset acquisition, with explicit limitations. It is not enough to claim a mature-scene pilot, useful motion learning, or readiness for formal training. Keep the next engineering gate at: actual Genesis scene import + pre-registered geometry/texture/motion coverage evidence, followed by a paired multi-window training pilot. The HQF figures remain report-only and are not to be reused as feedback for configuration changes.
