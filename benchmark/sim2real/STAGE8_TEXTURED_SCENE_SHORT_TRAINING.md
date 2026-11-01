# Stage 8: textured dynamic scene paired short training

**Status:** complete. The paired synthetic diagnostic finished at 2026-11-01 15:25 JST (06:25 UTC); final artifacts were checked at 16:32 JST (07:32 UTC). This is a small synthetic learning diagnostic, not a Sim-to-Real result.

## What ran

- Genesis loaded two separately registered ReplicaCAD Interactive v1.5 apartment layouts: `v3_sc0_staging_00` for training and `v3_sc2_staging_00` for validation. The full layout, its renderer depth and segmentation, four radiance frames, camera poses, object poses, shared flow labels, and EventGenesis events were rendered/generated in Genesis. No recorded RGB video was reused as a physics bank.
- Both layouts use the same registered 30 ms camera translation (0.06 m along the selected Habitat x axis) and a geometry-visible `frl_apartment_tvstand` probe translated by 0.08 m. The selected physical link is ID 3; source segmentation was checked before event generation. The visible object region grew from 1,000 to 1,128 pixels over the four saved timestamps in the training layout.
- Pinned V2E consumed the exact same saved radiance frames for each layout. Both HDF5 outputs passed event shape, timestamp-order/range, and image-coordinate checks. Event counts were not used to select layouts or labels.
- One event-only STE-FlowNet paired GT-supervised diagnostic is registered at 100 updates per generator. Model, initial seed, input representation, optimizer, learning rate, sample order, shared flow/validity, and budget are paired. This is a short synthetic diagnostic, not the formal self-supervised objective and not a generator ranking. Real data and real scores were not used.

## Scene and signal evidence

The checked-in [training-layout contact sheet](evidence/20261101-stage8-replicad-dynamic-sc0-preview.png) and [independent validation-layout contact sheet](evidence/20261101-stage8-replicad-dynamic-sc2-preview.png) show Genesis start/end RGB, renderer depth, shared 30 ms flow labels, and event support from both generators. The layouts contain textured floor/rug and multiple visible depth layers. Foreground furniture also occludes a large part of the central view, so this pilot does not establish broad visibility or scene coverage.

The pre-training audit uses a common event-active union of the normal EventGenesis and V2E inputs intersected with shared valid GT. It reports:

| Layout | Shared valid pixels | GT-motion pixels (>=0.5 px) | Textured-motion pixels | Common active pixels on motion | Active coverage of GT motion | Active coverage of textured motion |
|---|---:|---:|---:|---:|---:|---:|
| sc0 train | 73,626 | 73,626 | 26,298 | 8,882 | 12.06% | 30.46% |
| sc2 validation | 74,496 | 74,050 | 31,530 | 7,852 | 10.60% | 24.52% |

Event counts, shown for audit only, were 32,274 / 26,915 (EventGenesis / V2E) on sc0 and 31,019 / 13,871 on sc2. No count matching, event-count threshold, or generator-specific sample filtering was applied. The union-active mask retained 8,882 valid pixels on sc0 and 7,871 on sc2.

The renderer and object-flow result labels record only one 30 ms camera-plus-kinematic-object probe per layout. The object probe is not a calibrated natural-motion distribution; the ReplicaCAD import result itself calls this an appearance/motion pilot, not a complete validated scene conversion. Treat the coverage numbers as sample statistics, not simulator-quality claims.

## Training execution and result

- The first attempt (`.../20261101-stage8-replicad-dynamic-v2`) reached model evaluation but failed before optimizer update 1. The pinned STE-FlowNet source has legacy no-argument `.cuda()` calls in its forward path, which prevented the configured CPU fallback from running. The failure and traceback remain at the g1 run root.
- Added a narrow process-local CPU placement shim. When CUDA is unavailable, a no-argument `.cuda()` call on an already-CPU tensor returns that tensor; the shim leaves model parameters, operations, network code, and all CUDA behavior unchanged. The pinned model source was not modified. The CPU tensor placement probe passed, and ordinary model evaluation and backward now run.
- The corrected run is `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-stage8-replicad-dynamic-v3-cpucompat`, on g1, exact code commit `da019d2f85650267ca5ba50460cb14ca5b10b2bb`, execution worktree `/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage8-da019d2-20261101`.
- The corrected run completed all 100 optimizer updates for each arm (200 total), exit code 0, on CPU. It started at 2026-11-01 12:37:59 JST and finished at 15:25:13 JST; EventGenesis took 5,088 seconds and V2E 4,941 seconds. Six GPUs were occupied during the bounded pre-launch checks, so the launcher used its registered two-thread CPU fallback. No other job was stopped or altered.
- Both arms used the same initialization hash, paired samples and labels, event-only input, model, optimizer and 100-update budget. The final optimizer step is 100 for both. Each saved checkpoint restored to an equal model state, and its restored evaluation output matched the saved evaluation exactly (`max_abs=0`). The 580,021,553-byte checkpoints remain on g1; their hashes are in the evidence index.
- Both models fit the single training rollout: common-active EPE fell from 6.499 to 0.731 px/30 ms for EventGenesis and from 6.286 to 0.649 for V2E. On the one independent validation layout, final results were:

  | Arm | Validation EPE on full valid region (zero-flow) | EPE on frozen common event-active region (zero-flow) | Mean direction cosine on active moving pixels | Predicted active mean magnitude vs GT active mean |
  |---|---:|---:|---:|---:|
  | EventGenesis | 2.254 (3.498) | 2.638 (2.715) | 0.987 | 5.134 vs 2.721 px/30 ms |
  | V2E | 2.609 (3.498) | 2.600 (2.715) | 0.993 | 5.064 vs 2.721 px/30 ms |

- The predictions point in nearly the correct direction and beat zero flow by 2.9% / 4.3% on the event-active region. Their active-region magnitude is about 1.9 times the validation GT mean. The GT active mean is 5.729 px/30 ms in the training layout but 2.721 in validation, even though the scripted camera/object translations match. This is direct evidence of a large motion-scale shift between these two layouts. The stronger full-valid improvements (35.5% / 25.4%) include the much larger mostly static region and should not be read as equally strong motion learning.
- This establishes that STE-FlowNet can fit the fixed synthetic training sample and produces directionally aligned, modestly better than zero-flow predictions on one independent ReplicaCAD layout. It does not establish broad generalization, a generator ranking, or a real-data benefit. The validation trace varies across checkpoints; formal checkpoint selection was disabled. More independent trajectories and motion scales are needed before expanding training.
- The run summary records the exact source/config/data hashes and restoration check. The full JSONL update logs and compact validation curve are archived under [the run evidence directory](evidence/20261101-stage8-replicad-dynamic-v3-cpucompat/); full checkpoints remain at the registered g1 paths. At the post-run resource snapshot (2026-11-01 16:32 JST), GPUs 0–2 reported 0% utilization and 1–2 MiB memory, while GPUs 3–5 were occupied. This is a time-stamped observation, not a reservation.

## Reproduction and artifacts

The run used the committed launcher:

```bash
EXEC=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage8-da019d2-20261101
SIM2REAL_EXEC_CHECKOUT="$EXEC" bash "$EXEC/benchmark/sim2real/scripts/retry_stage8_replicad_dynamic_gtfit_g1.sh"
```

The bank is `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-stage8-replicad-dynamic-v2/flow_bank/training_bank_manifest.json`. The run manifests bind the scene paths, labels, event files and source hashes. The evidence directory contains the run summary, both complete JSONL logs, an update/evaluation CSV, exit and environment records, and a SHA256 index. The large final checkpoints remain at the g1 paths listed in the summary and index.

The Stage 8 code commits are on `exp/public-sim2real-v1`. The g1 execution checkout remains pinned to `da019d2`; this run did not change its code or configuration. The final report and evidence index now describe both completed arms. Real-data scoring remains closed.
