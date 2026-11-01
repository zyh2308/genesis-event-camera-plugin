# Stage 8: textured dynamic scene paired short training

**Status snapshot:** 2026-11-01 12:50 JST (03:50 UTC). The 100-update paired synthetic diagnostic is running in the isolated g1 checkout. This is a progress record, not a completed training report or a Sim-to-Real result.

## What ran

- Genesis loaded two separately registered ReplicaCAD Interactive v1.5 apartment layouts: `v3_sc0_staging_00` for training and `v3_sc2_staging_00` for validation. The full layout, its renderer depth and segmentation, four radiance frames, camera poses, object poses, shared flow labels, and EventGenesis events were rendered/generated in Genesis. No recorded RGB video was reused as a physics bank.
- Both layouts use the same registered 30 ms camera translation (0.06 m along the selected Habitat x axis) and a geometry-visible `frl_apartment_tvstand` probe translated by 0.08 m. The selected physical link is ID 3; source segmentation was checked before event generation. The visible object region grew from 1,000 to 1,128 pixels over the four saved timestamps in the training layout.
- Pinned V2E consumed the exact same saved radiance frames for each layout. Both HDF5 outputs passed event shape, timestamp-order/range, and image-coordinate checks. Event counts were not used to select layouts or labels.
- One event-only STE-FlowNet paired GT-supervised diagnostic is registered at 100 updates per generator. Model, initial seed, input representation, optimizer, learning rate, sample order, shared flow/validity, and budget are paired. This is a short synthetic diagnostic, not the formal self-supervised objective and not a generator ranking. Real data and real scores were not used.

## Scene and signal evidence

The checked-in [training-layout contact sheet](evidence/20261101-stage8-replicad-dynamic-sc0-preview.png) shows Genesis start/end RGB, renderer depth, shared 30 ms flow labels, and event support from both generators. The layout contains textured floor/rug and multiple visible depth layers. The foreground furniture also occludes a large part of the central view, so this pilot does not establish broad visibility or scene coverage.

The pre-training audit uses a common event-active union of the normal EventGenesis and V2E inputs intersected with shared valid GT. It reports:

| Layout | Shared valid pixels | GT-motion pixels (>=0.5 px) | Textured-motion pixels | Common active pixels on motion | Active coverage of GT motion | Active coverage of textured motion |
|---|---:|---:|---:|---:|---:|---:|
| sc0 train | 73,626 | 73,626 | 26,298 | 8,882 | 12.06% | 30.46% |
| sc2 validation | 74,496 | 74,050 | 31,530 | 7,852 | 10.60% | 24.52% |

Event counts, shown for audit only, were 32,274 / 26,915 (EventGenesis / V2E) on sc0 and 31,019 / 13,871 on sc2. No count matching, event-count threshold, or generator-specific sample filtering was applied. The union-active mask retained 8,882 valid pixels on sc0 and 7,871 on sc2.

The renderer and object-flow result labels record only one 30 ms camera-plus-kinematic-object probe per layout. The object probe is not a calibrated natural-motion distribution; the ReplicaCAD import result itself calls this an appearance/motion pilot, not a complete validated scene conversion. Treat the coverage numbers as sample statistics, not simulator-quality claims.

## Training state and issue found

- The first attempt (`.../20261101-stage8-replicad-dynamic-v2`) reached model evaluation but failed before optimizer update 1. The pinned STE-FlowNet source has legacy no-argument `.cuda()` calls in its forward path, which prevented the configured CPU fallback from running. The failure and traceback remain at the g1 run root.
- Added a narrow process-local CPU placement shim. When CUDA is unavailable, a no-argument `.cuda()` call on an already-CPU tensor returns that tensor; the shim leaves model parameters, operations, network code, and all CUDA behavior unchanged. The pinned model source was not modified. The CPU tensor placement probe passed, and ordinary model evaluation and backward now run.
- The corrected run is `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-stage8-replicad-dynamic-v3-cpucompat`, on g1, exact code commit `da019d2f85650267ca5ba50460cb14ca5b10b2bb`, execution worktree `/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage8-da019d2-20261101`.
- At the snapshot above, EventGenesis had completed 14/100 optimizer updates; V2E had not begun because this runner executes the two arms sequentially. The latest recorded loss was 2.38 with finite aggregate gradient norm 4.94. This is training-rollout fit only. Synthetic validation will next be recorded at update 20; no checkpoint has been written yet.
- At run start the six g1 GPUs all had active workloads. The launcher completed its five one-minute GPU checks, then used the registered CPU fallback: 128 logical CPUs, load1 6.46, about 436 GiB available memory, and two PyTorch/OpenMP threads. No other job was stopped or altered. The measured pace is about 50 seconds per update, so the fixed 100-update-per-arm run is CPU-bound and will take a while.

## Reproduction and artifacts

The run used the committed launcher:

```bash
EXEC=/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage8-da019d2-20261101
SIM2REAL_EXEC_CHECKOUT="$EXEC" bash "$EXEC/benchmark/sim2real/scripts/retry_stage8_replicad_dynamic_gtfit_g1.sh"
```

The bank is `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-stage8-replicad-dynamic-v2/flow_bank/training_bank_manifest.json`; scene data and complete logs remain under the g1 Stage8 v2/v3 run directories. The run manifests bind the configured scene paths, labels, event files and source hashes. The detailed update logs are `training_output/{eventgenesis,v2e}/updates.jsonl`; final checkpoints, restore comparisons, and summary are written after each arm completes.

The Stage 8 code commits are on `exp/public-sim2real-v1` and pushed to GitHub. At this snapshot the g1 execution checkout is pinned to `da019d2`; later documentation commits do not alter that running checkout. Update this report and evidence index after both arms finish.
