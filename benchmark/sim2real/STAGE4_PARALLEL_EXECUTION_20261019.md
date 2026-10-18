# Stage 4 parallel execution report — 2026-10-19

Report time: **2026-10-18 17:39:12 UTC / 2026-10-19 02:39:12 JST**; latest direct g1 training/resource snapshot: **17:37:55 UTC**. Server observations and output timestamps are UTC. This report covers work performed from the shared g1 Stage 4 task. The frozen simulator core `e79e656ac00f7847b98db9e12bf43b093f3afff1` was not modified. All training, validation and checkpoint selection remained synthetic-only; real-data scoring remains disabled.

## Executive status

| Line | Completed in this stage | Live state / blocker | Next delivery |
|---|---|---|---|
| E2VID+ reconstruction | Built exact 10 ms supervision from the same physical states; paired data, loader, joint transform, temporal-consistency gradient and actual process-resume preflights passed. Both seed-0 arms completed 20 updates and continued from the saved boundary in the registered full run. | Full 200-epoch / 12,000-update runs continue detached on g1. Direct live status and append-only optimizer logs at 2026-10-18 17:37:55 UTC show both arms at epoch 10, update 600/12,000, with saved update-600 checkpoints. | Keep both runs alive and report periodic, timestamped progress. Complete validation/checkpoint archive after their budgets finish. |
| STE-FlowNet dt1 | Both corrected-v2 seed-0 runs completed 45 epochs / 8,910 updates. Final synthetic-best paired diagnostic ran on all 64 registered windows and all four scene strata. The 90-row full curve archive, checkpoint hashes and individual exit receipts are checked in. MVSEC evaluator passed hand-calculated contract checks; real-score gate rejects unauthorized runs. | No confirmed timing/label/unit/output-crop defect in the final diagnostic. Learning is weak overall; event-active union improvement is modest and covers only 0.487% of full valid pixels. MVSEC candidate protocol still awaits research freeze and exact fixed test registration. | Protocol/manifest review. Keep real scores closed pending review. |
| SingleShotPose / E-POSE | Rechecked the official publisher and alternate public search routes. The paper confirms the raw H5 plus per-frame MAT contract and documents an event-only three-channel input used with YOLO-6D. Fixed the confidence-warmup and BN-inference-state mechanism on a small synthetic fit using ordinary `model.eval()`, standard confidence argmax, train-only BN recalibration, and save/restore. | The data are publicly described and the author's CAD repository is available, but the current WSL route still returns HTTP 202/403 and no original event/pose sample has been retrieved. The available category-named `wood_block` mesh cannot yet be tied to an exact sample. No formal E-POSE training or real inference. | Continue verifiable access attempts. After sample/CAD contract audit and held-out synthetic normal-eval validation, gate paired seed-0 training. |

## 1. Reconstruction: actual training launch and integrity evidence

### Data and labels

- Shared physical source bank: g1 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v3_geometry_corrected_20261017_32x2s/shared_trajectory.h5`, SHA-256 `cca882ba912fda3b3243d49bc7e29e28186046e3b6591c424f385b0506047884`.
- New E2VID adapter: g1 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/e2vid_corrected_10ms_20261019_v1/`, manifest SHA-256 `06ecadb1b4e90bc434ce30148c5d50704385cdab15a24e88ff9366f7c58edcb3`. It contains 32 rollout pairs per generator (24 train / 8 validation), with 10 ms event windows and exactly shared endpoint images, trajectory IDs, labels, split and validity for the two arms. The manifest and all 64 generated HDF5 files were hash-checked on g1.
- Reconstruction temporal-consistency displacement was recomputed by direct adjacent 100 Hz endpoint projection from the same depth/pose/K/visibility state. The 30 ms flow labels used by the completed flow study were not divided by three or replaced; those flow-run inputs remain unchanged.
- Each arm exposes 120 disjoint 40-frame training clips (60 effective-batch-2 updates/epoch) and 8 full 200-interval validation rollouts. Validation consumes indices 000–199, not the terminal duplicate endpoint. The upstream loader produced zero sequences at this exact-length boundary; the benchmark adapter now allows exactly one full sequence only for the registered 200-sample case and asserts its first/last sample IDs and rollout identity.

### Recipe and preflight

The run uses `configs/stage4_corrected_v2_flowdt1_seed0.json` reconstruction settings unchanged: pinned E2VIDRecurrent, five signed event bins, 40-frame clips, 112×112 crops, effective batch 2, Adam `1e-4`/AMSGrad, 200 epochs and 12,000 updates per arm. Selection is minimum equal-rollout synthetic-validation LPIPS-VGG; temporal consistency is recorded separately. The two arms share seed and initial state hash `48426d94f4159592155eda873d4e71e2f06439c10eade990de5979231ed07a0f`.

Preflight evidence confirmed:

- joint crop/flip of events, reconstruction targets, physical displacement and validity with four fixed transform seeds;
- temporal-consistency loss executed on valid neighbors; at update 1 it had nonzero gradient L2 (`0.0239571821` EventGenesis; `0.0239218361` V2E), with 76 valid TC calls recorded over the first update's two microbatches;
- finite losses/gradients and six recurrent-state tensors carried through the 40-frame sequence; training clip reset and validation-rollout reset are registered;
- 20 real optimizer updates per arm, then saved checkpoint restore, identical next sample IDs/content digests, matching next-batch loss probe and actual optimizer step 20→21;
- run-time real scoring gate closed.

The boundary-resume oracle confirms a checkpoint boundary and the next optimizer update; it is not a bitwise equivalence proof for an entire future trajectory. Update-20 checkpoints: EventGenesis SHA-256 `cb4f4000b435195cff1530bfc0600162ae5cbb689eb4ec39f5e57b156f6de9a0`, V2E `776ee6b18d984fe566f36360e9d83c7460b6050a076608fe12c4ebdd9aeb484b` (each 171,427,458 bytes). Resume oracle records same next sample indices `[55,4]`; restored probe losses were `[0.1372501254,0.2234207988]` and `[0.1345708221,0.2246799767]`, respectively.

The first adapter launch attempt is retained separately: the upstream exact-length validation loader returned an empty dataset and exited before optimizer updates. The adapter-only correction was preflighted before the successful v2 run; no frozen core or model recipe changed. A resume RNG restoration issue was also fixed before the full job was started: RNG ByteTensors are explicitly restored on CPU after `map_location=device`.

### Live run

- Run root: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2/{eventgenesis,v2e}/`.
- g1 environment: Python 3.11.15, PyTorch 2.5.1+cu121, CUDA build 12.1, Tesla V100S-PCIE-32GB. EventGenesis runs on physical GPU 2; V2E on physical GPU 4. Each held about 3.2 GiB at the latest resource observation.
- Detached tmux sessions: `s4e2vid_eg_resume_20261019` and `s4e2vid_v2e_resume_20261019`. Commands resume each arm's `last.pt` and complete the remaining registered 12,000-update budget. Real scoring is false in checkpoints/live status.
- At 2026-10-18 17:37:55 UTC, direct `live_status.json` and append-only metrics both showed EventGenesis and V2E at update 600/12,000 (epoch 10, batch 60); each status reported 76 valid temporal-consistency calls and 30 finite gradient tensors on the latest update. Both saved update-600 checkpoints were present and hash-checked. Detached tmux panes/Python processes remained alive (pane PIDs 246943/246948; Python PIDs 246981/246982); GPU 2/4 each used 3,252 MiB. GPU 4 was active at the resource snapshot, while GPU 2 was between work/validation; no other user's process was changed. The next periodic save is update 900. Full logs remain on g1 and current size/hash snapshots are indexed. See [`TRAINING_LIVE_STATUS.md`](TRAINING_LIVE_STATUS.md) for the concise status.

## 2. Optical flow: final synthetic diagnostic and public test preparation

### Completed 45-epoch run

Both corrected-v2 dt1 seed-0 runs are complete at g1 under `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/`: `full_training_retry1/{eventgenesis,v2e}`. Each has 8,910/8,910 updates and final state `complete`; the resumed full-training processes and successful `resume_retry1` supervisor each have exit code `0`. The original campaign wrapper exited `2`, and its first post-data continuation exited `1` after the strict resume-equivalence gate failed; those historical failures are retained and are not the exit status of the successful retry training processes. The retry's boundary resume oracle and preflight passed before the full run. Selected checkpoints remain the synthetic-validation best, independently chosen at EventGenesis epoch 33 and V2E epoch 20.

| Arm | Synthetic best epoch | Best validation EPE, px/30 ms | Synthetic final epoch EPE, px/30 ms | best checkpoint SHA-256 | final checkpoint SHA-256 |
|---|---:|---:|---:|---|---|
| EventGenesis | 33 | 0.2832538118 | 0.2837044896 | `a4e6c7e1d10a4d76562a046c364a5e17de6452d5b648fc8ccbaa3bd43c824a8e` | `d9efa865bdfd33e6c5f7ff0f75a2b1654f57982839efb0457495b99b8403d5ac` |
| V2E | 20 | 0.2828732346 | 0.2842815535 | `11041863685b417306f534b8e12a1177b1337fc2818d255ec7b76fa6e20cd474` | `8fd0d7379f46827a6dc39eaa1377b8dea953e85210abae4055dea41248553d79` |

Both best checkpoints are 580,038,858 bytes; both final `last.pt` files are 580,035,686 bytes. Initial model state hashes match. The zero-flow full validation reference remains `0.2842080519` px/30 ms. These close synthetic values do not establish a generator ranking or real-data advantage.

### Fixed final-best diagnostic

Run output: [`evidence/S4/20261019-final-best-motion-diagnostic/motion_diagnostic.json`](evidence/S4/20261019-final-best-motion-diagnostic/motion_diagnostic.json). It ran on g1 GPU 3, Python 3.11.15 / PyTorch 2.5.1+cu121 / V100S, 64 fixed windows from all 8 held-out rollouts, exit code 0, elapsed 90.9 seconds. It restored the epoch-33 and epoch-20 synthetic-best weights separately, checked each checkpoint SHA and its own generator/seed/update/epoch/best EPE/scheduler boundary, and verified the same paired initialization. No real data was read.

The first attempt exited 1 before inference because the older diagnostic loader required `resume_phase=train`, `resume_allowed=true`; a synthetic-best inference checkpoint correctly records `resume_phase=inference`, `resume_allowed=false`. The attempt and log are retained as `attempt1_*`. The adapter now checks the inference-only lifecycle and the full model-state hash explicitly, then calls ordinary eval-mode inference. The fixed sample/mask plan and base diagnostic hash are unchanged.

| Fixed region, pooled over four registered scene strata | Pixels | EventGenesis EPE / zero-flow EPE | V2E EPE / zero-flow EPE | Interpretation |
|---|---:|---:|---:|---|
| Full physical valid region | 5,703,272 | 0.276812 / 0.277367 | 0.276169 / 0.277367 | About 0.20% and 0.43% better than zero flow; 98.6% and 98.5% of predictions are below 0.1 px. |
| Shared EventGenesis∪V2E active pixels ∩ valid | 27,751 (0.4866% coverage) | 0.302791 / 0.352044 | 0.315132 / 0.352044 | EPE improves 14.0% and 10.5% on this common active support. It is informative for event-supported motion, not dense coverage. |
| GT motion ≥0.5 px/30 ms | 993,694 (17.43%) | 0.816378 / 0.821715 | 0.818644 / 0.821715 | Only 1.82% / 1.58% of these predictions reach 0.1 px; average gain remains under 0.7%. |
| GT motion ≥1.0 px/30 ms | 243,297 (4.27%) | 1.245190 / 1.255078 | 1.249889 / 1.255078 | Only 2.20% / 1.73% reach 0.1 px; the overall improvement is under 0.8%. |

On the common active union, EPE improves in camera-motion/static-scene, camera-plus-object, and occlusion strata. It regresses in independent rigid-object motion with a static camera: EventGenesis `0.3451` vs zero `0.2805`; V2E `0.3032` vs zero `0.2805`. Event support is only 0.15% of valid pixels in this stratum, so the small count and fixed synthetic diagnostic are not a basis for tuning. Overall, the checkpoints contain some useful event-supported motion response, but most flow remains near zero and gains over zero flow are weak; this is a learning limitation, not a confirmed contract failure.

All paired IDs, GT, validity, gray endpoints, timestamps, intrinsics and label hashes matched. Every event interval was verified as 30 ms; output decoding remained `[u,v]` at native pixel displacement per interval. No timing, label-direction, unit, resize or crop mismatch was detected. The old epoch-23 diagnostic remains untouched.

### MVSEC dt1 candidate

Candidate protocol: [`protocol/mvsec_dt1_protocol_candidate_20261019.json`](protocol/mvsec_dt1_protocol_candidate_20261019.json); test manifest: [`manifests/mvsec_dt1_test_manifest_candidate_20261019.json`](manifests/mvsec_dt1_test_manifest_candidate_20261019.json). The evaluator's synthetic contract test passes hand-calculated 2D flow, finite/nonzero GT, paired real-event support, outdoor top-190 crop, component-shape mismatch and empty-support cases. Unauthorized real scoring is rejected before predictions/GT are opened.

The current candidate registers indoor_flying1/2/3 and outdoor_day1, pinned STE timestamp propagation, native `[u,v]` displacement, shared real-event support, AEE and `EPE > 3 px AND EPE > 0.05|GT|` Outlier, per-sequence means and equal-weight sequence aggregation. Actual file hashes and complete APS index lists remain to be refreshed into the manifest; research must freeze those support/aggregation rules before real scoring. MVSEC scores remain closed.

### Flow curve archive

The full run was assembled from a completed epoch-1 paired pilot (`pilot/{eventgenesis,v2e}`: updates 1–198) and the continuation (`full_training_retry1`: updates 199–8,910). The continuation JSONL alone therefore begins at epoch 2. `scripts/archive_stage4_flow_seed0.py` verified each optimizer update appears once, each of 45 epochs has 198 updates and one synthetic validation, both per-arm resumed trainer exits are `0`, and final states and paired initial hashes agree. The 90-row curve and external-artifact index are in [`evidence/S4/20261019-flow-final-archive/`](evidence/S4/20261019-flow-final-archive/). The original campaign wrapper exit `2` and first post-data continuation exit `1` followed a failed strict resume-equivalence gate and are retained; they are not the exits of the successful retry training processes. The retry boundary oracle passed before the final run.

## 3. Pose: E-POSE data access and normal eval behavior

Full access/model evidence: [`evidence/S4/20261019-pose-eval-access/POSE_EVAL_AND_ACCESS_AUDIT.md`](evidence/S4/20261019-pose-eval-access/POSE_EVAL_AND_ACCESS_AUDIT.md), with per-file hashes and source pins in `evidence_index.json`.

- The official paper confirms that E-POSE uses the DAVIS346c sensor (346×260), its example discretization is 10 ms / 100 Hz, the raw source archive is H5, and per-frame MAT labels include `Pose = T_dvs^obj`, `IntrinsicMatrix`, `cameraPose = T_base^dvs`, timestamp and projected keypoints. The paper's benchmark uses an event-only three-channel image (positive count, negative count, most recent-event time) and evaluates YOLO-6D, PoseCNN and DGECN. The author repository says its current MATLAB scripts consume ROS bags and direct raw-H5 processing was future work. These are published format/method facts, not a successful local sample parse. [Scientific Data paper](https://www.nature.com/articles/s41597-025-04536-5)
- Latest checked official publisher route returned HTTP 202 with an empty landing body and HTTP 403 from the Figshare article API. A fresh search found no verified mirror or sample file. This is route-level access failure, not evidence that the publicly described dataset is unavailable.
- The author repository is pinned at `b96ae783fcb6de84e1e9c602e91bce941cab0234`; its category-named `wood_block/google_16k/textured.obj` is 1,430,303 bytes with SHA-256 `a1cccc1d4be3fa5ef3b12b780edf15d4f8978bfae93ecfb0265314ce1d7c4e0c`. This establishes that an author CAD asset exists, but not that it exactly matches an accessible event/pose sample or its metric scale. No actual sample-level K, time, transform, class, scale or symmetry audit has passed.
- The pinned SingleShotPose confidence loss starts only after 15 warmup epochs. The earlier 16-epoch/8-window pilot supervised confidence for only 8 updates. A separate synthetic train-only fixture showed that 64 more confidence-supervised updates plus BN buffer recalibration from training event inputs made standard confidence-argmax inference stable under `model.eval()`: final two-window keypoint EPE was 1.42 px, and the decoded corners passed ordinary PnP with 1.83°/3.95° rotation and 4.0/13.6 mm translation errors. Save/restore reproduced eval outputs. This was a two-window overfit check on one generic cuboid rollout, not held-out-rollout or E-POSE performance; formal paired training remains gated on data/CAD and held-out synthetic normal-eval checks.
- A separate two-interval timestamp audit was rerun at 17:39:10–17:39:12 UTC. Independent raw-event recount matched the adapter counts for EventGenesis and V2E on both `(0.0,0.1]` and `(0.1,0.2]`; the maximum selected timestamp equaled the right-end label time. Counts were `3,3` and `4,3`, respectively. This is synthetic fixture evidence, not a parse of E-POSE raw H5.
- No GT-cell selection, train-mode inference, GT crop/mask/box, segmented-event input, or real score was used. Generic cuboid findings remain only implementation diagnostics.

## 4. Reproduction commands and scope

On g1, activate `Eventbased_WAM` from the repository root. Current full reconstruction jobs are already detached; do not launch duplicate jobs. The exact run logs record the complete resumed commands.

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Eventbased_WAM

# Recheck current reconstruction progress without changing the jobs.
python3 - <<'PY'
import json, pathlib
root=pathlib.Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2')
for arm in ('eventgenesis','v2e'):
 d=root/arm
 rows=[json.loads(s) for s in (d/'training_metrics.jsonl').read_text().splitlines() if s]
 updates=[r for r in rows if r.get('event')=='optimizer_update']
 print(arm, 'updates', updates[-1]['optimizer_update'], '/ 12000', updates[-1]['recorded_utc'])
PY

# Re-run only on a currently idle GPU after checking ownership/resources.
CUDA_VISIBLE_DEVICES=<idle-gpu> python benchmark/sim2real/scripts/stage4_final_best_motion_diagnostic.py \
  --config benchmark/sim2real/configs/stage4_final_best_motion_diagnostic_20261019.json

# Rebuild the flow archive from the preserved g1 logs/checkpoints (use a new
# empty output directory; the existing archive script refuses overwrite).
python3 benchmark/sim2real/scripts/archive_stage4_flow_seed0.py \
  --campaign-root /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5 \
  --output-dir benchmark/sim2real/evidence/S4/20261019-flow-final-archive

# Verify the MVSEC evaluator contract; this does not open real samples.
python benchmark/sim2real/scripts/stage4_mvsec_dt1_metrics.py --self-test
```

The flow full diagnostic used GPU3 and took 90.901 seconds. The two main reconstruction jobs were left on GPUs2/4. The flow training has finished, so its large checkpoints/logs remain read-only on g1. No command in this report scores real HQF, MVSEC or E-POSE ground truth.

The pose mechanism and timestamp fixtures ran locally in WSL using the pinned source at `/home/科研/Eventbased_WAM/external_sources/SingleShotPose` and local RTX 2050 CUDA. Reproduce with:

```bash
python benchmark/sim2real/scripts/audit_pose_eval_contract_fixture.py \
  --pose-source /home/科研/Eventbased_WAM/external_sources/SingleShotPose
python benchmark/sim2real/scripts/audit_pose_event_windows.py
```

The PnP follow-up and timestamp-window run outputs, commands, exits and UTC times are preserved under [`evidence/S4/20261019-pose-eval-access/`](evidence/S4/20261019-pose-eval-access/).

## 5. Remaining work and validity

1. Keep reconstruction training running to 12,000 updates per arm and collect epoch-level synthetic validation, best/final checkpoints and SHA-256 values. The 20-update launch and actual resume are valid training-chain evidence; they are not complete task performance evidence.
2. Complete exact MVSEC data/checkpoint hashes and eligible dt1 sample indices. Submit the candidate protocol/manifest for research freeze; do not open real scores yet.
3. Continue E-POSE public access resolution and sample-level CAD contract work. Do not start formal paired pose training on the generic cuboid or a merely category-named mesh.
4. HQF is the currently connected reconstruction candidate; HS-ERGB remains an original candidate with no completed Stage 4 loader/alignment. Keep that dataset-role decision explicit for research review.

No evidence currently indicates that the active reconstruction or completed optical-flow runs are invalid. Optical flow is weak on the fixed synthetic diagnostic but passes the checked input, label, checkpoint and output contracts. Pose remains blocked on an original sample/CAD match. No real-data ranking has been run.
