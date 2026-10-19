# Stage 5 parallel handoff — 2026-10-19

Updated **2026-10-19 10:42:13 UTC / 19:42:13 JST**. The latest direct L1 log rows are 10:42:01 UTC; GPU ownership was checked at 10:42:13 UTC. Flow diagnostics completed through the fixed four-scale audit at 10:23:46 UTC. E-POSE's official API/page/downloader were rechecked at 10:41:48 UTC. The active branch is `exp/public-sim2real-v1`, local and GitHub commit `e4b6cec453938bc6b6852d778f8c207c0917a5f8`. The reviewed Stage 4 baseline is `61554fb0031609200bc5ddbf2340ec8d20722725`; the frozen simulator core remains `e79e656ac00f7847b98db9e12bf43b093f3afff1`. No simulator core files were changed. All optimization and checkpoint choice remain synthetic-only; all real scores remain closed.

## Three-line status at a glance

| Route | Actual state at this snapshot | Main blocker | Next delivery |
|---|---|---|---|
| **L1 Reconstruction / E2VID+** | TC geometry, exact 10 ms labels, augmentation, recurrence, bad-manifest rejection, and update-20 process-boundary restore followed by update-21 passed. At 10:42:01 UTC, full seed-0 runs were active at EventGenesis **1,390/12,000** and V2E **1,350/12,000** on g1 GPU2; latest rows show finite gradients, six recurrent tensors and 76 TC calls. | Runs are not complete and have no final synthetic validation or Stage 5 real inference. HQF has one prior sequence IO/inference check; HS-ERGB loader/alignment is not ready. Research still needs to freeze dataset role and target-time/score protocol. | Keep both fixed-budget runs alive; archive full curves and checkpoints. Real scoring stays closed. |
| **L2 Flow / STE-FlowNet dt1** | Existing paired run remains complete at 45 epochs / 8,910 updates per arm. Perturbation, both bounded GT pilots, common-active-mask evaluation, and four-scale label/output audit are complete. | Both arms remained worse than zero on the tiny fixed train/validation samples and active support; no hard scaling/unit issue was confirmed. The root cause remains uncertain. | The separate balanced 8-rollout coverage bank and statistics pipeline are committed and pass static/runtime import checks. Generation remains queued: at 10:42:13 UTC GPU0 had 8.3 GiB used at 95% utilization, and GPU2 is reserved for L1. No formal retraining or MVSEC score has started. |
| **L3 Pose / SingleShotPose → E-POSE** | Ordinary-eval confidence/keypoint/PnP and checkpoint restore ran on two independent synthetic validation rollouts, with synthetic-train-only BN statistics frozen at validation. Official code/CAD candidates are local; the mustard XYZ hash is indexed. | At 10:41:48 UTC, read-only WSL HEAD probes found Figshare API 403 and both publisher page/direct downloader 202 WAF challenge. No original event HDF5 or paired MAT was obtained; generic-box errors remain large and sample-to-CAD identity is unverified. | Keep the sample/CAD contract audit open. A matched original event excerpt, exact MAT sidecar and object identity are needed; no formal generic-box training. |

Training completion, implementation correctness, and real-data results are distinct. L1 is running after its correctness gate; the prior L2 training is complete while its cause remains unresolved and its scene pilot is queued; L3 has an executable generic-object mechanism but no matched E-POSE sample/CAD contract.

### Current parallel resource use

At 10:42:13 UTC, GPU2 had 6.4 GB allocated to the two reconstruction processes (PIDs `873967`, `873971`); the latest training logs at 10:42:01 UTC reached 1,390/1,350 updates. GPU0 had 8.3 GiB used at 95% utilization, so it was left untouched. GPUs 1, 3, 4 and 5 also had existing allocations; none were changed. The runtime is Python `3.11.15`, PyTorch `2.5.1+cu121`, CUDA runtime `12.1`, Genesis `1.2.2`; the new clean g1 checkout passed Python compilation and `genesis_event_plugin` import with the checkout on `PYTHONPATH`. Pose access checks used read-only HTTP HEAD requests from local WSL; no pose training is active. The new 8-rollout coverage pilot estimate remains about 2 GPU-hours of V2E simulation plus approximately 80 minutes of EventGenesis CUDA projection/integration, excluding Genesis rendering and writing. It is queued until a suitable GPU slot is safe.

## L1 — Reconstruction

### Correction and checks

The previous 200-epoch runs are retained as historical results and are not accepted under the corrected temporal-consistency (TC) contract. The bug was in the benchmark adapter: the sampling grid was created with the `align_corners=True` normalization formula, while two `grid_sample` calls used the default `False`. The compatibility fix sets the sampler convention explicitly and leaves the frozen core unchanged.

The preflight checks include zero-flow identity, known integer translation, subpixel translation, non-square images, direction/sign, valid borders, and the actual model TC path. The geometry checks are available at [`evidence/S5/20261019-tc-aligntrue-preflight/`](evidence/S5/20261019-tc-aligntrue-preflight/). The new pair preflight at g1 reports `preflight_pass_training_authorized`; its manifest is SHA-256 `2a855979f89475a3fc7914de2fdc437d19b5f6c127d239d47f8915cccf7445c4`. The 10 ms adapter data manifest is SHA-256 `06ecadb1b4e90bc434ce30148c5d50704385cdab15a24e88ff9366f7c58edcb3` (built from physical-bank manifest `cca882ba912fda3b3243d49bc7e29e28186046e3b6591c424f385b0506047884`); the same exact 10 ms image displacement and validity labels are used for both generators.

The real model probe produced nonzero, finite gradients through the weighted TC term: EventGenesis TC loss `0.0091513451`, gradient L2 `0.241010`; V2E TC loss `0.0142047405`, gradient L2 `0.350011`. Each update logs loss components, 76 valid TC calls, finite gradient tensors, and six recurrent state tensors. The joint transform receipt, train/validation rollout split (24/8), model reset/carry check, and paired initialization check passed. The negative gate was exercised with a bad manifest: it exited 1 and produced no training metrics or checkpoint.

Both arms first completed 20 actual updates, exited cleanly, restored their saved update-20 checkpoint in a separate process, and produced the next update (21). The V2E resume oracle verified the next sample hashes and loss exactly and the optimizer step count changed 20→21; EventGenesis also observed update 21 and passed. The source boundary checkpoints were:

- EventGenesis update 20: SHA-256 `8afafb5c9efcf1816b5473b2ca12b524ed1d47136e2e940fec7f7252eeeefd02`.
- V2E update 20: SHA-256 `6f0140889b27e4c328ca647a2b04e2fd631762f11fc628fa8072a5d7a8bad017`.

This verifies a process-boundary checkpoint restore and the next training step, not strict bitwise equivalence over an entire trajectory.

### Current training process

At 10:37:17 UTC / 19:37:17 JST, the full-training logs recorded EventGenesis update 1,290 and V2E update 1,250. Both rows had finite gradients, six recurrent state tensors, and 76 valid TC calls. The tmux sessions remained active. They use the same registered paired random initialization SHA-256 `48426d94f4159592155eda873d4e71e2f06439c10eade990de5979231ed07a0f`, shared rollout split, 200-epoch/12,000-update budget, and synthetic LPIPS-VGG checkpoint choice rule.

The immutable training checkout is g1 `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-stage5-exec-20261019`, commit `10a593c2040e58257b39a89d89bee6cdbde5e14b`. Run data, checkpoints, and append-only logs are under g1 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0/`. tmux sessions are `stage5_egfull` and `stage5_v2efull`; the latest direct GPU2 process snapshot showed only these two training processes on that GPU. No result from the first 260/230 updates is treated as a final reconstruction comparison.

### Real reconstruction data roles and preparation

The source handoff [`baselines/03_CHAT_SIM2REAL_HANDOFF.md`](research/input_2026-10-16/baselines/03_CHAT_SIM2REAL_HANDOFF.md) names HS-ERGB for Layer 1. The later versioned [`RESEARCH_PROTOCOL.md`](research/input_2026-10-16/protocol/RESEARCH_PROTOCOL.md) explicitly revises the candidate to HQF primary and HS-ERGB extension. This execution follows the current protocol for engineering preparation, but does not silently resolve that historical role change; research review must confirm the final role before real scoring. Earlier Stage 3 work downloaded and converted one preregistered HQF `bike_bay_hdr` bag: 2,972 event messages, 2,431 mono8 reference images at 180×240, 19,763,043 decoded events. Earlier synthetic-only checkpoints ran its 2,431 endpoints with a sequence-start reset. This validates one HQF sequence reader/inference path, not all 14 sequences, the new Stage 5 weights, or any score.

HQF candidate scoring covers all 14 sequences, resets recurrent state at sequence start, never supplies reference frames to the model, and uses LPIPS-Alex primary with SSIM/PSNR/MSE support metrics and equal-sequence aggregation. Exact event-to-reference endpoint pairing, crop/clipping, and exclusions still need freeze; no candidate score has been opened.

HS-ERGB has no complete loader or target-time alignment exercised. Its source task is video interpolation, so a valid event-to-reference reconstruction target still needs definition. No role change is being made in this engineering run; the historical HS-ERGB assignment versus later HQF protocol revision is listed for research confirmation.

## L2 — Optical flow

### Fixed input perturbation diagnostic

The already completed corrected-v2 seed-0 training remains unchanged: 45 epochs / 8,910 updates each. The existing synthetic-best checkpoints were reused, not retrained: EventGenesis epoch 33, checkpoint SHA-256 `a4e6c7e1d10a4d76562a046c364a5e17de6452d5b648fc8ccbaa3bd43c824a8e`; V2E epoch 20, SHA-256 `11041863685b417306f534b8e12a1177b1337fc2818d255ec7b76fa6e20cd474`.

The new test perturbed the five-bin tensor presented to the model, rather than the original unbinned `(x,y,t,p)` stream. It used normal input, a zeroed tensor, and reversal of the five time bins per polarity. Reversal preserves per-pixel counts and spatial support. The same valid mask and the same frozen union of normal EventGenesis/V2E active pixels were used for all conditions on each of eight fixed validation windows. No window was chosen by event count.

| Region (sample-macro mean, px per 30 ms) | EventGenesis normal / zero-flow | V2E normal / zero-flow |
|---|---:|---:|
| Full valid region | 0.10873 / 0.10608 | 0.10604 / 0.10608 |
| Frozen common event-active union | 0.16211 / 0.09308 | 0.09128 / 0.09308 |

On the active union, time reversal produced EPE `0.22439` for EventGenesis and `0.09156` for V2E. Zeroing the input changed active-region output by mean absolute flow `0.06169` and `0.02258` respectively; time reversal changed it by `0.07680` and `0.00918`. On full valid pixels, both models' mean predicted magnitudes were close to zero (`0.00498` and `0.00123` px per 30 ms), while the shared zero-flow EPE was `0.10608`. The active union covered 4,122 of 718,212 valid pixels (0.574%). The stored tensors contained 17,916 EventGenesis event counts and 1,037 V2E counts across the same eight windows.

These results confirm that an input perturbation can change the output, especially for EventGenesis, but they do **not** confirm correct motion decoding. EventGenesis is worse than zero flow on the active union; V2E is slightly better there but essentially indistinguishable over the full valid region. The active union is sparse, so full-frame zero-like output is not enough to determine if the model uses motion on event pixels.

The complete window-level results and exact masks are in [`evidence/S5/20261019-parallel-diagnostics/flow_input_perturbation_results.json`](evidence/S5/20261019-parallel-diagnostics/flow_input_perturbation_results.json); a compact summary is in the sibling `diagnostic_summary.json`. Run root on g1: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-flow-diagnostics/input_perturbation_v2/`.

### Bounded shared-GT diagnostic

The supervised diagnostic uses pinned STE-FlowNet and a separate loss, without changing the registered photometric/smoothness experiment. The first run used four fixed training rollouts, 64 shared-label windows, and one held-out window from each of eight validation rollouts. Both arms started from paired initial weights and used the same sampler order and 80-update budget. The pinned four-scale robust endpoint loss supervised training; model inputs remained events only. Synthetic labels were not used by the formal photometric experiment or checkpoint selection.

The scalar rollout-level `label_sha256` loader bug was fixed in commit `31f75221ffb6d50c12875269ff87dc3d700824fe`. The corrected first pilot completed 80 updates per arm and passed checkpoint state/output restore checks. Training loss fell from `0.8544 → 0.00735` (EventGenesis) and `0.6197 → 0.01038` (V2E), but the macro mean EPE over the 64 training windows was `0.29339` and `0.29040`, versus zero-flow `0.28622 px/30 ms`. On eight independent validation rollouts, EPE was `0.12471` / `0.10909`, versus zero-flow `0.10608`. Each arm saw only 80 updates over 64 windows (1.25 passes); thus this does not establish inability to fit, nor does the loss reduction establish useful flow. The pilot summary, full update curves, and external checkpoint hashes are in [`evidence/S5/20261019-parallel-diagnostics/gt_supervised_pilot/gt_supervised_80_update_v1_retry4/`](evidence/S5/20261019-parallel-diagnostics/gt_supervised_pilot/gt_supervised_80_update_v1_retry4/).


The second, isolated fit check began at `2026-10-19T10:02:24Z` and completed at `10:13:50Z` on g1 GPU2 in tmux `stage5_flowgt8`: two stratified windows (indices 16 and 48) from each of the same four training rollouts, 8 training windows total, 80 updates per arm (10 passes), paired initialization/order, the same four-scale loss, and the same eight held-out rollouts. It used source commit `7dfaa8028b1309ee10cb66bbab32173133718740`; exit code was 0. The optimizer objective fell from `0.52788 → 0.24788` for EventGenesis and `0.32594 → 0.25072` for V2E, with 70 finite-gradient tensors on every logged step. However, mean train EPE was `0.30348` / `0.30571` against zero-flow `0.29869`; mean held-out EPE was `0.12514` / `0.11253` against zero-flow `0.10608` px/30 ms. Neither arm demonstrated an EPE fit even on this 8-window subset under 80 updates. This bounded result does not establish that STE-FlowNet cannot fit or identify whether the limiting factor is training budget, sparse support, target/loss behavior, or data coverage.

Train-mode and ordinary `eval()` returned bit-identical dense flow for one fixed train and one held-out window per arm (`max_abs_difference=0`). Checkpoint restore state hashes and eval output hashes matched. This rules out a train/eval BatchNorm mismatch on those fixed examples; it is not a general proof of correctness. The 580 MB checkpoints remain external: EventGenesis `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-flow-diagnostics/gt_supervised_8window_80update_v1_7dfaa80/results/eventgenesis_last.pt`, SHA-256 `952de4a7503d291fb92aadea7be500749889283219b6c6255e9f48fd295586b1`; V2E at the sibling `v2e_last.pt`, SHA-256 `c910932c9415838c903c7abf717e04dd9ba3e42665d0bf0323f98b0776b85eb0`. Sizes are 580,024,698 and 580,022,034 bytes. Exact per-update logs, summary, command, UTC start/finish, and exit code are archived under [`evidence/S5/20261019-parallel-diagnostics/gt_supervised_pilot/gt_supervised_8window_80update_v1_7dfaa80/`](evidence/S5/20261019-parallel-diagnostics/gt_supervised_pilot/gt_supervised_8window_80update_v1_7dfaa80/).

A post-hoc evaluation of these same checkpoints completed at `10:16:45Z` in tmux `stage5_flowactive`, exit code 0. The mask was shared GT validity intersected with the union of event-active pixels from both normal paired inputs. Training support was 2,315/714,207 valid pixels (0.324%); held-out support was 4,122/718,212 (0.574%). On training windows, active-region EPE was `0.44570` EventGenesis and `0.37207` V2E against zero-flow `0.20376`. On the eight held-out windows it was `0.53246` and `0.13026` against zero-flow `0.09308`. Both arms are worse than zero flow on this active support, particularly EventGenesis. Therefore static background does not by itself account for the near-zero full-frame results in this bounded supervised pilot. This does not determine whether the cause is input representation, target/loss scaling, optimization budget, or scene coverage.

The script hashes the data adapter and checks identical labels, valid masks, sample IDs and scalar rollout label hashes for both generators before inference. It verifies each checkpoint's model-state hash and metadata, records event/GT shapes and finite values, and applies the same active mask to both arms. Its complete output is [`evidence/S5/20261019-parallel-diagnostics/gt_supervised_pilot/gt_supervised_8window_80update_v1_7dfaa80/posthoc_active_eval/summary.json`](evidence/S5/20261019-parallel-diagnostics/gt_supervised_pilot/gt_supervised_8window_80update_v1_7dfaa80/posthoc_active_eval/summary.json); source script SHA-256 is `7b9be6c276c55ae4157d1ca9acb74a493a046d97f9ebc67bb7e5633dacc4b82d`. The evaluation did not use real data or alter any checkpoint selection.

### Multi-scale target/output audit

A fixed read-only audit finished at `2026-10-19T10:23:46Z` on 16 windows: the same eight fixed training windows and one window from each independent held-out rollout. For each arm, it checked the registered checkpoint/data/config bindings, paired label equality, finite event/GT values, all four train-mode output scales, shared validity, scale-adjusted GT displacement, and ordinary `eval()` full flow. Native output shapes were 352×352, 176×176, 88×88 and 44×44. The first train-mode scale matched ordinary eval flow with maximum absolute difference `8.65e-6 px` over all samples. The first-scale robust endpoint loss matched independent full-resolution EPE to the expected `sqrt(EPE²+1e-6)` epsilon; target displacement scaled by output width/height ratios. No scale or flow-unit mismatch was confirmed in this path.

The same audit confirms the weak-fit observation: on all held-out samples, all four scale losses were above the zero-prediction loss; on the training subset the first three scales were above zero for EventGenesis, and all four were above zero for V2E. The fourth EventGenesis train scale was slightly below zero, without improving full-resolution or active-region EPE. This does not identify whether event support, scene coverage, or optimization budget is the remaining limit. Full per-scale measurements, checkpoint hashes and split IDs are in [`evidence/S5/20261019-parallel-diagnostics/gt_supervised_pilot/gt_supervised_8window_80update_v1_7dfaa80/scale_target_audit/summary.json`](evidence/S5/20261019-parallel-diagnostics/gt_supervised_pilot/gt_supervised_8window_80update_v1_7dfaa80/scale_target_audit/summary.json); source script SHA-256 is `79a023d7d893c7c955819d51df1be332706e4ba5b12bce3f04120c9fdd3984ee`.

The earlier 64-window Stage 4 best-checkpoint diagnostic measured a common active union of 27,751/5,703,272 valid pixels (0.4866%); shared GT motion ≥0.5 px/30 ms covered 993,694/5,703,272 (17.4232%). The existing bank generator uses uniformly colored boxes and a gray plane rather than the varied textures described in a config comment. A new isolated 8-rollout coverage pilot is registered with four motion strata and one train plus one held-out validation rollout per stratum. It balances texture pattern/contrast, speed, amplitude, foreground size and scene layout within each split. Both generators will use the same physical scenes and labels; sensor thresholds remain fixed, and event counts never select windows.

The configuration, generator, V2E finalizer and coverage-statistics script are committed at `e4b6cec453938bc6b6852d778f8c207c0917a5f8`. Static checks verified 8 unique rollouts, a 4/4 train/validation split, all four strata in each split, and both levels of each of the six factors per split. The clean g1 checkout `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-stage5-coverage-exec-20261019` passed Python compilation and `genesis_event_plugin` import in the registered environment. No pilot trajectory or event has been generated yet: GPU0 was at 8,467 MiB / 95% utilization at 10:42:13 UTC; GPU2 remains reserved for L1 and other GPUs already have allocations. The pilot is queued until a safe slot is available. It will measure per-generator event-active support, shared union coverage, GT-motion support and label validity; it will not train a formal model or open MVSEC scores.

**Next action recommendation:** run the registered scene-coverage pilot when a GPU slot becomes safe. The multi-scale audit found no scale or unit mismatch on the fixed path, while both arms remain worse than zero on the bounded active-region GT pilot. The new pilot tests the concrete coverage explanation without changing the formal objective or sensor thresholds. Until those measurements exist, current evidence does not justify another long flow run. Real MVSEC scores cannot select configurations.

**Current diagnosis:** confirmed facts include near-zero frame-wide output, 0.574% active-union coverage in the fixed validation subset, sensitivity to input perturbations, lower robust loss without lower EPE on the 8-window fit check, and no fixed-path scale/unit mismatch. On the same frozen active region, both pilot checkpoints also score worse than zero flow. These results do not identify one root cause. Event sparsity, scene coverage, optimization behavior and interactions between event encoding and supervision remain possible explanations; none is confirmed. The new scene pilot will test whether richer shared scenes improve event/GT support before any formal training change is proposed.

### MVSEC evaluation readiness

The evaluator's checkpoint-index lookup now uses the registered `manifest.checkpoint_binding` schema and checks artifact-manifest, exact checkpoint path, size, and SHA-256 before scoring. The self-test exercises hand-computed AEE/Outlier, crop edge mapping, empty masks, stale bindings, and rejection when real scoring is not authorized: [`evidence/S5/20261019-mvsec-evaluator-preflight/synthetic_self_test_v2.json`](evidence/S5/20261019-mvsec-evaluator-preflight/synthetic_self_test_v2.json).

The candidate dt1 test manifest fixes the registered checkpoint pair, event/APS endpoint interval, propagated GT endpoint, crop, common valid-GT/event-active support, AEE/Outlier and summary rules. The main recommendation is full native resolution for all registered sequences; the legacy STE code's 256×256 center crop/top-190-row mask should be reported as a secondary reproduction diagnostic. This difference is explicitly unresolved pending research freeze. The candidate protocol and manifest are [`protocol/mvsec_dt1_protocol_candidate_20261019.json`](protocol/mvsec_dt1_protocol_candidate_20261019.json) and [`manifests/mvsec_dt1_test_manifest_candidate_20261019.json`](manifests/mvsec_dt1_test_manifest_candidate_20261019.json). No real MVSEC score has been opened.

## L3 — Object pose and E-POSE

### Current E-POSE access result

The official-source checks at `09:15Z` and `09:59Z` were followed by fresh read-only HEAD requests from WSL at `10:41:48Z`. The Figshare API returned HTTP 403; the publisher dataset page and direct archive downloader both returned HTTP 202 with `x-amzn-waf-action: challenge` and zero content length. No archive bytes were received. The paper page at [Scientific Data](https://www.nature.com/articles/s41597-025-04536-5) confirms raw data is HDF5 with adjacent MAT pose metadata, while the official [Figshare DOI](https://doi.org/10.6084/m9.figshare.28106717) is the dataset record. A verifiable public mirror or sample was not found; no author was contacted. The latest machine record is [`evidence/S5/20261019-epose-access/epose_access_recheck_20261019T1041Z.json`](evidence/S5/20261019-epose-access/epose_access_recheck_20261019T1041Z.json); earlier records remain intact.

The paper defines raw events in HDF5 and a matching MAT sidecar with `T_dvs^obj`, intrinsic matrix, timestamp, camera pose, event window and keypoint projections. Its released tree also contains masks, segmented-event images and rendered event representations; these are not model inputs under this protocol. The official code repository at `/home/科研/Eventbased_WAM/external_sources/E_Pose` is commit `b96ae783fcb6de84e1e9c602e91bce941cab0234`; its README still describes generation from `.bag` files and says HDF5 handling requires adapting the scripts. It includes class-named point clouds/CAD candidates. For example, `xyz_data/mustard.xyz` is 474,668 bytes, SHA-256 `85c89f5eb25f25706e7ae1fa1acc17186c98c296fa0745e2e929f7c9125a9af1`. The code maps mustard to class index 6 and includes a DAVIS K candidate, but no raw sample exists locally to prove that a specific sequence is mustard or validate its sample-specific time offset, pose direction, units, or symmetry. No E-POSE `.h5/.hdf5/.mat` sample was found in the project data inventory. The minimum missing input is one original event HDF5 excerpt, its exact MAT sidecar, and sequence/object identity. Generic cuboids and unmatched class assets cannot substitute for this contract.

### Ordinary-eval mechanism diagnostic

The 31-epoch synthetic mechanism diagnostic used a generic Genesis box, 8 training windows from 4 training rollouts, and 4 windows from 2 independent validation rollouts. It completed 128 confidence-supervised updates per arm. BatchNorm statistics were rebuilt from the eight synthetic training inputs only and stayed unchanged during ordinary `model.eval()` validation. Confidence argmax, keypoint decode, PnP, checkpoint restore, and exact restored-state hash all executed; no train-mode inference or GT crop/mask/box/cell selection was used.

| Arm | Ordinary-eval validation keypoint EPE | PnP returned a solution | Mean rotation / translation error |
|---|---:|---:|---:|
| EventGenesis | 34.65 px | 4/4 windows | 55.5° / 0.878 m |
| V2E | 176.82 px | 4/4 windows | 115.5° / 1.154 m |

These large errors mean that the generic-box test proves the path executes, not that it predicts useful pose. It does not establish the E-POSE contract and is not used to rank generators. Full detail is [`evidence/S5/20261019-parallel-diagnostics/pose_generic_mechanism_result.json`](evidence/S5/20261019-parallel-diagnostics/pose_generic_mechanism_result.json); the large 386 MB per-arm checkpoints remain on g1 under `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-pose-generic-31epoch/`.

## Reproduction and artifacts

### Environment and immutable execution checkouts

- g1: `zhaoyuhan@g1.pkuml.org`, project root `/home/zhaoyuhan/Eventbased_WAM/`.
- Conda environment: `/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM`, Python 3.11.15. Only physical GPU2 was used for the new Stage 5 diagnostics/training; other users' GPU processes were left untouched.
- L1 execution checkout: `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-stage5-exec-20261019`, commit `10a593c2040e58257b39a89d89bee6cdbde5e14b`.
- L2 first diagnostic checkout: `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-stage5-flowdiag-exec-20261019-v3`, commit `31f75221ffb6d50c12875269ff87dc3d700824fe`. The current 8-window fit check uses clean checkout `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-stage5-flowdiag-exec-20261019-v4`, commit `7dfaa8028b1309ee10cb66bbab32173133718740`; the executed script SHA-256 is `d060e74495e7d31580bf432356d37dcc88c627463f5f7de7482ac2c96d2f6a88`.
- New coverage-pilot checkout: `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-stage5-coverage-exec-20261019`, commit `e4b6cec453938bc6b6852d778f8c207c0917a5f8`. It passed Python compilation and `genesis_event_plugin` import; the pilot has not started because no GPU was safely available at the latest resource check.
- Existing shared g1 path `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real` was not reset, cleaned, or overwritten. It remains at its prior dirty state; execution used independent clean worktrees.

### Commands

The active E2VID jobs were launched with the committed bounded wrapper, from the update-21 resume checkpoint. The wrapper writes the exact trainer command, PID, UTC times, and exit code under each run directory:

```bash
/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-stage5-exec-20261019/benchmark/sim2real/scripts/run_e2vid_stage5_segment.sh \
  eventgenesis \
  /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-stage5-exec-20261019 \
  /home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/event_cnn_minimal \
  /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/e2vid_corrected_10ms_20261019_v1 \
  /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-stage5-exec-20261019/benchmark/sim2real/configs/stage4_corrected_v2_flowdt1_seed0.json \
  /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0/preflight_v3/group_manifest.json \
  /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0/eventgenesis \
  /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0/full_training/eventgenesis.log \
  /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0/full_training/eventgenesis.exit \
  0 /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0/eventgenesis/last.pt
```

Use the same command with generator/run/log/exit paths changed to `v2e`. The saved `segment_command.txt` files are the authoritative invocation records.

The first bounded flow pilot used the following command (completed; all results are preserved):

```bash
/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python -u \
  benchmark/sim2real/scripts/stage5_flow_supervised_pilot.py \
  --model-root /home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/STE-FlowNet \
  --data-root /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/20261017-stage4-flowdt1-seed0-corrected-v5/steflow_30ms \
  --config /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-stage5-flowdiag-exec-20261019-v3/benchmark/sim2real/configs/stage4_corrected_v2_flowdt1_seed0.json \
  --output-root /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-flow-diagnostics/gt_supervised_80_update_v1_retry4 \
  --updates 80 --device cuda:0
```

The completed first pilot ran with `CUDA_VISIBLE_DEVICES=2` in tmux `stage5_flowgt4`; its evidence shows exit code 0. The new 8-window check was launched at 10:02:24 UTC by tmux `stage5_flowgt8`. It runs the same script from checkout v4 with the following changed arguments: `--train-windows-per-rollout 2 --output-root /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-flow-diagnostics/gt_supervised_8window_80update_v1_7dfaa80/results`. Its launch script, command, PID, start/finish timestamps, run log and exit code are under that g1 run root. The saved objective remains a separate synthetic diagnostic.

The new scene-coverage pilot is configured and queued; no generated artifacts exist yet. After a fresh resource/ownership check, launch in a new tmux session with a unique output directory:

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-stage5-coverage-exec-20261019
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PWD"
export GS_CACHE_FILE_PATH=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-flow-coverage-pilot/cache/genesis
export TI_OFFLINE_CACHE=1 TI_OFFLINE_CACHE_FILE_PATH=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-flow-coverage-pilot/cache/taichi
/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python -u \
  benchmark/sim2real/scripts/generate_stage5_flow_coverage_bank.py \
  --config benchmark/sim2real/configs/stage5_flow_scene_coverage_pilot_v1.json \
  --output-root /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261019-flow-scene-coverage-pilot-v1/shared_bank
```

Once the bank completes, run the pinned Stage 4 V2E shard runner on the eight registered IDs, then `finalize_stage5_v2e_coverage_pilot.py` and `stage5_flow_coverage_statistics.py`. Do not start while GPU0 is owned by another job; GPU2 remains reserved for L1.

### Evidence and external artifact index

- Compact input/pose results: [`evidence/S5/20261019-parallel-diagnostics/`](evidence/S5/20261019-parallel-diagnostics/).
- E2VID preflight, gate rejection, boundary-resume receipts: [`evidence/S5/20261019-tc-aligntrue-preflight/`](evidence/S5/20261019-tc-aligntrue-preflight/), [`evidence/S4/20261019-e2vid-parallel-launch/`](evidence/S4/20261019-e2vid-parallel-launch/).
- Flow checkpoint archive and MVSEC evaluation candidate: [`evidence/S4/20261019-flow-final-archive/`](evidence/S4/20261019-flow-final-archive/), [`evidence/S5/20261019-mvsec-evaluator-preflight/`](evidence/S5/20261019-mvsec-evaluator-preflight/).
- Full E2VID logs/checkpoints: g1 run root stated above. Flow GT pilot logs/checkpoints and the pose model checkpoints remain under `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-flow-diagnostics/` and `.../20261019-stage5-pose-generic-31epoch/`.
- Machine-readable external artifact/status index: [`evidence/S5/20261019-parallel-diagnostics/parallel_evidence_index.json`](evidence/S5/20261019-parallel-diagnostics/parallel_evidence_index.json). The compact diagnostic evidence and source-file checksum list are [`evidence/S5/20261019-parallel-diagnostics/SHA256SUMS`](evidence/S5/20261019-parallel-diagnostics/SHA256SUMS). Large g1 checkpoints remain outside Git; their paths and hashes are recorded in the index and per-run JSON summaries.

### Repository sync

The source/evidence commit `e4b6cec453938bc6b6852d778f8c207c0917a5f8` is on local and GitHub branch `exp/public-sim2real-v1`. L1 remains pinned to clean checkout `10a593c...`; the completed 8-window flow diagnostic used checkout v4 at `7dfaa80...`; the new pilot checkout is clean at `e4b6cec...` and passed compile/import checks. The shared g1 worktree has not been modified. This handoff update will be committed separately; the pilot has not yet generated data.

## Recommended next action

Continue the active corrected E2VID pair. Run the registered eight-rollout flow coverage pilot when a GPU slot becomes safe; the GT-fit, active-region and scale audits are complete, and present evidence does not warrant another long flow run. Continue E-POSE sample/CAD access and ordinary-eval mechanism checks; formal E-POSE training remains blocked on a verified sample/CAD match. Keep real scoring closed pending research review.
