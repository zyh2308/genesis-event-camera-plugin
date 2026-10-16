# Public-data Sim-to-Real status — Stage 3

Updated: 2026-10-16. Branch: `exp/public-sim2real-v1`. Server: `zhaoyuhan@g1.pkuml.org`, repository worktree `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real`. Frozen simulator core `e79e656ac00f7847b98db9e12bf43b093f3afff1` is unchanged.

Stage 3 repaired two independent correctness bugs, validated existing synthetic checkpoints, parsed preregistered public HQF/MVSEC samples, and ran event-only candidate inference. It did not select checkpoints or rank generators on real GT.

| Route | Actually completed | Current blocker / next gate |
|---|---|---|
| E2VID+ → HQF | Reused accepted Stage 2 synthetic checkpoints; converted public `bike_bay_hdr` bag; both candidates processed all 2,431 endpoints with one sequence-start reset and saved checkable reconstructions. No HQF target scoring. | Freeze HQF final-test sequence inventory, target/metric handling and aggregation. Review HQF as the main reconstruction candidate and HS-ERGB only as a separately aligned extension. |
| STE-FlowNet → MVSEC | Repaired full `[u,v]` EPE; deterministically restored and re-scored all four existing checkpoints over 60 dt1 / 54 dt4 synthetic intervals. Parsed a fixed public `indoor_flying1` fragment and ran all four candidates at 352×352 padded input, cropping outputs to 346×260. | The real support mask and aggregation remain provisional. Freeze zero-GT, event-support, invalid-region and per-sequence aggregation rules before formal real scoring. |
| SingleShotPose → E-POSE | Fixed pose event-window endpoints; independently recounted raw timestamps; ran a corrected paired synthetic pilot and a one-sample overfit/inference-state diagnostic. | The one-sample model/CAD contract remains generic. Confidence selection and batch-one BatchNorm running statistics are unstable. Official E-POSE landing/API currently return WAF HTTP 202/403; need one original sample plus matching CAD and a research decision on model/inference protocol. |

## Corrections and verified results

- Stage 2 optical-flow EPE values are invalid because only `u` was extracted and broadcast as `v`. Corrected synthetic means from the same checkpoints: dt1 EventGenesis `0.942304`, V2E `0.824453`; dt4 EventGenesis `2.963737`, V2E `2.858816` px per interval. The four restored full two-component outputs were bitwise equal under deterministic inference. No retraining.
- Stage 2 pose EPE values are invalid because selected event windows ended at the rollout tail instead of their associated label time. The repaired collector now stores `sample_id`, `rollout_id`, start/end, label time and event count; independent recounts match two distinct windows and exact boundary timestamps.
- The corrected pose pilot's standard eval-mode synthetic validation EPE was `35.79 px` (EventGenesis) and `245.00 px` (V2E). A one-sample diagnostic fit coordinate loss from `10.58` to `0.19`, but eval-mode decoding remained unstable while train-mode batch statistics changed outputs sharply. These small pilots are implementation diagnostics, not generator or E-POSE conclusions.

## Data and evaluation state

- HQF: actual `bike_bay_hdr` ROS bag parsed into EVREAL MemMap arrays: 19,763,043 events, 2,431 mono8 frames, 180×240, event polarity 0=OFF/1=ON. Both synthetic-only candidate checkpoints ran through the full sequence. Reference image pixels were not used as model input and no GT score was computed.
- MVSEC: original `indoor_flying1` data/GT and calibration are retained in WSL with hashes. A lossless fixed one-second slice is on g1: 7,294 events, 41 APS endpoints, 21 GT flow maps at 346×260. The preregistered fragment has 31 dt1 windows (`i→i+1`, mean 31.859 ms) and 28 dt4 windows (`i→i+4`, mean 127.437 ms). Pinned STE GT timestamp propagation was used. Per-window EPE was emitted under a provisional, explicitly unfrozen inspection mask; no aggregate or model choice was made.
- E-POSE: DOI metadata resolves to the 29,759,153,475-byte public record; the official landing page returned an AWS WAF challenge (HTTP 202), and the article API returned HTTP 403. No file-level manifest, raw sample, matching CAD, K, class record or pose annotation was obtained.

## Fairness and isolation

EventGenesis and V2E use the same six physical rollouts, labels, rollout split, paired initialization, architecture and synthetic budgets within each task. Training and checkpoint selection remain synthetic-only. Candidate inference inputs are event-only. The real samples and sequence were registered before weights were loaded; no real score changed a generator or checkpoint choice. No frozen-core file was edited. GPU3's other-user CARLA workload was left untouched.

## Handoff files

- Single-file research handoff: [`STAGE3_RESEARCH_HANDOFF.md`](STAGE3_RESEARCH_HANDOFF.md).
- Detailed execution report: [`reports/stage3_execution.md`](reports/stage3_execution.md).
- Stage 3 run/evidence index and external data/checkpoint paths: [`evidence/S3/20261016-stage3/`](evidence/S3/20261016-stage3/).
- Stage 2 historical report and independent review: [`reports/stage2_execution.md`](reports/stage2_execution.md), [`reviews/stage2_2834540/REVIEW.md`](reviews/stage2_2834540/REVIEW.md). The old flow and pose EPE are clearly marked invalid and retained only as history.

The formal real evaluation protocol remains a research-review decision. Do not interpret fixed-sample pipeline scores or the short synthetic pilots as sim-to-real performance evidence.
