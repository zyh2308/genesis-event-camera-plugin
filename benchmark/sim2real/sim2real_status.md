# Sim-to-Real status — Stage 2 training-chain pilot

Updated: 2026-10-16 (Asia/Tokyo). Branch: `exp/public-sim2real-v1`. The Stage 2 report and indexed evidence are the full record; this file is the handoff summary.

## Repository and execution

- Upstream: `https://github.com/zyh2308/genesis-event-camera-plugin.git`.
- Worktree: `/home/科研/Eventbased_WAM/repositories/gec-sim2real`; execution worktree on g1: `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real`.
- Experiments ran on g1 in conda `Eventbased_WAM`; WSL `Ubuntu-22.04` was used for repository preparation, evidence review, and commits. Model pilots used physical GPU 1 (RTX 3090) sequentially after idle checks. GPU 3 was occupied by another user's CARLA job and was left untouched.
- Frozen simulator core `e79e656ac00f7847b98db9e12bf43b093f3afff1` is unchanged and remains an ancestor of this branch.
- No candidate model was trained, validated, selected, ranked, or tuned on HQF, MVSEC, or E-POSE real test data. No real scores exist.

## Route status

| Route | Source inspection | Actually executed in Stage 2 | Current state and next gate |
|---|---|---|---|
| E2VID+ → HQF | Pinned E2VID/EVREAL source, HQF native loader/evaluator route, 14-sequence inventory | Six-rollout paired adapters; two epochs / 224 updates per generator; independent synthetic validation; R1/R2 fixes and boundary/direction fixtures; checkpoint restore | Synthetic-only chain passes. HQF files/license and candidate E2VID evaluation adapter remain unchecked. HQF/HS-ERGB rationale and metric/aggregation remain protocol candidates for research freeze. |
| STE-FlowNet → MVSEC | Pinned encoder/loss/GT timestamp association and dt1/dt4 reference entry points | Six-rollout paired adapters; dt1 and dt4, two epochs / 16 updates each per generator; synthetic validation EPE, correlation reference check and checkpoint restore; failed attempts retained | Synthetic-only chain passes for both dt values. MVSEC files and GT remain unchecked. Freeze primary mask/support and empty-frame rules before any real score. |
| SingleShotPose → E-POSE | Pinned SingleShotPose and E-POSE annotation source/paper; bundled RGB/mask/crop path rejected for this protocol | Event-only full-frame generic-cuboid paired training/validation pilot, twice; second run uses corrected window selection | Interface/gradient/checkpoint closure passes, but validation EPE is extremely poor and object/CAD is unmatched. Raw E-POSE sample and CAD access/contracts are unresolved; do not scale or replace model without research review. |

## Shared data and contract

- Six 0.3 s rollouts, 31 frames / 30 intervals at 100 Hz; split by whole physical rollout: train `train_00`–`train_03`, validation `validation_00`–`validation_01`.
- Shared bank: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage2/20261016-s2-pilot-v1/shared/shared_trajectory.h5`, 26,526,701 bytes, SHA-256 `68011b77d292ca5600fb10b3464af26c958c785d2258e8a8cfea0db35b15763d`.
- Camera K at 160×128: `[[137.24844291261175,0,80],[0,137.24844291261175,64],[0,0,1]]`; timestamps are seconds from rollout start; world is Genesis z-up / meters; camera is OpenCV x-right/y-down/z-forward; pose labels use the right endpoint; flow is forward source-to-target displacement per endpoint interval. Common validity does not depend on event observations. The pilot lacks target-occlusion z-buffering.
- Paired runs share physics labels, split, initialization, model, optimizer and budget. Inference is event-only; pose uses Genesis pose/K only for training targets. Sensor profiles and the V2E display mapping remain uncalibrated engineering settings.

## Data/evaluator readiness

- HQF: public route, pinned MemMapDataset and `eval.py` checked; no files opened. Candidate: all 14 sequences, continuous inference, no RGB seed, LPIPS-Alex primary with supporting reconstruction metrics. Public size estimates are 19.4 GB compressed / 94 GB uncompressed. License, data manifest and the candidate checkpoint adapter are still open.
- MVSEC: pinned HDF5/GT fields and timestamp helper checked; no files opened. Candidate sequences are three indoor_flying and outdoor_day1 at 346×260, dt1 adjacent endpoints and dt4 four intervals; estimated data+GT total 33.2 GB. Primary support mask/zero-GT/empty-frame rules are not frozen.
- E-POSE: paper fields and annotation source checked, but no raw sample/CAD parsed. Publisher lists 27.72 GB. g1 API returned HTTP 403; landing-page request returned HTTP 202 with empty body. This is an access limitation, not proof the dataset is unavailable. Event timestamp/polarity, K, transform, axes/units, sync, CAD scale and class mapping require one sample-level audit.
- Dataset formats and evaluator entries in `manifests/public_real_data_stage2.json` are source-level inspection. No real evaluator has been exercised against downloaded files.

## Evidence and recommendations

- Full report: [`reports/stage2_execution.md`](reports/stage2_execution.md).
- Stage 2 run logs, commands, environment, start/end times, exit codes, results, failed attempts and data manifests: [`evidence/S2/20261016-stage2-pilot/`](evidence/S2/20261016-stage2-pilot/).
- External g1 data/checkpoint paths, sizes and hashes: [`evidence/S2/20261016-stage2-pilot/external_artifacts.json`](evidence/S2/20261016-stage2-pilot/external_artifacts.json); per-file evidence index: [`evidence/S2/20261016-stage2-pilot/artifact_index.json`](evidence/S2/20261016-stage2-pilot/artifact_index.json).
- Stage 1 review and its audit files were preserved unchanged at [`reviews/stage1_88b0d14/`](reviews/stage1_88b0d14/). `REVIEW.md` SHA-256: `d8f739cf006b1ac2ca7924760868c85feb3267401174baf591468a2c6e974e78`.

Recommended next step: have the research session freeze HQF and MVSEC evaluation manifests, then scale synthetic-only training for those two routes. Keep E-POSE at its raw-sample/CAD and model-contract gate. Do not run real model scoring until the candidate protocols are frozen.
