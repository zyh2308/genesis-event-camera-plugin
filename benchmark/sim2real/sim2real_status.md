<!-- three-line-update-20261107 -->
> 2026-11-07 22:20 JST 固定核验：重建共同近黑退化已用配对短训定位到线性输出/时序损失的稳定性问题。启用共同原生sigmoid，保留原损失；120步65/66的失败保留，统一前5epoch/300步检查66/66通过，额外三基线seed2检查通过。新15组统一补跑已启动，首组update20→21恢复通过；旧HQF表全部保留，新真实评分尚未启动。光流、位姿原任务未修改，继续独立推进。详见[原因、合成证据与新补跑状态](reviews/reconstruction_repair_20261107/README.md)。以下保留历史记录。

# 2026-11-03 新版三线重训

当前部署：45 个正式训练作业（3 任务 × 5 管线 × 3 seed）已排队；仍处于新版事件数据生成阶段。正式训练尚未开始，实际短检与恢复通过后自动继续完整预算。真实评分关闭。

核心为 release 62da4f35ce3c5c96f04268bd1687a27657af9238；GPU3/4 在本轮作业结束后保留预约及低负载占位。此处是部署快照，服务器 CAMPAIGN_LIVE_STATUS.json 才是机器实时状态。

[部署报告与代码](campaigns/newcore_20261103/README.md) · [不可变部署证据](campaigns/newcore_20261103/DEPLOYMENT_VERIFIED_SNAPSHOT.json)

---

以下为历史状态，不代表新版训练完成。

# Public-data Sim-to-Real status — 2026-11-01 (Asia/Tokyo)

**New E-POSE real pilot (2026-11-01):** [Paired short training, fixed 200-window real test, demo and complete audit](reviews/epose_diverse_short_20261101/README.md). Both models pass synthetic localization, but strict real ADD-S success is only 0% / 1%; this does not establish a generator advantage.

**Latest handoff:** [ReplicaCAD textured-scene pilot and paired short training](STAGE8_TEXTURED_SCENE_SHORT_TRAINING.md), with [raw run evidence and hashes](evidence/20261101-stage8-replicad-dynamic-v3-cpucompat/evidence_index.json). The Stage 8 metric/pose progress and Stage 7 follow-up remain preserved at [Stage 8 metrics and pose](reviews/stage8_20261101/METRIC_AND_POSE_PROGRESS_20261101.md) and [Stage 7 follow-up](reviews/stage7_20261031/STAGE7_FOLLOWUP_HANDOFF_20261031.md); older snapshots are unchanged.

**Latest recorded g1 GPU check for flow:** `2026-11-01T07:32:30Z`, after the Stage 8 short run. GPUs 0–2 reported 0% utilization and 1–2 MiB memory; GPUs 3–5 were active. This is a point-in-time observation, not a reservation or guarantee of availability. The Stage 8 pair itself ran on CPU because no unowned GPU was available after its bounded wait.

| Line | Verified state | Next step |
|---|---|---|
| Reconstruction — E2VID+ / HQF | Existing seed-0 HQF scores cover 14 sequences / 15,498 frames and remain report-only. EventGenesis has lower LPIPS-Alex on 13/14 sequences; V2E has better PSNR/MSE on 13/14. Both show block/grid artifacts. Synthetic loader, state reset, and padding checks are recorded; the artifacts are not explained yet. No HQF re-score or retraining in this follow-up. | Use a fixed synthetic fixture to isolate remaining train/eval differences, then proceed with shared rich-scene paired training when registered and resourced. HQF stays the final real test; HS-ERGB stays an extension. |
| Optical flow — STE-FlowNet | Old 45-epoch / 8,910-update pair is unchanged. The Stage 7 self-supervised diagnostic and Stage 8 four-scale photometric oracle are preserved. A new ReplicaCAD pilot rendered two furnished layouts with depth layers and a moving TV stand; paired V2E used the same radiance frames. The separate GT-supervised short run completed 100 updates per arm with exact checkpoint restore/eval checks. Both arms fit the one training rollout, and on the one independent rollout their active-region EPE modestly beat zero flow (2.638 / 2.600 vs 2.715 px per 30 ms); direction cosine was 0.987 / 0.993. Predicted active magnitude was about 5.1 px while validation GT active mean was 2.721 px; training GT active mean was 5.729 px. This exposes a large motion-scale difference between these two layouts. Full-valid EPE improved more, but includes the mostly static region. CPU run took 2 h 47 min; no real data or score was used. | Expand the shared bank with multiple pre-registered rollouts and motion scales across the layouts, then repeat only a bounded paired diagnostic. Do not start a long training run or open MVSEC scoring from this two-rollout pilot. |
| Pose — SingleShotPose / E-POSE | The matched textured YCB/E-POSE wood-block paired pilot completed on g1 CPU: 2,400 additional full-network updates per arm, identical samples/augmentation/labels, both process exit codes 0. Independent source/data hashes, optimizer/RNG recovery and synthetic/real metrics passed. Synthetic validation (4 rollouts / 16 windows) met the registered localization gate: center EPE 5.44 / 13.81 network pixels. Frozen weights then scored all 200 registered native windows from real block/r_05_gl: corner EPE 91.61 / 61.11 native pixels, mean ADD-S 14.93 / 17.63 cm, ADD-S success 0/200 / 2/200; valid PnP 200/200 for both. Real pose quality remains weak and there is no overall generator-superiority claim. [Full report and evidence](reviews/epose_diverse_short_20261101/README.md). | Preserve this pilot; expand shared position, scale, independent-view and truncation coverage before a new fixed-budget study. Post-test diagnosis found 53% of real centers outside the actual augmented training range and 45% of GT projected boxes not fully in frame; this is a hypothesis, not a unique root-cause proof. r_01_gl and r_05_gl are seen; use reserved real sequences for later confirmation. |

The Stage 7 flow run remains indexed at [`evidence_index_20261031.json`](evidence/20261031-stage7-followup/evidence_index_20261031.json). The Stage 8 flow oracle logs, result, and failures are indexed at [`EVIDENCE_INDEX.json`](evidence/20261101-flow-photo-oracle/EVIDENCE_INDEX.json). The textured-scene paired run is indexed at [`evidence_index.json`](evidence/20261101-stage8-replicad-dynamic-v3-cpucompat/evidence_index.json); the 580 MiB checkpoints remain on g1. The pose run logs and remote checkpoint metadata remain indexed at [`evidence_index_stage8_pose_20261101.json`](evidence/20261101-stage8-pose-short/evidence_index_stage8_pose_20261101.json); the 386 MiB weights remain on g1. Frozen simulator core remains `e79e656ac00f7847b98db9e12bf43b093f3afff1`; MVSEC scoring remains closed. The E-POSE r_05_gl pilot is now scored, and r_01_gl/r_05_gl are recorded as seen; no model or preprocessing was changed after this score.
