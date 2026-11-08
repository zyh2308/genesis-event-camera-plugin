<!-- pose-advantage-and-failure-20261109 -->
# 2026-11-09 02:46–02:47 JST 三线与位姿短检

- 位姿：软聚焦停止；全量诊断发现暗光28/192次真正脱靶峰，集中于同一轨迹。中心影响平移与旋转裁剪，深度另有限制。可见mask辅助两方法三seed各288更新已完成、退出0，但仅29/36＋11/12检查通过，不采用、不新增真实E-POSE或长训，原正式30＋8不变。训练后1152配对峰检查和独立CPU2304图复核通过：暗光脱靶峰仅28→26，完整位姿20.31%→18.75%。
- 优势复验：32条新共享物理轨迹已完成，GPU4准备三光照两生成器事件；完成后自动接固定RAW C1评价。正常/半亮为事前登记primary，最暗完整报告；尚无新优势结论。
- 重建：4/15完整训练完成（EG三seed与V2E seed0各12000步），3组运行、8组等待；HQF三条为已见开发子集，待全部15组冻结后评价。
- 光流：15/15训练完成，每组8910步；固定MVSEC真实矩阵10/15完成、2运行、3等待，未读部分新真实分数调本轮候选。
- [本轮结果、完整数据与实际证据](reviews/pose_advantage_and_failure_20261109/README.md)；[后续操作边界](reviews/pose_advantage_and_failure_20261109/NEXT_ACTIONS.md)。

以下保留此前记录；当前状态以上述时间戳和对应实际收据为准。

---

<!-- pose-readout-control-20261109 -->
# 2026-11-09 位姿零更新配对结果

- 位姿：完整64窗×两生成器×三seed×三光照完成（1152配对），基线及full进程均exit0；0次更新，原模型/数据不变，独立CPU重算1152概率与2304完整pose通过。软聚焦机制门槛仅27/36通过，9失败，不采用；未开独立confirmation、长训或真实E-POSE，原30+8不改且未重测。
- 重建：01:11 JST快照4/15完整训练完成，3组继续、8组等待；EventGenesis三seed与V2E seed0均12000步完成。HQF三条开发序列评分仍待全部冻结。
- 光流：01:17 JST快照15/15训练完成，固定MVSEC真实评分8/15完成、2运行、5等待，未读新真实成绩调参。
- [完整数据表、实际收据、源代码和独立核验](reviews/pose_readout_control_20261109/README.md)

以下保留此前各时点记录；当前状态以本条及对应快照为准。

---

<!-- training-and-pose-diagnostic-20261109 -->
# 2026-11-09 00:39 JST 核查

- 重建：当前完整预算队列3/15完成、3运行、9等待。EventGenesis三个seed均200轮/12000步完成，最终退化检查均通过；V2E三个seed为11720、780、990/12000。新HQF三序列开发子集评价仍等待全15组冻结，不等同HQF14。
- 光流：15/15正式训练完成，每组8910步；固定MVSEC真实测试7/15完成、2运行、6等待。当前两个模型分别7000/8500、6000/8500窗，进程和心跳正常，未读取新真实成绩调参。
- 位姿：前景支持、注意力和尺度固定合成诊断完成，0次optimizer更新；576预定attention图独立核验通过。仍未确定单一训练根因，未开新训练/真实E-POSE评分，原30+8门槛不变。
- [重建与光流最新服务器证据](reviews/training_progress_20261109/README.md)
- [位姿诊断、源码与独立审计](reviews/pose_support_attention_scale_20261109/README.md)

以下保留此前各时点记录；当前状态以上述时间戳和对应快照为准。

---

<!-- pose-factor-review-20261108-final -->
# 2026-11-08 23:23 JST 研究核验

- 位姿：拆分中心、深度与旋转裁剪；深度投影21/30、BAF中心20/36、RAW教师BAF中心19/36，三个候选均未通过，不采用；正式短训均exit0、各6×288更新。原整体30+8不放宽，真实E-POSE评分关闭，无新增位姿训练排队。
- 光流：Census短训3/12，未通过、不采用；原冻结15组MVSEC评分已完成7组，两进程继续，未读新真实分数调参。
- 重建：g3全15组队列继续；EG seed0完整12000步，V2E seed0约9800、EG seed1/2约10960/11140步，新的完整配对HQF表尚未完成。选定三条HQF仍属开发子集。
- [研究分析、数据表与实际机器证据](reviews/pose_factor_illumination_20261108/README.md)
- [下一阶段固定窗口诊断](reviews/pose_factor_illumination_20261108/NEXT_RESEARCH_ACTIONS.md)

以下保留此前记录；当前进度以上方时间戳及所链接快照为准。

---

> 2026-11-08 20:50 JST 核查：[三线运行状态](reviews/flow_pose_status_20261108_2050/README.md)。位姿slow_features短训完成、退出0，18/30光照检查未通过，不采用、不新增真实评分；光流Census仍在四臂配对短训，MVSEC固定矩阵已完成4/15组；g3重建三队列继续，无全局失败。下方保留各时点快照。

> 2026-11-08 本轮新进展：[光流光度方向与位姿光照短检](reviews/flow_pose_short_20261108/README.md)。原采样加权未采用；实际诊断不支持原photo全局误导，Census共享TRAIN校准与四次更新/恢复通过，200步四臂短训运行。位姿soft retention完整22/30未采用，输入/时间/几何与饱和审计通过；slow_features双学习率12次更新/恢复通过，新12×288短训运行。g3重建完整训练不变；新候选真实评分关闭，既有MVSEC固定矩阵继续。完整15×528合成分解表已补齐，优势范围与失败条件均如实保留。下方为历史快照。

> 2026-11-08 18:32 JST 空卡追加：[九卡实际部署](reviews/parallel_priority_20261108/README.md)。g3三张4090重建完整预算继续；g1 GPU3光流采样短训、GPU4位姿冻结features短检、GPU2/5固定MVSEC；新空出GPU0/1两张3090已安全准入，补跑五方法三seed完整528窗的只读合成分解表，不改变原训练。位姿新候选EG seed0/1各两分支完成288更新，全30+8结果尚待完成。

> 2026-11-08 18:19 JST 最新执行：[三线并行部署与证据](reviews/parallel_priority_20261108/README.md)。重建按最新授权恢复五方法×三seed完整预算，在g3三张4090运行，156/156门禁通过；真实范围仅事后选定HQF三序列开发子集，不替代HQF14。g1 GPU3运行光流采样配对短训，GPU4运行位姿冻结features启动检查及后继短检；上一轮位姿20/30仍未采用，新E-POSE评分关闭。原MVSEC固定矩阵在GPU2/5继续，未用其新分数选择优化。以下保留历史快照。

> 2026-11-08 最新核验：[长连接恢复与三线进展](reviews/recon_photometry_detail_20261108/RECOVERED_CONNECTION_PROGRESS.md)。L1=0.25配对短训已完成：EventGenesis 9/9、V2E 8/9，仍未共同通过，不启动重建长训。光流15/15训练完成，MVSEC正确室外数据已完整校验；固定四序列8500窗真实测试改为GPU2跑seed0/1、GPU5跑seed2，每seed全部五方法同卡。位姿门槛仍14/30，新真实评分关闭。下方为保留的时间快照。

> 2026-11-08 VPN恢复后的最新部署：[首轮重建完整结果](reviews/recon_photometry_detail_20261108/FIRST_SHORT_TRAIN_RESULT.md)；[L1=0.25配对短训](reviews/recon_l1_quarter_20261108/README.md)在GPU2运行；[MVSEC四序列8500窗/15权重正式测试](reviews/mvsec_final_preparation_20261108/README.md)已在GPU5启动，最终真实成绩待完整矩阵完成。位姿光照门槛仍14/30，新真实评分关闭。

> 2026-11-08恢复核验：[六分支完整短训结果](reviews/recon_photometry_detail_20261108/FIRST_SHORT_TRAIN_RESULT.md)。EventGenesis光度/感知/细节均改善，V2E感知与细节保护失败，共同配方不采用；下一轮仍限短训。光流15/15完成，MVSEC正确GT重传中；位姿门槛14/30。以下记录以各自时间为准。

> 2026-11-08 最新研究核查：[三线状态与短训](reviews/recon_photometry_detail_20261108/CURRENT_STATUS.md)。光流15/15训练完成；位姿光照门槛14/30；重建300步配对优化已部署但完整分数未取得。g1连接重置及一次重试超时后停止连接，当前短训状态未能刷新。以下旧记录以其自身时间为准。

<!-- reconstruction-activation-budget-20261108 -->
> 重建输出激活×训练进度的合成短诊断已完成：两方法、seed0、两激活、每臂600步；没有新增真实评分或长训。所有配对检查通过，GPU预约已释放。详见[完整诊断结果](reviews/recon_activation_budget_20261108/README.md)。

<!-- hqf-repair-20261108 -->
> 2026-11-08 01:58 JST：两组seed0、第4,800步HQF真实开发测试已完成，未新增训练；全部14序列、每组15,498帧，GPU预约已释放。前后结果保留旧权重预算差异，不作为严格修复因果量或多seed排名。详见[真实测试结果](reviews/hqf_repair_pair_20261108/README.md)。

<!-- three-line-update-20261107 -->
> 2026-11-08 01:12 JST 更新：本轮重建仅做退化修复验证。20步因果消融、300步三seed的66/66检查及三基线seed2短检已完成。15组完整补跑已按用户澄清收缩：13组未启动任务禁用，当前两组安全保存并停止，GPU0/1预约释放。光流、位姿未改动；未做新真实评分。详见[范围调整与安全停止](reviews/reconstruction_repair_20261107/SCOPE_REDUCTION_20261108.md)。以下保留历史记录。

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
