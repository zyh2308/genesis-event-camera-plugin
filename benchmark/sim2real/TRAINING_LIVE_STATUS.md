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

<!-- three-line-update-20261107 -->
> 2026-11-07 核查更新：重建正式训练15/15，HQF五方法三seed评测已启动；光流12/15，剩余任务继续且MVSEC输入续传已恢复。位姿两轮修复短训均完成，但都未过全条件采用门槛，真实E-POSE未重测。详见[完整结果与证据](reviews/three_line_update_20261107/README.md)。以下保留历史记录。

# 2026-11-03 新版三线重训

当前部署：45 个正式训练作业（3 任务 × 5 管线 × 3 seed）已排队；仍处于新版事件数据生成阶段。正式训练尚未开始，实际短检与恢复通过后自动继续完整预算。真实评分关闭。

核心为 release 62da4f35ce3c5c96f04268bd1687a27657af9238；GPU3/4 在本轮作业结束后保留预约及低负载占位。此处是部署快照，服务器 CAMPAIGN_LIVE_STATUS.json 才是机器实时状态。

[部署报告与代码](campaigns/newcore_20261103/README.md) · [不可变部署证据](campaigns/newcore_20261103/DEPLOYMENT_VERIFIED_SNAPSHOT.json)

---

以下为历史状态，不代表新版训练完成。

# Stage 4 three-line training and evaluation status

## 2026-11-01 E-POSE completion update

The Stage 4 records below are historical snapshots, not current process status. Current three-line state is maintained in [sim2real_status.md](sim2real_status.md). The [matched-CAD E-POSE pilot](reviews/epose_diverse_short_20261101/README.md) completed 2,400 additional updates per arm and a fixed 200-window real test; strict ADD-S success was 0% / 1%, so real pose quality remains insufficient. Frozen core unchanged; CPU leases released.

**Latest direct g1 snapshot:** 2026-10-18 17:43:00 UTC / 2026-10-19 02:43:00 JST. Earlier Stage 3 and Stage 4 snapshots remain historical. Training jobs stay detached on g1; this page records the actual observation below, not an inferred completion estimate.

| Line | Latest verified state | Status | Next checkpoint / delivery |
|---|---|---|---|
| E2VID+ reconstruction, EventGenesis | Latest appended optimizer record **721 / 12,000**, epoch 13 / batch 1, `2026-10-18T17:42:58.133184Z`; TC valid calls 76 and 30 finite gradient tensors. | Running on g1 GPU 2; detached tmux pane/Python PID 246943/246981; resumed from saved update 20. `last.pt` snapshot SHA-256 `b8de3a5483a267d757779a1415528e6132d70acecc2b7a68b035163abff3e7ee7`, 171,427,906 bytes. Latest optimizer log hash `5d111500024d41f248085d371be23e82fe79085557f906a326f57ceb566add42`. | Continue fixed budget and record synthetic validation. |
| E2VID+ reconstruction, V2E | Latest appended optimizer record **729 / 12,000**, epoch 13 / batch 9, `2026-10-18T17:43:00.417939Z`; TC valid calls 76 and 30 finite gradient tensors. | Running on g1 GPU 4; detached tmux pane/Python PID 246948/246982; resumed from saved update 20. `last.pt` snapshot SHA-256 `48e18fccb5c3f9f4e73a52c2b9f6bbc2076987bc05754abb90c7a772a56e1247`, 171,427,906 bytes. Latest optimizer log hash `2cd988af92c742d1ab350d81fc39618e3f07b45dc7dd8342357db92a456be55c`. | Continue fixed budget and record synthetic validation. |
| STE-FlowNet dt1, EventGenesis | 45/45 epochs, **8,910 / 8,910** updates, final status `complete`, per-arm resumed trainer exit code `0`; synthetic-best epoch 33 / EPE `0.2832538118` px/30 ms. | Complete; real MVSEC score closed. | Final-best fixed synthetic diagnostic and 45-epoch curve archive complete. Await protocol freeze before real scoring. |
| STE-FlowNet dt1, V2E | 45/45 epochs, **8,910 / 8,910** updates, final status `complete`, per-arm resumed trainer exit code `0`; synthetic-best epoch 20 / EPE `0.2828732346` px/30 ms. | Complete; real MVSEC score closed. | Same as EventGenesis. |
| SingleShotPose / E-POSE | No E-POSE training process. Dataset publisher recheck and synthetic eval-mode confidence/BN mechanism check completed. | Waiting on original event/pose sample and exact sample-matched CAD; no formal task train or real inference. | Resolve public access, then sample contract and held-out synthetic normal-eval gate. |

## Reconstruction seed-0 live run

- Run root: `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2/{eventgenesis,v2e}/`; detached jobs use CUDA-visible physical GPUs 2 and 4. At the latest query each reserved about 3.2 GiB; GPU3 was idle after the flow diagnostic.
- Target: 200 epochs × 60 updates = **12,000 updates per generator**, with the frozen corrected-v2 reconstruction recipe and synthetic-validation LPIPS-VGG checkpoint selection.
- Data: `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/e2vid_corrected_10ms_20261019_v1/`; adapter manifest SHA-256 `06ecadb1b4e90bc434ce30148c5d50704385cdab15a24e88ff9366f7c58edcb3`. Shared source bank SHA-256 `cca882ba912fda3b3243d49bc7e29e28186046e3b6591c424f385b0506047884`.
- g1 environment: Python 3.11.15, PyTorch 2.5.1+cu121, CUDA build 12.1, Tesla V100S-PCIE-32GB. E2VID source bundle is pinned to event_cnn_minimal commit `78aa471a66593f55e349303563997297fbc816e8`; trainer SHA-256 `73cc8a1bfc2d8a4cdcf86e32ca4bdeaa0091239e93f499108aaac02cf7da2d58`.
- Seed-0 paired initial state SHA-256: `48426d94f4159592155eda873d4e71e2f06439c10eade990de5979231ed07a0f`. Update-20 checkpoints: EventGenesis `cb4f4000b435195cff1530bfc0600162ae5cbb689eb4ec39f5e57b156f6de9a0`; V2E `776ee6b18d984fe566f36360e9d83c7460b6050a076608fe12c4ebdd9aeb484b`; each 171,427,458 bytes.
- Actual resume oracle passed on each arm: the saved cursor reproduced the next two clip indices `[55,4]`, identical content hashes and the same two forward loss totals as the independent probe; optimizer counters advanced from 20 to 21. This checks boundary restore and the next update, not strict bitwise equivalence over a whole training trajectory.
- TC ran during the 40-frame clips (76 valid calls per optimizer update in the recorded case), its loss was separately logged, and a direct autograd probe measured nonzero TC-to-model gradient norms `0.0239571821` / `0.0239218361` on EventGenesis / V2E. The shared crop/flip preflight passed four fixed seeds for event tensors, images, flow-vector spatial channels and validity. Recurrent preflight passed for both models: state continued within a clip and the reset first-step output matched exactly.
- First v1 loader attempt and its failure remain in `evidence/S4/20261019-e2vid-parallel-launch/attempt_v1/`; it exited before training because the upstream dataset returned zero validation sequences at an exact-length boundary. The v2 adapter admits exactly the registered 200 samples and asserts IDs `000..199`. The original 30 ms optical-flow bank is untouched.
- Compact copied launch/resume evidence and its hash index: [`evidence/S4/20261019-e2vid-parallel-launch/`](evidence/S4/20261019-e2vid-parallel-launch/). Full and growing stdout/JSONL logs and weights stay at the g1 paths above. No real test is used.

At 17:43:00 UTC, both launcher panes and Python trainers were live. GPU 2 and GPU 4 each used 3,252 MiB of 32 GiB; utilization was 76% and 0% at the instant of the query. GPUs 0/1/3/5 were effectively idle. Append-only metrics had reached update 721 (EventGenesis) and 729 (V2E); both latest rows recorded finite gradients and 76 valid TC calls. The `live_status.json` files still reported update 720, so the append-only optimizer log is the authoritative step count for this snapshot. Current checkpoint and log hashes are in the timestamped observation and evidence index; no checkpoint cursor was inferred from its filename.

### Recheck live progress

Run from g1 in the `Eventbased_WAM` conda environment:

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Eventbased_WAM
python3 - <<'PY'
import json, pathlib
root=pathlib.Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2')
for arm in ('eventgenesis','v2e'):
    status=json.loads((root/arm/'live_status.json').read_text())
    print(arm, status['optimizer_update'], '/', status['optimizer_budget'], status['updated_utc'], status['status'])
PY
```

Full-run commands are the recorded `resume_stdout.log`'s launcher invocation, equivalent to:

```bash
CUDA_VISIBLE_DEVICES=2 python -u benchmark/sim2real/scripts/train_e2vid_stage4.py \
  --model-root /home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/event_cnn_minimal \
  --data-root /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/e2vid_corrected_10ms_20261019_v1 \
  --config benchmark/sim2real/configs/stage4_corrected_v2_flowdt1_seed0.json \
  --output-root /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2/eventgenesis \
  --generator eventgenesis --device cuda:0 --seed 0 \
  --resume /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2/eventgenesis/last.pt

CUDA_VISIBLE_DEVICES=4 python -u benchmark/sim2real/scripts/train_e2vid_stage4.py \
  --model-root /home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/event_cnn_minimal \
  --data-root /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/e2vid_corrected_10ms_20261019_v1 \
  --config benchmark/sim2real/configs/stage4_corrected_v2_flowdt1_seed0.json \
  --output-root /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2/v2e \
  --generator v2e --device cuda:0 --seed 0 \
  --resume /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage4-e2vid-seed0-corrected-v2/v2e/last.pt
```

Use these commands only to reproduce in a separate authorized run directory; do not duplicate the active processes.

## Optical flow final delivery

- Full 45-epoch curve (90 CSV rows, one row per arm/epoch), exit receipts, checkpoint hashes, external log hashes and reproduction command: [`evidence/S4/20261019-flow-final-archive/`](evidence/S4/20261019-flow-final-archive/).
- Final-best diagnostic JSON and retained initial failed-load attempt: [`evidence/S4/20261019-final-best-motion-diagnostic/`](evidence/S4/20261019-final-best-motion-diagnostic/). The diagnostic compares both synthetic-best weights on the same 64 validation windows and shared event-active-union mask. Overall full-valid improvement over zero flow is 0.20% / 0.43%; common active-union improvement is 14.0% / 10.5% at 0.487% support coverage. This indicates limited event-supported motion learning, not a real-data result.
- MVSEC dt1 protocol and evaluator remain candidates; `real_scores_authorized=false`. Controlled metric self-test passed; real scoring is blocked pending research freeze and exact data manifest hashes/index registration.

## Pose data and inference readiness

- E-POSE official record recheck: HTTP 202 landing and HTTP 403 API from the current WSL route. Author `wood_block` mesh is verified but not linked to an original event/pose sample; sample-level K, time, transform convention, unit, exact CAD scale and symmetry remain unaudited.
- SingleShotPose's short old budget supplied only 8 confidence-supervised updates after its 15-epoch warmup. A synthetic-only fixture subsequently showed stable ordinary `model.eval()` confidence-argmax inference after added confidence updates and train-only BN buffer recalibration, with normal save/restore. Held-out rollout and E-POSE results are still missing.
- Details and public-access evidence: [`evidence/S4/20261019-pose-eval-access/POSE_EVAL_AND_ACCESS_AUDIT.md`](evidence/S4/20261019-pose-eval-access/POSE_EVAL_AND_ACCESS_AUDIT.md).

Full three-line status and evidence interpretation: [`sim2real_status.md`](sim2real_status.md) and [`STAGE4_PARALLEL_EXECUTION_20261019.md`](STAGE4_PARALLEL_EXECUTION_20261019.md).
