# Stage 9 位姿线只读核查

2026-11-05。依据本地仓库与既有导出资料；未SSH、未使用GPU、未改代码、核心、数据或训练。服务器最新进度由主会话另行核验，以下不得当作实时完成矩阵。

## 当前可确认内容

- 正式新版core为release `62da4f35ce3c5c96f04268bd1687a27657af9238`。Stage9计划是5管线（EventGenesis、V2E、Vid2E、FILM+ESIM、RIFE+ESIM）×3seed，每run30000次optimizer更新、batch8，fresh shared random initialization。不是旧v10短训暖启动。
- 本地顶层sim2real_status/TRAINING_LIVE_STATUS仍写11月3日“正式0步”，已经陈旧。`reviews/native_auto_precision_20261105/RESULTS_REVIEW.md`第5节明确记录EventGenesis seed0位姿30000完成；其余14个run未由本次本地核查证明完成，须查服务器completion/log/checkpoint。
- 物理bank为匹配YCB036木块CAD的128条200ms轨迹、96train/32validation，native100Hz端点，前40ms预热。实际recovery adapter统一剔除CAD中心出画9窗，最终1529train/510validation，而非部署README原1536/512。剔除依据只依共享合成标签，不依事件数或生成器成绩。
- `pose_physical_geometry_audit.json`对640帧通过：GT PnP最大平移回算误差约1.31e-7m；CAD/hash和物理投影已有检查。不需要回退旧无纹理立方体或修改冻结core。

## 实际训练入口和选权重规则

服务器campaign根为`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1`（下称R）。

- trainer：`R/recovery_20261103/train_pose_stage9.py`。
- wrapper：`R/recovery_20261103/run_pose_training_job.py`。
- data adapter：`R/recovery_20261103/build_pose_adapter.py`。
- selection：`R/recovery_20261103/pose_window_selection.json`。
- actual recipe：`R/sources/training_recipe.json`；按run metadata/receipt SHA核对。
- data：`R/adapters/<method>/pose/adapter_manifest.json`，raw events来自`R/pose_bank/events/<method>/manifest.json`。
- outputs：`R/training/pose/seed<0|1|2>/<method>/`。

实际trainer用全帧事件图，9点坐标损失加中心网格CE；普通`model.eval()`下按预测confidence logits argmax选点。GT cell只出现在监督损失，不进入推理。合成增强共享scale/translation策略，中心出画则统一回退该次增强。

**实际synthetic_best选择是等rollout的mean_keypoint_epe_px，不是早期planning JSON写的ADD-S。** 不能为了正式真实分数好看，训练后改checkpoint选择为另一指标；应确认actual training_recipe是否已在首次正式更新前登记，并如实报告这个共享适配。

preflight20→独立进程resume21检查Adam计数、初始化hash及恢复预测，然后完整预算。每个成功completion声明30000更新、last和selected checkpoint SHA、recipe SHA、未用真实训练或评分。查进度应优先看completion+追加metrics+实际checkpoint cursor，不能只看live_status文件名。

## 真实数据与历史暴露范围

旧core v10、r05的200窗初测仅为历史结果：EventGenesis ADD-S成功0/200，V2E2/200；两组最终有效PnP均200/200。它不是新版Stage9正式训练结果，不能与新表混用。

r01_gl、r05_gl已见，必须与其余木块序列分开。旧post-test覆盖诊断不能再拿这两段调参后称新盲测。最终应标“E-POSE wood-block subset”，不是全E-POSE所有物体。

旧真实根目录公式为`P/data/real/epose/figshare_28106717_v1/extracted/block/<protocol real_sequence>`，旧real_sequence本身含`block/r_05_gl`，实际可能出现嵌套block目录；须按现有raw manifest核对，不猜目录层级。

旧`independent_verification.json`只对r05的5个固定packet核对MAT/raw HDF5事件坐标极性计数一致，不是所有真实完整序列已审计。不得将此局部检查泛化为全测试输入资格。

## 目前真实评价的具体缺口

本地已有`R/recovery_20261103/real_evaluation_contract.json`，但`score_launch_enabled=false`，pose的exact未见完整序列清单仍写“freeze before scoring”，不是已经固定的样本manifest。

`freeze_evaluation_checkpoints.py <campaign> pose`要求该task全部5方法×3seed的15个formal completion、共同seed初始化和recipe SHA合格，才写pose_frozen_checkpoints；输出仍`scoring_authorized=false`。不能因为某一组先完成而绕过此gate直接开分数。若研究侧决定先交付seed0完整配对评估，应先形成明确的protocol amendment，不能默默改脚本。

现成`evaluate_diverse_epose.py`是旧v10 CPU初测，硬编码两组+旧v1control、固定200个r05窗口、读取MAT['events']并`assert len(samples)==200`。它不是Stage9正式完整序列runner。

可以复用：事件编码、同坐标域去畸变、confidence-only decoder、固定EPNP、ADD/ADD-S、已知1cm平移的独立oracle。

必须补：

1. 全部注册木块完整序列的只读manifest、raw HDF5/MAT时间配对、已见/未见分组、数据hash及固定窗口定义，空事件窗口不能按方法过滤。
2. 原生raw全帧事件loader，MAT pose/K仅做时间对齐与评分；不读取有几何问题的官方keypointprojections作标签，不用GT crop/mask/box或GT cell。
3. 加载Stage9 completion指定的synthetic_best及SHA，支持5method×3seed、同一预处理与PnP约定。
4. ADD-S成功0.1D与全样本分母、无效PnP失败计数；ADD-S AUC必须在评分前写明积分区间和单位，不能自行混同0.1m与0.1D；对称旋转约定也须明确。旧脚本只有raw rotation和单阈值success，没有AUC及对称旋转。
5. per-sequence/per-seed原始记录与missing/failed run状态，之后做三seed均值、样本SD及paired sequence cluster bootstrap。相邻窗口不能当独立样本。

## 可立即并行推进

CPU无需等训练：冻结真实序列/packet清单与区间、审计CAD/K/单位/去畸变/因果性、准备正式loader、补ADD-S/AUC/symmetry/invalidPnP数值fixture、生成空结果schema和训练曲线表。

GPU阶段：现有未完成训练安全恢复/继续；已完成模型的完整合成普通eval定位与3D/PnP诊断。不能根据真实结果追加训练或选权重。

通过所有评价门禁后，完整真实评估可在空闲GPU加速也可CPU分批推理；模型必须eval，样本清单、seed、检查点先锁定，评分和曲线整理与其余两线并行。

## 位姿需要的结果图表

- 新版5method×3seed训练和synthetic validation曲线，选择点和失败记录。
- 主表：ADD-S@0.1D、预注册ADD-S AUC、平移/对称旋转误差及validPnP比例，已见与未见分组。
- 全序列×seed原始数值与全部分母，三seed均值±样本SD。
- 成功率阈值曲线、按速度/照明/平移旋转序列分组的成对差异；保持未评分单元格缺失，不用旧200窗短训填充。

## 本次优先核验资料

- `outputs/sim2real_retraining_week_20261103/CAMPAIGN_README.md`。
- `.../server_export_20261104/recovery_20261103/{train_pose_stage9.py,run_pose_training_job.py,build_pose_adapter.py,pose_window_selection.json,real_evaluation_contract.json,freeze_evaluation_checkpoints.py,aggregate_paired_real_scores.py}`。
- `.../server_export_20261104/recovery_20261104/repair_contract.json`：Vid2E pose纳秒级事件行顺序修复任务26_0，保留全部值和计数；它仅证明修复协议，是否全bank完成须看服务器receipt。
- `.../timestamp_diagnostics/timestamp_probe_vid2e_pose/result.json`：旧raw出现最大4ns逆序，source feed时间范围正常，不能把此诊断当修复完成。
- 仓库`benchmark/sim2real/reviews/epose_diverse_short_20261101/`：只作历史输入/metric实现和暴露范围证据。
