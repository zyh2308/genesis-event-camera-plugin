# Stage 7 follow-up handoff — 2026-10-31

**更新时间：** 2026-10-31 02:44 JST (2026-10-30 17:44 UTC)
**范围：** 光流自监督短诊断及实时 GPU 复核；重建和位姿的既有状态单独注明。
**仓库：** `exp/public-sim2real-v1`, HEAD at start `63de465b9cca457c0857c7fe3a26bdfbd01e6d27`.
**冻结核心：** `e79e656ac00f7847b98db9e12bf43b093f3afff1` (unchanged).

## 结果摘要

在共享 ReplicaCAD 合成 bank 上，使用实际 STE-FlowNet 和既定自监督目标，EventGenesis、V2E 各完成 100 次 optimizer 更新。任务在 g1 RTX 3090 上运行，退出码 0；权重保存后重新恢复，模型/优化器状态及重复验证输出均一致。真实数据没有进入训练或诊断。

这轮说明模型会响应事件输入，但没有证明它稳定学到了有用的光流：验证时，最终预测在两个独立验证路径的共同事件活跃运动区都劣于零光流基线。损失均值虽下降，EventGenesis 在第 99 次更新出现一次很大的有限值尖峰。没有发现新的、足以判定本次结果无效的确定性接口错误；也没有证据支持启动长程自监督训练。

## GPU 使用与实时资源

- g1 运行前检查时间为 `2026-10-30T17:34:33Z`：6 张卡均为 0% 利用率、1–2 MiB 显存、没有 compute-app 进程；CPU load `0.47/0.65/1.02`，可用内存约 493 GiB。该 run 的 GPU UUID 在正式预约后再次检查，仍空闲，随后启动。
- 运行占用 RTX 3090 `GPU-fd9f1bd4-83c4-cd74-0c88-241db6ec7e23`。EventGenesis 训练用时 134.28 秒，V2E 用时 136.77 秒；两组共 200 次更新。整项 run 从 `17:34:43Z` 到 `17:39:30Z`，约 4 分 47 秒。资源预约在任务退出时释放。
- `17:41:18Z` 与 `17:44:17Z` 的后续实时检查均显示 6 张卡空闲、无 compute-app、项目预约目录为空。最后一次检查记录在 `evidence/20261031-stage7-followup/gpu_live_check_20261030T174417Z.txt`。
- 本次候选模型训练在 GPU 上执行。CPU 仍用于文件、manifest、哈希和已有数据统计等不需要模型计算的工作；这不是把训练留在 CPU 上等待算力。
- 运行时 PyTorch 记录了 CUDA `grid_sampler_2d_backward` 没有确定性实现的 warning（配置为 `warn_only=True`）。因此相同初始化和批次顺序已绑定，但本轮不声称跨运行逐位复现。

## 光流：固定协议、结果和解释

协议已在运行前登记于 `evidence/20261031-stage7-followup/replicad_selfsup_diagnostic_config_v1.json`。它使用已登记的 ReplicaCAD bank manifest `995aac5c0163ef8efacfb1eb2599b02d05538ee8f05bc98104e4e25690522717`：完整布局族 sc0/sc1 的 4 个训练 rollout，独立布局族 sc2 的 2 个验证 rollout；每条 rollout 一个 30 ms 窗口。两组共享同一初始化 SHA256 `acc6bbd11488a21868c5f147c7f397521480166c24914cc7988a6428d6ed71cf`、样本顺序、模型、输入编码和预算。训练只用 STE-FlowNet 四尺度 photometric loss 加 10 倍二阶平滑；合成 GT 没有用于优化。事件为推理输入，GT 只用于合成验证。两组使用同一张从正常配对事件输入预先固定的 `valid GT ∩ event-union ∩ motion` 掩码；扰动输入后没有重算掩码。

这批输入来自实际导入 Genesis 的 ReplicaCAD `v3_sc0/sc1/sc2` 室内布局，不是下载的视频。sc2 的登记预览展示场景外观、depth、30 ms GT flow、EventGenesis/V2E events 和相机平移：[`paired_flow_event_montage_sc2_run16_run21.png`](../../evidence/20261030-stage7-followup/genesis_env_torch28/replicad_layout_runs/v3_sc2_staging_00/paired_flow_event_montage_sc2_run16_run21.png)。生成器输入使用 Genesis float32 scene-linear radiance；预览为了显示将 radiance clip 到 1 后再编码，**不把预览称为原生 HDR**。基于预登记的纹理与 GT 运动区域，sc2 固定验证路径的共享事件 union 覆盖约 `4.92–9.96%`；sc0/sc1 固定路径约 `9.61–17.22%` / `12.53–20.98%`。sc2 的低事件支持是当前可观测性限制之一，但不能据此单独断言其造成了模型误差。对应统计见 [`sc2 coverage`](../../evidence/20261030-stage7-followup/genesis_env_torch28/camera_motion_coverage_audit_v3_sc2_run16_run21.json)、[`sc0 coverage`](../../evidence/20261030-stage7-followup/genesis_env_torch28/camera_motion_coverage_audit_v3_sc0_run14_run17.json)、[`sc1 coverage`](../../evidence/20261030-stage7-followup/genesis_env_torch28/camera_motion_coverage_audit_v3_sc1_run15_run20.json) 和 [`geometry/texture audit`](../../evidence/20261030-stage7-followup/genesis_env_torch28/camera_motion_geometry_texture_audit_v3_sc0_sc1_sc2_run14_run15_run16.json)。这些是相机运动、静态场景的合成样本，不覆盖独立运动物体或遮挡变化。

### 最终更新的独立验证

单位为每 30 ms 端点区间的像素位移。完整有效区和共同事件活跃运动区分别与各自零光流 EPE 比较：

| 组别 / 验证路径 | 全有效区 EPE / 零流 | 共同活跃区 EPE / 零流 | 活跃区预测 / GT 平均幅值 | 方向余弦 |
|---|---:|---:|---:|---:|
| EventGenesis / sc2-X | 3.538 / 3.500 | 4.113 / 2.674 | 5.733 / 2.674 px | 0.788 |
| EventGenesis / sc2-Y | 2.578 / 2.361 | 3.310 / 1.666 | 4.364 / 1.666 px | 0.794 |
| V2E / sc2-X | 2.968 / 3.500 | 3.034 / 2.674 | 4.078 / 2.674 px | 0.766 |
| V2E / sc2-Y | 2.294 / 2.361 | 2.873 / 1.666 | 3.646 / 1.666 px | 0.709 |

共同活跃区占有效区约 10.3%（sc2-X）和 8.0%（sc2-Y）。方向余弦为正，但预测幅值偏大；两组在共同活跃区均未优于零流。V2E 的全有效区误差略低于零流基线，但这不代表其活跃运动区域恢复正确。0/20/40/60/80/100 的完整曲线和分路径数据保存在 `evidence/20261031-stage7-followup/remote_compact/flow_selfsup_diagnostic_v1/training_output/summary.json`。

同一验证输入分别使用正常事件、全零事件和反转五个时间 bin 的事件。零事件输入与反转输入的哈希均变化；反转保留空间支持和每像素/极性的总计数。两组模型输出也随干预变化，说明输入确实到达模型。但最终 checkpoint 下，零事件输入的活跃区 EPE 反而低于正常输入：EventGenesis 为 `1.984/1.535 px`，V2E 为 `2.084/1.373 px`（X/Y）；这些值低于正常事件输入，也低于零流误差。这说明“对事件敏感”不足以证明学到了正确运动，模型先验/偏置仍有明显影响。反转时间 bin 有时也得到比正常输入更低的误差。完整逐项数据见下方的修正后扰动审计结果。

训练损失前 20 次到后 20 次的均值从 EventGenesis `1403.86` 降至 `1150.86`（约下降 18%），V2E 从 `1357.58` 降至 `1034.46`（约下降 24%）。EventGenesis 第 99 次更新的总损失为 `4657.11`、photometric 分量 `4510.17`、梯度 L2 范数 `36495.71`，均为有限值；第 100 次回到 `1025.08`。V2E 本轮最大总损失 `2020.77` 出现在第 2 次更新。全部 100 次更新有有限 loss/gradient，最终 Adam step 均为 100。

### 已证实与未证实

- **已证实：** 正常、清零、反转事件的模型输入确实不同；输出也会改变。两组完整训练执行了 100 次更新；checkpoint 恢复后状态及验证输出相同。共享场景标签与掩码在两组间一致。
- **尚未证实：** 这个短自监督目标是否能在更丰富数据上泛化；错误来自 renderer / radiance、可见性、photometric warp 假设，还是训练不足。GT 监督短试验显示网络在某些路径能改善误差，但 sc2-Y 活跃区未优于零流；这也不能区分模型、监督与观测信号问题。
- **本轮有效性：** 作为预登记的 100-update 诊断，本轮数据和恢复证据有效；其验证表现不支持模型已稳定学会运动，也不替代或重选旧 45-epoch 正式权重。未进行 MVSEC 或其他真实评分。

此前一次输入扰动尝试因沿用 sc2-Y 标签检查 sc2-X 而被标为无效，保留在 `flow_input_perturbation_audit_retry01/INVALID_RUN_NOTE.md`。修正后的 retry02 将每个样本与其自身标签绑定，并要求复核旧的 prediction/EPE/零流哈希后才通过。该结果中正常输入的预测 hash 与既有独立验证一致，EPE 与零流基线误差小于 `2e-5`。使用 retry02，不使用初版或 retry01 作结论。

### 推荐的下一步

不启动另一轮长程自监督训练。先用同一实际四尺度目标对每个固定合成验证窗口计算正确 GT flow、零 flow、反向 flow 对应的 photometric 分量，并结合固定可见区核对 warp 后的灰度残差、纹理和共同事件支持。若真实 GT 在当前 radiance/warp 假设下也不能降低 photometric loss，应先定位数据/损失语义；若正确 GT 的 loss 合理而网络仍不能改善，优先排查优化与输入表示。此诊断不使用真实数据，也不切换训练目标。

## 另外两条线

### 重建 / HQF

本次没有重训或重复 HQF 评分。研究审查已核实现有 seed-0 权重在 HQF 全 14 序列、15,498 帧的结果：EventGenesis 的 LPIPS-Alex 胜 13/14，PSNR/MSE 则 V2E 胜 13/14；预登记图像显示两组都有明显块状/格状伪影，不能把指标差异写成高质量重建。现有训练/评测 loader 数组一致性、200 窗口有限状态、起点重置与画幅 pad/crop 核验已记录在 `reviews/stage7_20261030/ENGINEERING_PROGRESS_20261030.md`。HQF 仅作既有结果报告；本轮未用其分数调整模型、数据或映射。HS-ERGB 仍是未接通的扩展候选，不影响已冻结的 HQF 主评测角色。

### 位姿 / E-POSE

本次没有进行 E-POSE 下载探测、位姿训练或真实评分。已核验样本的投影域 sidecar 及 Brown–Conrady 往返精度仍有效：独立 round-trip 误差约 `6.36e-14 px`，原 MAT 未改；两个样本的旧标签与理想投影均值差约 `16.45/16.41 px`。尚缺匹配 CAD 上的独立合成 rollout 训练、普通 eval 置信度选点/PnP 和保存恢复闭环。原始事件时间同步、对象尺度/对称性等未全部冻结。当前结论是“几何适配候选已核查”，不是“位姿任务已跑通”。

## 产物、复现和版本绑定

- 运行目录：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1/flow_selfsup_diagnostic_v1`。
- 运行时间：`2026-10-30T17:34:43Z` 至 `2026-10-30T17:39:30Z`；exit code `0`；g1，Python `3.11.15`，隔离环境 `/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28`。
- 运行 checkout：`/home/zhaoyuhan/Eventbased_WAM/execution_checkouts/sim2real-stage7-flow-resume-d0e3679-20261030`，commit `d0e36796f04826a14a0931878405abed943a49a3`。模型源锁定为 STE-FlowNet `eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6`；关键 helper hash 和完整执行命令见运行 `preflight.txt` 与 `run_stage7_replicad_selfsup_diagnostic_g1.sh`。
- EventGenesis checkpoint：远端 SHA256 `704ea0a1d769b05341ceb18f876c8d84c687ed3f97e014d6f45018e9223c9039`，580,024,945 bytes。V2E checkpoint：远端 SHA256 `f6756378fa76e49c68a74b9618ce0316af02634f1f60ee41746e1f6435605dc5`，580,024,945 bytes。权重保留在 g1，不进入 Git；紧凑 summary、逐更新日志、命令、GPU 预约记录和 hashes 已复制到本仓库。
- `summary.json` SHA256 `0ad3fb0d16d8f8bad3099840d3b64d3aab0aca6f613598c19cdbd800254075c7`。配置 SHA256 `335c4616b57a2d5d03d1acc5fe7f82bba34bd7fe9650a4547cda92d2cc611139`；训练脚本 SHA256 `7ff0881374ba3b15240c1e9dd38489996e332b9c130881e73f69945bb50d4471`。
- 逐文件大小和 SHA256 汇总在 [`evidence_index_20261031.json`](../../evidence/20261031-stage7-followup/evidence_index_20261031.json)。索引同时包含前一日 ReplicaCAD 导入、三布局覆盖、短训、失败尝试、紧凑日志、HDF5 小样例和预览；约 53 MB 的 renderer `.npy` 中间张量不入 Git，实际服务器路径和哈希仍在对应 artifact manifest 中。

复现入口（只会建立新 run，不覆盖现有输出）：

```bash
cd /home/zhaoyuhan/Eventbased_WAM
bash runs/sim2real/20261031-stage7-replicad-multiaxis-pathpilot-v1/flow_selfsup_diagnostic_v1/start_stage7_replicad_selfsup_diagnostic_g1.sh
```

重新运行前需确认共享 bank、STE-FlowNet 锁定源码、隔离环境和同配置仍存在，并重新实时检查/预约 GPU。真实评分关闭。此 100-update 诊断不得作为正式 checkpoint 选择依据。

## Stage 7 尚未完成

| 方向 | 实际进度 | 尚缺的关键证据 |
|---|---|---|
| 光流 | 成熟 ReplicaCAD 多布局 bank 已生成；配对 GT 监督与自监督各有短诊断；新自监督实际使用 g1 RTX 3090。 | GT warp 对实际多尺度 photometric loss 的正确性；更多独立场景的稳定泛化证据；MVSEC 真实文件/评分协议仍只按既有准备状态管理，真实评分关闭。 |
| 重建 | 旧 HQF 完整结果与合成 loader/递归接口审计已核实。 | 解释伪影的合成因果诊断；共享丰富场景上的修正版配对训练尚未在本次启动。 |
| 位姿 | 一个 E-POSE 样本的投影域适配候选已通过数值 round-trip。 | 精确 CAD 对应、多个独立合成 rollout、普通 eval/PnP、恢复验证与指标冻结。 |

不要把场景导入、短程训练或现有 HQF 分数写成仿真器真实性或 Sim-to-Real 优越性结论。
