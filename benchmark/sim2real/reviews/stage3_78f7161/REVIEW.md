# Stage 3 独立研究验收

日期：2026-10-16（Asia/Tokyo）。被审查分支：`exp/public-sim2real-v1`。GitHub、WSL 与 g1 均为 `78f71615acc7824072b0888eb16972c9a908ea44`。

**结论：重建与光流通过本阶段管线验收；上一轮两项结果失效的代码问题均已修复。位姿的时间窗错误已修复，但可用的模型推理及 E-POSE 真实接入仍未完成。可以让重建、光流进入协议冻结及正式合成训练准备，位姿独立处理，不再阻塞主线。当前没有正式公共真实数据上的 generator 优劣结论。**

本次直接检查仓库、WSL 原始公共数据、g1 转换数据/模型/输出，复算哈希，CPU 安全加载 checkpoint，独立重跑 19 项测试及针对性数据检查。没有重新运行 GPU 模型、训练或 Genesis 仿真。训练/推理执行事实由相互一致的日志、checkpoint、真实输出文件及代码支持；不将该核查表述为独立 GPU 全量复现。

## 1. 交付真实性与版本

| 核查项 | 结果 |
|---|---|
| 三处版本 | GitHub / WSL / g1 HEAD 全部一致；WSL 和 g1 核验开始时工作区干净 |
| frozen core | 相对 e79e656，`genesis_event_plugin/` 无差异；integration 仅 README 改动 |
| 证据索引 | S3 索引 360 条、总索引 668 条全部匹配大小/SHA256；两个索引有重叠，不能相加当成独立文件数 |
| 原始公共文件 | WSL 中 HQF bag、MVSEC data、GT、calibration 共 4 个文件均存在且大小/哈希正确 |
| g1 派生真实数据 | HQF 7 个 MemMap 文件、MVSEC fragment/GT/calibration 共 10 个文件全部匹配 |
| checkpoint | 8 个登记文件全部匹配大小/哈希，`weights_only=True` CPU 加载成功，参数均有限；E2VID/STE 与前阶段已核实的文件哈希一致 |
| 下游源码 | 5 份锁定副本共 686 个文件均匹配登记哈希 |
| 运行代码 | g1 与 WSL Python 运行脚本哈希一致 |
| 运行记录 | 最终 flow rescore、HQF 两组、MVSEC 四组、pose paired、最终 controlled-checks 的 wrapper 均有 exit 0；失败重试及缺失历史退出记录如实保留 |

报告未掩盖失败尝试，也未把 self-test 随机权重文件冒充实际候选模型。HQF/MVSEC 实际候选 checkpoint 均指向原 synthetic-only 训练产物。

## 2. 光流二维指标错误：修复通过

准确执行当前模型输出提取和 EPE 函数的独立例子：

- `(u,v)=(3,4)` 的完美预测：EPE=0。
- 仅 v 增加 2：EPE=2。
- 缺失向量分量的 `[H,W]` 输入：明确拒绝，不再静默广播。

独立重跑仓库 19 项 CPU 测试，全部通过，覆盖单位、mask、NaN、padding/crop、两通道输出、时间窗边界和官方 GT helper 对照。real adapter 空 mask 返回显式无分数；synthetic helper 空 mask 明确报错，均没有伪造零误差。

四个旧 STE checkpoint 的文件和模型状态不变，支持未为 metric 修复重新训练。重评分代码恢复完整 `[2,H,W]` 后评测全部 60 个 dt1 / 54 个 dt4 合成验证区间，恢复 probe 还检查完整两通道结果。当前报告的合成诊断为：

| dt | EventGenesis 合成 EPE | V2E 合成 EPE |
|---|---:|---:|
| 1 | 0.942304 | 0.824453 |
| 4 | 2.963737 | 2.858816 |

已核对归档逐区间记录与汇总、checkpoint 哈希及恢复 probe。GPU 前向结果未由研究会话重新计算，因此对这些合成数值的验收属于代码及产物支持的重评分，不是独立完整模型复现。它们仍是短程 pilot，不能写成稳定的 generator 比较。

## 3. 位姿时间错位：修复通过；模型可用性未通过

当前 `collect_samples()` 对每个选中区间重新设置 `label_time_index=interval_index+1`，并保存 sample ID、起止时间、标签时刻和事件数。

研究会话在 g1 CPU 上执行准确 collector，对两组全部 24 个独立 train/validation 样本另行按 raw timestamps 复算：

- 全部窗口均为约 10 ms。
- 事件终点、标签时刻与 shared bank 对应端点全部一致。
- 实际事件数、独立原始事件重数和归档日志全部一致。
- 上轮错误样例 EventGenesis `train_00/interval_007` 现已恢复到正确短窗，未继续读到 0.30 s。

两组各有 128 条训练更新记录，checkpoint epochs=16，参数有限且哈希匹配。SGD 没有 Adam 式 step 字段，更新次数的证据是日志及代码，不将其误写为直接从 optimizer step 读出。

报告的模型限制属实且没有被伪装为实验成功：标准 eval/置信度选格仍不稳定；单样本诊断显示 train/eval 状态下大幅差异。代码中的 GT-cell 解码与 train-mode 检查被单独标注为合成诊断，标准 validation 使用 eval 和置信度选格。独立检查发现两组训练后与诊断后的 BN buffers 完全一致，诊断没有污染保存的模型状态。

这些证据支持“存在推理状态和选格问题”，不证明更换模型是唯一解法，也不证明 BN 是全部误差来源。本次未独立重跑 GPU 单样本拟合，不能对所有诊断误差数字作独立模型复现声明。

E-POSE raw events、对应 CAD/class/K/pose 的样本级契约仍没有建立；当前 generic cuboid 不能作为其真实验证结果。数据访问失败的历史日志只支持该环境当时未取得样本，不支持全球不可获取的结论。

## 4. HQF：真实序列推理入口通过

原始 `bike_bay_hdr.bag` 文件已在 WSL 复算哈希。转换代码从 ROS embedded event timestamp / image header timestamp 读取数据，不使用 bag 消息到达时间替代事件时间。g1 上的 19,763,043 个事件、2,431 个图像端点及转换文件哈希均存在并匹配。

两组候选实际推理 manifest 与登记哈希一致。逐样本检查确认：

- 两组均有 2,431 个按图像端点排列的记录，目标时间与实际数组一致。
- 最晚输入事件不晚于标签端点。
- 每组递归状态只在序列开始 reset 一次。
- 所有登记的 26 张输出 PNG/组均存在、哈希正确、尺寸 180×240；已抽看一张输出，存在可见结构，但本验收不对重建质量作结论。
- 实际加载原 synthetic-only checkpoint；参考像素没有进入模型，也没有计算 HQF 指标。

保留的 EVREAL 索引语义、首帧空事件窗、额外处理最后端点和 PNG clipping 都已在代码/manifest 中说明。本阶段可验收 IO/递归推理；正式评分的帧集合、warm-up、输出裁剪/归一化、指标与聚合必须统一冻结。保存的 8-bit preview 不应直接替代尚未冻结的正式评价输入。

## 5. MVSEC：真实输入、二维输出及候选评价入口通过

研究会话在 WSL 将 fragment 与完整原始文件逐字段对照：events、APS 图像/时间/索引、flow maps、flow timestamps 均等于对应原始前缀。独立按预定规则求得 first common APS index=9，dt1 31 窗、dt4 28 窗，与登记完全一致。

g1 四组共 118 个 NPZ 全部匹配大小/哈希，数值有限、形状正确。研究会话逐窗独立核验：

- 事件数与原始时间戳窗口重数一致。
- `[2,352,352]` prediction 按 y=46、x=3 裁剪后，逐元素等于 `[2,260,346]` 保存输出。
- 保存 GT 与准确锁定的官方 STE 时间传播 helper 在实际 fragment 上吻合。
- 同一个 dt/窗口的两模型事件输入、GT 和 mask 哈希完全相同。
- 不调用工程 EPE 函数，直接用两分量差的 Euclidean norm 重算，与已保存逐窗误差最大差约 7.1e-9（浮点差异）。核查只记录一致性，没有新汇总真实分数或排名。
- 本 fragment 中 encoded support 与 raw-event support 相同；这不证明所有后续稀疏/高计数窗口都会一致，正式协议应明确支持区域从何处生成。

因此光流真实输入及候选评测链路已接通。native 分辨率推理内存有实际记录；这仅支持 inference，不保证 native 分辨率 dt4 training 的显存预算。

**测试隔离记录需明确：MVSEC pilot 已计算并保存逐窗真实 EPE。** “没有 aggregate ranking”不等于“从未查看真实 GT 分数”。该固定片段应登记为已暴露的工程检查范围，禁止再据此调参、挑 checkpoint 或改 generator；正式协议/论文披露其用途及处理规则。当前检查未发现用这些分数选择模型或调整 generator 的代码路径。

## 6. 验收决定与后续工作

| 路线 | 决定 | 后续重点 |
|---|---|---|
| 重建 | **管线通过** | 由研究会话冻结 HQF 主实验协议，再准备正式合成训练与完整 test inventory |
| 光流 | **管线通过，主线可推进** | 冻结 MVSEC GT/support/聚合与已暴露片段记录；明确训练时长/数据规模/种子与资源预算 |
| 位姿 | **时间窗通过，完整任务仍未完成** | 获取一个真实样本+CAD；合成域解决标准 eval 的模型状态问题，必要时提交有依据的成熟模型替代方案 |
| 最终 Sim-to-Real 结论 | **尚无** | 当前权重仍来自极小短程合成 pilot，不能据真实推理存在就宣称仿真器泛化更好 |

这次没有发现与 Stage 2 同等级、会推翻重建/光流管线验收的新错误。后续不必再反复重做已通过的单步/小规模 smoke；应把主线资源投入明确协议与足量 synthetic-only training。输入 radiance/display mapping、遮挡 validity、场景/运动覆盖等研究问题仍须在正式实验协议中说明。

## 7. 独立证据与文件位置

- `local_verification.json`：索引、WSL 原始文件哈希、独立二维 EPE 例子、冻结核心差异。
- `source_slice_verification.json`：完整 MVSEC 与 fragment 的逐字段对照及固定窗规则复算。
- `remote_audit_stdout.json`：g1 数据/8 checkpoint/118 prediction 文件、manifest、版本与脚本哈希。
- `targeted_stdout.json`：24 pose 样本重数、BN 状态、118 窗真实输出/GT/EPE/crop 检查、HQF 时间/52 preview 核查、686 源文件验证。
- 同目录脚本为核验方法；`audit_summary.json` 提供简要审查结果。

报告和证据同步保存到共享 WSL 仓库 `benchmark/sim2real/reviews/stage3_78f7161/`。新增审查目录是本次核验产物，尚未提交或推送；没有改变原始交付的索引、工程实现、原始数据或 checkpoint。
