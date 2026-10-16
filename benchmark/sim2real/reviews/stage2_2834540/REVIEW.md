# Stage 2 独立研究验收

核验日期：2026-10-16（Asia/Tokyo）。GitHub / WSL 提交：`2834540c45c1b50c2d19d5fab078752f2350c3bf`。g1 提交及证据基线：`d9a9a3f08b45ef1eb2faaabddc8513c361d25a4a`。分支：`exp/public-sim2real-v1`。

**结论：第二阶段部分通过，尚不能认定全部完成。重建合成训练链路及 Stage 1 R1/R2 修复通过；光流训练有实物证据，但二维 EPE 提取有错误，现有 EPE 数字不能验收；位姿存在事件窗与标签时间错位，其训练运行不能作为正确任务闭环验收。三个真实数据集仍未完成数据级接入或真实评分。**

本次独立读取 GitHub、WSL checkout、g1 数据和 checkpoint，复算哈希，安全 CPU 加载 checkpoint，并执行了针对性 CPU 复现。没有重新运行 GPU 训练、真实模型评分或 Genesis 渲染。本次只增加研究审查材料，不改工程实现，不更新远端分支。

## 1. 已核实的事实

| 项目 | 独立结果 |
|---|---|
| GitHub 与 WSL | 分支均指向 2834540；WSL 核验开始时工作区干净 |
| g1 | HEAD 为 d9a9a3f，核验时工作区干净；比交接提交少 README 链接和交接文档，运行脚本哈希与 WSL 全部一致 |
| 冻结实现 | `genesis_event_plugin/` 相对 e79e656 无差异；integration 仅 README 改动 |
| 归档索引 | 135 个条目中 134 个匹配当前文件；唯一不匹配是 README，其索引准确对应 d9a9a3f 版本，后续新增交接链接导致过期，不是运行数据损坏 |
| 外部数据 | 33 个文件的大小及 SHA256 全部匹配 |
| checkpoint | 10 个文件（含保留的旧 pose 尝试）大小及 SHA256 全部匹配；CPU `weights_only=True` 加载成功，参数均有限，模型 state hash 与归档 result 一致 |
| 优化状态 | 两个 E2VID checkpoint 的 Adam step 为 224；四个 STE checkpoint 为 16；pose SGD 含 momentum 状态但不记录 Adam 式 step，不能据此独立声称精确更新次数 |
| 下游源文件 | 5 份锁定源码共 686 个文件，全部符合登记哈希；g1 是源码副本，不将其误称为全部具有 Git HEAD 的 checkout |
| 运行结果归档 | 四份 result JSON 与 g1 原始运行目录中的文件哈希一致；失败尝试及非零退出码保留，成功重试与报告一致 |
| 配对及划分 | 6 个完整物理 rollout，4 train / 2 validation；两 generator 的重建图像/flow/mask 及 STE 图像/flow/mask/时间信息逐元素相同 |
| 实际 K | 已保存实际 160×128 相机 K 和坐标/时间约定，补齐上一轮归档缺口 |

这些证据支持“训练运行和产物确实存在”，不自动证明指标或时间对齐正确。下面的两个新问题来自代码与数据的独立复现。

## 2. [P1] 光流 EPE 只取了水平分量，现有指标无效

位置：`benchmark/sim2real/scripts/run_steflow_stage2_pilot.py:328`，恢复验证也在第 387 行重复。

锁定的官方 `STEFlow.py` 在 eval mode 返回 `flow1`，形状 `[B,2,H,W]`。当前提取为：

```python
prediction = pred[0].detach().cpu().numpy()[0]
```

第一个 `[0]` 已去掉 batch，第二个 `[0]` 又去掉了向量分量维度，只留下 `[H,W]` 的 u。`compute_epe()` 没有检查形状，NumPy 将它广播到 `[2,H,W]`，使同一 u 同时与 GT 的 u、v 比较。

研究会话执行准确提取表达式和准确 metric 函数：当 GT 为 `(u,v)=(3,4)`、模型预测完全等于 GT 时，现有代码返回 EPE=1，正确结果为 0。这不是模型未收敛造成的偏差。

影响：报告中的 dt1 `0.8301 / 0.8082` 和 dt4 `2.8537 / 2.8379` **不能作为正确二维 EPE 使用**；同样错误出现在 checkpoint 恢复评分，两次相等不能验证 metric 正确。训练 forward/backward、photometric/smoothness 更新及 checkpoint 实物证据仍有效。

修正要求：保留 `[2,H,W]`，在指标入口明确断言 prediction/GT/mask 形状、数值有限性和单位。用 u/v 不同、仅 v 错误、完美预测等样例核验，然后读取现有四个 checkpoint 重算全部 synthetic validation。**此错误在评测阶段，无需仅为它重跑光流训练。** 更新所有 result、报告、交接和证据索引；保留旧值并标注作废。

同时，dt4 只测了 160×128 场景填充到 256×256 的内存，不能用它保证 MVSEC 原始分辨率可运行。目标分辨率仍需小型资源验证。

## 3. [P1] 位姿事件窗终点沿用上一循环变量，标签与输入时间错位

位置：`benchmark/sim2real/scripts/run_pose_stage2_pilot.py:203–217`。

前一个遍历全部区间的循环把 `label_time_index` 留为 30。随后遍历选中的区间时，直接读取 `times[label_time_index]`，没有按当前 `interval_index+1` 重设终点。因此早期样本输入一直积累到 rollout 最后时刻，而 keypoint / pose 标签仍取当前选中区间右端点。

研究会话在 g1 CPU 上执行当前 `collect_samples()`，并按正确物理区间独立重编码：

- EventGenesis `train_00` 区间 7：标签时刻为 0.08 s，期望输入 `(0.07,0.08]`，实际输入 `(0.07,0.30]`。
- 期望事件数 351，实际 24,093，归档训练记录也为 24,093；重编码后的输入最大绝对差为 1.0。
- 每个 generator 的 12 个独立 train/validation 窗口中，6 个早期窗口发生同类错误；另外 6 个末窗口自然一致。两组共检查 24 个窗口，12 个错误。

影响：当前 pose pilot 有输入/监督时间错位，并包含标签时刻之后的事件。不能把它作为正确的 10 ms event-to-pose 闭环。它可能影响异常大的 keypoint 误差，但不能据此断言这是误差的唯一原因，也不能据该误差判断模型或 generator 不行。

修正要求：以每个样本的物理窗口显式生成输入起止时间、标签时间和共享 sample ID，检查右端一致性，补一个至少包含两个不同区间的时间契约检查。修复后重跑同等级的小型 pose pilot；不要扩大错误样本训练。E-POSE raw sample / CAD / class / K / transform 契约未完成的问题仍独立存在。

## 4. 重建：上一轮 R1/R2 已实际修复

**R1 通过。** 序列长度已变为 3，官方 temporal objective 在第三步执行。日志记录每组 224 次 train、56 次 validation 调用，两个 checkpoint 的 Adam 状态也为 224。研究会话另行执行官方 objective，确认前两步返回 None、第三步有有限非零 loss 和有限非零梯度；这项不再是未触发的配置。

**R2 通过。** 研究会话独立重新运行空事件、两事件、同时间戳 fixtures，三种情况 flow 标签哈希完全相同、均保留 64 个 valid pixels。并使用准确 adapter 读取两 generator 的全部 12 份 HDF5，逐区间对照 shared bank：物理 flow 最大误差为 0，validity 全部一致。

方向检查也独立重跑：forward flow 的未屏蔽 warp L1 为 0.004762，反向为 0.052381。官方 TC 自带可见性处理可能屏蔽错误对应，原报告如实说明 scalar 不足以判断方向，这个解释成立。

边界说明：少于 3 个事件产生零 voxel 是保留的 upstream 行为；本次验收的是标签不随事件改变，不将其宣称为最优稀疏事件编码。完整训练目标链路可通过本阶段验收，但训练预算、场景覆盖及 HQF 评测仍未正式冻结。

## 5. 相关算子与真实数据准备

研究会话对新的 memory-efficient correlation 使用独立 `F.unfold` 参考，覆盖 B=2、小于 patch 的尺寸及 1×1 输入；CPU float64 forward 最大误差为 0，gradient 最大误差约 7.1e-15。其数学操作通过；没有独立重跑 GPU 性能或原生 CUDA extension 对照。

HQF、MVSEC、E-POSE 在登记的 g1 real 路径均没有数据目录；仓库 manifest 也是来源/字段和候选规则清单，还不是已验证的真实文件 inventory。HQF candidate checkpoint adapter、MVSEC actual GT/support/alignment、E-POSE raw sample/CAD 均未在真实文件上运行。不能将“评测入口已准备完成”作为本阶段已验收事实。

检查的模型入口为 event-only，实际训练文件来自 synthetic bank，未发现真实测试或 depth/seg/GT pose 被作为额外模型推理输入。这是代码和路径范围内的审查，不是对所有历史文件访问的全量审计。

shared bank 仍缺目标遮挡 z-buffer，且只含六条短时落方块轨迹。EventGenesis 使用 linear radiance，V2E 使用固定 Reinhard+sRGB uint8 映射；这一协议差异已被交接文件正确披露。正式比较前由研究会话明确其结论范围和处理方式，不允许工程会话通过真实测试分数选择输入映射。

## 6. 验收状态与下一步

| 路线 | 本次验收 | 下一步最小动作 |
|---|---|---|
| 重建 | **合成链路通过**，R1/R2 通过 | 保留现有结果，接入实际 HQF 文件并校验候选评测适配 |
| 光流主实验 | **训练通过、EPE 未通过** | 修二维提取及形状检查，用现有 checkpoint 重评分；补目标尺寸资源测试 |
| 位姿 | **正确时间对齐的闭环未通过** | 修窗口时间契约，重跑小型 pilot，同时解决一个 E-POSE raw sample 与 CAD 对应关系 |
| 公共真实验证 | **尚未完成接入，无真实结果** | 完成真实文件级核查与协议冻结后再开启正式评分 |

优先做有限修正和真实文件接入，三线并行；不重复已通过的重建训练，不要求为光流 metric 错误重训，也不扩大尚有时间错位的 pose 语料。

E-POSE 当前访问失败只证明本环境当时没有取到数据，不证明全球不可获取；保留这一方向性阻塞，不擅自换任务。此次独立核验不对其在线可达性作新的结论。

## 7. 独立证据与交付边界

- `local_verification.json`：135 条索引、实际日志摘要、独立 CPU correlation 检查。
- `remote_audit_stdout.json`：g1 HEAD、33 个外部文件、10 个 checkpoint、exit code、共享标签逐元素核查。
- `targeted_verification.json`：准确 EPE 代码的反例、24 个 pose 实际窗口、E2VID boundary / direction / 标签和梯度复验。
- `remote_metadata.json`：686 份源文件哈希、运行脚本哈希、真实目录状态、原始 result JSON 哈希。
- 同目录 `audit_*.py` / `remote_*program.py`：独立核验方法。最初一次针对性审查因选错归档路径失败，之后改读实际 run 路径并完成；这不是工程训练运行失败，最终证据文件来自完成后的核查。

此报告和证据另存到共享 WSL 仓库 `benchmark/sim2real/reviews/stage2_2834540/`，供工程会话直接读取。新增审查材料不属于原交付提交，尚未 commit/push。g1 的原训练数据与 checkpoint 未改动。
