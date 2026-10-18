# Stage 4 运动学习诊断与配对训练状态

状态快照：**2026-10-18 02:01:22 UTC / 11:01:22 JST**，由 g1 训练日志和 GPU 状态刷新。训练仍在进行。固定合成诊断在 **01:44:51–01:46:09 UTC / 10:44:51–10:46:09 JST** 完成。诊断只使用固定 synthetic validation 子集；未读真实数据，未改变训练、验证、模型选择或 generator 配置。

## 结论

- **训练正常推进。** g1 上 EventGenesis/V2E 两个训练进程仍运行于 GPU 2/GPU 4。实时记录为 4,900/8,910 与 4,950/8,910 次更新，约 55.0% 与 55.6%。预算仍为每组 45 epoch、每 epoch 198 次更新。
- **当前模型没有学到足够稠密的有用光流。** 固定 64 个验证窗口中，全有效区预测幅值中位数仅为 0.000219/0.000232 px per 30ms，GT 中位数为 0.202197；98.93%/98.38% 的预测幅值小于 0.1 px。全有效 EPE 几乎等于零流。
- **观察到局部运动信号，但覆盖很小。** 双方共用事件活跃区的 EPE 比零流低 6.19%/6.57%，该区域只覆盖有效像素的 0.487%。独立刚体运动、静止相机的 GT 动态前景（幅值 ≥0.5 px/30ms）仅占该类别有效区 0.138%；此处 EPE 比零流低 12.54%/11.11%。这是部分局部学习信号，不足以证明已学会稳定稠密光流。
- **没有发现新的确定性工程错误。** 配对时间、灰度端点、K、标签、有效区、单位、方向、输出裁剪及物理标签来源通过固定样本契约核对。弱表现的底层原因仍未确定，不能仅凭此诊断归因于数据稀疏或训练不足。
- **继续完成本轮训练。** 诊断没有证明训练无效的证据；短程学习偏弱不触发中止或调参。真实数据评分继续关闭。

## 当前训练快照

两组配置、采样、增强、初始化配对和 checkpoint 选择协议保持不变；模型只接收 event volume。GT flow 仅用于合成验证，不参与优化。checkpoint 仍只按完整 synthetic validation 指标选择。

| Generator | 实时 optimizer update | 最近完整验证 | 最近验证 EPE | 零流 EPE | 当前最佳验证 EPE |
|---|---:|---:|---:|---:|---:|
| EventGenesis | 4,900 / 8,910 | epoch 24 / update 4,752 | 0.2838378343 | 0.2842080520 | 0.2838365008（epoch 19） |
| V2E | 4,950 / 8,910 | epoch 24 / update 4,752 | 0.2831938486 | 0.2842080520 | 0.2828732346（epoch 20） |

单位为 px per 30ms。最佳值分别只比零流低约 0.13% 和 0.47%。主验证值来自各自最近一次完整验证，实时更新数来自更晚的训练记录。固定子集诊断只覆盖 64 个窗口并使用像素池化，统计口径不同，不应与完整验证 EPE 直接横比。

快照时监督器 tmux stage4-flow-corrected-20261017-v7（PID 3590105）、训练 PID 3594965/3594967 仍存活。完整训练日志位于 g1：

- /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/full_training_retry1/eventgenesis/training_metrics.jsonl
- /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/full_training_retry1/v2e/training_metrics.jsonl

02:01:22 UTC 读取时，两日志大小与 SHA256 分别为 4,006,813 B / a04de0598594661764f936afbbc8a1dd6a3819866a2409590a4ff249e08eb6f0，以及 4,036,751 B / 873059b30d717085b59d59f0b37e22a86165ee3fffdac8659ecbec3019fc609f。日志还会持续追加，这些摘要只标识该次快照，并非最终摘要。

## 固定 synthetic 诊断

预注册方案在读取模型输出前已记录于 [DIAGNOSTIC_PLAN.md](evidence/S4/20261018-motion-diagnostic/DIAGNOSTIC_PLAN.md)。双方 checkpoint 均为 seed 0、update 4,554、scheduler epoch 23 的验证后边界状态，并保留共同初始化。诊断使用全部 8 个固定验证 rollout；每个 rollout 取 adapter 行 0、9、18、27、36、45、54、65，合计 64 个配对窗口。样本 ID 对应 dt1_000、027、054、081、108、135、162、195。无样本筛选或重加权。

运行前登记的区域与阈值：

- **完整有效区**：物理 flow_valid mask。
- **共同事件活跃区**：flow_valid 与两组五 bin event volume 的非零空间位置并集；双方预测共用同一个 mask。
- **固定 GT 动态区**：flow_valid 且 GT 向量幅值 ≥0.5 或 ≥1.0 px per 30ms。
- **近零预测**：幅值 <0.1 或 <0.5 px per 30ms。方向仅在预测幅值 ≥0.1 的像素上统计。

数据契约为 (t_i,t_i+3] 的 30ms 窗口、共享 100Hz 端点、标签时刻等于事件窗右端、flow 单位 px per 30ms。模型输出 [1,2,352,352]，只裁剪已登记 padding y=46:306、x=3:349，不缩放 u/v。64 对样本的 ID、GT、valid、灰度端点、时间戳、K 和 label hash 全部一致；配对灰度端点和 GT 的最大绝对差为 0。共享 HDF5 的抽查物理端点、flow 字段及窗口时间也与 adapter 记录相符。

### 汇总指标

相对改善为 (zero-flow EPE - model EPE) / zero-flow EPE。以下 EPE 对固定诊断集合中的像素池化计算。

| 共用区域 | 有效区覆盖 | 零流 EPE | EventGenesis EPE / 改善 | V2E EPE / 改善 |
|---|---:|---:|---:|---:|
| 完整有效区 | 100% | 0.277367 | 0.277371 / -0.0014% | 0.277173 / +0.0699% |
| 共同事件活跃并集 | 0.4866% | 0.352044 | 0.330247 / +6.19% | 0.328912 / +6.57% |
| GT 幅值 ≥0.5 px/30ms | 17.423% | 0.821715 | 0.816592 / +0.623% | 0.819365 / +0.286% |
| GT 幅值 ≥1.0 px/30ms | 4.266% | 1.255078 | 1.245155 / +0.791% | 1.250986 / +0.326% |

完整有效区预测幅值 p50 为 EventGenesis 0.000219、V2E 0.000232 px/30ms，而 GT p50 为 0.202197。GT≥0.5 区中，预测幅值仍有 98.41%/98.41% 小于 0.1 px；达到 0.1 的像素分别只有 1.59%/1.59%。只在这些非近零预测上计算方向时，平均方向 cosine 为 0.841/0.530、中位角误差为 17.8°/46.2°。条件方向统计覆盖很小，不能代表整个动态区。

### 四类运动场景

全区、事件活跃区及 GT≥0.5 区的改善均相对该类别对应的零流 EPE。事件活跃覆盖率为双方共同并集占该类别有效像素比例。

| 类别 | EventGenesis：全区 / 活跃区 / GT≥0.5 | V2E：全区 / 活跃区 / GT≥0.5 | 活跃区覆盖 |
|---|---:|---:|---:|
| 相机运动、静态场景 | -0.003% / +4.12% / +0.013% | -0.080% / +5.58% / -0.014% | 0.472% |
| 静止相机、独立刚体运动 | -68.75% / -24.46% / +12.54% | -34.28% / -9.74% / +11.11% | 0.148% |
| 相机与刚体共同运动 | +0.196% / +13.51% / +0.94% | +0.172% / +9.84% / +0.40% | 0.606% |
| 遮挡组合运动 | +0.090% / +5.54% / +0.84% | +0.260% / +6.66% / +0.41% | 0.725% |

独立刚体类别的全区指标受静止背景主导，动态前景只占 0.138% 有效像素。因此全区 EPE 变差与固定动态前景上改善可以同时出现。其余类别的 GT≥0.5 区预测仍有约 97.6%–99.5% 小于 0.1 px。分层结果仅供合成诊断，不改变本轮训练或 checkpoint 规则。

## 证据、尝试与边界

逐样本结果 JSON：[motion_diagnostic.json](evidence/S4/20261018-motion-diagnostic/motion_diagnostic.json)，保留 64 个样本的窗口时间、事件数、四区指标和配对校验。机器可读运行收据：[run_receipt.json](evidence/S4/20261018-motion-diagnostic/run_receipt.json)。代码、配置、bank 与 adapter 指纹在仓库内；完整 bank、权重和训练日志留在 g1，绝对路径、大小和 SHA256 收录在收据中。

第一次诊断在任何模型 forward 前被 ID 断言拒绝：脚本把 adapter 行号当成物理 window ID，未乘以 3。没有模型输出或训练变化。修正映射并提交后，第二次固定样本运行在 RTX 3090 GPU 1 上耗时 77.751 秒、退出码 0。代码和数据未改变，也没有重新挑样本。

严格整段 CUDA 轨迹逐位复现仍保留为失败证据；安装版 PyTorch 的 grid_sampler_2d_backward_cuda 没有确定性实现。恢复门禁的 resume_equivalence_epoch_boundary 实际核验 checkpoint 边界恢复、下一 batch 重建、一次真实 optimizer 更新和 forward loss 对照；它**不等价于严格整段轨迹恢复**，也没有独立重写 Adam 以比较完整参数更新。determinism 诊断报告里的日志文件名已修正，嵌套索引记录了实际路径、大小和哈希。

另在独立 GPU 1 上补齐了 V2E 的 checkpoint 边界恢复核验，没有中断 GPU 2/4 的主训练。通过正式 resume-check launcher 从 V2E pilot update 198 启动，gate exit 0，真实恢复进程推进到 update 199 后正常暂停；独立 oracle 检查样本 ID、增强 seed、70 个 Adam step counter（198→199）、scheduler 边界、模型状态变化及 source-checkpoint forward loss。三项 loss 的绝对差均为 0，oracle PASS。第一次补充尝试因 Python 环境路径写错在进入 launcher 前以 127 退出，stderr 保留在 g1；修正路径后的第二次尝试成功。逐项结果见 [v2e_resume_checkpoint_oracle.json](evidence/S4/20261018-motion-diagnostic/v2e_resume_checkpoint_oracle.json)，两次运行的日志路径与摘要列在 run_receipt.json。该补充仍是边界恢复核验，不会改变严格整段复现失败的结论。

本诊断没有访问 HQF、MVSEC、E-POSE 或任何真实 GT，不进行真实域排名。

## 复现

在 g1 的 Eventbased_WAM 环境、仓库根目录执行。运行时设 CUDA_VISIBLE_DEVICES=1，因此脚本 cuda:0 对应物理 GPU 1：

    export CUDA_VISIBLE_DEVICES=1
    python -u benchmark/sim2real/scripts/stage4_motion_learning_diagnostic.py --model-root /home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/STE-FlowNet --data-root /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/20261017-stage4-flowdt1-seed0-corrected-v5/steflow_30ms --shared-bank /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v3_geometry_corrected_20261017_32x2s/shared_trajectory.h5 --config benchmark/sim2real/configs/stage4_corrected_v2_flowdt1_seed0.json --eventgenesis-checkpoint /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/motion_diagnostic_epoch23_update4554/checkpoints/eventgenesis_epoch23_post_validation_update4554.pt --v2e-checkpoint /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261017-stage4-flowdt1-seed0-corrected-v5/motion_diagnostic_epoch23_update4554/checkpoints/v2e_epoch23_post_validation_update4554.pt --output benchmark/sim2real/evidence/S4/20261018-motion-diagnostic/motion_diagnostic.json --device cuda:0

本次环境：Python 3.11.15、PyTorch 2.5.1+cu121、CUDA 12.1、NumPy 2.4.6、h5py 3.14.0、RTX 3090。原始 stdout/stderr、起止时间、命令及退出码在 g1 的 .../motion_diagnostic_epoch23_update4554/attempt_02/；失败尝试保存在同目录 attempt_01/。

