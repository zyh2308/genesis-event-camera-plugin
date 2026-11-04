# 相关性算子 CUDA 加速核查与安全部署

核查时间：2026-11-04 14:28 UTC（23:28 JST）。状态：隔离验证通过，已部署条件切换入口；目标 V100 的正式 batch 8 门禁尚未执行，加速版尚未进入正式训练。

## 结论

原 STE-FlowNet 相关性算子逐个计算 81 个位移，反向也逐个执行乘法与累加。找到可提速部分，但不能把单算子加速倍数当作整项实验加速倍数。

直接采用官方专用 CUDA 前向/反向时，单算子前向及梯度容差检查通过，模型预测相对 L2 差异为 6.69e-7、三项损失相同，但一次 Adam 更新的相对 L2 差异为 **1.4424%**，超过事先设定的 1% 门槛。该方案被拒绝，未部署；失败报告见 `model_results.json`，没有放宽阈值重判。

最终保守方案保留原 PyTorch 前向，只把反向换成一个专用 CUDA kernel。kernel 逐项保持原有 dy/dx 累加顺序，并以 `--fmad=false` 编译，避免将原来分开的乘法和加法合并。它没有改变模型、损失、数据、标签、训练预算或冻结仿真器。

## 已完成的验证

- 在 g1 临时租用空闲 GPU 1（RTX 3090），与正式作业隔离，测试后已释放。没有停止、恢复或修改任何在跑训练，也没有向共享环境安装包。
- 11 组算子样例全部通过，覆盖四个实际训练尺度、极小/奇数尺寸、零值、单点、非连续输入及上游梯度、FP32 和 FP64。新方案的前向和两个输入梯度在这些样例上均 **逐位相同**；双精度有限差分 gradcheck 通过。
- 用固定的两个合成训练窗口、相同增强、相同初始化，运行实际 STE-FlowNet、自监督损失、70 组参数梯度检查及一次 Adam 更新。预测与损失相同；梯度相对 L2 差异 8.46e-10、Adam 更新相对 L2 差异 2.35e-7，处于参考实现自身的重复运行波动量级。没有声称完整长训练轨迹逐位等价。
- 后端选择入口的 7 项 CPU 测试通过：有效加速选择、回退、错误 seed、源码哈希损坏、报告损坏、比较组不完整、缺少决策文件。后五种情况均在启动训练前被拒绝。

## 实测性能

以下为同一张 RTX 3090 的隔离短测，FP32，禁止 TF32，无 AMP；算子 batch 8，每项预热 3 次、计时 10 次。耗时包括前向和两个输入的反向。

| 特征形状 B×C×H×W | 原实现 ms | 保守 CUDA 反向 ms | 加速比 |
|---|---:|---:|---:|
| 8×64×128×128 | 45.589 | 16.343 | 2.79× |
| 8×128×64×64 | 26.210 | 8.318 | 3.15× |
| 8×256×32×32 | 19.461 | 4.623 | 4.21× |
| 8×512×16×16 | 19.914 | 4.355 | 4.57× |

实际模型整步短测使用 **batch 2**，包括相同前向、损失、反向、有限梯度检查及 Adam 更新；1 次预热、5 次计时中位数：**1.412 → 1.183 秒/步**，吞吐约提高 **19.3%**，步耗时减少约 **16.2%**。这不是正式 V100 batch 8 或完整训练的速度结论；数据加载、checkpoint、验证耗时尚未包含，因此不能据此承诺整轮训练快 19%。原始计时列表在 `model_ordered_results.json`。

## 部署边界与公平性

1. 当前运行及已完成的 seed 0/2 光流保持原后端。五个尚未启动的 seed 1 方法统一使用同一个不可变后端决策，避免生成器与实现差异重合；训练种子之间的后端差异须在最终复现记录中披露。
2. 在 g1 GPU 3 的当前 V2E seed 2 训练自然结束后，队列先执行 `25_correlation_v100_fullbatch_gate`，使用 **V100、batch 8** 对相同模型、损失、梯度、Adam 更新及整步速度复检。该任务有 900 秒超时和显存检查，不会抢占或中断当前任务。
3. 只有该检查通过且整步加速超过预设 1.10×，共同决策才选择加速版；否则自动选择原实现，五组一致回退，不阻塞后续实验。
4. 五组 seed 1 继续执行原有 20→21 checkpoint 边界恢复预检，再按 45 epoch / 8,910 更新训练；加速入口会拒绝混用原后端的 checkpoint。原训练源码和 recipe 的 SHA256 已核对未变。
5. 只改五个待运行队列入口并加入门禁，原始队列规格保存在 `queue_backups/`。独立训练副本的全部差异见 `training_source_diff.patch`，运行源码、二进制及证据哈希见 `deployment_manifest.json`、`external_artifacts.json`。
6. 训练和本次检查仅使用合成数据，真实评分保持关闭。本次性能验证不构成任何生成器质量或 Sim-to-Real 优势结论。

## 三线状态

核查结束时，g1 两个原光流任务继续运行；g3 重建 RIFE seed 0 的心跳更新到 14:28:22 UTC，日志到 6,208/12,000；g1 位姿已自然推进到最后一个 Vid2E seed 2。累计完成记录为光流 2/15、重建 4/15、位姿 14/15。光流 V2E seed 0 最新已完成 epoch 34（6,732/8,910，记录时间 14:24:32），seed 2 为 5,832/8,910（14:28:43）。这些是独立时刻的日志观测，不是同步完成快照。

## 代码、证据与重现

服务器独立目录：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/correlation_cuda_20261104/`。

- `operator_results.json` / `model_results.json`：直接使用官方 CUDA 版的结果及失败门槛。
- `operator_ordered_results.json` / `model_ordered_results.json`：最终保守方案的算子与模型证据。
- `ordered_backward.cpp`、`ordered_backward.cu`、`ordered_adapter.py`：最终实现；`build_ordered.py`：隔离构建。
- `target_v100_gate.py`、`choose_seed1_backend.py`、`prepare_deployment.py`：门禁、共同决策与队列部署；`deployment_verification.json`：部署后核查。
- `parallel_status_at_audit_end.json`：三线日志观测；`test_dispatch.py`、`dispatch_test_results.json`：控制入口安全测试。
- 构建失败及被拒绝的原始日志均保留；二进制留在服务器，仓库只记录路径、大小和 SHA256。

官方比较实现来自 [ClementPinard/Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension)，固定提交 `4aa4431bf2f6057c9d40d4990227e3e655699ed3`，原始源码与 MIT LICENSE 保存在 `upstream/`。最终有序反向为本次独立实现，前向直接调用冻结参考实现。构建使用已有 Torch 2.8 环境和私有 CUDA 12.6/GCC 12 工具链，目标架构 sm_70 与 sm_86，没有修改共享工具链。
