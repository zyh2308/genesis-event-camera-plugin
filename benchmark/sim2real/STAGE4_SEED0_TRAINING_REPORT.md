# Stage 4 光流 seed 0 配对训练结果

报告更新时间：2026-10-17。训练运行 ID：`20261016-stage4-flowdt1-seed0-v4`。本报告固化已完成训练状态；其结果仅为 synthetic-only 工程训练与合成验证证据，不是 Sim-to-Real 结论。

## 结论摘要

- EventGenesis 和 V2E 的 STE-FlowNet dt1 seed 0 训练均完成 **45 epochs、8,910 次 optimizer 更新**，各训练进程退出码 `0`，campaign 退出码 `0`。
- 每组训练日志有 8,955 条 JSONL 记录：8,910 个更新记录和 45 个 synthetic validation 记录。两组完整训练状态 JSON 均为 `complete`。
- 两组在第 16/17 epoch 达到各自 synthetic validation 最优值：EventGenesis `0.9389129835000911`、V2E `0.9406893614490628` px / 30 ms。两者差约 `0.00178` px，单 seed 下没有可据以支持生成器排名的证据。
- 未读取 MVSEC 或其他真实数据做训练、checkpoint 选择、调参或评分。本报告不声称任何真实泛化优势。

## 固定实验设置

| 项 | 实际设置 |
|---|---|
| 服务器与目录 | g1；项目 `/home/zhaoyuhan/Eventbased_WAM`；仓库 `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real` |
| 分支 / 训练实现提交 | `exp/public-sim2real-v1` / `1fab7e17e201943c77703aa8a1c0c6eaefbb8f5a` |
| 作业 UTC 时间 | 启动 `2026-10-16T16:34:32Z`；campaign 完成 `2026-10-17T04:03:48Z` |
| 配置 | `benchmark/sim2real/configs/stage4_training_launch_v1_torch_cuda.json`，SHA-256 `c0ab0d17f024b00106957c3e44d7a013e920f6e760f70266c43a9cf904634685` |
| 训练 bank | 32 个共享物理 rollout，24 train / 8 validation；完整 HDF5 SHA-256 `e9e57131c84059047bddd0d34ac3271f23f5e4ff0e4fa5e0b9f8579772ee337a`；split SHA-256 `60722e124b24c8da2306f3a8a4115ec8588949653a2f014b3687c2a2dfc59aa1` |
| V2E 导出 | 32 rollout，25 μs；complete manifest SHA-256 `5d79bad2f9fe81e1fad8af0676b9048aefaf73c0b6edd8db7f39f31ef82beb2f` |
| 下游样本 | 两组共用 1,584 train / 528 synthetic-validation windows；adapter manifest SHA-256 `c9da4158d9775997f08dd75ce0ac2ccc33873d983422ef87c931e834f6da0624` |
| 配对设置 | seed 0；两组初始模型 state SHA-256 均为 `acc6bbd11488a21868c5f147c7f397521480166c24914cc7988a6428d6ed71cf`；effective batch 8；Adam learning rate `4e-4`；预算每组 45 × 198 = 8,910 updates |
| 模型目标 | 固定 STE-FlowNet；event-only 模型输入。使用共享合成灰度端点和 photometric + `10 ×` second-order smoothness 目标；合成 GT flow 只用于 synthetic validation EPE，不送入优化器 |
| 真实数据隔离 | 真实数据评分/反馈关闭；训练和模型选择仅使用登记的 synthetic train/validation |

独立审查核对了完整共享 bank、64 个 adapter NPZ 哈希、32 对 rollout 的共享灰度/flow/mask/时间/K 数组，并确认前 298 个配对 update 的 sample 顺序相同。首个 epoch 每组各覆盖 1,584 个训练窗口一次。Torch CUDA 是本轮实际 generator backend；配置沿用了已审查的运行时修订，不应将输出描述为与旧 NumPy backend bitwise 相同。详细背景见 [`STAGE4_TORCH_CUDA_RUNTIME_AMENDMENT.md`](STAGE4_TORCH_CUDA_RUNTIME_AMENDMENT.md) 和 [`reviews/stage4_1895a15/REVIEW.md`](reviews/stage4_1895a15/REVIEW.md)。

## 训练与验证结果

| 生成器 | GPU | 最后更新训练 loss | photometric | smoothness | 第 45 epoch mean EPE | 最佳 synthetic validation EPE | 最佳 epoch | 峰值分配显存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EventGenesis | 2 (V100S 32 GiB) | 1081.1363525 | 1080.7971191 | 0.03392557 | 0.9466441114 | 0.9389129835 px / 30 ms | 16 | 24,948,515,328 B |
| V2E | 4 (V100S 32 GiB) | 1078.3039551 | 1077.9970703 | 0.03068791 | 0.9438607459 | 0.9406893614 px / 30 ms | 17 | 24,948,515,328 B |

最后训练 loss 是优化目标的诊断值，不是两种 generator 的独立性能指标。验证 EPE 按 8 个留出 rollout 等权汇总，单位是像素 / 30 ms。Epoch 45 EPE 和每组的 best EPE 都只来自 synthetic validation；best checkpoint 选择未读取真实分数。

## Checkpoint 与日志证据

所有大文件保留在 g1，不进入 Git。训练 run root：

`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/flow_dt1_seed0/`

| 生成器 | Artifact | 大小 | SHA-256 |
|---|---|---:|---|
| EventGenesis | `eventgenesis/last.pt` | 580,035,558 B | `83e2788f6c0506d22f28e212b6c65faaf56d2858e541940be91d2da4edf6ab8a` |
| EventGenesis | `eventgenesis/synthetic_best.pt` | 580,038,730 B | `10c8785439ee9b8b61497381e013b3b022e9c50eaa83509f877140d597682f85` |
| V2E | `v2e/last.pt` | 580,035,558 B | `a08a5de6bf7fd92dd4d0e7c78dcdc82b88e60bb2792125d5e2db1bbfb17ce587` |
| V2E | `v2e/synthetic_best.pt` | 580,038,730 B | `af451d42dc69236554aa5e4a34dadf0fb29554e135e8e39e7a02a30d023bb57c` |

完整 metrics JSONL：

- EventGenesis：`.../eventgenesis/training_metrics.jsonl`，8,955 行，SHA-256 `a582977145da62eaaa95eb97a3b039eadfe9575def244e2b099fa77ad538c29c`。
- V2E：`.../v2e/training_metrics.jsonl`，8,955 行，SHA-256 `eac74897f8c57027608b331b2303f48531233d4ab7cb3b39568df8b66e65939a`。
- 两组 `train.exit_code.txt` 均为 `0`；campaign 的 `training_retry3.exit_code.txt` 为 `0`，完成标记时间为 `2026-10-17T04:03:48Z`。
- 原始 stdout/stderr、每轮合成验证明细、`live_status.json`、run metadata、PID/退出记录均保留在上述 run root 的各 arm 子目录。`live_status.json` 最终 state hash：EventGenesis `ef39e6dd6b41917b9181e99adf07b6a57b1998eb1c49d60075f3a912309356c5`；V2E `619e0795efe8215defc96e22212fd8264198e55411cbe714ae998918f8aed092`。
- 首 20 update 的不可变日志和 checkpoint 恢复探针见 [`evidence/S4/20261016-stage4-flowdt1-seed0-v4-update20/README.md`](evidence/S4/20261016-stage4-flowdt1-seed0-v4-update20/README.md)。该处已记录 update-20 checkpoint 的哈希、优化器恢复及预测最大误差。

## 复现与状态核验

本轮记录的启动入口与命令如下。它用于这次运行的历史复核；启动脚本默认绑定本 run ID，并且恢复脚本要求当时的 `training_retry2` 失败状态，所以**不可在已完成 run root 上再次执行**：

```bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real
bash benchmark/sim2real/scripts/start_stage4_training_after_shape_fix_tmux.sh
```

成功训练实际使用的代码提交为 `1fab7e17e201943c77703aa8a1c0c6eaefbb8f5a`；检查当前结果不需要启动训练。对现有外部结果进行状态和哈希核验的命令：

```bash
RUN=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4
cat "$RUN/training_retry3.exit_code.txt"
cat "$RUN/flow_dt1_seed0/eventgenesis/live_status.json"
cat "$RUN/flow_dt1_seed0/v2e/live_status.json"
sha256sum "$RUN"/flow_dt1_seed0/{eventgenesis,v2e}/{last.pt,synthetic_best.pt,training_metrics.jsonl}
```

本轮更新后的仓库状态报告见 [`TRAINING_LIVE_STATUS.md`](TRAINING_LIVE_STATUS.md)；历史启动失败和修复过程见 [`STAGE4_V4_RECOVERY.md`](STAGE4_V4_RECOVERY.md)。

该 runner 是一次性故障恢复入口，不是通用新实验 launcher。若要重复训练，应先为新 run 建立明确的 bank/config/run ID 和 preflight 记录，再用独立 run root 启动；本报告不把现有恢复脚本冒充为安全的通用复现实验命令。

## 仍需处理 / 解释边界

1. 独立审查确认现有 `--resume` 实现将 checkpoint 的 Torch CPU RNG ByteTensor 随 `map_location=device` 移到 CUDA，随后传给只接受 CPU tensor 的 `torch.set_rng_state()`。本轮训练未使用 `--resume`，update-20 probe 检查的是模型、optimizer、scheduler 和预测一致性，**没有覆盖完整 RNG/采样器断点恢复**。这不影响本轮成功完成的训练，但在下次断点续训前须修复并验证 next sample、augmentation、optimizer step 和 scheduler 状态。审查详情：[`reviews/stage4_1895a15/REVIEW.md`](reviews/stage4_1895a15/REVIEW.md)。
2. 此结果是 dt1、单 seed 的 Stage 4 工程训练。dt4、其他 seeds、正式真实评测和统计比较均未由本轮覆盖。
3. EventGenesis 与 V2E 的 synthetic validation EPE 差距很小，epoch 最优点也不同。不能把该差距外推为真实数据排名或 simulator 效果。
