# Stage 4 训练状态

状态更新时间：`2026-10-17T04:03:48Z`。Run ID：`20261016-stage4-flowdt1-seed0-v4`。

## 状态

本轮 **STE-FlowNet dt1 seed 0 配对训练已完成**。EventGenesis 与 V2E 各完成 45 epochs × 198 batches = 8,910 次 optimizer 更新；两个训练进程及 campaign 的退出码均为 `0`。独立训练日志每组共 8,955 行（8,910 update 记录和 45 个 synthetic validation 记录）。

训练在 g1 的 `/home/zhaoyuhan/Eventbased_WAM` 上运行，EventGenesis 使用 GPU 2、V2E 使用 GPU 4（Tesla V100S 32 GiB）。有效 batch 为 8；两组的初始化 state SHA-256 均为 `acc6bbd11488a21868c5f147c7f397521480166c24914cc7988a6428d6ed71cf`。登记预算、固定配置和完整结果见 [`STAGE4_SEED0_TRAINING_REPORT.md`](STAGE4_SEED0_TRAINING_REPORT.md)。

## 合成验证结果

| 生成器 | 最佳 synthetic validation EPE | 最佳 epoch | 第 45 epoch EPE |
|---|---:|---:|---:|
| EventGenesis | 0.9389129835 px / 30 ms | 16 | 0.9466441114 px / 30 ms |
| V2E | 0.9406893614 px / 30 ms | 17 | 0.9438607459 px / 30 ms |

checkpoint 只根据 synthetic validation 选择。当前结果是一个 seed 的短阶段合成训练结果；不构成真实数据性能、生成器优劣或 Sim-to-Real 结论。真实测试未用于训练、选择或调参。

## Artifact 位置

- 服务器 run root：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4`
- 完整训练日志、metrics JSONL、状态 JSON、退出码和模型权重：`flow_dt1_seed0/{eventgenesis,v2e}/`
- 首 20 次更新的已提交独立证据：[`evidence/S4/20261016-stage4-flowdt1-seed0-v4-update20/README.md`](evidence/S4/20261016-stage4-flowdt1-seed0-v4-update20/README.md)
- 完整结果、checkpoint 哈希、运行命令和复现信息：[`STAGE4_SEED0_TRAINING_REPORT.md`](STAGE4_SEED0_TRAINING_REPORT.md)
- 工作树中独立审查的进度快照保留于 [`reviews/stage4_1895a15/REVIEW.md`](reviews/stage4_1895a15/REVIEW.md)；该审查早于训练完成，不能将其中的进度表当作最终状态。

## 后续注意

训练完成后没有活动训练进程。独立审查发现 `--resume` 路径恢复 Torch CPU RNG state 的设备类型问题；本轮训练未使用该路径，不影响已完成结果，但在下次使用断点续训前应修复并单独核验。详见独立审查报告和完整训练报告。
