# Stage 4 训练实时状态

最近刷新：工程入口实现中；尚未在服务器创建 Stage 4 运行进程。

## 固定配置

- 分支：`exp/public-sim2real-v1`；基线提交：`78f71615acc7824072b0888eb16972c9a908ea44`。
- 配置：[`configs/stage4_training_launch_v1.json`](configs/stage4_training_launch_v1.json)；flow dt1 seed 0，EventGenesis/V2E 配对，synthetic-only。
- 共享 bank：32×2 s，346×260，100 Hz endpoint，1 ms Genesis step；24 train / 8 独立 synthetic-validation rollouts。
- 真实测试评分与训练反馈关闭。

## 共享物理 bank

- 状态：尚未启动。计划写入服务器 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v1_32x2s/`。
- 标签按同一物理端点直接投影，flow 窗口 `(t_i,t_i+3]`，30 ms；common validity 由投影几何、目标画幅和目标深度可见性共同确定。

## V2E 25 μs 导出

- 状态：尚未启动；固定采用锁定的 `SuperSloMo39.ckpt` 和 25 μs timestamp resolution，不降级到 1 ms。

## STE-FlowNet dt1 seed 0

- EventGenesis：尚未训练；optimizer updates 0/8910。
- V2E：尚未训练；optimizer updates 0/8910。
- 训练前置条件：完整 paired bank、V2E raw-event manifest、1584 train / 528 validation windows per generator，以及相同标签 hash 核验通过。

## 运行与证据

- 运行入口：`benchmark/sim2real/scripts/run_stage4_campaign_server.sh`（通过 tmux 启动）。
- 首个交付点：两组各自真实完成至少 20 个 optimizer updates，并保存可恢复的 update-20 checkpoint、逐更新 loss/gradient 日志与恢复核验。
- 当前文件是启动前状态；不得将代码准备或进程创建描述为训练已开始。
