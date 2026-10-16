# Stage 4 训练实时状态

最后刷新：`2026-10-16T16:38:00.550116+00:00`

## 固定配置

- 分支：`exp/public-sim2real-v1`；代码提交：`1fab7e17e201943c77703aa8a1c0c6eaefbb8f5a`。
- 配置：`/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real/benchmark/sim2real/configs/stage4_training_launch_v1_torch_cuda.json`（SHA-256 `c0ab0d17f024b00106957c3e44d7a013e920f6e760f70266c43a9cf904634685`）。
- 合成 bank：`/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v1_torch_cuda_32x2s`；计划 32×2 s，346×260，100 Hz endpoints，1 ms physics。
- 真实数据评分/反馈：关闭。两组只用各自合成事件输入和共享合成灰度/标签。

## 共享物理 bank

- 进度：32/32 rollouts; 状态 `complete`; 更新时间 `2026-10-16T14:00:06.525750+00:00`.

## Previous attempts

- `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v1`：campaign exit code `90`，结束/更新时间 `2026-10-16T10:40:17Z`；启动器和标准错误见 `/home/zhaoyuhan/Eventbased_WAM/logs/sim2real_stage4_flowdt1_seed0_tmux.log`。
- `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v2`：campaign exit code `4`，结束/更新时间 `2026-10-16T10:49:41Z`；启动器和标准错误见 `/home/zhaoyuhan/Eventbased_WAM/logs/sim2real_stage4_flowdt1_seed0_tmux.log`。
- `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v3`：campaign exit code `143`，结束/更新时间 `2026-10-16T11:04:28Z`；启动器和标准错误见 `/home/zhaoyuhan/Eventbased_WAM/logs/sim2real_stage4_flowdt1_seed0_tmux.log`。
- 完整 shared HDF5 SHA-256：`e9e57131c84059047bddd0d34ac3271f23f5e4ff0e4fa5e0b9f8579772ee337a`；split SHA-256：`60722e124b24c8da2306f3a8a4115ec8588949653a2f014b3687c2a2dfc59aa1`。

## V2E 25 μs 导出

- 完成：32 个 rollout；manifest `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/v2e_events/v2e_complete_manifest.json`，SHA-256 `5d79bad2f9fe81e1fad8af0676b9048aefaf73c0b6edd8db7f39f31ef82beb2f`。

## 30 ms STE-FlowNet seed 0

- 两组 adapter：1584 train / 528 synthetic-validation windows；manifest `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/20261016-stage4-flowdt1-seed0-v4/steflow_30ms/training_bank_manifest.json` SHA-256 `c9da4158d9775997f08dd75ce0ac2ccc33873d983422ef87c931e834f6da0624`。
- **eventgenesis**：`training`，GPU `2`，optimizer updates `58/8910`；loss `1190.74560546875`（photo `1186.852783203125`；smooth `0.38927948474884033`）；有效 batch `8`；checkpoint `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/flow_dt1_seed0/eventgenesis/last.pt`；SHA-256 `3149c07b7d13226dd63ca72603d3880ef6e44297d02b807beba3d5f3a075221d`；更新时间 `2026-10-16T16:35:49.986771+00:00`。
  - update 20 checkpoint reload/optimizer/scheduler probe 通过；prediction max abs `7.152557373046875e-07`。
- **v2e**：`training`，GPU `4`，optimizer updates `58/8910`；loss `1207.3978271484375`（photo `1203.6732177734375`；smooth `0.3724561929702759`）；有效 batch `8`；checkpoint `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/flow_dt1_seed0/v2e/last.pt`；SHA-256 `45047da79159dc0e8f1a53c246c2695b6896d2b250b3c97932eba06041af5924`；更新时间 `2026-10-16T16:35:49.624643+00:00`。
  - update 20 checkpoint reload/optimizer/scheduler probe 通过；prediction max abs `6.705522537231445e-08`。

## 运行与证据

- Run root：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4`。运行日志、PID、exit code 和 checkpoint 均留在服务器 run root。
- 最近刷新器 PID：`2515402`。
- 当前结果仅证明训练链路和更新确实发生；不构成 generator 优劣或 Sim-to-Real 结论。
