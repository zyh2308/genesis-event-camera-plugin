# Stage 4 训练实时状态

最近刷新：2026-10-16 UTC。保留 v1、v2 启动失败记录；v3 已通过环境和源码核验并启动共享 bank，但首个物理区间未完成，campaign 已结束。当前仍无有效训练 bank，两个模型均为 0 optimizer updates。

## 固定配置

- 分支：`exp/public-sim2real-v1`；基线提交：`78f71615acc7824072b0888eb16972c9a908ea44`。
- 配置：[`configs/stage4_training_launch_v1.json`](configs/stage4_training_launch_v1.json)；flow dt1 seed 0，EventGenesis/V2E 配对，synthetic-only。
- 共享 bank：32×2 s，346×260，100 Hz endpoint，1 ms Genesis step；24 train / 8 独立 synthetic-validation rollouts。
- 真实测试评分与训练反馈关闭。

## 共享物理 bank

- 状态：v3 曾启动，首帧 endpoint 0 成功；约 8 分钟未完成首个 10 ms 区间，后退出（run 中记录 campaign exit code 4）。未写成任何 rollout；只留下不完整 `.partial` HDF5 和首帧 PNG 输入，均保留。计划目标路径 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v1_32x2s/` 当前不能作为有效 bank 使用。
- 标签按同一物理端点直接投影，flow 窗口 `(t_i,t_i+3]`，30 ms；common validity 由投影几何、目标画幅和目标深度可见性共同确定。

### 吞吐诊断

- 服务器检查时间 `2026-10-16T11:07:03Z`：GPU 1、2、4、5 空闲；GPU 3 有其他用户作业，未触碰。v3 的 EventGenesis 使用 GPU 1 做 Genesis 渲染，CPU NumPy 做 warp；环境、配置 SHA 和 V2E 源码 pin 均已通过。
- 代码推导：300 Hz cutoff 的 `tau=1/(2π·300)`，10 ms 窗、`sensor_eps_target=0.05` 要求 `ceil(dt/tau/0.05)=377` 个采样，sampler 上限截到 300。NumPy 路径对 300 个中间时刻逐项投影并执行双向遮挡 splat。实际 v3 在首个区间未产生下一 endpoint，因此不能把它写成一个已测吞吐值。
- 新增 target-size bounded diagnostic：在 346×260、同一 10 ms/300-sample sampler 下计时已有 Torch batch 后端和 DVS 累积，并在选定 alpha 上与 NumPy 参考投影/warp 比较。诊断不生成训练 bank，也不改事件模型参数。需要根据其速度、误差和显存记录，再决定是否采用并将实际运行后端锁入配置/manifest。

## 失败尝试记录

- `20261016-stage4-flowdt1-seed0-v1`：campaign exit code 90，2026-10-16T10:40:17Z 结束。配置哈希检查通过，未创建 bank、V2E 事件或训练进程。根因是启动器引用了不存在的项目内 miniconda 路径；服务器实际环境位于 `/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM`。启动器现使用 `/home/zhaoyuhan/miniconda3/bin/activate Eventbased_WAM`。完整预检与错误日志保存在 g1 的 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v1/` 和 `/home/zhaoyuhan/Eventbased_WAM/logs/sim2real_stage4_flowdt1_seed0_tmux.log`。
- 一次后续只读启动前核验因同一 miniconda 路径假设退出，未创建 run directory 或远端进程；该核验不触及 GPU，也没有修改远端文件。
- `20261016-stage4-flowdt1-seed0-v2`：campaign exit code 4，2026-10-16T10:49:41Z 结束。环境版本和下游源 bundle 校验通过，尚未创建 bank。根因是服务器 V2E 源码为去除 `.git` 的固定副本，而启动器错误地用 `git rev-parse` 验证；现改为逐文件 SHA-256 manifest 校验，并同时核验 107 个源文件及 SuperSloMo checkpoint 的大小和 SHA-256。日志同见上述 g1 tmux 外层日志；逐文件清单为 `manifests/v2e_source_pin.json`。

## V2E 25 μs 导出

- 状态：尚未启动；固定采用锁定的 `SuperSloMo39.ckpt` 和 25 μs timestamp resolution，不降级到 1 ms。

## STE-FlowNet dt1 seed 0

- EventGenesis：尚未训练；optimizer updates 0/8910。
- V2E：尚未训练；optimizer updates 0/8910。
- 训练前置条件：完整 paired bank、V2E raw-event manifest、1584 train / 528 validation windows per generator，以及相同标签 hash 核验通过。

## 运行与证据

- 运行入口：`benchmark/sim2real/scripts/run_stage4_campaign_server.sh`（通过 tmux 启动）。
- 首个交付点：两组各自真实完成至少 20 个 optimizer updates，并保存可恢复的 update-20 checkpoint、逐更新 loss/gradient 日志与恢复核验。
- v3 run root：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v3/`；银行目标及残留位置：`/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage4/training_bank_v1_32x2s/`；外层日志：`/home/zhaoyuhan/Eventbased_WAM/logs/sim2real_stage4_flowdt1_seed0_tmux.log`。所有 partial 产物保留，不作为训练数据。
- 训练状态：EventGenesis 0/8910；V2E 0/8910。此处只报告真实已完成更新；代码准备、bank 生成进程和 endpoint 0 均不算训练开始。
