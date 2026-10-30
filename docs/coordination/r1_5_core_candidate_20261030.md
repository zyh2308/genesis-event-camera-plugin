# R1.5 核心修复候选：验收记录

**结论：候选通过 CPU 数值控制与缓存回放，但暂不建议冻结或切换正式默认 core。** 覆盖率 splat、端点恒等、未知支持和真实光度锚点接口均有可复核实现。严格 1 ms 缓存回放准确；单个 5 ms 中点不足以重现机器人片段的事件时间与数量，默认运动采样器在该 10 ms 窗口还要求额外 6 个真实状态。CUDA 未能验证，候选 warp 的 CPU 成本也远高于 10 ms 控制周期。

## 运行范围和隔离

- 分支：`exp/r1_5-core-candidate-20261030`，核心候选代码提交 `52d4df15b0c3e89da972471c784278c0cfcbbb32`。结果运行登记的代码提交与该提交一致。
- 改动只在隔离 worktree；正式工作目录 `exp/robot-closed-loop-v1`、冻结版本、HDR overlay、A/B/C 场景和共享环境均未修改。无新 renderer、物理推进、训练或事件排名。
- 本次验证使用 CPU 和已存 episode 004/005 缓存、已有 g1 阴影 × MSAA 因子渲染。完整本地验证脚本耗时 12.19 s。结果目录仅 628 KB。
- g1 于 `2026-10-30T11:05:55Z` 可连接；六张 GPU 均有活跃 compute 进程，利用率 81–91%，未启动 CUDA。g3 检查因 Windows OpenSSH 无权读取指定 `known_hosts`、主机校验失败而停止；没有重试或连接后执行远程命令。

## 候选具体改动

1. NumPy 与 Torch 都先按双线性足迹近似累计前景覆盖，再决定深度层和亮度。极小正权重不再因先参与 z-buffer 而单独排除主要表面。覆盖率只是当前重建核的近似足迹，**不是** renderer 的精确像素覆盖率。
2. t0/t1 直接返回已观测端点 radiance；不再让带浮点投影的恒等变换改变端点。覆盖不足保留 `unknown` 标记，不将微小贡献提升成可信可见像素。
3. 两端表面标签/深度一致且共同可见时，使用分段 log-radiance 随时间插值；跨表面和支持不足处不假设可恢复。`hold` 仍只是显式近似策略。
4. strict 路径复用 RenderRequest 与缓存输入：要求精确时间戳、radiance 来源、可用时间及每段最大 5 ms 实测锚点间隔；只接收当前区间内真实可用的样本。输出最早可用模拟时刻不早于区间右端，另计渲染/计算墙钟时间。没有中间状态时会请求真实采样，不把 warp 子步数冒充光度观测。

## 预先固定的验收判据和结果

门槛写在版本化配置中，之后运行；完整字段见结果 `config.json`。

| 判据 | 结果 | 判断 |
|---|---:|---|
| 两后端精确端点误差 | NumPy/Torch 均为 0 | 通过 |
| CPU NumPy/Torch 光度与覆盖差、支持掩码差 | 最大差 0；支持/可见性不一致 0 px | 通过 |
| 连续斜面 ±1e-6 px 微动 | 内部最大亮度变化 2.98e-8 | 通过 |
| 1e-6 足迹邻点 | coverage 9.54e-7、visible=true、supported=false | 通过；未知与可见被分开 |
| 前景遮挡与部分足迹 | 前景 0.8、背景 0.02；边缘 0.605/0.215 与期望一致 | 通过构造控制 |
| 10 ms 已知 log-linear ramp，需真实中点 | 48 vs 48 事件；最大时间误差 3.81e-10 s；静止控制 0 事件 | 通过该已知曲线，不外推到未采样动态光照 |
| 缓存因子图 unknown fraction | 4.37–4.67%（移动轮廓 0.19–0.21%） | 低于预设 15% |
| 支持区内部 log 误差 P95 | 两机器人缓存各配置最高 2.45e-4 | 低于预设 0.2 |
| CUDA 与真实 renderer 成本 | 均未测得 | 冻结门槛未满足 |

深度 tie probe 在约 1 m 深度、间隙 0.86–1.05 µm 下输出连续；配置的 z tie 容差约 0.954 µm。它说明该最小探针未复现亮度跳变，不证明所有尺度、斜面或材质下无边界问题。轮廓附近仍观察到高尾误差：机器人因子格中有 0–120 个 supported 像素超过 0.2 log，最大值达 2.156；这不违反预先限定为“支持区内部 P95”的门槛，但不应被其掩盖，需在后续契约中继续报告。

## 实际轨迹 / 阴影 / MSAA 对照

两条机器人轨迹的状态时刻均为 0.5805、0.5855、0.5905 s。先用两端点生成中间图像，再对照实际缓存中间图；阴影开关与 MSAA 开关在每个条件内部匹配。表中事件是三张 5 ms 间隔图像送同一理想 DVS 的**稀疏对照**，不是 1 ms 事件真值。`L1` 是逐像素、逐极性的事件计数差。

| Episode | 阴影 | MSAA | 三帧参考事件 | warp 中点事件 | 计数 L1 | 中点未知比例 | 支持区内部 log P95 |
|---|---|---:|---:|---:|---:|---:|---:|
| 004 | 开 | 开 | 413 | 397 | 16 | 4.37% | 1.68e-5 |
| 004 | 开 | 关 | 513 | 490 | 29 | 4.42% | 1.67e-5 |
| 004 | 关 | 开 | 40 | 40 | 0 | 4.37% | 1.02e-5 |
| 004 | 关 | 关 | 97 | 97 | 0 | 4.42% | 7.20e-6 |
| 005 | 开 | 开 | 615 | 582 | 59 | 4.55% | 2.25e-4 |
| 005 | 开 | 关 | 927 | 941 | 118 | 4.67% | 2.44e-4 |
| 005 | 关 | 开 | 252 | 252 | 0 | 4.55% | 1.75e-4 |
| 005 | 关 | 关 | 613 | 631 | 42 | 4.67% | 1.57e-4 |

shadow-off 只用于诊断，不能据此把正式阴影关掉。匹配渲染显示开关会显著改变静态地面与全图事件；移动轮廓控制中，开阴影时 38 个受支持像素差超过 0.2 log，关阴影为 0。它支持“真实光度过程需采样”的判断，但不表示 shadow 是全部残差的唯一来源。MSAA 也改变事件数，说明 resolve 设置必须固定记录。

把每格已存的实际中间 HDR radiance 样本通过 strict 接口输入后，八格输出都与三帧 Direct-HDR 稀疏事件逐项完全一致；因为缓存已存在，额外 render count 为 0。此项验证接口和样本回放，不是候选生成器独立重建成功。

### 采样密度与事件时间

- 两条机器人片段使用全部已有 1 ms Direct-HDR 帧时，10 张图像序列逐像素相同，理想 DVS raw events 完全一致（004：625；005：939）。这些是已有帧缓存，不是本轮重新渲染。
- 仅在 10 ms 区间加入一个 5 ms 中点时，004 只有 413 事件（对 1 ms 参考计数 L1=212，配对时间 P95=4.013 ms）；005 只有 615 事件（L1=324，时间 P95=4.275 ms）。它不满足动态片段精度。
- 同一窗口默认运动采样器实际要求 6 个额外时间点：0.58175、0.58300、0.58425、0.58675、0.58800、0.58925 s；加已有 5 ms 中点和两个端点共 9 个状态。这不是一个中点即可解决的窗口，但当前证据只有两条短轨迹，不能宣称“大多数窗口”都需如此。
- ramp 控制的 provider 返回公式已知的准确中点，仅验证已知平滑曲线。对于 <5 ms 的未采样变暗/恢复等过程没有精度保证；末端不变也不能证明中间无瞬态。

## 成本、时序与限制

- 本轮在缓存上的 strict API CPU ready 延迟约 0.94–0.96 s/10 ms 区间；每个 warp 中点约 0.25–0.28 s（320×240）。它不实时，且不是 CUDA 性能估计。
- 本轮新 renderer 调用数为 0，因此候选独立 renderer 成本**未测**。复用的 2026-10-29 g1 factorial 记录每种配置 6 次渲染总计约 0.051–0.095 s，即该旧设置下约 8.5–15.8 ms/次；这只是历史参照，不能当成本轮候选在线实测。若每 10 ms 需要 7–10 个中间样本，按此历史速率会超过仿真区间长度，尚未包含 Python warp 和事件处理。
- 因果接口让采样输出最早在区间右端可用；另有墙钟渲染与处理延迟。离线缓存能够回放过去，不代表机器人闭环可在对应时刻无延迟收到这些事件。
- 下游学习网络没有用本轮新事件重训，旧 checkpoint 也没有拿来当候选正确性的判据。当前结论仅覆盖生成图像/事件 API 的这些控制和缓存片段。

## 候选冻结建议

**不切换正式默认，不冻结为通用 core。** 可把 `52d4df1` 留作可回退候选，作为后续 CUDA/renderer 核验的基线。数值覆盖与端点修复已有一般性 CPU 证据；端到端的动态光度时序和可接受成本尚未证明。

下一步有两个清晰方向，供研究侧决定优先级：

1. **高保真、近 direct 路径：** 对动态或采样器判定不可靠的 10 ms 窗口实渲染约 1 ms 光度锚点，再验证实际 CUDA、renderer 耗时、缓存/控制可用时延。优点是时间信息有来源；代价按既有渲染速率可能超过实时预算。
2. **收窄近似路径的支持范围：** 只在已验证的平滑、稳定光度区间使用 5 ms 分段 log-linear warp；阴影、新显露/遮挡或 motion sampler 要求更多采样时明确标 unknown / request render，不回退成“准确”。优点是成本可控且边界诚实；代价是机器人动态片段可能频繁不可用。

当前没有充分依据推荐把全量补渲染变成默认，也没有依据把 5 ms 间距用于机器人动态片段。若研究侧要求冻结，应先完成 g1 CUDA parity + 阈值附近事件对照、至少一条未调过的动态片段严格采样成本，并书面决定实时延迟预算。

## 复跑命令与证据

在 WSL 本地复跑，不需要访问服务器：

```bash
cd /home/科研/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1_5-core-candidate
r15_py=/home/科研/Eventbased_WAM/repositories/genesis-event-camera-plugin-bounce-catch-20261024/.venv-robot/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" "$r15_py" \
  benchmark/robot_closed_loop/diagnostics/validate_r1_5_core_candidate.py \
  --cache-root /home/科研/Eventbased_WAM/repositories/genesis-event-camera-plugin/benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059 \
  --factorial-root /home/科研/Eventbased_WAM/repositories/genesis-event-camera-plugin/benchmark/robot_closed_loop/results/r1_5_shadow_multisample_cached_states_20261029_g1_gpu0_previewfix_855b4b0 \
  --output benchmark/robot_closed_loop/results/r1_5_core_candidate_20261030b_cpu_cached
```

输出目录内的 `SHA256SUMS` 覆盖 JSON/CSV/NPZ 和复制配置；本轮校验全部通过。关键哈希：

- 候选代码模块：`physics_warp.py` `8c23c601677cf25c1df445ea65235b154f5895353ba9bc67f37da40a3ae3fe41`；`torch_physics.py` `aafdfcb2c7498592e6e9b2149af4515d53755aa2caea88574270672b3e15e7ad`；`physics_interpolator.py` `3361a5b554e802fc1d68515c2c28c5ed1aea3d2ea274baee45846c6040ee35f9`；`physics_plugin.py` `85226f8e0fa57c26512a77d76ee3444580ebe3566a4871682d1c108a1a449bba`。
- 配置 SHA256：`bde0882f0a70e14dfea4d70f94557579bcb089aaa78bfca85a5c18d6bdc0b7b4`；验收脚本 SHA256：`e39b4f8cd8d0395492dba5f4130de2e90e773abe3dda3523f9dfe0e3a0d20ff0`。
- episode 004 state cache / direct evidence：`0525c18c8d89f5831ef0cb71ac9e3b6b130d60579daf374f06d5cfcee5b879d7` / `b3afe5b5badee3d2606d343bb750aa970887928c41b8001a5fc4e9ff32b9c321`。
- episode 005 state cache / direct evidence：`91c1397642e317d6bb51150b4107c518b52e4fa9fcf5f1caf6a2a91315c7d434` / `f161079a24c1bf9fee838a6fbb2a0fa5d785435b116f0ea5628e6efbba497236`。
- 因子渲染输入源提交 `855b4b0bc7de96170b4bfee8d7002b370782f346`；原 summary SHA256 `060c991de176e2129f4c6f284b6ac76b85f03e9ee02d1cb9374ac81728f03200`。
