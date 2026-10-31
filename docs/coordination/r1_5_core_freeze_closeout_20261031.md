# R1.5 core 收尾验收与冻结建议（2026-10-31）

## 结论

对当前 `exp/r1_5-core-candidate-20261030` 分支完成了收尾证据审计。**strict sampled Genesis-linear-HDR 候选的既定验收项均有通过证据，建议按限定范围冻结。** 本轮没有改动 core 算法，也没有重跑已通过的 GPU 实验；新增审计器核验 F/I/K/B 四组归档哈希、门槛结果和实际运行提交中的核心源码是否一致。

这项审计不证明所有未采样光度过程都能恢复。冻结主张仅适用于有时间戳、来源和可用时间的真实线性 HDR 样本；严格路径在真实采样不足时请求采样或保留未知，不把密集 warp 子步伪称为观测。

## 核心版本一致性

CUDA F、HDR I、真实因果 K、CPU B 与当前候选分支的四个核心文件逐字节相同：

| 模块 | SHA256 |
|---|---|
| `physics_interpolator.py` | `0ac9174f5e3a0635de0dc8cb7a28c51f4d6ac0eb690bfeb1ae8486bf44a5941d` |
| `physics_plugin.py` | `85226f8e0fa57c26512a77d76ee3444580ebe3566a4871682d1c108a1a449bba` |
| `physics_warp.py` | `8c23c601677cf25c1df445ea65235b154f5895353ba9bc67f37da40a3ae3fe41` |
| `torch_physics.py` | `8a409765268746fefedb15a805c4d95db857bf6a4359c899cd65e6ac4dfe5955` |

代码基线为 `90f792d166f000ef4f9e072a97633cc4ad39c8b7`；F/I/K/B 实际执行提交分别是 `9e6b5d7e2689d541087a56ff1fbf0fc904571b69`、`26de5e410ecc550a9dc7f61b25aa2926d6399e0a`、`7559893ea4168b979679ee32eac03806050a9b03`、`e71f2b7b7369994d714f540ff498f3902b90081c`。它们都在当前分支历史中，核心文件哈希完全一致。

## 已通过的证据

| 项目 | 实测结果 | 证据 |
|---|---|---|
| CUDA/CPU 支持与事件数值一致性（F） | 106 个历史差异位置统一为 69 个共同支持、37 个共同未知；两集 native mask XOR 为 0；四种重建/DVS 组合及阈值邻近探针通过。 | [run F](../../benchmark/robot_closed_loop/results/r1_5_cuda_support_forensics_20261031f/README.md) |
| 真实 Direct-HDR 中间帧（I） | episode 004/005 各新渲染 9 个 1 ms 状态，共 18 帧；HDR 最大误差为 0、分割差为 0，DVS 事件逐项匹配 625/625 与 939/939。 | [run I](../../benchmark/robot_closed_loop/results/r1_5_strict_direct_rerender_20261031i/README.md) |
| renderer 与真实运行成本（I） | 使用 `gec-robot/.venv-robot`、Genesis 1.2.2、EGL rasterizer；相机模块 SHA256 `78c18ffada07a98657ddf11326e833fc928431ecead674434aadb25dc023b714`，包含 `radiance` 参数。18 次 render+readback 为 288.7 ms，平均 16.04 ms/帧；完整进程 wall 为 76.08 s。 | [run I manifest](../../benchmark/robot_closed_loop/results/r1_5_strict_direct_rerender_20261031i/output/run_manifest.json) |
| 实际物理与因果顺序（K） | 两条相反方向、各 0.3 s 的正常 Genesis 物理轨迹；1,200 步、1,202 帧、300 个控制读取。样本先被真实捕获，事件包在右端点可用，event-only close 命令在下一物理步生效；无 GT、分割、深度输入或物体状态重设。 | [run K](../../benchmark/robot_closed_loop/results/r1_5_live_causal_hdr_20261031k/README.md) |
| 通用 CPU 控制与严格采样（B） | 静态零事件、微小双向运动、精确端点、前后景遮挡、CPU NumPy/Torch 一致；已知 ramp 48 个事件，时刻误差最大约 `3.81e-10 s`。两集 1 ms 缓存序列精确回放。5 ms 单中点不足，因此不能把该间距宣称为普遍充分。 | [run B](../../benchmark/robot_closed_loop/results/r1_5_core_candidate_final_20261102b/README.md) |

新增的 manifest 级审计通过：`python benchmark/robot_closed_loop/diagnostics/audit_r1_5_core_freeze_closeout.py --output benchmark/robot_closed_loop/results/r1_5_core_freeze_closeout_<new-run-id>`。本次输出位于 [`run audit`](../../benchmark/robot_closed_loop/results/r1_5_core_freeze_closeout_20261031a/README.md)，逐项校验了四个结果目录的 95 项哈希、固定接受门槛和上述五个提交的核心文件哈希。它是归档一致性审计，不是新的 Genesis 仿真运行。

## run G 失败任务收尾与资源

run G 在 `2026-10-30T17:30:14Z` 以 exit 1 结束，原因是错误选择基础 conda 环境，其 `Camera.render()` 不支持 `radiance`；没有生成有效 HDR 帧、事件或成本数据。该失败已由后续 run I 使用已核验的 HDR 环境替代，旧日志保留。

2026-10-31T04:56Z 在 g1 复核：run G 记录的 worker PID `1940344` 和 renderer PID `1940351` 均不存在，tmux session 不存在，GPU1 UUID 锁和 R1.5 公共 slot 都不存在。失败任务已经收尾，无需再清理。该时刻 GPU0/GPU1 有其他用户的计算进程；GPU2–GPU5 未见 compute 进程，但这只是瞬时观测，不代表已分配，本轮没有新增占卡或启动重复 renderer。

## 冻结范围与未完成主张

建议将 `r1.5-strict-sampled-20261031` 保持不动，并在本收尾审计提交后增加一个独立 closeout tag。适用范围是：

- 双线性足迹覆盖率是已声明的近似；0.995 支持门槛不降低，支持不足明确为 unknown。
- 精确端点、float64 投影及已测 CPU/CUDA 支持决策和 DVS 数值路径一致。
- strict 光度路径只接收实际、带时间戳的 scene-linear HDR 样本；输出不早于 bracket 右端点。
- 机器人参考配置 [`r1_5_core_strict_sampled_20261102.json`](../../benchmark/robot_closed_loop/configs/r1_5_core_strict_sampled_20261102.json) 将光度锚点最大间隔限制为 0.5 ms；低于采样间隔的变化仍不在保证范围内。
- 为兼容旧调用者，Python 构造器默认仍为 `strict_fidelity=false`；科学运行须显式使用 strict 配置和真实样本 provider。主分支合并继续留待验收。

尚未证明：旧学习 checkpoint 定位恢复、R2 抓取、EVK4 绝对光度标定、墙钟实时。K 中固定地面出现局部亮度变化，但没有同一 live 轨迹的 shadow-off 对照，不能将其全部归因于阴影。CPU shadow/MSAA 离线对照显示阴影能在简单控制中造成残差，但不等价于 live 因果隔离。

## 复跑入口与哈希

- 归档审计：运行上述 `audit_r1_5_core_freeze_closeout.py` 命令并为每次重跑使用新 run ID；输出 `SHA256SUMS` 覆盖 audit README/JSON。
- renderer、因果运行和资源/PID/配置复跑命令分别见 run I 与 run K README；启动新实验必须使用新的 run ID，并通过当前资源门槛。
- strict profile SHA256：`3525cad9e84982a2b835b1b7d32428b94be24508fe28faf9f0d79002dfb10818`。
- 本次 audit summary SHA256：见 audit 结果目录 `SHA256SUMS`；四个历史原始结果及其 SHA256SUMS 均保留。
