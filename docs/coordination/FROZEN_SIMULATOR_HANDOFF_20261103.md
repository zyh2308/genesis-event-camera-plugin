# EventGenesis 仿真器修复与冻结交接报告

日期：2026-11-03。接收方：等待新版仿真器的研究与执行会话，包括机器人 A/B/C、事件质量和下游训练任务。

**交接结论：原稀疏刚体重建路径已完成修复，CPU/CUDA 与定位验收通过，并完成一次严格事件驱动物理抓取。各会话可按本文接入固定版本，恢复相应实验。核心停止继续调优；场景、数据、感知模型与控制器分别维护。**

## 1. 唯一版本入口

| 项目 | 固定值 |
|---|---|
| 仓库 | `https://github.com/zyh2308/genesis-event-camera-plugin` |
| 冻结分支 | `freeze/robot-sparse-cuda-r2-20261102` |
| 完整发布提交 | `62da4f35ce3c5c96f04268bd1687a27657af9238` |
| 核心代码来源提交 | `6ac417c7a03e125f40ebf967796bf28191d55859`；发布提交包含同一核心及最终配置、控制与证据 |
| 生成配置 | `benchmark/robot_closed_loop/configs/event_sparse_reconstruction_20261102.json` |
| 原方块任务控制配置 | `benchmark/robot_closed_loop/configs/shared_control_r2_verified_20261102.json` |
| 冻结描述 | `benchmark/robot_closed_loop/results/r1_6_sparse_cuda_freeze_20261102/FROZEN_BASELINE.json` |
| 最终独立复核 | 审计提交 `d3415f38e4e3bee1d8d19e9b0c732889863d4a52` 中 `docs/coordination/freeze_audit_20261103/` |

**应固定“发布提交 + 生成配置”，不要只拉最新分支或只 cherry-pick 最后一笔发布提交。** 修复代码分布在此前提交中；最后一笔提交本身不是独立补丁包。只采用旧 coverage-splat 候选、旧端点/hold 处理，也不等于采用本次最终版本。

[固定版本](https://github.com/zyh2308/genesis-event-camera-plugin/tree/62da4f35ce3c5c96f04268bd1687a27657af9238) · [原始冻结报告](https://github.com/zyh2308/genesis-event-camera-plugin/blob/62da4f35ce3c5c96f04268bd1687a27657af9238/docs/coordination/ROBOT_SPARSE_CORE_FREEZE_20261102.md) · [最终复核与哈希勘误](https://github.com/zyh2308/genesis-event-camera-plugin/tree/d3415f38e4e3bee1d8d19e9b0c732889863d4a52/docs/coordination/freeze_audit_20261103)

## 2. 为什么修，以及最后修了什么

最初是感知模型在共同事件输入上经常把方块位置看错。诊断逐步追到稀疏端点之间的图像重建：既有投影覆盖处理错误，也有亮度随时间变化处理不合理。早期只改覆盖规则，图像指标改善却未恢复定位；该候选不作为本次交接版本。

最终验收以定位像素误差下降为主要任务指标，同时保留几何、时间和数值检查约束，避免只迎合一种参考事件统计。

| 问题 | 已冻结的处理 | 作用 |
|---|---|---|
| 很小的边缘投影权重先抢赢深度竞争，替换整像素主要亮度 | 按前后表面的覆盖面积合成，保留足够支持的判定 | 减少整像素跳变与错误事件 |
| 已知端点再投影，可能改变本来准确的输入 | 区间两端保持真实输入值 | 消除端点被重建过程改写的问题 |
| 区间内过早使用后一帧亮度、未知区突然切换 | `coverage_time_log_linear`：根据时间与投影覆盖连续融合 log 亮度，并连续处理近似未知区 | 修正亮度变化及阈值事件时序，减少人为突发 |
| CPU/CUDA 投影精度及边界判断不一致 | 投影几何参数保留双精度；统一端点容差与张量输出/回写处理 | 保持两后端支持判定和事件结果一致 |

支持门槛仍为 **0.995**；未知区域仍保留未知标记。近似亮度不被标成真实观测。独立 strict-sampled 模式仍保留，但它不是本次稀疏算法修复路径。

**数据流仍是：原生低频线性 HDR + depth/segmentation + 相机/物体位姿 → 几何中间图像重建 → DVS → 事件。** 生成与训练未使用高频 HDR 补图，未使用目标 GT 掩码筛事件。

## 3. 修复效果与验收证据

### 3.1 定位效果

误差单位为原始 **320×240 图像像素**，统计包含全部查询，不按可达判断或置信度删除难例。P95 表示 95% 查询的误差不超过该值。

| 评估材料 | 模型 | 中位数 ↓ | P95 ↓ | 大于 10 px 或缺失 ↓ |
|---|---|---:|---:|---:|
| 9 条固定开发轨迹，581 次查询 | 旧 Ours | 8.18 | 106.72 | 225/581 |
| 同一批 581 次查询 | 最终稀疏修复模型 | **2.33** | **5.77** | **3/581** |
| 另外 3 条归档轨迹，193 次查询 | 旧 Ours | 9.08 | 112.38 | 87/193 |
| 同一批 193 次查询 | 最终稀疏修复模型 | **2.20** | **5.54** | **0/193** |

旧/新模型分别由对应事件训练；网络仍为同一 `heatmap_v2`，训练 seed 606、40 epoch 预算、batch 32、拆分与模型选择规则相同。新模型有 1,203 个训练窗口、566 个验证窗口，按自身验证选第 36 epoch，测试前锁定权重。这不是仅更换输入但保持同一权重的比较，也不是正式 Ours/V2E 排名。

所有模型读取相同的固定 Direct-HDR **评估事件**。高频参考事件没有进入候选生成、训练或 checkpoint 选择。九条训练/验证轨迹共读取原有 **1,814 张端点图**，新增渲染与高频帧读取为零。归档三条轨迹是补充开发复查，不宣称正式独立测试集。

### 3.2 CUDA 与受控检查

g1 的 RTX3090、V100S 各通过 33 项受控检查、4 段真实缓存的 16 个稀疏区间，以及 16 次固定模型查询。每种后端产生 19,725 个事件：事件位置、极性、数量一致，支持掩码分歧为 0，事件时间最大差约 0.260 µs。CUDA 生成差异导致的定位变化为 0 px；CPU/CUDA 推理数值变化最大约 0.000212/0.000205 px。

另有静态零事件、端点精确、平移、覆盖不足保留未知、严格模式拒绝未知、公开入口只访问两个端点等检查。已知 log 亮度 ramp 应在 2.22/4.44/6.67/8.89 ms 触发阈值，修复后回到应有时刻。最终复核又用冻结配置重新构造 CPU 公开入口，320 个 ramp 事件与解析时刻一致、静态 0 事件、额外 HDR 读取为 0。

### 3.3 严格事件抓取

用户约定的最低冻结条件是“至少完成一次严格事件驱动抓取”。最终达到：

| 项目 | 已验收结果 |
|---|---|
| 条件 | seed `2026110202`；0.50 m/s；横向 +55 mm；观测延迟 10 ms |
| 感知 | 固定的修复稀疏事件训练模型，在线读取独立 Direct-HDR 评估事件 |
| 控制输入 | 模型估计状态、过去观测、公开滑动标定、机器人自身反馈；目标 GT 仅供评分 |
| 严格保持 | 连续 0.500 s，即 1,000 个 0.5 ms 物理步 |
| 保持窗口最低底面净空 | 80.082 mm，门槛 80 mm |
| 最大相对滑移 | 3.202 mm，限制 10 mm |
| 接触 | 合格窗口持续双侧接触，桌面支撑样本 0 |
| 因果与复算 | 未来观测违规 0；500 条 1 ms 轨迹记录的合格判定复算无分歧 |

这是一个已用于开发的 seed 的功能验收，不是正式抓取成功率。此前 13 次开发运行含重复 seed 和多个控制版本，失败全部保留，不合并成独立样本成功率。

R2 为此另行修正了下游速度估计、闭合速度参考与确认时机、机器人自身横向跟随，以及运动中/稳定后的夹持保护。它们是**控制链路修改**，不属于所有任务必须继承的事件仿真器核心。原严格物理评分没有放宽，模型没有在 R2 中重新训练。

### 3.4 最终归档复核

GitHub、本地与 g1 的 17 个核心文件一致；4 个最终下游文件、checkpoint、Panda 资产与接触求解源码与运行记录一致。11 个运行目录清单中的 **1,598 个文件**重新逐项校验通过。最终复核不修改代码、不重训、不重复占用 GPU。

## 4. 接入时必须使用的配置

入口类是 `genesis_event_plugin.physics_plugin.GenesisPhysicsEventPlugin`。历史构造器默认仍为旧模式；**更新代码后必须显式选新配置**。

| 关键参数 | 冻结值 |
|---|---|
| `warp_composite` | `coverage_time_log_linear` |
| CUDA 后端 | `warp_backend="torch"`, `device="cuda:0"`；这里是当前可见 GPU 的局部编号 |
| CPU 等价后端 | `warp_backend="numpy"`, `device="cpu"` |
| 支持门槛 | `min_coverage_fraction=0.995` |
| 稀疏近似模式 | `strict_fidelity=False`, `disocclusion_policy="hold"` |
| 辐射输入 | 原生 float 线性 HDR，`input_space="linear"`；不能替换为 PNG 显示灰度 |
| 亮度响应 | `physical_log`，scale/white level 均 1，floor `1e-6` |
| 中间真实状态补采接口 | `intermediate_state_provider=None`；本配置不依靠高频补渲染 |
| 验收传感器 | 正/负阈值 0.2，理想、无噪声、无带宽限制；不是 EVK4 实物标定 |

最小构造示例如下。两个 provider 必须来自同一时刻、同一相机和场景；此处只是连接示例，不替各场景实现数据采集。

```python
import json
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.noise_model import DvsNoiseConfig

with open(PROFILE_PATH, encoding="utf-8") as f:
    profile = json.load(f)
kwargs = dict(profile["plugin_kwargs"])
# CPU 复核时才覆盖：kwargs.update(device="cpu", warp_backend="numpy")
plugin = GenesisPhysicsEventPlugin(
    noise_config=DvsNoiseConfig(**profile["noise_config_kwargs"]),
    frame_provider=frame_provider,
    motion_state_provider=motion_state_provider,
    **kwargs,
)
plugin.attach(scene, camera)
```

`frame_provider(scene, camera)` 返回含 `radiance/depth/seg` 的映射；`motion_state_provider(scene, camera)` 返回 CV 约定的 `camera_to_world` 及按 segmentation ID 索引的 `object_to_world` 4×4 矩阵。使用完整物体/机器人 link 位姿，不用只有实体线速度的旧近似。场景时间戳以真实模拟状态为准。

离线生成可以在拿到后一端点后生成整个区间事件；**在线控制不能提前使用后一端点尚未到达时才生成的事件**。分别记录事件时间和可用时间，再叠加协议规定的延迟。训练数据生成可用的几何真值不等于下游控制可以读取目标真值。

现有任务宜在空闲时建立固定提交的独立 worktree，或在自己的分支审查吸收对应核心与配置后逐文件核对。不要切换、reset 或覆盖另一会话正在运行的 checkout，不要直接导入另一会话会继续改写的工作目录。共享缓存、模型和环境继续只读复用；输出按任务/run ID 隔离。

## 5. 哪些旧结果需要重跑

| 已有成果 | 处理 |
|---|---|
| 旧生成路径产出的 Ours 事件、体素和相应模型 | 如需作为新版结果，重新生成相关事件、重训相应模型并在固定评估协议上复评；旧结果保留为历史 |
| 与生成事件无关、输入契约和内容均一致的物理 rollout/HDR/depth/seg/位姿缓存 | 可按哈希共享复用，无需仅为换核心重复物理仿真 |
| 原共同 Direct-HDR 参考 | 若场景、相机、响应、采样与协议未改，继续固定使用；不能为追求新版优势更换评估源 |
| 原 Genesis 场景可行性、GT/状态控制录像 | 保留，不因事件生成核心升级而作废；不能改标为事件闭环 |
| 旧 Ours/V2E 等比较结果 | 不直接继承新版结论。Ours 重建及训练发生变化，正式比较重新锁定所有组协议；公平条件成立且输入/协议未变的对照产物可复用 |
| 旧 checkpoint | 不能仅因核心升级自动变成新版模型；只有本文固定验收模型对应已修复稀疏事件 |

对每个任务先做一段代表缓存接入和本任务原有的小规模验证，通过后再启动大批量。记录发布 commit、17 文件核心哈希、生成配置及内容指纹、输入缓存、事件源、模型、采样率和输出位置。不要只记录笼统的“Ours 新版”。

## 6. 不同会话如何继续

| 会话 | 可以继续的工作 | 必须区分的事项 |
|---|---|---|
| A：机器人调反弹板入容器 | 在冻结核心上接入刚体/机器人 link、相机和事件采集，验证感知与调板任务 | 原 90° 腕部刚性连接的入杯 demo 是状态条件驱动。高速碰撞前后要检查输入采样是否足以表达运动，不能把旧物理 demo 当事件闭环验收 |
| B：柔性条带 | 继续已有 Genesis 物理场景、状态跟随和录像；评估变形几何适配需求 | 当前核心按 object/link 刚体变换重建，未验收连续形变。不能把同一 segmentation ID 的整个条带当刚体就宣称已支持 |
| C：流体 | 继续实际 Genesis 渲染、物理质量记录和任务验证；评估流体表面表示与适配需求 | 未验收流体重建。质量账本或静止杯画面不是机器人移动接取事件闭环 |
| 其他事件质量、训练或迁移任务 | 固定版本后开展与自己任务相关的接入验证，按上表选择重跑范围 | 本报告证明刚体基线修复有效，不替代各任务的独立评估与传感器协议 |

**B/C 不需要停止已有物理探索，但事件仿真器尚不能被当成已经验证的柔性/流体工具。** 新能力如需核心扩展，应另立版本与验收，不覆盖本冻结版本。

## 7. 已知适用范围与冻结规则

已验收输入为 100 Hz、320×240、机器人/方块刚体场景。覆盖率仍是近似；HDR 与 depth/segmentation 尚未统一为完全相同的 renderer 子像素采样。端点之间没有被观察到的独立光照、阴影或短闪不能唯一恢复。换高球速、强光变、分辨率、噪声或形变输入，要验证本任务的数据契约，不把原验收结论无限外推。

这不阻止冻结当前基线。今后执行：

1. 本发布提交、配置、既有证据保持不变。后续分数不好，先排查输入、数据、模型与控制，不立即回到核心调参。
2. 场景参数、相机、采样、传感器 profile 与控制变化分别记录，所有比较组按同一协议执行；不要把为某一生成器定制的控制改动算作生成器收益。
3. 只有发现违反既定契约的可复现错误，或明确批准扩展新能力，才另开版本；不静默覆盖本论文基线。
4. 抓取正式成功率、多训练 seed 排名、真机泛化与 EVK4 标定属于后续研究，不是本次验收已经得到的结果。

## 8. 复现位置与哈希

以下均为已归档路径，不代表当前 GPU 空闲或已给接收会话分配资源。访问服务器仍按最新 `/home/科研/AGENTS.md`：当前只考虑 g1，g3 禁止连接直至用户解除；不要停止他人任务、升级共享环境或重复复制大缓存。

| 内容 | g1 位置 |
|---|---|
| 冻结指针与配置 | `/home/zhaoyuhan/Eventbased_WAM/frozen_baselines/robot-sparse-cuda-r2-20261102/` |
| 固定核心源码 | `/home/zhaoyuhan/Eventbased_WAM/runs/robot_sparse_cuda_acceptance_20261102c/core` |
| 稀疏生成、训练与定位证据 | `/home/zhaoyuhan/Eventbased_WAM/runs/robot_r1_5_sparse_time_repair_20261101a/` |
| RTX3090 CUDA 与最初 R2 | `/home/zhaoyuhan/Eventbased_WAM/runs/robot_cuda_r2_server_20261102a/` |
| V100S CUDA | `/home/zhaoyuhan/Eventbased_WAM/runs/robot_sparse_cuda_v100_20261102b/` |
| 最终严格成功、原始事件与轨迹 | `/home/zhaoyuhan/Eventbased_WAM/runs/robot_R2_loaded_guard_20261102o/` |
| 最终只读复核 | `/home/zhaoyuhan/Eventbased_WAM/reviews/robot_sparse_freeze_20261103/review_result.json` |

模型为上述稀疏训练目录的 `fit/spatial_event_state.pt`，SHA256：

`dda51f2d5a9665295eeeeccd98682183946f8b080ceb610f82c2baeec3f8d896`

生成配置按 Git/g1 的 UTF-8 LF 字节校验：

`dbcf85fe3c86ae21972d4f5b79638018ca44f46a39b8cb1e14e8b6cff3a31a8a`

控制配置 SHA256：

`67602d8906945447e04dca4d157bceaf26c3715fa64d33555d91e9bca125e2d8`

原 `FROZEN_BASELINE.json` 中生成配置的 `profile_sha256` 为 Windows CRLF 导出件哈希 `bee4a7129232809a6c224e77b82ac4bc81f919a30f3f3175eecf41596d4c2f6f`，不是 Git/g1 LF 文件的字节哈希。已确认仅换行不同、参数相同；审计封存 `AUDIT_SEAL.json` 记录两者及语义指纹。不要为此改冻结参数或重跑数据。校验 Python 源码时同样按冻结描述约定的 UTF-8 LF 表示。

仓库内最终抓取的复跑命令、评分、失败账本与哈希位于固定发布提交的 `benchmark/robot_closed_loop/results/r1_6_sparse_cuda_freeze_20261102/evidence/`。以 `R2_plan.json` 和 `strict_event_grasp_audit.json` 为入口，使用新的独立输出目录。

## 9. 可直接转发给执行会话的通知

> EventGenesis 已完成原稀疏刚体路径修复并冻结。请读取本报告，固定仓库发布提交 `62da4f35ce3c5c96f04268bd1687a27657af9238`，显式使用 `event_sparse_reconstruction_20261102.json` 的 `coverage_time_log_linear`，不要只更新代码或继续旧 coverage-splat 配置。先核对输入与哈希、做本任务最小接入验证，再按报告决定重生成事件/重训/复评范围；物理可行性录像和兼容只读缓存保留。记录版本、配置、输入、输出与事件来源。原刚体定位和 CUDA 已验收，已有一次严格事件抓取；B/C 变形/流体事件重建仍需单独适配验证。保持冻结核心不动，重大方向问题向研究会话汇报。
