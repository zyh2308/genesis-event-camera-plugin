# 新版冻结仿真器对 Sim-to-Real 实验的影响核查

核查日期：2026-11-03（日本时间）。本报告区分本会话独立复核、交接方既有验收，以及尚待本任务验证的推断。

结论：新版确实修复了会污染物体边缘与事件时序的重建行为，并已显著改善机器人开发材料上的像素定位。它值得作为下一轮三线实验的冻结核心。但这不等于真实 E-POSE 定位、MVSEC 光流或 HQF 重建已经改善；这些结果仍需对应事件重生成、模型训练和固定协议评估。

## 1. 固定版本及本次实际检查

- 用户提供的交接文档位于 `e44472f23de2910461fea1e7b8528860d2559622`，它是说明文件的版本。
- 实际采用的完整发布版本为 `62da4f35ce3c5c96f04268bd1687a27657af9238`。
- 核心来源版本为 `6ac417c7a03e125f40ebf967796bf28191d55859`。
- 显式生成配置：`benchmark/robot_closed_loop/configs/event_sparse_reconstruction_20261102.json`。
- 最终审计版本：`d3415f38e4e3bee1d8d19e9b0c732889863d4a52`。
- 旧 Sim-to-Real 结果继续标记 `e79e656ac00f7847b98db9e12bf43b093f3afff1`，不重写其历史身份。

本会话直接完成：

1. 阅读并逐段检查新旧 NumPy warp、interpolator、公开 plugin 入口，以及新版 Torch 几何/覆盖/融合实现。
2. 核对 GitHub 的 DVS 与 noise_model：新旧 Git blob 完全相同，说明这次核心修复的直接作用层是 DVS 前的图像重建；生成配置是否改变仍需另行控制。
3. 从 g1 冻结指针读取实际核心，17/17 个文件的规范 LF SHA256 与发布描述一致；生成配置原始 LF 哈希也一致。
4. 使用固定旧/新代码在本地独立运行 5 项解析输入检查；源码缓存的 Git blob SHA1 均核对通过。
5. 用 g1 冻结配置和公开 GenesisPhysicsEventPlugin 入口独立复测 CPU ramp/static：320 个 ramp 事件，静态 0 事件，每个区间只访问两个端点；最大解析时间误差约 8.89e-10 秒。
6. 读取服务器逐次预测 CSV，以 GT/predicted u,v 重新计算新版 Ours 与 V2E 的全部 581 + 193 次误差，和记录的逐次误差最大差为 0。
7. 检查我们旧 E-POSE 生成脚本及服务器 HDF5 工件，以及光流生成脚本的端点、传感器、几何缓存与入口。

本会话没有重训、没有重新评分公共真实数据，没有改共享核心或环境。归档 CUDA 验收和 1,598 项大产物检查沿用审计方记录，本次未重复执行这些检查。旧机器人模型的误差取自归档 summary，本次未逐次重算旧模型原始输出。

证据入口：`review_evidence.json`、`controlled_review_result.json`、`source_verification.json`。

## 2. 到底修复了什么

### 2.1 边缘小覆盖不再决定整像素亮度

旧 `physics_warp.py::_splat_projected_diagnostics` 对所有落到目标像素、权重大于 0 的样本先做最小深度竞争。极小前景贡献也可能先赢得 z-buffer、排除背景；随后按剩余权重归一化，就把那一点前景放大成整像素亮度。

新 `_splat_projected_coverage_diagnostics` 按深度从前到后累计近似覆盖，前景只占其覆盖份额，剩余面积可由后层填充。双线性 footprint 是面积近似，不是精确 renderer 子像素覆盖。支持门槛仍是 0.995。

独立受控例：前景亮度 1、背景 0.2，前景只覆盖目标像素 1%。按该覆盖模型应为 0.208；旧代码得到 1.0，新代码得到 0.207999989。这个例子证明具体机制已改变，不是任务误差的数值预测。

定位模型依赖物体边缘事件。整像素错误亮暗跳变会形成错误边缘、额外事件或错误活动位置；修复可以改善模型收到的空间线索，而不需要在输出端平移预测坐标。

### 2.2 亮度不再过早跳到后一端点

旧 Sim-to-Real 核心的公开 plugin 默认 `warp_composite="next_primary"`：只要下一端点的投影可见，就采用它的完整亮度；旧 unknown/hold 在 alpha=0.5 附近切换端点。这会使阈值事件时序偏离期望，并可能制造集中突发。

最终冻结模式为 `coverage_time_log_linear`。令两端点重建亮度为 I0/I1、覆盖为 c0/c1，权重为 w0=(1-alpha)c0、w1=alpha*c1。条件亮度满足：

`log I(alpha) = (w0 log I0 + w1 log I1) / (w0+w1)`

未知位置仍保留 unknown 标记；有有限条件几何估计时保留该近似，没有投影的洞才使用连续的端点 log 融合回退。它不把近似亮度升级成真实观测。DVS 仍可能从近似帧产生事件，不能将 unknown 标记理解为已经完全排除该类事件。

独立受控例：10 ms 内亮度由 1 变为 exp(0.9)，在 2.5 ms 时应为 exp(0.225)=1.25232。旧 next-primary 已是 2.45960，新版为 1.25232。阈值 0.2 的事件应依次出现在 2.222/4.444/6.667/8.889 ms；公开入口复测符合解析值。

### 2.3 精确端点、数值精度与支持约束

新模式在区间两端直接返回真实输入，避免已知图像再经投影改写。NumPy/Torch 对端点采用同一容差；Torch 的相机、K、物体 SE(3) 和 alpha 几何计算保留 float64，投影接口再转 float32；CUDA 回到 NumPy 的副本经过 unknown 回退后，对 tensor 输出显式写回。

旧/新 NumPy `compute_se3_flow`、`project_endpoint_to_alpha` 等解析几何函数的 AST 相同。本次受控刚体位移中，GT flow 的新旧差为 0。因此无需因为升级核心就擅自改变光流标签的方向、单位、时间跨度或半像素约定。

完整 CUDA 一致性由原验收支持：RTX3090/V100S 各 33 检查、16 区间、16 固定模型查询；该结论有已声明的场景和输入范围，本次未重复 GPU 运行。

对应代码：

- [NumPy 覆盖处理](https://github.com/zyh2308/genesis-event-camera-plugin/blob/62da4f35ce3c5c96f04268bd1687a27657af9238/genesis_event_plugin/physics_warp.py#L336)
- [新模式时间融合和 unknown](https://github.com/zyh2308/genesis-event-camera-plugin/blob/62da4f35ce3c5c96f04268bd1687a27657af9238/genesis_event_plugin/physics_warp.py#L574)
- [插值器 unknown 回退](https://github.com/zyh2308/genesis-event-camera-plugin/blob/62da4f35ce3c5c96f04268bd1687a27657af9238/genesis_event_plugin/physics_interpolator.py#L202)
- [Torch 几何精度](https://github.com/zyh2308/genesis-event-camera-plugin/blob/62da4f35ce3c5c96f04268bd1687a27657af9238/genesis_event_plugin/torch_physics.py#L120)

## 3. 定位收益有多扎实，边界是什么

单位都是机器人开发场景原生 320×240 图像的像素。P95 是 95% 查询误差不超过的值。全部查询均计入；新版与 V2E 无缺失预测。

| 材料 | 模型 | 中位误差 | P95 | >10 px 或缺失 |
|---|---|---:|---:|---:|
| 9 条开发轨迹，581 查询 | 旧 Ours | 8.1800 | 106.7202 | 225/581 |
| 同上 | 新稀疏 Ours | 2.3265 | 5.7743 | 3/581 |
| 同上 | 归档 V2E 参考 | 2.1492 | 5.2330 | 1/581 |
| 另 3 条归档开发轨迹，193 查询 | 旧 Ours | 9.0800 | 112.3846 | 87/193 |
| 同上 | 新稀疏 Ours | 2.2004 | 5.5448 | 0/193 |
| 同上 | 归档 V2E 参考 | 2.3214 | 5.2176 | 1/193 |

主材料的 Ours 中位误差下降约 71.6%，严重误差比例从 38.7% 降至 0.52%。最显著的改善是减少偶发的大幅失定位。

必须保持四个解释边界：

- 这是同一 heatmap_v2 网络、seed 606、40 epoch 预算、batch 32、既定拆分与选择规则下，以各自事件重新训练的模型。新模型在自身合成验证选 epoch 36。这不是同一权重只换输入的实验。
- 所有模型在共同 Direct-HDR 参考事件上评估。该参考是 Genesis 高频渲染得到的合成参考，不是真实事件相机或公共真实数据。
- 9 条材料含开发训练/验证轨迹；另外 3 条也按归档开发复查报告。581/193 是相关查询，不能当作 581/193 个独立实验样本。
- 新版已接近 V2E，但这组记录不支持全面优于 V2E：主材料中 V2E 的中位数和 P95仍略低；补充材料新版中位数/严重误差更好，V2E P95略低。这不是正式生成器排名。

一次严格事件驱动物理抓取是有价值的功能证据，但同时包含下游控制修正，且是开发 seed；不能把全部收益只归给核心，也不能宣称正式成功率。它与我们 E-POSE 的公开真实 6DoF 实验不是同一条证据。

## 4. 对我们此前三条实验的具体含义

| 任务 | 与修复的联系 | 仍未解决的因素 | 缓存和下一步 |
|---|---|---|---|
| STE-FlowNet 光流 | 边缘覆盖与事件时间会影响像素运动学习；新核心值得优先接入 | 原场景运动/活跃区域少、全图零流占优、训练与留出运动幅值不同，不能只靠修复补足信息 | Stage 8 ReplicaCAD 保存 float HDR、depth、segmentation、相机/物体位姿，100 Hz/320×240 刚体条件最贴近本次验收。先在这些完全相同的端点上重放旧/新 Ours |
| E2VID+ 重建 | 时间错位和假边缘可能影响重建光度、结构与时序一致性 | 真实域差距、块状伪影、传感器响应与训练覆盖仍需评估；既有 HQF 分数属于旧核心 | Stage 4 的完整物理 bank含 radiance/depth/seg/pose，可先检查契约和哈希再复用。新版事件需要对应重训 |
| SingleShotPose / E-POSE | 更干净的物体轮廓可能改善中心/角点学习 | R10 合成位置/尺度/截断覆盖不足、真实域差距、小数据短训，修复不自动消除 | R10 HDF5 实查仅有位姿、K、时间戳、旧 events、uint8 V2E灰度，没有原生 HDR/depth/seg。需按保存物理轨迹补采/重建这些端点；不能用显示灰度冒充 HDR |

我们旧 Stage 4、Stage 8、E-POSE 生成脚本均没有显式传入 `warp_composite`，在旧 e79 核心上实际走的是 next_primary。因此这次修复与我们旧路径有直接关系。

但频率要分清：旧 Stage 4 和 R10 E-POSE 原生端点为 1 ms（1000 Hz）；新机器人验收为 10 ms（100 Hz）。Stage 8 动态 ReplicaCAD 已是 100 Hz。不能把机器人修复的收益幅度直接搬到高频 E-POSE；也不能一边降采样、一边换核心，再把变化全部归因于修复。

另外，R10 脚本把旧 run 的 sources 插到 sys.path 最前面。未来只改安装包或仓库 HEAD 仍可能导入旧核心。每次新运行都应记录模块 `__file__`、17 文件实际哈希、最终传入配置、缓存内容指纹，而不只打印一个版本字符串。固定几何相机代理及 K 的半像素适配继续独立核对。

## 5. 新一轮的接入和比较安排

1. **先接通新版冻结核心。** 用独立固定源码入口，显式选择 coverage_time_log_linear。公开验收 profile 是理想无噪声、阈值 0.2；先按它复核入口，保持冻结配置原件不变。不要直接导入另一会话可写的工作目录。
2. **优先光流最小闭环。** 复用已保存的丰富 ReplicaCAD 物理端点，同时生成旧 Ours、新 Ours、V2E；固定模型初始化、表示、采样、增强、损失、训练预算、GT与留出 rollout。单窗口能拟合只是接口证据；独立 rollout 超过零流、对输入/运动变化有响应，才支持进入长训。监督诊断和原自监督主实验必须分开报告。
3. **把修复收益与传感器变化分开。** 我们旧 Ours/V2E 的 sigma=0.03、cutoff=300 Hz 等非理想参数已明确记录；冻结机器人 profile 是无噪声/无带宽限制。若识别核心修复收益，旧/新 Ours必须保持传感器响应、端点频率等不变。这应登记成独立任务 profile 并通过最小检查，不静默改机器人冻结配置。若同时改传感器与端点频率，只能报告整套新方案的结果。
4. **再安排重建和位姿的配对训练。** 兼容物理缓存按哈希只读复用。E-POSE 补采后必须保存原生 HDR、深度、segmentation、完整 link/object/camera 位姿，以便以后重放，不再只留显示灰度。扩大位置/尺度/截断覆盖若同时进行，要另列实验版本，不能充当纯核心消融。
5. **真实测试隔离继续执行。** checkpoint仅由合成验证选择；GT crop/mask/box/cell 不进入预测输入。已有 HQF14序列和 E-POSE r01/r05 已见，后续复评明确标为已见复评；未见材料预先登记再开启。真实成绩不好时，先检查数据和模型，不回头用真实分数调冻结核心。

对照产物只有在输入、生成配置、模型初始化与预算、预处理、拆分和选择规则都一致时才可复用。旧 checkpoint不能直接改称“新版模型”。升级核心不需要改我们的研究总目标：训练在仿真，测试在现实，以 event generator 为核心变量。

本次原始服务器只读取证时间约为 2026-11-03 02:33–02:36 JST。GPU 3/4 的预约记录在读取时仍为 holding_idle；本次没有切换占位进程或启动训练。

## 6. 复现本次受控核查

在含 NumPy 的 Python 环境执行：

`python controlled_review.py`

脚本只读取本目录固定源码副本并写受控结果 JSON，不访问服务器或真实测试。5 项检查分别为前景覆盖、时间 ramp、精确端点、unknown支持、解析 GT flow不变。

本目录的源代码副本仅用于该受控核查，不是部署完整包；完整仿真器以发布提交和17文件哈希为准。

[用户交接说明](https://github.com/zyh2308/genesis-event-camera-plugin/blob/e44472f23de2910461fea1e7b8528860d2559622/docs/coordination/FROZEN_SIMULATOR_HANDOFF_20261103.md) · [冻结生成配置](https://github.com/zyh2308/genesis-event-camera-plugin/blob/62da4f35ce3c5c96f04268bd1687a27657af9238/benchmark/robot_closed_loop/configs/event_sparse_reconstruction_20261102.json) · [归档独立复核](https://github.com/zyh2308/genesis-event-camera-plugin/blob/d3415f38e4e3bee1d8d19e9b0c732889863d4a52/docs/coordination/freeze_audit_20261103/review_result.json)
