# EventGenesis 运算与部署效率核查

日期：2026-11-05 JST（取证发生于 2026-11-04 UTC）。对象：冻结核心 `62da4f35ce3c5c96f04268bd1687a27657af9238`、Stage 9 实际传感器配置及真实生成收据。

本轮完成论文/代码调研、冻结代码只读检查、服务器历史成本统计，以及五个合成训练区间的 CPU 采样诊断。没有修改冻结核心、正式数据、训练配置或运行队列，没有启动训练或占用 GPU。这里给出的是优化依据与实施顺序，尚无新版仿真器的实测加速比。

## 1. 核心结论：传感器积分频率放大了几何重建工作量

当前配置为 100 Hz 物理端点、300 Hz 光感受器截止频率。公开插件在 `attach()` 时自动为采样器设置滤波时间常数和数值步长约束：

\[
\tau=\frac{1}{2\pi\cdot300}=0.5305\ \mathrm{ms},\qquad
N_{sensor}=\left\lceil\frac{\Delta t}{\tau\epsilon}\right\rceil
 =\left\lceil\frac{10\ \mathrm{ms}}{0.5305\ \mathrm{ms}\cdot0.05}\right\rceil=377.
\]

最后的重建采样数取亮度、运动、可见性、深度与传感器约束的最大值。因此传感器积分需要的小步，也要求执行对应的完整几何投影、覆盖合成和中间图像生成。在这个约束主导的区间，100 Hz 端点会产生约 **37,700 次中间状态更新/模拟秒**；这不是新增真实渲染，而是稀疏端点间的重建计算。

五个预先固定的合成训练区间，用实际公开插件的 `attach()` 配置进行 CPU 检查，结果如下。每例在采样数算出后立即停止，没有生成事件或运行模型。

| 输入 / 区间起始索引 | 不含传感器积分约束的其余采样要求 | 传感器积分要求 | 实际公开入口最终采样数 |
|---|---:|---:|---:|
| rich `s0_00` / 100 | 6 | 377 | 377 |
| rich `s1_00` / 100 | 3 | 377 | 377 |
| rich `s2_00` / 100 | 15 | 377 | 377 |
| rich `s3_00` / 100 | 17 | 377 | 377 |
| pose `train_000` / 10 | 16 | 377 | 377 |

**证据范围：**这证明五个被检查区间中存在采样工作量放大，不证明整个 bank 的每个区间都由这一因素主导，也不证明 3–17 次重建足以保持全部事件质量。完整耗时占比仍须独立分段计时。

**诊断修正记录：**第一次诊断直接构造 `PhysicsInterpolator`，没有经过插件 `attach()` 的传感器参数派生，得到的是“去掉传感器约束”的控制结果，不能作为正式入口的采样数。随后通过实际插件重新检查，得到五例均 377。首次脚本、日志和报告保留在 `evidence/probe_sampling_cpu.py`、`sampling_probe_results.json`；正式入口复检在 `probe_sampling_public_cpu.py`、`sampling_probe_public_results.json`。两者都未修改核心。

代码依据：[插件自动派生时间常数](https://github.com/zyh2308/genesis-event-camera-plugin/blob/62da4f35ce3c5c96f04268bd1687a27657af9238/genesis_event_plugin/physics_plugin.py#L160)、[采样器计算传感器步数并取各约束最大值](https://github.com/zyh2308/genesis-event-camera-plugin/blob/62da4f35ce3c5c96f04268bd1687a27657af9238/genesis_event_plugin/adaptive_sampler.py#L242)。

## 2. 实际生成成本与训练成本分开看

从本轮已经完成的 EventGenesis 原始生成 manifest 统计：

| bank | 条数 / 每条模拟时长 | 每条 worker 墙钟中位数 | worker 布局 | 记录到的单进程显存峰值 |
|---|---|---:|---|---:|
| rich | 32 / 2.0 s | **806.79 s，约 13.45 分钟** | 同一 V100S 上两个 worker | 约 12.02 GB |
| pose | 128 / 0.2 s | **75.19 s** | TITAN RTX 上一个 worker | 约 16.89 GB |

rich 共生成 27,658,134 个事件；pose 共 6,910,972 个事件。所有统计及 manifest SHA256 在 `evidence/generation_costs.json`。

这些墙钟包含缓存读取、构造、生成、验证、保存及哈希等工作；rich 两个 worker 还共享一张卡。它们证明当前离线生产成本较高，**不是纯 CUDA kernel 计时、受控 RTF 或与其他生成器的公平速度排名**。单进程峰值可能累计到先前 rollout，不能把两个进程的峰值简单相加当作同时显存峰值；worker 墙钟和也不能当作总任务墙钟或 GPU 小时。

当前正式训练读取已生成的事件/适配数据，不在每个 optimizer 更新中重新运行 EventGenesis。仿真器优化主要改善下一轮数据生产与在线仿真；STE-FlowNet 训练提速仍由独立的相关性算子工作解决。没有理由仅因本报告重新生成本轮数据或重启训练。

## 3. 代码中已确认的开销来源

检查了七个关键源文件，与当前 Stage 9 profile 中的冻结 SHA256 全部匹配；文件列表在 `frozen_source_check.json`。这是一轮重点代码检查，不声称完成全仓动态性能剖析。

| 位置 | 已确认事实 | 优化方向 / 限制 |
|---|---|---|
| `physics_interpolator.py:455–508` | 中间深度及多组诊断搬回 CPU；逐个中间帧转 NumPy；请求张量输出时还会把处理后的图像拷回 GPU | 将未知区域处理、状态和诊断摘要留在 GPU；按需导出。不能省略未知标记、hold 策略或质量检查 |
| `physics_plugin.py:327–350` | NumPy 中间帧逐个送进 DVS，区间内事件连接后再排序 | 建立完整传感器语义的张量接口，按块一次输出；先证明时间区间有序、读出延迟及同时间戳顺序可兼容，再去掉冗余排序 |
| `dvs_emulator.py:405,477–490,673–675` | 每个子帧创建 GPU 张量；取最大事件数回 CPU；Python 按事件层循环；事件回 CPU 后排序 | 像素并行、时间顺序的 CUDA 状态机；事件计数、前缀和、写出；保留不应期与残余对比度状态 |
| `torch_physics.py:435–495` | 覆盖合成包含 GPU 标量回读、深度层数循环、多个 `bool(any())`、scatter 和临时数组 | 分块/分段深度处理和复用缓冲区。保留深度 tie、前后覆盖、0.995 支持门槛及 surface 一致性 |
| `dvs_emulator.py:721–762` | 现有批量 DVS 只支持理想子集，明确拒绝低通、泄漏、shot noise 和不应期等功能 | 当前正式配置开启这些项，不能直接换到这个理想批量类来声称等价提速 |
| `genesis_adapter.py` | 当前集成入口存在 GPU→NumPy，且适配器尚不提供完整多环境批量契约 | 在线采集可评估原生 GPU buffer；Genesis 的具体可用接口仍需验证，不能直接假设兼容 Isaac 的接口 |

NumPy 参考、Torch 投影堆叠和自适应采样已经存在。下一步应补齐端到端数据驻留、状态推进和调度，而不是重复实现已有功能。

## 4. 论文与代码中可复用的做法

### ESIM（CoRL 2018）：按信号变化安排几何采样

ESIM 将渲染与事件生成结合，按预计亮度变化或运动幅度决定下一采样时刻。源码中分别有基于亮度导数和最大光流的调度。对应我们的启发是：**几何采样应有自身的误差依据，传感器数值积分另行推进**。我们已有类似几何约束，进一步工作是解除调度耦合；不能据此放松当前误差界。

[项目与论文入口](https://rpg.ifi.uzh.ch/esim.html) · [调度源码](https://github.com/uzh-rpg/rpg_esim/blob/master/event_camera_simulator/esim_data_provider/src/data_provider_online_render.cpp#L251)。

### Vid2E（CVPR 2020）：像素并行与紧凑事件写出

其 GPU 实现先计算各像素事件数量，再前缀求和得到输出偏移，最后并行写出事件。可以借鉴这种内存组织，减少 Python 层逐事件次数循环。完整 DVS 有额外状态：计数与写出两遍必须共用初始状态和相同噪声样本，不能推进两次 RNG 或滤波状态。

[论文](https://rpg.ifi.uzh.ch/docs/CVPR20_Gehrig.pdf) · [代码](https://github.com/uzh-rpg/rpg_vid2e/blob/master/esim_torch/src/esim_torch/esim_torch.py#L56)。

### Prophesee / OpenEB：直接输出事件体素、批量常驻状态

官方 `GPUEventSimulator` 同时提供 AER 事件和直接事件体素接口；后者绕过完整事件列表的物化。源码使用按视频/像素组织的常驻状态及 CUDA/Numba kernel。这适合训练、在线感知只需要体素时使用，但须验证它与“同一完整事件流→原适配器”结果相符，保留原窗口、极性、时间插值和归一化。正式原始事件归档仍保留独立输出路径。

[API](https://docs.prophesee.ai/stable/api/python/core_ml/video_to_event.html#gpu-event-simulator) · [源码](https://github.com/prophesee-ai/openeb/blob/main/sdk/modules/core_ml/python/pypkg/metavision_core_ml/video_to_event/gpu_simulator.py#L324)。

### Real-time event simulation（ICRA 2023）：将计算集中在变化区域

论文提供稀疏插值方法，可借鉴活动区域计算、任务分解和减少中间结果的思路。该工作采用线性差分，并简化噪声，其速度收益不能全部归为相同传感器模型的工程加速。我们的平静区域仍可能有滤波尾迹、泄漏或噪声事件，因此稀疏计算需要保守界限与状态更新，不能只保留运动物体区域。

[论文/项目](https://cogsys-tuebingen.github.io/realtime_event_simulator/) · [源码](https://github.com/cogsys-tuebingen/event_simulator/blob/main/src/SparseInterpolatedEventSimulator.cpp)。

### EsaacSim（ECCVW 2026）：GPU 渲染产品接入

公开代码有在 post-render 阶段取得 GPU buffer、避免图像回主机的接口，值得借鉴。其当前公开 Python 展开路径每个活跃像素只发一个事件，Warp 路径也存在相应简化和容量限制；因此不能复制其事件展开逻辑作为我们的等价实现，也不能直接拿论文延迟与当前完整语义对比。引用它作为技术参考不改变本轮五种生成器的正式比较名单。

[论文](https://arxiv.org/abs/2608.08522) · [GPU 接入](https://github.com/ibugueno/esaacsim/blob/main/exts/esaac_sim.sensor/esaac_sim_sensor/async_omnigraph.py) · [事件数量取舍](https://github.com/ibugueno/esaacsim/blob/main/src/isaac_event_sim/sensors/event_backends.py#L427)。

### Mamba / 并行扫描：借鉴递推算法，不更换下游模型

像素的一阶滤波可写成 `y[k]=a[k]*y[k-1]+b[k]`。仿射变换可以组合后做前缀扫描；Mamba 的 CUDA 扫描代码有相应组合算子。它可作为滤波阶段的后续算法参考，但浮点运算重排仍要验收；阈值发放、参考电压复位和不应期是有状态的非线性过程，不能一起当作普通线性 scan。优先实现单 kernel 内的逐像素顺序递推，减少改动。

[论文](https://arxiv.org/abs/2312.00752) · [仿射扫描实现](https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_common.h#L138)。

### 最新快速导航预印本：理解适用范围

2026 年 10 月的两篇工作分别研究 CUDA 简化渲染和免渲染几何代理，用于低分辨率避障。它们证明按任务需求简化观测可以显著改善策略训练效率，但这种代理不是我们当前用于重建、光流和物体位姿的完整事件流替代品。本轮不采用其高倍数速度作为 EventGenesis 的可达承诺。

[CUDA 事件避障仿真](https://arxiv.org/abs/2610.38967) · [Rendering-Free Event Frame Simulation](https://arxiv.org/abs/2610.37543)。

外部文件的读取时间、URL 与 Git blob SHA 见 `external_source_metadata.json`；动态分支 URL 只作为入口，后续复现应核对保存的 blob SHA。

## 5. 推荐实施顺序

### 第一优先：执行等价优化，先保持全部 377 步

1. **分段计时。** 独立缓存回放中分别测 HDF5、主机/设备传输、端点几何准备、投影、覆盖合成、未知处理、DVS 状态推进、事件展开、排序、输出。记录每个区间最终采样数及主导约束。采样诊断本身不等于耗时剖析。
2. **端到端 GPU 数据驻留。** 中间图像、unknown/support、滤波与比较器状态保持张量；诊断摘要与 AER 按块导出。CPU NumPy 保留为参考接口。继续执行必要的验证，避免逐像素/逐子步回读。
3. **融合有状态像素循环。** 像素之间并行，像素内按原时间顺序推进，保留原有低通、噪声、泄漏、不应期、被抑制事件的残余对比度与噪声复位规则。先不改 dtype、积分公式或噪声分布。
4. **有界缓冲区及紧凑写出。** 计数/前缀和/写出，临时状态与容量复用；溢出显式扩容、分块或回退，禁止静默丢事件。按原时间顺序处理时间块，padding 不推进状态、不消耗随机数。
5. **减少重复工作。** 缓存固定 K、像素射线、可复用矩阵及静态索引；动态深度/位姿变化时正确失效。只在确证有序且兼容读出和同时间戳处理后合并排序阶段。

这些步骤主要改变执行方式。任何实际事件差异仍需核验，不能仅凭公式相同认定实现等价。

### 第二优先：独立几何与传感器时间尺度

保留足够精细的像素积分，在变化较缓区域用更少几何关键时刻描述连续亮度，遮挡/出现/碰撞区域提高几何密度。这是从本次 377 对 3–17 诊断得到的设计假设，**会改变中间信号近似，属于候选算法变更**，要在独立版本做质量—速度消融。不能直接把 377 改成 16，也不能关闭低通、增大 `sensor_eps_target` 或降低几何精度来冒充等价优化。

闭式/指数积分可另行研究；当前亮度依赖时间常数、噪声和阈值复位使它不能直接替换为一个固定系数公式。并行扫描也只先用于可线性表达的滤波部分。

### 工程部署

- 离线 HDF5 按块预取，一次传入 GPU；训练用缓存、原始归档与压缩写盘分别计时。共享 `/home` 是 NFS，节点本地缓存应按哈希只读复用并保留归档；不盲目把已在 RAM 的训练数据再次搬盘。
- 在线采集检查 Genesis 能否导出对应 HDR/depth/seg 的 GPU buffer及时间戳。保持渲染线程归属，数据可用时间与事件时间分开记录；异步处理不能给控制器未来端点信息。
- 批量运行采用环境维度、独立状态与 seed；不只增加多个竞争同一 GPU 的 Python 进程。比较单 worker、双 worker、真批量的实测吞吐和延迟。
- 构建产物缓存并固定源码、Torch/CUDA、目标架构；不升级正式任务的共享环境。动态采样、事件数和 Python 分支整理后才试局部 `torch.compile` / CUDA Graph，避免整条动态管线盲目捕获。[PyTorch 2.8 约束](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs)。
- 并行噪声可参考 [Random123](https://github.com/DEShawResearch/random123) 的计数器设计，按 seed/episode/pixel/time-index/channel 编址；改变已有 RNG 样本路径需要单独版本与统计验收，不能当作逐事件相同。

## 6. 验收和成文指标

先在当前分辨率 346×260、固定输入/配置/硬件上做受控短测，包含静止、相机运动、物体运动、强纹理、遮挡/再出现、快速变化及完整噪声设置。固定输入提前选择，不按某生成器有利的事件密度挑片段。

| 层次 | 必须检查 |
|---|---|
| 几何与亮度 | 端点不改写；支持/unknown、surface 与深度边界一致；无高频 HDR 或 GT 目标 mask 补入 |
| 事件与状态 | 数量、地址、极性、时间；滤波/比较器/不应期终态；不丢事件；固定 seed、分块与恢复一致 |
| 噪声 | 先同一预抽样随机张量验证；若换 RNG，再做隔离的样本路径/分布检查 |
| 下游回归 | 固定模型和合成缓存，复核定位、光流及重建输入差异，防止速度优化重新引入像素偏移 |
| 公平性 | 当前训练使用原冻结数据和实现；候选另行版本化；相同 seed 比较组遵循共同训练后端策略 |

性能表应同时给：

- **生成器本体时间**：同一冻结端点缓存到原始事件，包含必要转换与排序；首次初始化另列。
- **端到端时间**：物理/渲染、生成、传输、落盘分别报告，再报告完整墙钟；不能只报告其中最快的一段。
- `RTF = 模拟时长 / 墙钟时间`，越高越快；单环境在线延迟与多环境合计吞吐分开。
- `MEvents/s`、实际事件数量/密度、峰值显存、P50/P95 区间延迟；warmup 后重复测量，保留失败/溢出记录。
- 有噪声和理想设置分开，完整质量设置下的优势优先；不把改变分辨率、事件数、噪声或输出形式的速度提升称为同条件工程加速。

当前正式训练与真实评分协议保持原样。本轮新增的是明确的性能问题定位与有证据的优化路线，没有新 Sim-to-Real 分数或新仿真器速度结果。
