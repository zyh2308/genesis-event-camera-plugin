# EventGenesis 计算效率优化交接报告

交接日期：2026年11月5日，日本时间。接收对象：独立的插件计算效率优化会话。

本报告将计算效率支线从公共真实数据Sim-to-Real研究中独立出来。接手会话负责降低事件生成的计算、搬运和资源开销，维护可复核的正确性与效率证据；重建、光流、位姿三条正式实验、训练协议及真实评分仍由原研究和执行会话维护。

**当前最适合继续工作的起点，是在指定确定性配置与已测范围内通过验收、尚未自动升为生产冻结版的index-reuse v1几何候选。它相对上一GPU版速度倍率约1.242，耗时减少约19.5%；各自原生自动采样下，其耗时仍是V2E的约28倍。先继续减少重复计算、同步和内存开销，保留既有信号与噪声规则。采样、积分和随机过程的改变应作为另外的算法版本，不与等价工程加速混报。**

本次仅整理文档并核查仓库证据，没有连接服务器、启动测试、修改仿真器或操作训练。下列性能和资源状态均来自此前记录；接手时必须重新确认服务器实时状态。

## 1 职责和不可变条件

### 1.1 本支线交付目标

1. 在相同输入、功能、精度、噪声和遮挡规则下，减少插件生成事件的实际墙钟时间。
2. 给出同硬件、明确计时边界、同时报告显存的对照表，保留失败、退化、回退及没有收益的方案。
3. 给出可单独使用的代码包、实际入口、配置和源哈希；验证优化实际启用，避免测到参考回退。
4. 对适用范围作明确声明，经过约定验收后才提出冻结或生产接入建议。

本支线不负责重启下游训练、不决定公共真实测试何时打开，也不以真实测试结果选择采样参数。不得为效率实验重写既有bank、标签、划分、检查点或结果。

### 1.2 用户已指定的约束

- 保持数值精度、完整噪声参数与过程、遮挡/覆盖/unknown规则，不以降分辨率、丢事件或关闭功能换速度。
- 已完成和在运行的三线工作不能因本支线中断、替换或重来。生成器变快不会直接加快读取离线事件的optimizer更新。
- 原冻结核心、限定高速发布版及候选分开管理。不得将候选文件直接覆盖冻结目录，或悄悄提升为正式数据生产版本。
- 新代码、新编译目录、新测量输出隔离。只读复用大缓存、模型和已登记环境，不重复复制整个数据集或安装升级共享依赖。
- 不恢复旧2026年9月30日方案；早期历史冻结标识不能替代下面列出的当前实测版本。
- real-test isolation、共同物理轨迹/标签、event generator为核心变量、下游禁止特权信息泄漏等研究约束继续有效。

工程细节可自主处理。只有会改变物理/噪声模型、数值合同、冻结边界、正式数据或总体研究方向的决定，需要先用具体证据向研究侧汇报。

## 2 仓库和版本入口

### 2.1 工作位置

| 项目 | 已登记位置 |
|---|---|
| 共享Git仓库 | `https://github.com/zyh2308/genesis-event-camera-plugin.git` |
| 共享研究分支 | `exp/public-sim2real-v1` |
| 本报告整理前的证据锚点 | `b44891e156721a67f3b042433cebcd01d4cdcb1c` |
| WSL研究仓库 | `/home/科研/Eventbased_WAM/repositories/gec-sim2real` |
| WSL跨会话规则 | `/home/科研/AGENTS.md`，必须读取实际文件 |
| g1项目根目录 | `/home/zhaoyuhan/Eventbased_WAM` |
| 既有研究Python | `/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python` |

共享仓库在整理前有95个未跟踪文件，必须原样保留。不要`git clean`、`reset --hard`或全量暂存未知文件；代码改动使用自己的worktree。

### 2.2 三层版本不能混淆

**第一层，17文件参考。** 本轮实际比较的参考提交是`62da4f35ce3c5c96f04268bd1687a27657af9238`。17文件使用canonical-LF哈希验证。它是正确性比较对象，不是被优化代码覆盖的目录。

**第二层，限定高速发布版。**

- tag：`freeze/eventgenesis-gpu-stage9-20261105`。
- 源码证据提交：`3e68e3b727ec2c21d9aa947a7989ebc2efd14e75`。
- 包内容SHA256：`21480a95fe934bc385d37f075c4fbb366b9ccb9d0c33654ebddac83902f82ce9`。
- 仓库代码：`frozen_versions/eventgenesis-gpu-stage9-20261105/`。
- 资格描述：`frozen_baselines/eventgenesis-gpu-stage9-20261105/BASELINE.json`。
- g1登记核心：`/home/zhaoyuhan/Eventbased_WAM/frozen_baselines/eventgenesis-gpu-stage9-20261105/core`。

这是一份有配置边界和原始浮点非确定性限制的发布版，不能称为全场景、全preset、全接口逐位确定的最终物理仿真器。

**第三层，最新index-reuse候选。**

- 仓库增量：`benchmark/sim2real/reviews/sampler_cost_20261105/release/`。
- g1完整候选：`/home/zhaoyuhan/Eventbased_WAM/runs/simulator_sampler_optimization_20261105e/candidate_release`。
- 只有`genesis_event_plugin/gpu_geometry.py`改变，其余发布文件逐字节一致。
- 该文件SHA256：`e74793c4805eabfa2a19d9db02b5d829c73a12a61d3fed6b4cbeb8e5995a060b`。
- 候选文件清单哈希：`74c68a4dbe232e9d5ddc8b955456302718a55bff380d5e7e779c779827ce7fcc`。
- 完整差异以`release/manifest.json`为准。仓库嵌入的是增量，不能把它单独当完整可运行包。
- 它没有自动升为第二层的tag，也没有接管三线生产数据。

**不要重新运行探索builder覆盖归档的v1。** 探索脚本后来包含v2，但实际选中并完整验收的是`histfree+deadmix+indexreuse_v1`。恢复候选时应按清单复制精确基包到独立目录，再只应用允许的增量，检查其他文件仍相同。

完整哈希应从版本化JSON读取，不要手抄。此前多次门禁已拦截手抄哈希多字符的夹具错误。

## 3 计算为什么昂贵

### 3.1 377来自数值步长约束

当前已登记物理端点为100Hz，即每区间10ms。光感受器截止频率为300Hz，时间常数约0.530516ms。原采样器按`N_sensor = ceil(Δt / τ / sensor_eps_target)`计算传感器步数下界；实际`epsilon_target=0.05`，得到377步。

最终N再与运动、辐照度、深度和可见性等要求取最大值。五个已检查区间中，其他要求分别为6、3、15、17、16，均由传感器项主导至377。它是原生自动采样结果，不是手动固定377；给V2E强制377则是另一个受控工作量实验。

每10ms的377步都触发完整中间几何/亮度重建，相当于约37,700次中间更新/模拟秒。对346×260图像，每区间约3391万个像素采样位置。**这些是稀疏端点之间的计算，不是377次新的真实Genesis渲染。**

0.05是数值设计参数，不是相机标定量；377也不是已经证明的最高质量采样数。直接更改N会改变滤波近似、噪声抽样次数和不应期联合过程，不能只保持噪声率参数不变就称输出分布相同。

### 3.2 CPU和GPU往返的根因

原接口围绕NumPy图像和CPU事件设计。几何虽在GPU计算，中间图像、深度、mask和诊断会被搬回CPU；未知区处理后，中间帧又逐个上传GPU进入DVS。逐子步标量读取和Python循环还带来同步等待。

已实现的常驻路径将图像、mask、状态和诊断保留GPU，仅按需导出诊断，最后返回CPU稀疏事件。代表窗口的CUPTI硬件追踪为：

| 字节或次数 | 原路径 | 常驻事件输出路径 |
|---|---:|---:|
| CPU到GPU | 274,940,688 bytes | 3,636,408 bytes |
| GPU到CPU | 2,136,685,534 bytes | 51,840 bytes |
| 下行复制次数 | 5,580 | 704 |

完整诊断请求仍产生约1.93GB下行，成本只是推迟到使用时，没有消失。`.cpu()`阻塞时间包含等待GPU，不能全部当PCIe传输时间，也不能与CUDA阶段耗时直接相加。

### 3.3 为什么还没有完全GPU化

CPU端点预处理、自动采样、少量标量控制和最终CPU输出仍在。真正缺失投影的稀疏洞填保留原NumPy log/exp，再上传数值；纯GPU替代分支曾出现约4.77e-6的HDR最大差，未进入推荐路径。

因此当前目标是减少不必要往返，不是为了“全GPU”标签删除数值兼容步骤。Genesis原生HDR/depth/seg采集的GPU buffer接入尚未完成完整效率验收；缓存回放加速不等于渲染器已零拷贝。

## 4 已完成优化和实测收益

下列阶段的输入长度、计时边界和聚合方式不同，**不能将倍率相乘**得到总加速，更不能跨表挑最好值拼接。

### 4.1 融合完整传感器核

像素间并行，像素内部按原时间顺序执行低通、阈值、噪声、不应期与状态更新；保留Torch RNG调用形状、顺序、dtype。异常/容量不足时恢复RNG并执行原参考回退，不丢事件。

最终五场景完整capture汇总为1.252倍，按五场景中位数之和之比计算，耗时减少20.1%；传感器部分各场景约2.45–5.05倍。速度版没有省显存；分块省显存版约4.64GiB，但汇总速度收益较低，约1.160倍。

证据：[融合核与首轮基准](https://github.com/zyh2308/genesis-event-camera-plugin/blob/b44891e156721a67f3b042433cebcd01d4cdcb1c/benchmark/sim2real/reviews/simulator_efficiency_20261105/BENCHMARK_REPORT.md)。

### 4.2 GPU常驻路径

五场景最终接口汇总提速2.905倍，按五场景中位数之和之比计算，耗时降低65.6%；相对上一融合路径约2.311倍。单条连续1秒轨迹原版324.358秒、常驻版109.361秒，均302,999事件；该长测各一次，仅作连续状态回归。

同卡部署中，最终单worker和双worker总吞吐分别约0.9396、0.9427区间/秒，双worker几乎没有额外收益。后续默认从单worker起，不以增加进程数冒充算法加速。早期融合版本双worker的结论不能直接搬到常驻版本。

该阶段单窗口峰值约7.45–11.25GiB，没有降低显存。强制完整诊断导出后收益降至约1.551倍。

证据：[GPU常驻实测](https://github.com/zyh2308/genesis-event-camera-plugin/blob/b44891e156721a67f3b042433cebcd01d4cdcb1c/benchmark/sim2real/reviews/simulator_efficiency_20261105/GPU_RESIDENT_REPORT.md)。

### 4.3 限定高速发布版与固定高密度V2E

同V100S、五个固定200ms片段，每方法三次测量；V2E强制377倍SSM插帧。EventGenesis与V2E耗时分别约12.079–22.281秒、45.897–48.953秒，配对提速几何平均2.556倍，五场景重采样区间为[2.263,3.140]。

它只证明指定高密度工作量下的缓存生成优势，不证明相同事件质量、官方默认设置全面领先或完整Genesis实时运行。该表EventGenesis峰值7.182–11.000GiB，V2E约0.754–0.757GiB，显存代价必须同时展示。区间也只描述固定五场景的变化，不是全领域置信区间。

证据：[限定高密度比较与冻结](https://github.com/zyh2308/genesis-event-camera-plugin/blob/b44891e156721a67f3b042433cebcd01d4cdcb1c/benchmark/sim2real/reviews/simulator_efficiency_comparison_20261105/EFFICIENCY_COMPARISON_REPORT.md)。

### 4.4 最新几何索引复用候选

最终选中复用相同mask的选行索引、删除未使用混合计算及只用于循环上界的直方图构造。正常模式代表区间的`aten::nonzero`从418降至157次，`cudaStreamSynchronize`从875降至501次。剖析耗时未混入主计时。

最新主表从每片段初始化到排序后的CPU事件，包含setup、生成及传感器首帧初始化；网络与私有核预热，不含渲染、磁盘读取或写出。每片段200ms、每方法三次中位数：

| 片段 | 上一GPU版 秒 | index复用版 秒 | V2E原生auto 秒 | 旧版除以新版 |
|---|---:|---:|---:|---:|
| s0_00 | 19.2574 | 15.3076 | 0.4610 | 1.258 |
| s1_00 | 12.3437 | 10.7225 | 0.4446 | 1.151 |
| s2_00 | 21.6929 | 17.1086 | 0.5389 | 1.268 |
| s3_00 | 22.4145 | 17.8320 | 0.5407 | 1.257 |
| train_000 | 20.4049 | 15.9594 | 0.7717 | 1.279 |

跨片段比值几何平均1.242，约减少19.5%耗时。新版峰值7.341–11.545GiB，V2E约0.758GiB；新版相对旧版峰值约增加1.6–4.9%。最新两者原生自动运行点的耗时比为28.042，即EventGenesis仍明显更慢。

V2E这里是auto/minfactor2，并沿用登记的300Hz与完整噪声参数，不能称它所有参数均为官方默认。rich每200ms实际40张中间图像、最大间隔5ms；pose为88张、最大间隔2.5ms。s2_00额外minfactor63控制为8.3707秒、1260帧、158.73微秒最大间隔，也未证明质量相同。

证据：[最新候选完整报告](https://github.com/zyh2308/genesis-event-camera-plugin/blob/b44891e156721a67f3b042433cebcd01d4cdcb1c/benchmark/sim2real/reviews/sampler_cost_20261105/README.md)。

### 4.5 已尝试但不应再当新建议的方案

- 单独deadmix约1.001倍、histfree约1.014倍，收益小。
- full-coverage prune约0.990倍，变慢。
- tile16/tile32约0.657/0.808倍，均比reference慢。
- indexreuse_v2组件约1.212倍，v1约1.206倍；新增收益不足1%且组件峰值更高，没有选择v2完整升级。
- 纯GPU洞填未满足原数值一致要求；多worker也未给常驻版明显吞吐收益。

组件结果只用于选型，不能代替实际插件入口基准。未选候选不能继承v1的完整验收。

## 5 正确性资格必须分层表述

### 5.1 已有通过证据

- 同一中间帧输入下，完整传感器事件、状态及RNG逐位一致。
- 原AER微秒写出一致，以及特定固定计数/定位输入检查通过。
- 最新候选35项组件压力检查通过，覆盖HDR/低照度、灰度/RGB、空支持、部分覆盖、遮挡翻转和异常边界。
- 指定确定性配置下，五类各2区间短测，以及仅s3_00的100区间连续长测通过；长测302,999事件。
- 指定确定性设置下，实际重建/光流/位姿loader在已登记窗口通过逐位核对。
- 默认模式固定E2VID检查点的有限连续合成窗口预测逐位一致；它是回归探针，不是质量提升或真实测试结论。

### 5.2 仍然失败或没有证明的事项

**默认CUDA完整原始float64事件逐位门禁没有通过。** FP32并行归约的末位波动可放大到事件时间；旧版自身重复也有波动，但不能据此宣布所有差异无害。历史3ns诊断线也曾失败，原失败状态没有改记为通过。

正式loader不能用微秒文件一致代替。已有默认模式重建输入检查：原版/新版64/500个组合不逐位一致，原版独立重复68/500，最大voxel差1.1920928955078125e-6；最新索引候选的正常模式正式输入全等门禁也保留失败。

最新默认模式s2_00的旧新版最大时间差约4.9229e-10秒，s3_00约1.99202e-10秒；地址、极性和顺序一致。这个量很小，但严格失败仍是失败，不可量化时间或改比较容差让它通过。

固定模型输出相同只能说明已测consumer对这些差异未表现出变化，不能覆盖所有网络、窗口、增强、seed和未来输入。

**工程执行等价与物理真实性是两份不同合同。** 即使某确定性配置完全保留原输出，也不等于原中间亮度、噪声或真实传感器模型已完全正确。优化验收不能借机把二者合成一个“精确版”口号。

### 5.3 确定性设置及计时隔离

已测设置为`torch.use_deterministic_algorithms(True)`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。配置必须在正确的进程/CUDA初始化边界设置。它只支持已覆盖范围的等价性，不覆盖默认路径失败。

确定性与正常模式的性能要分表，不混合计时。保留实际dtype：输入/状态的部分边界是FP64，内部投影与覆盖合成存在FP32，传感器状态FP64；不是整条管线全FP64。

CUDA核使用既有私有工具链、`--fmad=false`、无fast-math，目标V100S为sm70。不要为提速悄悄更改编译数学设置。

## 6 后续采样和V2E测试只作为效率验收边界

这一部分是最近扩展出来的质量诊断。交接后不应继续无限扩展成新的主研究线；只保留与“能否安全减少计算”直接相关的结论和固定回归用例。

### 6.1 不能直接把377改成63

在共同合法亮度域、相同光学数组及时间网格下，分别相对各自原生连续传感器模型，377步的最坏事件时间误差为：

| 信号 | EventGenesis 微秒 | V2E 微秒 |
|---|---:|---:|
| 亮度突变 | 46.77 | 45.37 |
| 平滑变化 | 37.14 | 28.02 |

降低采样后两者误差均增大。20微秒等门槛是本项目预登记的工程诊断预算，不是ESIM/V2E官方标准。V2E部分指标更好；各自连续模型的数值误差也不是跨生成器真实保真度排名。现有测试没有给出全局最佳N或无损降采样许可。

### 6.2 中间图像错误不一定由N导致

独立新s3_00片段1.0–1.1秒、101张1ms真实Genesis帧，只给候选100Hz端点。EventGenesis从63增到754步，原始亮度误差基本不变，几何时间从5.88增至30.95秒。静止自重投影仅浮点舍入差，真实配对最终观测端点逐位一致。

统一显示映射及最终量化后的PSNR：EventGenesis约47.45dB，V2E原生auto约37.04dB，端点线性插值约49.13dB。简单控制更好，该片段不能作为复杂动态/遮挡下的论文优势证据。也不能用这一片段独自决定正式数据规格。

约67%的查询点未通过严格双端表面一致性mask，**不代表67%真实遮挡、没有深度、被hold或停止更新**。这份参考深度全部有效；有限的conditional混合亮度仍进入传感器，只有未知且非有限的hole才采用补值。单纯放宽mask通常不会修正已有亮度。

1ms参考未做更密真实渲染收敛，因此只用于几何诊断，不是20微秒事件真值。新片段与历史bank端点逐位重放失败仍保留；它是明确的新独立片段，不能冒称旧bank逐位复现。

### 6.3 噪声数值限制是双方共有问题

完整短检每档只有307.2像素·秒，事件计数极少，不能证明ISI、Fano或完整噪声分布等价。FP32 uniform与FP64极小ON概率的上端比较，在本fixture的189/377步下存在结构性不可触发问题，已在EventGenesis和锁定V2E中分别复现；不是几何加速引入。

泄漏抖动累计方差随步长改变，但已测短时间绝对幅度很小。更改RNG、噪声离散化或积分公式属于独立算法变更；不能为拿更小N在当前冻结版本中直接修补。

证据：[采样质量报告](https://github.com/zyh2308/genesis-event-camera-plugin/blob/b44891e156721a67f3b042433cebcd01d4cdcb1c/benchmark/sim2real/reviews/sampling_quality_20261105/SAMPLING_QUALITY_REPORT_ZH.md)；[V2E对照及论文核查](https://github.com/zyh2308/genesis-event-camera-plugin/blob/b44891e156721a67f3b042433cebcd01d4cdcb1c/benchmark/sim2real/reviews/v2e_sampling_quality_20261105/V2E_COMPARISON_REPORT_ZH.md)。

## 7 接手时真正需要阅读的代码

相对仓库根目录：

| 代码入口 | 用途和检查点 |
|---|---|
| `frozen_versions/eventgenesis-gpu-stage9-20261105/genesis_event_plugin/optimized.py` | `OptimizedGenesisPhysicsEventPlugin.attach/capture`；实际快路径与回退，检查`optimization_status` |
| 同包`gpu_interpolator.py` | `ResidentPhysicsInterpolator.interpolate_device/_interpolate_pair`，`finish_geometry`，延迟诊断及精确NumPy洞填 |
| 最新增量`release/genesis_event_plugin/gpu_geometry.py` | `project_endpoint_to_alpha_stack_torch`，`_splat_projected_coverage_batch_torch`，`depth_aware_bidirectional_warp_projected_batch_torch`；已做索引复用的起点 |
| 同发布包`gpu_sensor.py` | `supported`与`generate_device_sequence`，预抽噪声、状态、容量和参考回退 |
| 同包`gpu_kernel.py`及`csrc/fused_sensor.cu` | 私有编译、完整时间状态递推，保留数学设置和RNG合同 |
| 同包`adaptive_sampler.py`及`physics_plugin.py` | 原生auto与attach派生的时间常数/epsilon；只构造interpolator会遗漏正式约束 |
| 同包`radiometry.py` | 原生linear HDR、灰度权重及floor；默认输入空间不能误当linear |

采样器上限是实际配置项：Stage9为512，其他默认值可能300。高N诊断必须检查实际N、capped与fallback；改变epsilon还可能使限定快路径退回参考。不能把慢路径运行或被上限截断的结果当候选成功。

私有构建环境从已登记收据程序化读取`CUDA_HOME/CC/CXX`，保留`MAX_JOBS=2`、`TORCH_CUDA_ARCH_LIST=7.0`及自己的`EVENTGENESIS_GPU_BUILD_DIR`。此处不另造安装命令或猜测编译器路径。

## 8 下一轮应该怎样推进

### P0 先确认起点 不重新做整个项目

读取本报告、最新candidate manifest、当前AGENTS；建立独立worktree和run目录。复核候选精确SHA、实际配置、V2E完整源码及checkpoint。先读已有验收，只有因新改动、环境变化或未覆盖风险才重跑对应检查，避免重新耗费完整旧测量成本。

若要计算，先实时检查g1资源并取得GPU2原子预约；短检实际插件快路径确实启用，无静默cap/参考回退。交付首份收据应包含commit、profile、实际N、device/dtype、真实调用数、fallback及源码哈希。

### P1 优先做不改变输出合同的计算组织优化

在最新候选上重新定位剩余前三项成本。可考察稳定K/射线与固定索引缓存、正确失效的端点准备缓存、减少重复分配和标量同步、在原运算顺序内复用中间量、减少无必要诊断导出、紧凑事件输出。

profile独立于正式计时；只改一个明确瓶颈再作配对短测。缓存必须按分辨率、K、深度、分割、位姿与episode等真实依赖失效，不能在运动/遮挡变化后复用旧值。不要假设CUDA Graph、`torch.compile`或更多worker必然提速。

此前tile/prune失败可复用为反例。新实现若只是换名重做，应先说明为什么本次条件会不同。每项记录收益、显存、相同输入验收及失败原因；没有测得收益就不进入最终候选。

### P2 工程部署优化另列口径

先核对缓存是否已在内存，再考虑HDF5预取、固定形状缓冲区、pinned-memory异步上传与安全重叠。共享`/home`为历史NFS记录，实际挂载应重新确认；节点本地缓存按哈希复用，不盲目复制所有数据。

端点上传、插件计算和保存可各自计时，但总墙钟必须保留真实依赖与同步，异步流水线不能使用未来端点冒充在线因果观测。若研究Genesis GPU buffer，先证明HDR/depth/seg、坐标、像素中心和时间戳接口一致，再评估成本；原有渲染覆盖层保持冻结。

### P3 只有等价空间收益耗尽后 才研究算法版本

多速率几何与传感器解耦、误差有界的关键时刻采样、指数/闭式积分、parallel scan、counter-based RNG和直接体素输出，都可能有价值，但均不能自动称为当前事件流等价。若选择其中之一，先单独登记改变的近似、信号/随机合同、质量预算与验收，再做小规模原型。

几何仍可能有近似误差，不应通过增加采样、放宽mask或改噪声参数掩盖。论文效率比较需要质量证据，但不要求本支线接管三项下游长训或公共真实评分。

### P4 冻结前的最小证据集

至少包含已登记压力例、五类真实插件输入短程、一个连续长程、异常/溢出回退、独立进程状态恢复、完整状态/RNG，以及实际三loader。默认和确定性模式分别保留状态；严格失败不以微秒写出或容差替代。

性能使用同一方法同一输入至少三次、预热、CUDA同步、轮换顺序，保存全量计时与显存。若要升为生产版，还需明确目标输出合同、正常模式数值风险与配置支持范围，并将候选、参考、release三者分别归档。

## 9 公平效率实验怎么做

**主表采用各自原生自动运行点，并保留实际N、最大步长、事件数及功能配置。** 固定377属于次要工作量控制；同采样数不等于同质量。V2E原生auto配合登记300Hz有IIR步长警告，但警告本身不是物理质量判决。

完整计时分三种边界，不能互相相除：

1. warm cached generation：已加载CPU物理缓存/视频，到排序后CPU原始事件；计入几何/插帧、传感器及必要搬运；排除渲染、进程/权重启动、读写盘。
2. native end-to-end：物理步进、真实Genesis渲染、各方法输入准备、事件生成和统一格式写出。额外depth/seg等采集成本应归EventGenesis，不能借共享缓存名义隐藏。
3. cold start：新进程到首段输出，权重加载、初始化、JIT和I/O分项记录；首次编译与缓存命中分开。

推荐指标：秒/模拟秒或RTF（二者互为倒数，RTF=模拟时长/墙钟时间），峰值allocated/reserved显存、CPU进程树内存、实际帧数和事件数。MEvents/s仅作补充，不能以输出事件越少算优势。单环境延迟、多环境吞吐和在线可用延迟另列。

质量未匹配时只报告多个运行点，不能写“同精度快若干倍”。当前单片段显示图像质量也不能替代含噪事件真实性或三项Sim-to-Real成绩。

已拟定但未完成的正式扩展包括20条独立rollout、按场景分层的配对统计、五条生成管线、Genesis端到端与cold-start表。它们是方案，不是现成结果。Vid2E/FILM/RIFE需注明是插帧后端加共同ESIM，不包装成三个独立传感器模型；不要为凑完整论文表无控制扩展本支线。

详见[论文效率协议](https://github.com/zyh2308/genesis-event-camera-plugin/blob/b44891e156721a67f3b042433cebcd01d4cdcb1c/benchmark/sim2real/reviews/paper_efficiency_protocol_20261105/EFFICIENCY_LITERATURE_AND_PROTOCOL.md)。该文末尾已修订：原生auto优先，固定高密度降为次要控制；早期更宽的措辞不再沿用。

## 10 论文和代码调研如何复用

此处是已有研究索引，本轮没有新增检索或复测。具体版本和读取证据在原研究报告中保存。

- ESIM的自适应调度启发几何采样与传感器积分分工；它不授权直接放松当前误差/噪声合同。[作者项目](https://rpg.ifi.uzh.ch/esim.html)
- Vid2E GPU实现的计数、前缀和和并行写出可复用内存组织思路；完整有状态DVS的两遍处理不能重复推进状态或RNG。[官方代码](https://github.com/uzh-rpg/rpg_vid2e)
- OpenEB的GPUEventSimulator提供常驻状态和直接体素接口；若采用，必须与完整事件流经原loader的表示相核对。[官方实现](https://github.com/prophesee-ai/openeb/blob/main/sdk/modules/core_ml/python/pypkg/metavision_core_ml/video_to_event/gpu_simulator.py)
- ICRA2023 Real-time event simulation可借鉴活动区域与模块计时；其简化模型和不同输入范围不能直接承诺同倍率收益。[作者项目](https://cogsys-tuebingen.github.io/realtime_event_simulator/)
- Mamba的仿射scan只可启发线性滤波递推，不可把阈值复位、不应期和随机发放一起当线性scan。[官方代码](https://github.com/state-spaces/mamba)
- FILM、RIFE等计时方法支持同步、预热、同设备与显存并报；不应抄不同硬件论文速度与本机结果混排。

V2E固定源码为`cac638444557a224ba6ee5c29260916f152c00bd`，107项源文件已核对；SSM权重SHA以版本化`v2e_source_pin.json`读取。其lin-log和带宽规律与EventGenesis不同，`hdr=True`也不自动等价为我们的physical-log。原生中点时间戳必须保存，不能使用含auto TODO的固定时间helper去处理变倍率auto。

## 11 服务器安全与复现起步

截至最近规则，g3仍明确禁用，不得SSH/SCP或只读探测。仅考虑已授权g1；规范身份为`zhaoyuhan@g1.pkuml.org:22`，保持主机密钥校验，不硬编码历史IP。

WSL连接g1显式使用Windows OpenSSH客户端`/mnt/c/Windows/System32/OpenSSH/ssh.exe`，沿用已验证的配置/密钥/known_hosts。GPU2历史UUID为`GPU-a02affe0-7598-579e-ca39-f551f35ad6e9`，它是身份记录，不是当前空闲证明。

开GPU任务前实时检查UUID、进程、显存、负载、CPU、内存、磁盘和项目预约；取得原子租约后再复核。原记录在`/home/zhaoyuhan/Eventbased_WAM/.gpu_reservations/`，并使用GPU UUID文件锁及全局预约锁。优先复用项目机制，不让不同会话凭同一空卡截图各自开跑。

不要停止、迁移或降级其他任务，同用户名也不等于本任务。连接超时后按现有规则至少冷却60秒，最多再验证一次；鉴权/host-key问题停止排查，不反复换身份。用持久连接减少短连接风暴。不得重新启用已禁用插件、修改Codex会话历史或触碰无关配置。

本轮最后测试已完成并释放自己的GPU2租约，不保留空占进程。接手不能把历史released记录当作现在已取得资源。

建议最初只执行以下本地只读检查，然后再按当前规则连接服务器：

```bash
repo=/home/科研/Eventbased_WAM/repositories/gec-sim2real
cat /home/科研/AGENTS.md
git -C "$repo" status --short
git -C "$repo" rev-parse HEAD
```

这些检查不构成启动训练、替换数据或升级冻结核心的授权。

## 12 原始证据导航

以下均为g1已登记路径，不表示本轮重新验证了远端存在性或当前进程：

| 路径后缀 位于项目runs下 | 内容 |
|---|---|
| `simulator_efficiency_20261105a` | 融合核、速度/显存候选、worker和回归 |
| `simulator_gpu_resident_20261105b` | 常驻路径、CUPTI、连续状态、实际接口 |
| `simulator_efficiency_comparison_20261105c` | 静态发布候选、固定高密度V2E、冻结验收 |
| `simulator_formal_acceptance_20261105d` | auto、正式loader、固定checkpoint和确定性隔离 |
| `simulator_sampler_optimization_20261105e` | 最新索引复用候选、五场景计时与完整资格 |
| `simulator_sampling_quality_20261105f` | 解析/积分参考、独立真渲染、采样与噪声诊断 |
| `simulator_v2e_sampling_quality_20261105g` | V2E原生传感器、插帧及共同输入核对 |

仓库轻量证据对应`benchmark/sim2real/reviews/`、`benchmark/sim2real/evidence/`及`benchmark/sim2real/experimental/`。各阶段大数组/trace通过`external_artifacts.json`或`raw_artifact_index.json`定位，不必把完整服务器目录复制入Git。

随本报告提供的`EVIDENCE_INDEX.json`列出建议优先阅读的文件、相对路径、字节数和SHA256。它只证明本次读取的仓库证据版本，不是全仓或远端重新审计。

接手会话第一份汇报应回答：使用哪个精确版本、实际快路径是否启用、剩余前三个瓶颈是什么、准备改变哪项计算组织、如何验收，以及是否确有额外收益。完成一个具体优化和相应对照后再决定是否扩大，不把本支线继续扩展成新的下游实验项目。
