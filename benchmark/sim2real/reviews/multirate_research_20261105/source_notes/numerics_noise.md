# 数值传播、稀疏噪声与 RNG 的可迁移机制

研究日期：2026-11-05 至 2026-11-06。此文件为研究笔记，没有实现候选、运行数值试验或给出测速承诺。未连接服务器、使用 GPU、修改冻结核心、环境、bank、标签或模型。

核对的独立工作树 HEAD：`d3e8787e62128494391d4c18a28003b0abc318d8`。现有严格失败保持失败，本文不授予任何候选新的资格。

## 1. 对当前瓶颈的直接判断

377 是传感器显式 Euler 精度要求给出的步数下界，而当前程序把这一时间网格用于昂贵的中间几何/辐照度重建。首先值得研究的是**低频更新几何、高频运行传感器**。它能让我们先保留完整现有传感器、噪声和不应期实现，单独量化几何输入近似产生的误差。其成功条件是遮挡、深度、可见性和亮度变化可由独立的误差控制处理，不能把粗网格跨过遮挡跳变。

更进一步的闭式/事件驱动传播有理论基础，但它不是把 `eps` 换成指数这么简单。当前带宽随亮度改变，阈值复位、泄漏、shot noise、不应期还会互相影响。去掉传感器小步需要设计、验证新的数值或随机过程合同。

## 2. 当前实际合同：从冻结实现核对

代码：`frozen_versions/eventgenesis-gpu-stage9-20261105/genesis_event_plugin/gpu_sensor.py`，以及同包 `csrc/fused_sensor.cu`。

- `gpu_sensor.py:28–32`：每个时间步依次抽 `randn(float32)` 泄漏扰动、`rand(float32)` shot 判定、`rand(float64)` 噪声时刻，最后把所有步堆叠。调用形状、dtype、交错顺序和最终 generator 状态均属于已有 RNG 合同。
- `gpu_sensor.py:35–45`：`eps=min(intensity_fraction*dt/tau0,1)`；泄漏增量与 `dt*(1-jitter*Z)` 成比例；ON/OFF 判定共用一张 FP32 uniform，噪声时刻在当前步均匀分布。
- `fused_sensor.cu:18–44`：逐步更新低通、减泄漏、处理全部信号阈值事件，再处理最多一个 shot 候选；ON 优先于 OFF。不应期使用最后已接受事件时刻。接受信号后按接受个数移动阈值参考，接受噪声则把参考复位到本步信号终值。

因此更改步数会同时改变滤波离散误差、随机数消费和随机过程。另一个容易忽略的事实是：每步先处理信号，再处理噪声；它不是所有候选按连续时间统一归并。把它改为按时刻归并，会改变部分同一步内的竞争结果，不能称为相同实现。

## 3. 线性子系统可以精确传播，但输入与时间常数必须满足条件

Rotter–Diesmann 1999 给出线性时不变子系统的矩阵指数传播，非线性发放机制单独处理。它解决的是给定输入表示下的线性状态演化，并不保证输入变化或阈值时刻也被精确表示。[作者机构论文](https://brainworks.biologie.uni-freiburg.de/1999/journal%20papers/RotterDiesmann99.pdf)，DOI `10.1007/s004220050570`。

NEST v3.9 的官方教程把满足有限维线性微分关系的输入扩充进状态，以 `y(t+h)=exp(Ah)y(t)` 传播。可读的实现入口是 `models/iaf_psc_exp.cpp` 的 `pre_run_hook()` 和 `update()`：前者预计算指数传播系数，后者更新状态。但这个具体网格模型仍把不应期及发放检测放在时间网格上，不能作为连续阈值时刻精确性的证据。[固定版本教程](https://nest-simulator.readthedocs.io/en/v3.9/neurons/exact-integration.html)，[固定提交代码](https://github.com/nest/nest-simulator/blob/a32cd8728fbb100cc1ce2bc7955196b2bd15403c/models/iaf_psc_exp.cpp)。协调主代理已由 GitHub API 核对 v3.9 指向 `a32cd8728fbb100cc1ce2bc7955196b2bd15403c`。

下面是对我们的 RC 模型的推导，公式不是测速结果。令 `u=log irradiance`，低通状态为 `x`，`a(t)=intensity_fraction(t)/tau0`：

\[
\dot x=a(t)[u(t)-x],\quad A(s)=\int_0^s a(v)\,dv,
\]

\[
x(s)=e^{-A(s)}\left[x_0+\int_0^s a(v)u(v)e^{A(v)}\,dv\right].
\]

当 `a=1/tau` 恒定，且区间输入为常数 `u` 时：

\[
x(h)=u+(x_0-u)e^{-h/\tau}.
\]

当 `a` 恒定，输入是线性函数 `u(s)=u0+m*s` 时：

\[
x(s)=u_0+m(s-\tau)+(x_0-u_0+m\tau)e^{-s/\tau}.
\]

这可以去掉显式 Euler 为稳定性/精度而强制的小步，但仅对所假设的输入与系数成立。**当前 `a(t)` 随亮度变化**，不能把全区间套一个固定 `tau` 指数就宣称精确；把 `a` 冻结于几何节点，或将 `eps` 改成 `1-exp(-eps)`，只是各自局部保持模型的传播，仍是新数值版本，而且即使 `a,u` 恒定也不与旧 Euler 输出逐位相等。一般变化的 `a(t)` 需要进一步积分、适用的特例闭式，或自适应求解器。

几何误差预算与传感器积分误差预算应分开。更精确地积分插值输入，不能恢复插值中已丢失的遮挡或亮度变化。

## 4. 从“推进时间步”转向“求下一次阈值事件”

Brette 2007 的事件驱动精确积分明确要求：事件之间的状态演化可写出，以及能预测是否/何时发放。作者同时提供事件驱动和时钟驱动的 C++ 示例，是可追踪的实现入口；本轮核对了作者页面，未下载审计 ZIP。[论文与作者代码入口](https://romainbrette.fr/exact-simulation-of-integrate-and-fire-models-with-exponential-currents/)，DOI `10.1162/neco.2007.19.10.2604`。

对固定 `u,tau`、固定阈值参考 `b`，解 `x(s)=b±C` 可得：

\[
s=-\tau\log\frac{b\pm C-u}{x_0-u}.
\]

只接受比值合法、根处于当前区间、极性及运动方向正确的根。线性输入对应“线性项+指数项”，一般需求根；先处理单调区间或极值，不能只检查两端而漏掉先上穿后下穿。若确定性泄漏使参考 `b(s)` 移动，应求 `x(s)-b(s)=±C`。阈值复位后需从新状态继续；不应期到期时如果已超过阈值，也必须规定如何处理，不能只沿用过去的根。

完整传感器还必须在几何节点、噪声候选、阈值、参考复位、不应期到期处断段。事件队列和分支的 GPU 代价需要实测；低事件率时有机会减少空步，不能预先断言在当前分辨率和场景下更快。

## 5. 稀疏 Poisson 可以替代逐步掷骰，但不能继承当前噪声合同

Lewis–Shedler 1979 的 thinning：用上界强度产生候选，再按目标强度与上界之比保留。它提供对非齐次 Poisson 的方法；若上界很松，候选也可能很多。本轮核对的是出版社摘要，不把全文中未核实的细节当结论。[原始论文](https://onlinelibrary.wiley.com/doi/abs/10.1002/nav.3800260304)，DOI `10.1002/nav.3800260304`。

NEST 官方 `poisson_generator_ps` 明确提供不受网格约束的时刻及固定 dead time，说明“精确时刻+死区”是已有成熟设计；它是机制参考，并非当前 DVS 信号/噪声联合过程的现成替换。[官方模型文档](https://nest-simulator.readthedocs.io/en/stable/models/poisson_generator_ps.html)。

推导上，齐次候选过程的下一间隔是 `-log(U)/lambda`。非齐次情况可解累计强度 `integral(lambda dt)=-log(U)`，或做 thinning。仅有噪声、恒定强度、非延长型固定不应期 `r` 时，接受事件间隔为 `r+Exp(lambda)`，长期率为 `lambda/(1+lambda*r)`；这是验证用特例。信号事件也会更新不应期时，这个简单公式通常不再适用。

当前 shot 实现是每像素每步最多一个候选，概率及时间来自不同精度的随机数；ON/OFF 共用 uniform，并受信号先行和参考复位影响。连续 Poisson 与有限时间箱 Bernoulli 即使率参数相同也不是同一分布。新过程需明确候选/接受事件、强度计算时刻、死区中是否推进随机过程、信号与噪声的时序竞争。修复既有极小 ON 概率的 FP32 上端不可触发问题也要单列，不能混为等价加速。

泄漏 jitter 还有独立问题：旧每步随机泄漏增量与 `h*Z` 成正比，所以独立步累计方差与 `sum(h^2)` 成正比。改变时间网格会改变它的噪声强度；换成 Brownian 的 `sqrt(h)` 缩放或桥接是新的物理噪声合同。保留完整噪声意味着保留或明确重新定义这些现有细节，不能只保留几个 rate 数字。

## 6. Counter-based RNG 减少随机池与状态，但必须登记映射版本

Salmon 等 SC11 / Random123 把 RNG 写成 `random_bits=F(counter,key)`。官方 CUDA 可直接调用；`include/Random123/philox.h` 是 Philox 入口，`uniform.h`/`uniform.hpp` 管实数转换，`boxmuller.hpp` 管高斯转换。它支持按逻辑索引重现样本，不必先物化整张 `T×H×W` 随机池。[作者论文](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)，[官方 README](https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/README.md)，[官方 Philox 代码](https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/include/Random123/philox.h)。DOI `10.1145/2063384.2063405`。协调主代理由 GitHub API 固定 main 为 `9545ff6413f258be2f04c1d319d99aaef7521150`。

适合的新映射可按 `(episode,pixel,noise_kind,candidate_index)` 分配唯一地址；被拒绝候选也应占有自己的索引，不能只按接受事件计数而让分支改变样本。随机数位、uniform 开闭区间、FP32/FP64 转换、高斯变换、键域和溢出规则都要固定。高斯变换还涉及数学库跨设备重现性。

同为 Philox 不代表与 Torch `rand/randn` 相同。要继承严格 RNG 资格，必须复现安装版本的调用形状、offset 预留、分布转换、交错顺序和 generator 末状态；目前没有这项证据。因而按新索引在融合 kernel 内采样先作为新 RNG 版本研究，不能承诺字节等价。即使随机池省内存，生成成本及吞吐也仍需测量。

## 7. 仿射 parallel scan 能做低通部分，不能包办整个 DVS

Mamba 的官方 `SSMScanOp` 合成两个仿射映射；这是机制参考，不是事件相机验证证据。已查 `csrc/selective_scan/selective_scan_common.h:126–158`。[论文](https://arxiv.org/abs/2312.00752)，[官方实现](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/csrc/selective_scan/selective_scan_common.h#L126)。协调主代理由 GitHub API 固定 main 为 `e9594ce1c732d97440f0332fdc43170a2294dbfa`；已查实现是 float/complex-float。

当 `x_j=A_j*x_(j-1)+B_j`，且 `A_j,B_j` 已知，合成为：

\[
(A_1,B_1)\circ(A_0,B_0)=(A_1A_0,A_1B_0+B_1).
\]

实数运算下可结合，因此低通递推可写成 `A=1-eps,B=eps*u` 进行 prefix scan。但阈值参考复位、最后接受时刻、不应期筛选、噪声后复位是依赖历史分支的状态机，不能放进这个固定大小的仿射算子。原逐步浮点加乘重关联也会改变严格输出；源实现并非 FP64 即插即用。scan 仍需要每个时间点输入和系数，自动消除几何重建的说法不成立。当前已逐像素并行融合，只有低通部分 scan 的收益未证实，优先级低于几何/传感器分频。

## 8. 建议的边界与验收顺序

| 路径 | 可保留什么 | 必须重新验收什么 | 定位 |
|---|---|---|---|
| 现有 index-reuse 精确工程路径 | 相同每步输入、原运算次序、RNG 调用和状态机 | 原严格门禁、回退、内存与计时 | 已授权工程延续；保留已有失败 |
| 独立几何节点，仍用原 377 个 sensor 时刻 | 离散滤波、完整噪声消费次序与不应期代码；在相同输入下逐位可比 | 几何插值输入误差、遮挡/深度/可见性与事件误差 | 最窄的研究增量；全管线不是字节等价 |
| 指数/自适应/阈值求根传感器 | 原物理参数可作为起点 | 变化带宽、连续参考、阈值复位、跳变和不应期 | 新数值版本 |
| 连续稀疏 Poisson + counter RNG | 可设计物理意义明确、可寻址可复现的完整新过程 | 率、ISI、Fano、极性、联合竞争、RNG 与泄漏合同 | 新随机/物理算法版本 |
| 低通 affine scan | 给定输入的低通代数结构 | FP64 误差、浮点重关联、整体收益 | 后续机制试验 |

先保留全部噪声合同研究几何分频；随后对无噪声传感器相对现有连续模型/DOP853 证据做误差收敛；完整新噪声合同单独评审后再接入。每步都区分几何、积分、事件定位和随机误差。禁止用放宽容差、时间量化或缩短不应期来通过旧严格门禁，也不把生成器加速解释为读取离线事件的 optimizer 更新加速。
