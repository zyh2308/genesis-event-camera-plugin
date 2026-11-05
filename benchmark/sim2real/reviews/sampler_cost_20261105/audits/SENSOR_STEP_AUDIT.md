# 377步的传感器数值约束与可安全减算方向

本文件为本地源码只读审计；未访问服务器、未运行GPU、未修改发布包或训练。源码来自 `outputs/efficiency_comparison_results_20261105/release/genesis_event_plugin`。

## 1. 377的精确来源

- `physics_plugin.py:164–170`：若调用者未显式配置采样时间常数且cutoff>0，自动注入 `tau=1/(2*pi*cutoff_hz)` 和 `noise_config.sensor_eps_target`。
- `adaptive_sampler.py:244–248`：`sensor_samples=ceil((dt/tau)/sensor_eps_target)`；最终与对比度、边界位移、深度变化、相机/物体运动、辐照度残差等独立项取max。
- 当前dt=0.010秒、cutoff=300Hz、target=0.05，故tau约0.530516毫秒，dt/tau约18.84956，ceil(18.84956/0.05)=377。
- `gpu_interpolator.py:222`调用相同冻结sampler.compute，没有固定U覆盖。

这个target是数值更新系数的上限目标，不是已经推导出来的“最终事件时刻误差≤某值”的证明，也不是实测相机标定参数。它是保守的全幅最坏亮度界。

## 2. IIR实际是Euler递推，并非解析指数积分

`utils/dvs_core.py:81–142` 与 `gpu_sensor.py` 保留如下公式：

```
eps = clamp(inten01 * delta_time / tau, max=1)
LP_new = (1-eps)*LP_old + eps*log_input
```

`dvs_emulator.py:271–280`的HDR强度因子是 `clamp(pixel_values/radiance_white_level,0,1)`。因此当前采样界使用max(inten01)=1，不是实际场景的亮度上界。在全幅偏暗情况下，377可能比维持同一eps上限所需的步数更多，但这不意味着可直接减少所有更新。

代码文档使用“指数移动平均”这个术语，但这里的eps是`k*dt`，不是`1-exp(-k*dt)`。把它替换成解析指数公式会改变冻结实现、阈值穿越和噪声交互；即使对连续ODE更准确，也不能未经独立验证称为现有输出无损优化。

## 3. 减少sensor步数会改变噪声，不仅改变信号积分

`gpu_sensor.py:28–32`为每个时间步依次保留三个原始随机调用：

1. 漏电抖动：H×W FP32 `torch.randn`；
2. shot触发：H×W FP32 `torch.rand`；
3. shot时间：H×W FP64 `torch.rand`。

`utils/dvs_core.py:145–194`中的漏电量为：

```
delta_leak = dt * leak_rate * pixel_mismatch * (1 - jitter * normal_random)
```

固定总时长T、M个等长子步，独立抖动的累计方差正比 `M*(T/M)^2=T^2/M`。因此377→63而保持rate/jitter数字不变，会使这一离散漏电抖动的累计方差约增六倍。不能把“参数未改”描述为“噪声过程未改”。

`utils/dvs_core.py:365–440`是逐子步Bernoulli shot概率；改变步长会改变每步至多一个事件、多个事件与不应期的联合关系。`dvs_emulator.py`随后还会因噪声事件重置base状态。即使总平均噪声数近似相同，完整时序过程也不保证相同。

若将来使用已校准相关噪声，其AR更新也按采样次数推进，不可随意改变步数而保持alpha不动。当前发布快路径不支持该功能，会明确回退；不应据当前profile推广。

## 4. 建议本轮首先保留377个sensor时刻与RNG序列

不减少sensor步数，不改target、带宽、阈值、不应期、噪声率/抖动/随机call形状与顺序，先减少每个几何/合成步骤中不影响结果的工作。这样不需要凭新的误差容差声明质量等价。

### 优先A：删除完整但被覆盖的旧合成运算

`gpu_geometry.py`的 `coverage_time_log_linear`分支已计算：

```
conditional_mixed = exp(weighted_log)
```

随后scalar与RGB分支又计算一次：

```
mixed = exp((1-alpha)*log(safe0) + alpha*log(safe1))
if composite == 'coverage_time_log_linear':
    mixed = conditional_mixed
```

第一条旧mixed的结果完全未使用。可改为coverage模式直接选conditional_mixed，只有其他模式才计算旧mixed。必须保留surface_match、coverage、support、unknown、z-buffer等诊断和后续规则。这个优化可以去掉377×H×W整栈上的两个log、一个exp及中间算术，无需改变采样或噪声。

### 优先B：精确剪去已经完全遮满目标像素的后方候选

`gpu_geometry.py::_splat_projected_coverage_batch_torch`中：

```
accepted = min(layer_weight, clamp(1-coverage,min=0))
out += layer_value * (accepted/clamp(layer_weight,...))
coverage = clamp(coverage+accepted,max=1)
```

coverage已经严格等于1时，后续accepted为0，first_layer=False，后方候选无法再影响有效输出；zbuf/occupancy已在层循环前计算。可在每层coverage更新后，仅将 `coverage[idx] < 1.0` 的候选保留到下一层，减少后方无效遍历与scatter。

**不能用0.995支持阈值提前停止**，那会改变已有遮挡合成。必须在真实GPU上检查有限HDR、极端大值/overflow、signed zero、多个深度层、surface标签冲突及严格诊断。若公开函数允许负值/极端值，需限制快路径或精确回退，不能因物理输入通常非负便改变通用函数合同。

### 次选：流式时间块减内存，但必须维持RNG顺序

377帧目前一次材料化几何/辐照度、log、eps、leak、shot掩码与随机时间，显存开销很大。可研究按固定时间块流式处理，但仍须依次draw三个原调用、保持全部377时刻、维护跨块传感器状态、原事件排序键和异常回退的完整回滚能力。chunk改变batch形状也可能改变浮点累加最后几位，需重新严格比对，不可凭“都是相同计算”免验收。

## 5. 本轮不应直接采用的捷径

- 直接把377改为63或把sensor_eps_target从0.05放宽到0.3；
- 在安静/无事件像素跳过随机抽样或漏电状态更新；
- 合并M次随机调用为一次T×H×W调用而不证明Philox输出/offset完全相同；
- 用“事件数差不多”替代原始事件/状态/RNG与正式loader核查；
- 每隔若干sensor时刻只算一次几何、其余复用或线性补帧，但不证明几何与遮挡不变；
- 关闭/粗化coverage、unknown、surface-depth与遮挡检查，或用min_coverage_fraction替代完全coverage；
- 使用fast-math、改变dtype或移除异常fallback来取得提速；
- 仅以相同sampling因子声称与V2E同质量。

本审计支持先做无损结构优化，再用原始事件、全部sensor状态/RNG、实际三loader和冻结checkpoint输出验证。单纯减少sensor步数属于带噪数值模型变更，不能与这种工程减算混在同一“无损”结论中。
