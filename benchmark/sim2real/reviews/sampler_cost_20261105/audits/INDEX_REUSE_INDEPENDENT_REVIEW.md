# Index-reuse v1 独立行为审查

审查对象：`histfree+deadmix` 上的第一版 index-reuse 补丁，即每个 tap 内按逻辑行展平后 `index_select`，尚未加入 `source_rows` 预物化的版本。父会话提供的服务器源码 SHA256：`e74793c4805eabfa2a19d9db02b5d829c73a12a61d3fed6b4cbeb8e5995a060b`。本审查未访问服务器，因此该服务器哈希不是本审查独立测得。

结论：未发现当前正式事件生成链路需要修复的确定行为变化。候选仍应以五类固定短片段与一段连续长片段门禁全部通过为发布前提。优化减少重复的布尔索引选行，没有减少传感器步数，也没有改变噪声、可见性或遮挡公式。

## 逐项判断

| 检查项 | 判断与边界 |
| --- | --- |
| Boolean mask 与选行次序 | 原来的 `src[keep]` 使用 mask 对前三维的逻辑行顺序；`keep.reshape(-1).nonzero().reshape(-1)` 得到同样的升序逻辑位置，`src.reshape(total,C).index_select(0,selected)` 得到相同通道行。目标 `idx` 的重复值也保留同样的出现顺序。 |
| 非连续、expand、stride=0 | `reshape` 必要时物化逻辑顺序；不会按底层 storage 顺序错误选行。该物化可能增加开销和峰值内存，但不应改变数值。v1 在 tap 内重新物化与 v2 预物化仅是工程差别；这里不把 v2 的门禁结论套给 v1。 |
| active/take 索引 | 两者都是一维 mask，替换为一次升序 nonzero 后复用。所有 scatter 的源元素、目标索引、调用顺序保留，未排序、合并或删去贡献。 |
| 首层 surface_id/consistent | `first_pixels` 的位置唯一；`index_copy_` 写入同样的 long 值和 Boolean 比较结果。不存在重复写目标或浮点 reduction。旧 RHS 与新 RHS 均来自独立 label 数组，不存在覆盖后再读同一数组的别名问题。 |
| NaN、Inf、signed zero | index/index_select/reshape 是选取或拷贝，不引入浮点变换；应保留 payload 和符号位。后续 log、clamp、scatter 及 `inf*0` 仍按原式执行。已报告的极端 HDR、负值与 NaN 原始字节检查比仅 equal_nan 更有约束力。 |
| FP32 scatter | 贡献序列不变支持确定性设置下逐位等价。未启用确定性算法时 CUDA 原子浮点加法原有的调度差异仍存在；不得声称一次非确定性运行必然与另一次逐位相同。该限制是原路径已有的限制，不能为了“通过”而放宽事件精度。 |
| 非有限/负深度 tie 容差 | index-reuse 只修改 active helper，histfree 的 bounded-reference fallback 原文保持。NaN/Inf 走原有有界实现，负容差仍应同样抛 ValueError；不得把非有限容差描述为推荐物理配置。 |
| 空 mask | `nonzero` 返回空索引，`numel()==0` 跳过与原 `bool(mask.any())==False` 相同的主体。减少的 any/nonzero 仅涉及 Boolean/整数，不消费 RNG。 |
| Autograd | 数学上是相同 gather，但 reshape/expand/index_select 与高级索引的 backward kernel 可能不同，梯度求和顺序与内存开销也可能不同。本轮 no_grad stress 与事件门禁不证明 backward 逐位等价。正式插件输入来自 NumPy，经 as_tensor 形成不要求梯度的张量，因此不阻塞当前事件生成发布；不要扩大成“可微仿真梯度完全一致”的声明。 |

## 证据与发布要求

本审查实际完成的是代码与索引语义复核，没有运行 GPU。父会话报告：v1 已通过全部 35 项 stress，以及 s2 实际插件两段测试的全部帧、诊断、事件、状态、RNG 确定性逐位比较；五类固定短片段与一段连续长片段的完整门禁仍在进行。最终声明必须对应完成的原始证据，不能以本报告替代执行结果。

排除 `geometry_diagnostics/uncertainty/output_ready_after_right_endpoint_wall_s` 这个精确字段合理：它是执行耗时，本次目标就是改变耗时。必须继续逐项比较其他诊断、时间戳、有效区、事件顺序、传感器状态及 RNG；不能按名称泛化排除 uncertainty/runtime 一类字段。

行为层面未发现必须立刻修改的事项。发布前需要保证以下三点：

1. 五类固定短片段与一段连续长片段门禁全部完成且通过，原始失败与结果不覆盖；最终基准与质量门禁使用相同的 v1 源码哈希。
2. 归档实际选中的 v1 源码并锁定上述 SHA256。目前 workspace 的 builder 已包含后来预物化 `source_rows` 的 v2；直接再次运行它不能重建 v1。不得把 v2 builder 的输出无意当成已测的 v1 发布，最终复现入口应指向确切的被测源码。
3. 文档将等价范围限定为已核验的 float32、无梯度事件生成路径；若将来承诺公共可微接口，则补 backward/expand 梯度门禁或保留梯度路径的原实现。

本审查没有修改 builder、发布包、服务器、正在运行的训练或实验配置。
