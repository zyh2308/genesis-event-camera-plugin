# 原始 teacher 的 BAF 目标偏移：CPU 零更新诊断

只检验一个机制假设：原 e188 中心训练的 source-init teacher 如果看 BAF 正常光图，其目标是否已经偏离 RAW 目标与 GT，使 normal teacher 保留项把 student 拉向错误位置。此诊断不修改 teacher、损失、权重、输入阈值、仿真器或任何训练。BAF 中心训练的完整失败结果原样保留，不用于选择新训练权重或真实评分。

teacher 必须取已锁 `ctx.completion.results[method,seed].localization_path` / `localization_sha256`。载入原始 state 后，其全参数与 buffer SHA 必须等于旧 C1 center 训练记录的 `source_state_sha256`。**它不是训练后 C1、不是新 BAF candidate**。结束时再次核对模型状态、权重文件与源码 SHA。模型始终 eval、requires_grad=False、CPU，optimizer 更新数严格为0。

数据与编码直接复用已锁 BAF v2（manifest `1383a156…`）的 `prepare`，冻结 export（manifest `9e604e49…`）的 CPU `load_context` 与原 native tensors / estimate_centers。两个生成器、三个 seed，normal / half / eighth，原8个 TRAIN rollout各4窗，共32；原独立4个 VAL rollout各16窗，共64；共1,728行。RAW 与 BAF 原始图来自同一窗，RAW encoder 和普通 adapter 逐元素完全一致。GT只用于事后误差，绝不输入 BAF 或 teacher。

BAF规则沿用原字节：8邻居（不含自身）、strict `<2000µs`、不分极性、时间面零初始化、每个输入都更新时间面、全episode因果后取原 `(start,end]`。float64秒乘1e6，不round、不移动原时间戳；这仍是连续时间扩展，不宣称分数微秒下与 DV 整数版本逐位相同。[固定官方规则源码](https://gitlab.com/inivation/dv/dv-processing/-/raw/bf8c0ee8/include/dv-processing/noise/background_activity_noise_filter.hpp)。

逐窗输出 RAW teacher UV、BAF teacher UV、两种 GT 中心误差、RAW→BAF 目标位移及事件保留数。像素单位为原相机346×260，normalized UV分别乘对应宽高。按 rollout 等权汇总，每个生成器、seed、TRAIN/VAL、光照分别列出；不挑最好 seed，不把 RAW→BAF 位移自动当作因果根因。原 C1 GPU UV parity **不适用**，因为本轮使用不同权重；不复用或伪称 C1分数。源 hash 和 RAW 编码 parity 仍严格保留。

原 e188 实际 normal teacher 保留项只使用 normal 光照；half/eighth 的 teacher 前向在这里仅作响应诊断，不虚构原训练存在暗光 KD。若原 RAW teacher 也差，或 BAF 的额外位移很小，原样保留否定结果；即使位移明显，也只能说明目标偏移，不能证明它是全部误差的唯一原因，更不能推出完整 pose 成功率已经改善。

最长1,200秒，失败保留新目录，不重跑或调参；无 GPU、无真实数据、无训练、无新 candidate。根会话先登记并审核源码，再执行：

```text
python3 -B probe_teacher_targets.py --contract <factor-v1>/sources/contract.json --diagnostic-sources <factor-v1>/sources --baf-sources <BAF-v2>/sources --out <unique-teacher-diagnostic-root>
```

本地标准库 CPU 边界测试只验证有限数、像素尺度、rollout汇总和源码 SHA 拒绝，不虚称已读取服务器真实 teacher 或执行完整模型前向。根会话随后运行实际 CPU 数据与模型诊断。
