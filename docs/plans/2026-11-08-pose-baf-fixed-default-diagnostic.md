# 固定 BAF 2 ms：事件位姿中心的零更新机制诊断

目标：检验暗光独立验证中的中心定位差距是否对孤立背景事件敏感。两个生成器使用完全相同的下游过滤；不改仿真器、噪声、遮挡、原始事件文件、模型参数或正式评价协议。本轮不训练、不进行真实 E-POSE 评分，也不把滤波结果作为已完成的优化成果。

## 固定规则及依据

采用 DV 官方 BackgroundActivityNoiseFilter 的固定默认邻域与时长：8 邻居（不含自身），不分极性，过去相邻输入事件距离严格小于 **2,000 µs** 时保留。先判定，再将**每个输入事件**写入自身时间面，包括被拒绝事件。同一时间戳只允许按原始输入顺序已经出现的事件支持当前事件；不看后续事件。

- [官方 BAF 源码，固定版本 bf8c0ee8](https://gitlab.com/inivation/dv/dv-processing/-/raw/bf8c0ee8/include/dv-processing/noise/background_activity_noise_filter.hpp)：构造器默认 `dv::Duration(2'000)`，8 邻域与严格 `<`，判断后无条件更新。
- [官方 TimeSurface 源码，同版本](https://gitlab.com/inivation/dv/dv-processing/-/raw/bf8c0ee8/include/dv-processing/core/core.hpp)：默认 `Eigen::Matrix::Zero`；reset 使用 `setZero()`。
- [官方过滤基类，同版本](https://gitlab.com/inivation/dv/dv-processing/-/raw/bf8c0ee8/include/dv-processing/core/filters.hpp)：逐输入顺序处理，packet 边界不清空时间面。

因此保留官方**零初始化**：若原时间轴起点接近零，开始 `[0,2 ms)` 可能因尚未写入的零邻居而保留孤立事件。不得悄悄替换成负无穷初始化。输入事件都更新时间面，因此同一输入流在 `t≥2 ms` 后与负无穷初始化决策一致（CPU 测试覆盖）。正式 scored 窗口在原物理轨迹预热之后；本轮不平移时间轴、不删除预热事件，整段 episode 过滤后再取原 `(start,end]` 窗口。

官方 DV 使用整数微秒。此实现将原始秒时间戳**仅乘 1e6**，以 float64 微秒执行比较，保持原输出时间戳和事件顺序，不 round。这是连续时间扩展，**不宣称对分数微秒输入与 DV 整数实现逐位等价**。仅依赖过去输入的向量化前驱查找与逐事件标量实现、因果 prefix 和 packet 连续性做等价测试。

## 固定数据、模型与停止条件

- 基合同、9e604e… source manifest 和 fdd5d9… frozen export 原字节锁定；使用 `load_context(...,check_gpu=False)`，CPU、2 线程、0 optimizer 更新。
- 仅使用已冻结 C1 `physical_supervised` 独立中心模型，2 生成器 × 3 seed。中心、深度、旋转架构和现有训练均不改。
- TRAIN：原训练 bank 前 8 个 sorted rollout，每条从可见窗口按原 `linspace` 固定选 4 个，共 32；VAL：原独立 4 个 rollout，共 64。normal / half / eighth 全部同物理窗，合计 **1,728 行**。
- 过滤器不读取 GT。原 adapter 可见性筛选和固定样本保留原样；GT 只用于事后计算中心误差。过滤得到全零图也保留该窗口。
- encoder 回调保留 episode 数组强引用，整段只过滤一次；每个窗口用原 encoder、原 log/count_scale=8、原 resize 编码原始流与过滤流。另独立运行普通 adapter，要求原始图逐元素 exact parity。
- 18 个 VAL method/seed/condition 组的原始 CPU 中心预测，与先前 frozen GPU 导出的 C1 `pred_uv` 比较，normalized UV 绝对误差 **≤2e-6**。任何组不满足即失败，不放宽阈值。reference completion SHA 为 `fea5095c363b2d4a02602c4fe2dec6144058564be877c55caf0ef1dc24631093`。
- 全部输入只读，模型参数和 buffer 状态 SHA、源文件 SHA 前后复核。最长 1,200 秒；失败保留 failure receipt，禁止自动重跑、调阈值或转训练。

逐窗同时保留 raw/filtered 中心 UV、像素误差、中心位移和事件保留数；按 rollout 等权汇总，每个 seed 单列，保留改善和退步。保留率是固定窗口事件数加权比例（同窗跨 seed 重复，不声称独立事件样本）。本轮只检验噪声敏感性，不证明滤波训练有效、不证明因果根因，也不替代完整位姿成功率或真实测试。

运行入口（仅由根会话审阅并部署）：

```text
python3 -B diagnose_centers.py --contract /.../pose-factor-diagnostic-v1/sources/contract.json --diagnostic-sources /.../pose-factor-diagnostic-v1/sources --out /.../pose-baf-center-diagnostic-v1
```

上线前追加固定负对照：对同一冻结中心网络、相同batch shape输入全零张量，保留zero UV和中心误差。raw/filtered/zero三者并排，避免滤波后无信号却因中心先验看似改善。不增加训练、筛窗或阈值；不把此零更新诊断当作原30+8质量门槛。
