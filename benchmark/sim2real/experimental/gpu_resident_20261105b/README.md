# GPU 常驻事件生成管线：2026-11-05

本目录保存冻结核心之外的隔离适配层。正式三线训练、已有数据和 checkpoint 未接入该候选；本轮只在用户分配的 g1 GPU 2 上验证。

## 最终入口

`resident_final.make_plugin_class(GenesisPhysicsEventPlugin, 'exact')` 构建候选类。`capture()` 保留 NumPy 稀疏事件输出；内部走独立的 `interpolator.interpolate_device()` 通道。原 `interpolate()` 默认仍返回 NumPy 图像，`return_torch=True` 返回 CUDA 张量，并同步对应公开属性。

中间图像、深度/可见性诊断和传感器状态保持在 GPU。诊断通过 Mapping/Sequence 按需访问返回 NumPy 数组，`materialize_diagnostics()` 强制全部导出。诊断计算和遮挡规则保留。依赖具体 `dict/list` 类型而非数据访问协议的外部调用方尚未全面验证；这是本轮缓存回放入口，不是通用插件发布版。

当投影缺失且原规则需要端点亮度回退时，`exact` 模式保留原 NumPy 计算，只上传缺失像素的值。普通路径只同步小型计数和最终稀疏事件。纯 GPU `log/exp` 分支在 HDR 空洞样例中存在末位差异，未进入推荐路径。

## 精度和适用范围

- 使用 Stage 9 冻结配置和 17 文件核心 `62da4f35ce3c5c96f04268bd1687a27657af9238`。
- 保留 FP64 几何输入/传感器状态、完整时间采样、原 RNG 调用顺序、噪声、不应期、事件数量、未知区域和覆盖判据。
- 沿用上一轮已验证的 `frozen_sensor_fused` CUDA 扩展，记录二进制 SHA256；不修改冻结文件和共享环境。
- 当前适配范围是 `physical_log`、完整当前噪声配置、理想读出、正不应期、`coverage_time_log_linear` 与 `hold`，无相机 margin/额外状态 provider。其他选项显式拒绝，不静默关闭功能。
- 进程内几何适配在本轮单线程 worker 中使用；多进程有独立 Python 状态。尚未作为同一进程的并发多线程 API 验证。

## 文件层次

- `prior_candidates.py`：上一轮融合路径，作为独立对照。
- `resident.py`：第一版常驻图像路径，仍整批传出掩码。
- `resident2.py`：掩码也常驻显存，按需导出请求和诊断。
- `resident_final.py`：保持原 NumPy 接口，通过明确的设备入口接通事件管线；物理和传感器计算沿用第二版。
- `bench_pipeline_final.py` / `paired_suite_final.py`：最终公平计时；相同物理输入和配置，交替顺序、预热、CUDA 同步。
- `profile_transfers.py`：CUPTI 硬件复制追踪；其耗时不混入性能表。
- `test_*` / `final_audit.py`：边界、空洞、噪声、恢复、诊断、实际输入、AER 写出和固定模型回归。

所有带有 GPU 数学分支不通过的结果、完整浮点时间戳的非逐位情况，都单独保留。原版几何自身也有重复波动；严格原始数组门禁与有明确范围的回归通过状态分开报告。

## 复现

在取得独占 GPU 预约之后，使用记录中的 Torch 2.8 环境。数据和冻结核心路径保留在脚本中。`build/` 指向上一轮不可变 CUDA 扩展产物；外部清单记录绝对路径、别名、大小及 SHA256。其他 GPU 架构需要重新构建和验证。

```text
python test_holes_v2.py
python test_sensor_v2.py
python test_process_resume_v2.py
python test_deferred_final.py
python test_actual_sensor_final.py
python paired_suite_final.py
python analyze_final.py
python bench_long_final.py resident_exact
python test_model_final.py
```

输出目录要求不存在，以免覆盖证据。计时从已有 CPU 物理缓存到公开事件输出，不包括 Genesis 渲染、磁盘读取和结果落盘；完整诊断导出另有计时。性能数值不能直接当作整个 Genesis 仿真实时率，也不代表离线训练 optimizer 提速。
