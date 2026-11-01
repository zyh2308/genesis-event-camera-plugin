# 稀疏重建 CUDA 修复与验证（2026-11-02）

已完成原稀疏重建修复的 Torch/CUDA 接入，并通过下列实际检查。高频 HDR 没有参与候选事件生成或训练；模型保持此前通过像素定位验收的同一 checkpoint。

## 测得结果

| 检查 | 结果 |
|---|---|
| 本地 RTX 2050 实际 CUDA 基础检查 | 33 项通过；灰度/彩色、批量/时间维、遮挡、覆盖率边界、未知区域和公开接口 |
| g1 原始 Genesis 稀疏缓存 CPU 对照 | 4 段通过；中心及 +55 mm 条件；16 个原始 10 ms 区间 |
| 同四段缓存的本地 CUDA 对照 | 每条后端共 19,725 个事件；数量、位置、极性相同，支持掩码分歧为 0 |
| 事件时间最大差 | 0.260 微秒 |
| 固定模型，16 个时刻，CPU/CUDA 生成输入对照 | five-bin 输入相同；定位输出差为 0 像素 |
| 同一模型与输入，CPU/CUDA 推理对照 | 最大定位输出差 0.000165 像素 |

这里验证的是 CUDA 没有破坏已经得到的 CPU 定位收益。此前九集 581 个共同参考查询的定位中位数/P95 为 2.33/5.77 像素，本次没有重训，也没有重新宣称一次新的九集精度实验。

## 本轮修正

将 CPU 已验收的“表面覆盖面积 + 随时间连续变化的 log 亮度”移植到 Torch。补齐了端点规则：极接近已观测端点时，直接使用该端点，与 NumPy 一致。公开插值接口现在遵守返回张量的要求，未知区域的近似填充值也会写回显卡张量。

0.995 覆盖率门槛、严格模式的缺样拒绝规则、模型结构、checkpoint、噪声响应和任务评分均未放宽。稀疏候选使用显式 `coverage_time_log_linear`、`strict_fidelity=False`；未知区域仍标为未知，亮度是近似重建，不冒充额外真实观测。默认与旧冻结基线均未替换。

## 开销与剩余工作

基础对照的实际测试部分约 1.21 秒，显存分配峰值约 9.14 MiB；四段真实缓存回放共 45.27 秒（包括在本地重算 NumPy 对照），其中四段 Torch/CUDA capture 合计约 10.74 秒。16 个模型查询对照约 1.28 秒。以上是小片段正确性检查，不作为大规模生产吞吐量或实时性能声明。

g1 已核实环境和数据齐全，但检查时六张显卡均被他人训练占用。此次 g1 只用了两个 CPU 线程；实际 CUDA 在用户授权后使用本地 RTX 2050。尚未执行新的 R2 Genesis 抓取闭环，也没有宣布整体仿真器重新冻结。

下一步是取得 g1 显卡后，先复核服务器 CUDA 路径，再以固定模型、控制器和严格评分执行 R2 抓取；不再重做已通过的 CPU 修复搜索。只有出现新差异时才扩大排查。

## 复跑

保存的 `scripts/` 是独立诊断入口。源码来自本分支 `genesis_event_plugin/`，用 `PYTHONPATH` 指向仓库根目录。`check_sparse.py` 在该目录只提供原已验收的传感器配置。

```bash
PYTHONPATH="$PWD" python benchmark/robot_closed_loop/results/r1_5_sparse_cuda_20261102/scripts/validate_cuda.py --devices cpu,cuda:0 --output /NEW_OUTPUT/controls
PYTHONPATH="$PWD" python benchmark/robot_closed_loop/results/r1_5_sparse_cuda_20261102/scripts/replay_sparse_cache.py --devices cuda:0 --input /ARCHIVE/sparse_endpoint_blocks.npz --output /NEW_OUTPUT/cache
```

输入归档 SHA256：`e599c6196fefdb3c385ea2137fa3afa85d7979a9e07c06873ac7029ed62e2756`。原缓存仍在既有只读路径，没有复制全部数据或修改共享环境。

完整原始本地检查、CPU/CUDA raw events、source/config 哈希和服务器 CPU 结果由同目录清单索引。`acceptance_summary.json` 给出精确数值与冻结范围。早期失败尝试仍保留，不替换成成功记录。
