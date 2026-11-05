# 插件效率支线首次交付

2026-11-05，日本时间。范围：在交接的 index-reuse v1 上减少等价工程开销；未启动正式训练或真实评分。

## 结果

选中首层 z-buffer 复用候选。相同 V100S GPU2、原生 auto、完整 Stage9 信号/噪声/遮挡规则、指定确定性配置下，s2_00 的 200 ms 缓存片段每版独立新进程三次、预热后轮换顺序：

| 指标 | 原 index-reuse v1 | 新候选 |
|---|---:|---:|
| 初始化加生成中位数，秒 | 34.416673 | 33.875528 |
| 峰值已分配显存，GiB | 11.945336 | 11.653050 |
| 峰值预留显存，GiB | 13.994141 | 13.505859 |
| 每次输出事件数 | [46698, 46698, 46698] | [46698, 46698, 46698] |

耗时减少 1.572%，倍率 1.015975。已分配峰值减少 2.447%，预留峰值减少 3.489%。这是一条代表片段的实测，不宣称五场景总体倍率或端到端 Genesis 加速。计入 fresh plugin setup、生成、必要搬运和最终排序后的 CPU FP64 事件；不计渲染、缓存读盘、序列化、进程启动或首次 JIT。没有把 profile 时间用于此表。

耗时采用三次中位数；显存采用三次运行各自峰值的最大值。

正常 CUDA 模式另测 s2_00 的 20 ms 短片段，三次中位数为 v1 1.867690s、新版 1.844722s。片段长度和数值模式不同，不与上表相除；正常模式完整严格资格仍为失败。

## 精确版本与入口

- 交接仓库锚点：23761f2f5e3e68e0f50b3d7c5664be5690d93822；22 个 EVIDENCE_INDEX 文件大小/SHA 全通过。
- 原候选几何 SHA：e74793c4805eabfa2a19d9db02b5d829c73a12a61d3fed6b4cbeb8e5995a060b。
- 新候选几何 SHA：a5426aac911eec66c5552b99b4368cc7cc5fdde46557b64631122e1500c1f8e2；仅 genesis_event_plugin/gpu_geometry.py 改变，其他完整基包清单文件逐字节相同。
- 候选构建提交：cf08be39287c555123d7360a101785544ab938f2；实际计时代码和源哈希保存在 FIRST_DELIVERY.json 与各 result.json。
- profile SHA：fbaa4b17b0ac0bc5dd8f20ac8827be9668d62af0a82988dea6332d2de3d7de2f；实际 N=377/10ms、无 cap/fallback；CUDA_VISIBLE_DEVICES 为 GPU-a02affe0-7598-579e-ca39-f551f35ad6e9。
- 锁定 V2E 源码快照的 107 个文件与 SuperSloMo39 权重均通过完整大小/SHA 复核，详见 v2e_pin_verification.json；快照没有 .git 元数据，身份依据完整文件清单。
- 入口：candidate_release/genesis_event_plugin/optimized.py 的 OptimizedGenesisPhysicsEventPlugin.attach/capture；准备与核验命令见 source/README.md 及 source/run_checks.py。
- 完整可独立解包代码：zbuf_reuse_candidate.tar.gz，SHA256 fd932f3d3a2e25ae2c745ddabbf41f30b6ae7da043b0d3b241cf9194b518fe3a。包内保留 stage9_sensor_profile.json、许可证和 SOURCE_MANIFEST.json；它是候选，不是新冻结版。
- 指定 torch.use_deterministic_algorithms(True)、CUBLAS_WORKSPACE_CONFIG=:4096:8；FP64/FP32 边界与原包相同。私有 sm70 编译维持 --fmad=false、无 fast-math；MAX_JOBS=2。工具链从原安装收据读取，未安装/升级依赖。

## 改动与剩余瓶颈

覆盖合成构造候选时已经按原候选顺序计算 zbuf。首轮 unresolved 全 True，旧代码又以相同 idx/z 重算 nearest。新代码首轮只读复用 zbuf，后续层原式不变；没有跳过贡献、提前退出、改变容差或更换采样/噪声过程。不得再对这份 nearest 别名做原地修改，否则会破坏返回的 z-buffer 诊断。

正常模式独立代表区间 profile 中，剩余前三个阶段成本为覆盖合成、投影/对象坐标变换、完整传感器调用。覆盖阶段主要涉及 gather/index、逐层归约及 nonzero 引发的同步；投影涉及 FP64 bmm；传感器仍含原 RNG 与完整状态递推。profile 中 GPU annotation 是区间跨度，含等待且层级重叠，不能当作纯核执行时间相加，也不能与 .cpu() 等待重复计费。原算子自耗时排名及新旧计数完整保存在 profile.json。

本轮没有重测 CUPTI 传输字节或 CPU 进程树峰值，因此新增收益限于上述计时、GPU 显存及算子计数。工程等价验收不产生新的传感器物理真实性结论。

```json
[
  {
    "operator": "aten::nonzero",
    "reference": 157,
    "candidate": 149
  },
  {
    "operator": "aten::index_select",
    "reference": 312,
    "candidate": 296
  },
  {
    "operator": "aten::scatter_reduce_",
    "reference": 140,
    "candidate": 132
  },
  {
    "operator": "cudaStreamSynchronize",
    "reference": 501,
    "candidate": 493
  }
]
```

## 验收和保留的失败

- 35 个严格压力例通过，含 HDR/低照度、灰度/RGB、空支持、部分覆盖、遮挡翻转和异常边界；返回数组/诊断按原始字节比较。
- 五类各两区间实际插件短检通过：[{'case': 's0_00', 'events': 137, 'all_physical_observables_bitwise_equal': True}, {'case': 's1_00', 'events': 193, 'all_physical_observables_bitwise_equal': True}, {'case': 's2_00', 'events': 3575, 'all_physical_observables_bitwise_equal': True}, {'case': 's3_00', 'events': 3880, 'all_physical_observables_bitwise_equal': True}, {'case': 'train_000', 'events': 2765, 'all_physical_observables_bitwise_equal': True}]。actual attach/capture、每区间一次 sensor、N=377、零回退；事件、中间亮度/深度/覆盖/unknown、状态与 RNG 全等。
- s3_00 连续 20 区间/200 ms 通过，50474 个事件；实际重建/光流/位姿 loader 的所测窗口通过严格核对。只恢复物理时间原点，没有时间量化。
- 默认 CUDA 完整原始事件/loader 严格失败继续保留；本轮正常模式短计时不提升这一资格。没有改变相等门禁、噪声、遮挡规则、分辨率或事件数预算。
- 首个“共享齐次坐标存储”候选虽然精确通过，正常短测中位数 1.849194s vs v1 1.818727s，峰值也未下降，已淘汰；源码及测量保留在 candidates/projection_storage 和 evidence。
- 首次队列因前一任务结束后利用率采样尚未归零而退出，已释放；记录保留为 attempt1。随后有界等待并仍要求实际零负载、无外部进程，再取得原子预约。
- 本轮没有重跑 100 区间长程、独立进程恢复及传感器异常/容量回退；传感器/内核源码未变，既有结果作为引用。未满足新的生产冻结完整门槛，未修改或提升任何冻结目录。

## 资源与产物

g1 GPU2：已释放，状态 released_complete。使用全局预约锁和 UUID flock，检查真实进程、显存、利用率、CPU、内存及 NFS 磁盘；原预约和新预约均有收据。未连接 g3，未操作其他任务。

WSL worktree：/home/科研/Eventbased_WAM/repositories/gec-efficiency-continuation-20261105。

g1 worktree：/home/zhaoyuhan/Eventbased_WAM/repositories/gec-efficiency-continuation-20261105。

原始 run：/home/zhaoyuhan/Eventbased_WAM/runs/simulator_efficiency_continuation_20261105h。大数组、Chrome trace 和 loader 输出留在该目录，文件大小/SHA 由 external_artifacts.json 登记。原始逐次计时、失败、profile、完整包清单和资源收据随本报告归档。共享 bank、标签、checkpoint、冻结核心和共享环境保持原样。生成器加速不会直接加快读取离线事件的 optimizer 更新。
