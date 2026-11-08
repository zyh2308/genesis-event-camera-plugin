# MVSEC 统一真实测试：并行执行安排核验

本页记录一次由吞吐率触发的执行资源调整，不报告模型效果，也不宣称真实测试已经完成。科学协议、评价源码、真实窗口和模型权重保持冻结。服务器安全启动归档快照为 **2026-11-08 07:15:09 UTC / 16:15:09 JST**；这不是后续实时状态。

## 为什么替换原单卡安排

原 GPU5 v1 作业从第 400 到第 600 个窗口约用 133 秒，即约 1.5 窗口/秒。15 个模型各评估 8,500 个固定窗口，共 127,500 窗口；仅按这一观测速率粗算就约需 23.6 小时，尚未计入额外校验和准备时间，会超出原管理器的 18 小时上限。

因此调整的是资源分配和保险超时。该决定依据运行时间及窗口计数，没有读取部分成绩、选择表现较好的生成器、改变检查点或删选测试窗口。父执行会话对其自身原 GPU5 v1 管理器 PID 531276 发出 SIGTERM；其 handler 保留 `SystemExit: Own manager received signal 15` 失败记录，按运行交付记为 exit 1。原资源收据确认 `released`、`current_job=null`、`cleanup_error=null`。这是有记录的行政停止，不是正常完成，也没有改记为成功。参见 [替换及停止证据](runtime_v2/preserved_single_gpu_v1/runtime_supersession.json) 和 [原资源释放收据](runtime_v2/preserved_single_gpu_v1/resource_release.json)。

旧 v1 目录及部分产物保留原样，不读、不合并、不复用其分数。两个新分片都从新目录完整评估各自分配的模型，没有复杂续评或择优重试。

另有一次不同性质的失败：两片同时启动时，GPU2 在短时共享的全局 registry 非阻塞锁处收到 `BlockingIOError`，尚未取得 GPU 租约、没有启动评分。父会话保留仅含失败记录的原目录，然后按原源码、原参数顺序手动重启；这不是自动重试、模型数值失败或因分数不好重跑。参见 [原锁冲突](runtime_v2/preserved_gpu2_lock_race_failure.json) 与 [手工重启原因](runtime_v2/gpu2_launch_retry1_reason.json)。不能把本轮写成“没有失败”。

16:15:09 JST 的 [安全启动收据](runtime_v2/safe_launch_verified.json) 记录：GPU2 父 PID 548295、GPU5 父 PID 547711 均运行，各自在新目录写出 200 个窗口；对应新任务没有 `failure.json`。这仅证明统一测试已安全开始，不代表任何模型已经完成 8,500 窗口，更不代表整体 15 模型完成。只核查窗口行数，没有读取真实指标值。

## 固定分配及公平性

| 执行分片 | 设备 | seed | 方法数 | 完整模型数 | 新输出目录 |
|---|---|---|---:|---:|---|
| GPU2 | Tesla V100S-PCIE-32GB | 0、1 | 每 seed 5 | 10 | `20261108-mvsec-frozen-matrix-gpu2-v2` |
| GPU5 | NVIDIA TITAN RTX | 2 | 5 | 5 | `20261108-mvsec-frozen-matrix-gpu5-v2` |

两设备固定 UUID 分别为 `GPU-a02affe0-7598-579e-ca39-f551f35ad6e9` 和 `GPU-06ee62ba-c171-40d4-b4ae-ea63391dc885`。每个 seed 的 EventGenesis、V2E、Vid2E、FILM-ESIM、RIFE-ESIM 全部使用同一张卡；浮点类型、前向实现、TF32 关闭及冻结的确定性推理设置一致。设备差异作为运行元数据保留，不隐去。

科学矩阵仍然是三 seed × 五方法的全部 15 个已完成训练检查点。每个运行分片的只读准备和数值门禁仍检查完整 15 权重，不把冻结的 `SEEDS`、manifest 或数值预检缩减为本片子集。每个模型仍使用同样的 7,700 个室内窗口及 800 个室外窗口。

## 冻结及设备数值门禁

共同科学文件保持原字节：

- `frozen_protocol.json` SHA256：`fd6165b609061554e97eeb3a69da538a03e7b3f1c070a29dc4031bcf35ca2dd0`。
- `frozen_manifest.json` SHA256：`8595e9bc81fb5f5acebe8a1deddf70996b8e997e8e7ecf77d440b49b65d61043`。

这两项哈希来自本地已归档的首次冻结文件；新分片必须使用相同哈希。原文件中已有的 GPU5 数值证据作为历史保留，不为设备安排重写科学合同。

每个设备另有自己的授权文件，绑定共同冻结 protocol/manifest 及该设备完整数值报告 SHA。报告须覆盖 15 个相同检查点、每个两个固定合成输入，并符合全模型最大绝对误差 ≤0.001 px/interval、RMS ≤0.0001 px/interval；FP64 独立算子 oracle 的三个误差均须 ≤1e-12。两报告的合成输入指纹一致，模型十四文件及整树、训练参考、adapter、compat 和 probe 源码同时核对。

probe 读取的是 CPU 准备完成收据绑定的原 candidate manifest；授权在完整 probe 通过后才创建，并绑定最终冻结文件，避免哈希循环和提前开启评分。CPU 自测不能代替设备上的完整模型数值报告。

已从归档的 [GPU2 完整数值报告](runtime_v2/gpu2_probe/numerical_preflight.json) 核对 `passed`、15 检查点及最大绝对误差 2.86102294921875e-6。两个分片的只读预检均以 0 退出；报告、设备授权、预检、启动及停止证据均由父会话原字节拉取。本页及本地 CPU 自测不替代这些原始服务器收据；归档清单见 [服务器证据索引](runtime_v2/EVIDENCE_INDEX.json)。

## 完成声明和执行边界

每片完成只可声明 `shard_complete`，分别核对 10 或 5 个模型。`joint_runtime_manifest.json` 是固定分配计划，其全矩阵 union 必须为 15 个唯一 `method/seed`，不漏项、不重复；计划文件本身不是完成证据。最终联合核验必须读取两片成功退出、资源释放及全部结果身份和哈希，确认共同冻结文件一致，才能发布完整三 seed × 五方法结果。单片不得冒称完整矩阵。

每片总上限 36 小时、每个模型 4 小时只是异常保护边界，**不是预计耗时或训练预算**。不因此延长训练、改变评价精度或允许调参。执行继续使用实际进程核查及原子 UUID 租约；只清理自身子进程，结束释放，不设置空闲占位。

本地 v2 队列已经通过 12 项离线安全夹具、两片 10/5 分配和 union 无重复核验；九个核心门禁函数的 AST 与 v1 相同。冻结 evaluator 没有改动。真实成绩和原始输出只留服务器，论文聚合及出表随后单独完成。
