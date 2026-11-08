# v3：一份新的公共真实物理轨迹，所有方法共同严检

v3 是隔离的成本工具分支。父目录保留 v2，服务器 v1/v2 均不覆盖。**默认仍严检旧 utility 轨迹**；只有明确选择新增模式才改变物理核验的参考文件。核心、原策略权重、原始归档动作、种子、7° 相机、640×480 分辨率、combined 场景均不变。

## 为什么允许重新记录参考

成本比较需要所有方法处理相同的真实物理轨迹，不要求该轨迹必须逐位复现另一环境的旧 utility rollout。固定预先选定的开发回合和原始动作，在当前机器记录一次真实运动，可以避开“旧回合隐藏 IK/执行状态没有归档”的接口限制。新参考不是某方法的新闭环成果，也不能用于提高其成功率。

这不保证跨次运行必然一致。后续运行仍执行实际 Genesis 物理，按每个 25 Hz 控制步检查同一份公共 qpos/qvel/time；门槛继续为 1e-6、1e-5、1e-8。任何方法、帧率或重复不匹配都停止。**不为不同方法分别记录参考，不从多次轨迹中按输赢选一条，不放宽门槛。**

## 录制模式

新增 `--record-physics-reference`。预先固定录制配方为 Ours/25 Hz、legacy 采集、保留审计、原 worker，无详细 profiling。原 actor 与前端照常计算，但物理只接收原回合归档动作。该配方只定义一个公共成本输入，不作为方法对比。

录制时仍核验原控制时间表、完整源帧预算、源代码、核心、权重与统一终点，但不要求新 qpos 与旧 qpos 相同。0.80 s 正式时窗的位置仍是 0.32–1.12 s。原轨迹与新轨迹的差值保留为诊断字段；它不被写成通过误差。录制输出 `PHYSICS_REFERENCE.json`，状态 `PHYSICS_REFERENCE_RECORDED`，`utility_result=false`、`cost_measurement=false`。其 `COST_RESULT.timing` 和 RTF 为 null，CSV exporter 会拒绝将其当作成本点。

参考文件包含实际逐步 qpos/qvel/time、原动作、原动作文件 SHA、冻结输入 SHA、桥接代码 SHA、5 个原控制器源码 SHA、种子/初始条件/场景、host 与 CPU 亲和性。它的 SHA 是后续所有成本和 parity 收据中的 `physics_reference_sha256`。另保留 `reference_trace_sha256` 指向旧动作来源；两个字段含义不同。

## 后续严格测量

新增 `--physics-reference /absolute/PHYSICS_REFERENCE.json`。加载时核对原动作数值、完整时间表、状态维数和有限性，以及封存输入、控制器源码、种子、场景和主机。不同方法、帧率和前端均使用同一文件；允许前端不同，因为其新动作均被丢弃，但物理检查必须一致。

legacy/native 事件一致性比较增加公共物理参考 SHA 与 kind 核对。正式重复既需要公共物理轨迹检查，又需要对应方法/帧率/前端/GPU 的原始事件 parity 收据。CSV 增加 `physics_reference_sha256`、`physics_reference_kind`；同一成本曲线不能混入不同物理参考。

## 最小 CLI

沿用父版协议的 `CDIR/META/REF/FRONTEND/OUT/PY`，但 `CDIR` 指向本 v3 独立目录。任务须已经获得 GPU，不由本工具预约。

先只录制一次，使用新目录：

```bash
"$PY" -B "$CDIR/run_fixed_trajectory_cost.py" --source-meta "$META" --reference-run "$REF" --frontend-json "$FRONTEND" --method ours --rate 25 --repeat-id public_recording --output "$OUT/public_physics_recording" --acquisition legacy --retain-audit-io --record-physics-reference
PUB="$OUT/public_physics_recording/PHYSICS_REFERENCE.json"
```

随后先 Ours/25 Hz 单例严格匹配，不立即展开四档：

```bash
"$PY" -B "$CDIR/run_fixed_trajectory_cost.py" --source-meta "$META" --reference-run "$REF" --physics-reference "$PUB" --frontend-json "$FRONTEND" --method ours --rate 25 --repeat-id validation_legacy --output "$OUT/ours25_legacy" --acquisition legacy --retain-audit-io
"$PY" -B "$CDIR/run_fixed_trajectory_cost.py" --source-meta "$META" --reference-run "$REF" --physics-reference "$PUB" --frontend-json "$FRONTEND" --method ours --rate 25 --repeat-id validation_native --output "$OUT/ours25_native" --acquisition required
"$PY" -B "$CDIR/compare_cost_event_parity.py" --legacy "$OUT/ours25_legacy" --native "$OUT/ours25_native" --output "$OUT/ours25_parity.json"
```

通过后，每个方法/帧率使用相同 `--physics-reference "$PUB"`。V2E 仍须有该帧率的 `--v2e-parity-receipt`；其余正式重复、冷暖拆分、native/audit 边界和最少三个重复沿用原成本协议。不应跳过 legacy/native 原始事件检查。

若这份当前机器的新公共轨迹仍不能通过后续严格检查，v3 方案到此停止，不把参考录制当作成功验收。届时需另定“固定物理状态序列的原生采集＋生成＋前端成本、共同物理成本单列”的测量协议，不能将未通过的集成 wall 冒充完整成本曲线。
