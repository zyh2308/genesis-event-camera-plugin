# v4：成本路径的逐查询功能等价协议

本协议在任何正式成本重复开始前固化，对 Ours/V2E × 100/50/25/12.5 Hz 一致适用。正式效用盲测、冻结事件核心、原权重、相机和场景代码均不修改。v1–v3 文件及 v3 raw parity FAIL 继续保留。

## 已有实测依据与验收对象

v3 在 g1 用同一公共物理轨迹运行 record→legacy→required，28 个控制步的 qpos/qvel 差均严格为 0，29 个端点完整。Ours/25 Hz 的两条运行各有 110,828 个事件；每像素/极性次数一致，按像素/极性内时间顺序匹配时有 86 个时间不同，最大 39.4 ms，P99 为 0。全部感知样本和状态相同。

同一种 legacy 模式的 public_recording→legacy 复跑也出现相同规模的 86 个时间差。因此不能把差异直接归因于删除审计 I/O，也不把它称作纯舍入误差或已证实背景事件。v4 不修改事件或掩盖这些结果。

成本验收对象是：在同一固定成本输入上，省去非算法所需的审计存储及多余采集后，实际任务计算路径是否相同。v4 新增显式 `task_functional`，原 `raw_exact` 仍为默认。两种模式都要求完整逐查询功能证据；`raw_exact` 另外要求原始事件逐字一致。

## 保持不变的物理与输入合同

只使用 v3 已登记公共物理参考：

`41458da631174aad39846d96e0ab7b753aa45c79bfcc0046658ba913d02b32a9`

v4 禁止重新录制参考。仍真实推进物理、逐控制步严格核 qpos≤1e-6、qvel≤1e-5、time≤1e-8；仍核完整帧预算、原动作、源代码、17 个冻结核心文件、原策略权重、5 个原控制器文件、种子、7°/640×480/combined 场景。参考不符或后续物理不匹配即停止，不能换参考或调门槛。

计时仍为 0.32–1.12 s，共 0.80 模拟秒。所有 28 次控制查询（含预热）都进入功能验收，正式时窗中为 20 次查询。Ours 必需 HDR/depth/seg/姿态采集、V2E 必需原生 LDR/SSM、共同前端/actor/真实物理计入成本。V2E 多余审计 HDR 不作为其算法需求。实际全部源帧必须生成，不能提前结束变便宜。

## 直接捕获实际模型数据

`ACTOR_QUERY_WITNESS.json` 记录每次实际 `ActorCritic.act_inference` 的完整输入、输出，包括归一化后的 obs/obs_t/obs_c、dtype、shape、device 和原始数值字节（base64）；不以推算的目标位置代替实际模型输入。

同时记录实际查询时间、顺序、次数、是否调用模型、真实 `held`/`event_state is None`，以及 `fixed_step` 在替换归档动作**之前实际收到的** `ignored_actor_action` 字典和原 runner 的标准化动作。held/None/action 从已封存 `run_original.py` 的实际调用帧读取，若接口不符则报错，不猜测。无状态查询也必须占据其实际时间点，不能只比较成功调用模型的子集。

所有张量与数组只读复制，保持 dtype、shape 和字节；记录不改变随机流、参数、动作或图像。CPU 合约用小型 Torch CPU 网络核验前后输出、输入/动作字节、NumPy/Torch RNG 和 CUDA 未初始化状态不变；原接取模型的实际数据见证须在服务器运行时验收。轻量查询见证的内存复制开销保留在所有方法一致的 warm wall 中，JSON 与 raw 比较在计时结束后写入。

## 两种验收与每次重复的约束

- `raw_exact`：逐查询实际功能完全相同，并且 raw events 完全相同。收据状态为 `PASS_RAW_AND_TASK_EQUIVALENCE`。
- `task_functional`：逐查询实际功能完全相同；raw 是否相同与时间差统计如实保留。收据状态为 `PASS_TASK_FUNCTIONAL_EQUIVALENCE`。

均须 legacy 与 required 两条实际运行的全部查询逐项、逐字节相同。仅末次状态相同不通过；物理相同也不替代模型证据，因为本成本工具会用归档动作覆盖新动作。

收据绑定方法、帧率、前端、物理参考、源代码、权重、host/GPU、计时工具及验收器 SHA。每次正式重复必须重新生成自己的实际查询见证，其 payload SHA 与这份已验收的公共功能指纹完全相同才 `eligible_as_native_primary=true`。任何不一致都拒绝该点，不用误差容差近似放行。

`raw_event_exact` 在正式重复中单独记录该次 raw 与 legacy 参考是否相同。`task_functional` 不把 false 改成 true。原始事件保留，不改时间戳，不删事件，不改变核心。

## 最小运行顺序

沿用 v3 服务器环境与已分配 GPU。`CDIR` 改为本 v4 独立目录，`PUB` 必须指向上述 SHA 的 v3 原始 `PHYSICS_REFERENCE.json`。不运行 record 模式。方法和前端使用研究计划已锁定值。正式比较统一选 `task_functional`，不要混合两种验收规则。

```bash
"$PY" -B "$CDIR/run_fixed_trajectory_cost.py" --source-meta "$META" --reference-run "$REF" --physics-reference "$PUB" --frontend-json "$FRONTEND" --method "$METHOD" --rate "$RATE" --repeat-id validation_legacy --output "$OUT/${METHOD}_${RATE}_legacy_v4" --acquisition legacy --retain-audit-io --equivalence-kind task_functional
"$PY" -B "$CDIR/run_fixed_trajectory_cost.py" --source-meta "$META" --reference-run "$REF" --physics-reference "$PUB" --frontend-json "$FRONTEND" --method "$METHOD" --rate "$RATE" --repeat-id validation_native --output "$OUT/${METHOD}_${RATE}_native_v4" --acquisition required --equivalence-kind task_functional
"$PY" -B "$CDIR/compare_cost_event_parity.py" --legacy "$OUT/${METHOD}_${RATE}_legacy_v4" --native "$OUT/${METHOD}_${RATE}_native_v4" --kind task_functional --output "$OUT/${METHOD}_${RATE}_task_equivalence_v4.json"
```

V2E 各次运行继续加相应帧率的 `--v2e-parity-receipt`。先 Ours/25 Hz 单例证明 v4 记录不影响物理，再展开其他点；不因单例通过宣布全部帧率等价。

仅验收通过后启动正式重复：

```bash
"$PY" -B "$CDIR/run_fixed_trajectory_cost.py" --source-meta "$META" --reference-run "$REF" --physics-reference "$PUB" --frontend-json "$FRONTEND" --method "$METHOD" --rate "$RATE" --repeat-id r01 --output "$OUT/${METHOD}_${RATE}_r01_v4" --acquisition required --equivalence-kind task_functional --cost-equivalence-receipt "$OUT/${METHOD}_${RATE}_task_equivalence_v4.json"
```

CSV 仍仅从实际 COST_RESULT 导出。新增 `equivalence_kind`、`functional_query_exact`、`actor_query_payload_sha256`、`actor_query_count`、`raw_event_exact`、`equivalence_receipt_sha256`。正式行要求 functional_query_exact=true；同一方法/帧率/前端的重复须共用一个功能指纹，整表共用一个物理参考及验收 kind；其他 native-only、无审计、完整 0.80 s、无详细 profiling、同主机/GPU、至少三次重复等合同继续适用。

当前仅准备和 CPU 检查，尚未产生 v4 实测通过收据或正式成本点。若实际 actor 输入/输出不同，停止该点并保留差异；不得用此协议宣称生成器事件数值精度通过，也不改变效用盲测结果。
