# 固定轨迹效果—成本测量协议

本目录是独立计时外壳。它不修改已封存的 `4931807` 任务代码、17 个冻结事件核心文件、原 Catch It! 策略权重、7° 相机、640×480 分辨率和 combined 接触场景。本地目前只完成 CPU 合约检查，尚没有实际服务器计时结果；服务器首轮须先跑单个模式 smoke。

## 测量对象

闭环效果取正式盲测，成本取独立的固定动作轨迹；不能把失败后提前结束的 episode wall 当作更高效率。优先使用已有开发成功回合，**不根据盲测输赢选择轨迹**：

`/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/hdr_utility_diagnosis_20261108a/contact_paired25_windows10/cases/v01_normal25/ours`

该回合 seed `2026110831`，第一接触约 1.36 s，覆盖统一的接触前时窗。计时器委托原 `run_original.py`，前端和原 actor 正常计算；只在传给物理环境前，以作者实际 `action2dict` 解码归档动作并替换新动作。每个 25 Hz 控制步核对归档 qpos、qvel、时间，容差分别为 1e-6、1e-5、1e-8。任何分歧均使成本结果无效，不放宽门槛迁就结果。

初始化和预热为 0–0.32 s；主计时为 **0.32–1.12 s，固定 0.80 模拟秒**。计时从 0.32 s 的前端查询开始，到物理推进至 1.12 s 为止。结束使用专用异常通知 runner 的 `finally` 关闭真实 worker，不伪造任务成功或终止评分。100/50/25/12.5 Hz 含重置的完整端点数依次为 113/57/29/15。成本模式无条件采集到统一终点；不得在某方法自认为已抓住时停止采集。原任务正式评估的停止规则保持不变。

## 必需采集与审计采集分开

Ours 使用实际 HDR radiance、depth、segmentation、相机和物体姿态；这些均计入必要成本。V2E 原任务同时留存一份不用作输入的 HDR/depth/seg/全部物体姿态，实际算法只读原生 uint8 LDR→OpenCV 灰度和时间。将多余 HDR 读数算成 V2E 算法需求会不公平。

因此先在同一主机和 GPU 做两次固定轨迹：`legacy --retain-audit-io` 与 `required`。后者禁用展示录像、展示图片与审计 HDF 写入，V2E 只采实际必要 LDR。检查两条运行的**完整有序 raw events、dtype、shape 逐元素相同**，并保持所有物理检查、源帧率、前端与源码一致。通过后生成专属 parity 收据；只有持有相同源码/方法/帧率/前端/轨迹/窗口/主机/GPU 收据的无详细 profiling 运行，才可进入正式成本曲线。Ours 也执行此检查，验证禁用非算法输出没有改变结果。不创建假的黑色视频或空 HDR 文件冒充实测；被禁用的产物不存在。

## 报告什么

- 主成本：实际 warm host wall / 0.80 模拟秒；同时给 RTF、重复次数、均值与 min–max。无墙钟实时声明。
- 共同成本：真实物理步、PID、必要机器人状态读取、共享前端、原 actor。
- 方法成本：Ours HDR/depth/seg/物体姿态采集；V2E 原生 LDR 采集；各自真实 worker RPC（包括重建、SSM、事件响应与等待）。
- 冷启动另列：从启动器主函数进入到 warm 时窗起点，包含封存检查、导入、模型初始化与预热；不把解释器启动假装已测到。进程总 wall 含关闭、结果写入前的阶段，不能与 warm wall 混用。
- 最小计时外壳的 parent exclusive spans 可相加；RPC 等待已包含子进程工作，**不能再加一次 worker 的内部时间**。可另做 `--profile-worker` 分解 Ours 稀疏重建/事件响应、V2E SSM/事件响应。该详细 profile 不进入主成本曲线；如额外加 CUDA 同步，也只用于该 profile。
- 真实 GPU UUID、host、源码和输入哈希、Linux CPU user/system、峰值 RSS 均保留。没有累计 GPU kernel 活动测量时该项为 null/unknown；不能用 wall 代称 GPU 累计时间。

同源帧率给 100/50/25/12.5 Hz 成本—效果点。相同总预算比较只能在**已经实测、相同硬件**的点及其预算重叠范围内进行；列出预算下实际可用的操作点，不能外推出未测帧率或把固定动作成本当作新的闭环成功率。图中选择的 frontend_id 须与盲测最终锁定值相同。

## 最小服务器 CLI

以下是 Bash 示例，须在 root 已分配的 GPU 任务里执行；此脚本不预约 GPU。`PY`、`EGL_DEVICE_ID`、`CATCHIT_V2E_SOURCE`、`CATCHIT_TEMPORAL_PARITY_FILE` 沿用已验证任务环境。把本目录整体复制到独立 instrumentation 目录，FAIR_RUNTIME 的 source 仍指向原始封存快照，不能覆盖它。`FRONTEND` 必须指向 root 真正锁定的 JSON，示例配置不是选择结果。V2E 还须传入**相应帧率**的原生 SSM parity 收据；四档 minimum_upsampling 分别为 10/20/40/80。

```bash
CDIR=/absolute/isolated/cost_design
META=/absolute/FAIR_RUNTIME.json
REF=/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/hdr_utility_diagnosis_20261108a/contact_paired25_windows10/cases/v01_normal25/ours
FRONTEND=/absolute/actually_locked_frontend.json
OUT=/absolute/fresh_cost_runs
PY=/absolute/existing_Genesis_python
RATE=25
METHOD=ours
```

先以 Ours/25 做单次 legacy 冒烟，确认 qpos/time 和端点预算 PASS，再跑 required；**不启动完整四率前先确认单例通过**：

```bash
"$PY" -B "$CDIR/run_fixed_trajectory_cost.py" --source-meta "$META" --reference-run "$REF" --frontend-json "$FRONTEND" --method "$METHOD" --rate "$RATE" --repeat-id validation_legacy --output "$OUT/${METHOD}_${RATE}_legacy" --acquisition legacy --retain-audit-io
"$PY" -B "$CDIR/run_fixed_trajectory_cost.py" --source-meta "$META" --reference-run "$REF" --frontend-json "$FRONTEND" --method "$METHOD" --rate "$RATE" --repeat-id validation_native --output "$OUT/${METHOD}_${RATE}_native" --acquisition required
"$PY" -B "$CDIR/compare_cost_event_parity.py" --legacy "$OUT/${METHOD}_${RATE}_legacy" --native "$OUT/${METHOD}_${RATE}_native" --output "$OUT/${METHOD}_${RATE}_native_parity.json"
```

通过后独立跑至少三个重复，不把验证运行当作正式重复。Ours/V2E 交替或预先规定 ABBA 顺序，同一机器、同一 GPU、相同 CPU 亲和性与线程设置；避免并行干扰。每一率/方法/最终前端都要单独 parity。V2E 给两条 validation 和后续运行都加 `--v2e-parity-receipt /absolute/native_v2e_parity_for_this_rate.json`：

```bash
"$PY" -B "$CDIR/run_fixed_trajectory_cost.py" --source-meta "$META" --reference-run "$REF" --frontend-json "$FRONTEND" --method "$METHOD" --rate "$RATE" --repeat-id r01 --output "$OUT/${METHOD}_${RATE}_r01" --acquisition required --native-parity-receipt "$OUT/${METHOD}_${RATE}_native_parity.json"
```

详细子阶段确有需要才单独加 `--profile-worker`；其余输入不变。`--synchronize-worker-stages` 会改变执行节奏，只能与该 profile 一起使用。该 pass 输出 sidecar，且 `eligible_as_native_primary=false`。无需为主图先等 profile。

导出明确列出的实际结果，不自动捞取其他实验 lifetime：

```bash
"$PY" -B "$CDIR/build_cost_table.py" --run "$OUT/ours_25_r01" --run "$OUT/v2e_25_r01" --output "$OUT/FAIR_COST_RUNS.csv"
```

CSV 包含 `measurement_start_s/end_s`、相同轨迹哈希、source_snapshot 哈希、method/rate/frontend/repeat、actual host/GPU、完整窗口/必需采集/parity/详细 profile 标志。只有 `complete_window/native_only/primary_validated=true` 且 `audit_included/instrumented=false` 的行可用于主成本图。全部布尔列使用小写 true/false。没有实测结果时不产生示例数据或图。

## 已知部署检查点

本地 CPU 测试证明计时嵌套不重复加总、冷暖分离、未关闭 span 不可提前结束、禁用归档不生成假 HDR 文件；不等同服务器 Genesis/CUDA 验收。若服务器 smoke 的固定动作轨迹不复现、完整帧预算不符、关闭审计改变事件或实际源码符号不兼容，停止该模式并保留错误结果，先修计时外壳；不得改变冻结核心、原权重或正式盲测配置。
