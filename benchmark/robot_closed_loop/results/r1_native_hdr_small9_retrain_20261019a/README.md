# R1 原生 HDR 输入恢复与感知失效定位

g1 开发诊断，2026-10-19。目标是查明 Ours 在共同 Direct-HDR 事件上的感知失效，不是正式排名或独立测试。保留已审计的 `7fb72dc` 批次及原结果；本轮新配置和结果单独归档。冻结 simulator core 未改。

## 已核实的输入与生成路径

- Ours 训练事件来自同一九集缓存 rollout 的 `trajectory_cache.h5/radiance`，由 `camera.render(radiance=True)` 捕获为 `float32` scene-linear renderer radiance。不是 PNG 灰度，也没有把 display DN 重新命名为 HDR。样例 RGB 范围为 `0.01281–1.37402`，亮度 P1/P50/P99 为 `0.03626/0.14954/1.02344`。Genesis overlay 没有提供本任务的绝对辐射标定，故这些是 renderer-linear 数值，不声称单位为 W·m⁻²·sr⁻¹，也不声称阈值已按 EVK4 实物标定。
- Ours 使用冻结插件 `physical_log`：对正亮度取自然对数并使用 `1e-6` floor、gain `1`、ON/OFF 对比阈值 `0.2`；本轮关闭阈值扰动、噪声、滤波和 refractory。缓存深度、分割、相机/目标位姿只用于生成器内部的几何重建与遮挡处理。
- V2E 事件从已审计批次逐字节复用；保留官方显示视频输入、SuperSloMo 插帧和冻结的 V2E 理想响应 overlay，没有删事件或后处理。Ours HDR physical-log 与 V2E 显示视频响应仍不相同，因此本轮不能解读为生成器优劣排名。
- 下游网络只读取相同定义的因果五-bin 事件张量。GT 仅用于训练标签和推理后的离线评分；目标 GT、分割、深度和未来状态没有成为感知网络输入。

配置：[`r1_native_hdr_small9_development_v1.json`](../../configs/r1_native_hdr_small9_development_v1.json)，SHA256 `351c09ae190f41e27ce2262cdf4bd5eaa932a96e685191ae9a309ffc3c1d5e1b`。scene SHA256 `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`；control SHA256 `3d289c2bef1c562b26f9b713c2a826b813a286a1f7e21f9ead0a01c63c5e0d80`。固定 `heatmap_v2`、seed `606`、最多 40 epochs、batch 32、相同 episode split：train `[0,1,2,3,7,8]`、validation `[4,5,6]`；split contract SHA256 `0d45dbc930f76d73409f2b5d6b29f4676cb68c4394675039d8aac264cb7e6e56`。

## 第一个出错环节与已排除项

在已知 +55 mm 失败片段中，最先出现大误差的是网络图像定位，而不是速度回归。旧 Ours checkpoint 在同一 Direct-HDR 查询 `cutoff=0.5900 s` 的位置误差为 `185.8 mm`，中心对照片段为 `16.3 mm`。同一场景状态直接重渲染与参考关键帧逐像素一致；短窗离线密集 HDR 重放的平均 log-radiance 绝对误差，+55 mm 目标区域为 `0.00700`、背景为 `0.00218`。这些重放使用缓存中的精确状态行，不是新动力学 rollout、事件真值或插值帧冒充的 direct render。

原生 Ours raw events 在 100 Hz 源帧边界有明显时间聚集：+55 mm 片段边界 ±1 ms 窗内为 `31,116` events，同长度帧间中点窗为 `1,501`（`20.73×`）；中心对照为 `33,381/1,346`（`24.80×`）。这些窗口的 unknown/hold 像素平均仅 `0.00143%`，不支持将该 burst 归因于 hold 补全。事件坐标、时间单位、极性和五-bin 分配已核对；同一 source/checkpoint 的输入 tensor hash 一致。上述证据将首个明显失配定位在 raw-event 时序/分布进入网络之前或该处，但尚未证明具体根因在 frozen plugin 的哪一段，也没有把 seam bursts 证明为定位退化的因果因素。

少量图像：+55 mm [`failure_chain_montage.png`](../r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059/episode_005/failure_chain_montage.png)，中心对照 [`failure_chain_montage.png`](../r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059/episode_004/failure_chain_montage.png)。密集渲染与 raw event/tensor/trace 证据及逐项哈希在 [`r1_failure_chain_diagnosis_20261019a`](../r1_failure_chain_diagnosis_20261019a/)。

## 固定速度策略隔离

在相同 Direct-HDR 输入和原始 checkpoint 上，两模型均对照网络速度头、现有 50 ms 历史 OLS、稳健 pairwise-median。+55 mm Ours 58 个查询中，网络头速度误差 mean/P95 为 `0.060/0.116 m/s`，OLS 为 `2.037/7.843 m/s`，pairwise-median 为 `1.852/7.755 m/s`；在关键 `0.590 s` 查询，三者分别为 `0.006/2.625/2.167 m/s`。但是位置误差仍为 `185.8 mm`。这证明历史位置跳变会污染 OLS 速度，但替换速度策略不能修复定位。V2E 上三策略也使用相同接口；其中其网络速度头误差反而略高于历史估速，说明不能仅为 Ours 改专属速度 API。完整两 episode × 两模型 CSV 在 [`per_query_velocity_comparison.csv`](../r1_failure_chain_diagnosis_20261019a/velocity/per_query_velocity_comparison.csv)。

## 原生 HDR 小规模重训及共同参考结果

这是在看比较结果前固定的相同 40 epoch 预算、相同网络/seed/split 和 checkpoint 选择规则。9 个 Ours episode 共生成 `12,323,507` events；对应 V2E 训练事件从原归档逐字节复用。Ours/V2E 各用 566 个 validation windows：原生 HDR Ours 自身 validation XY P95 `38.8 mm`、速度 P95 `0.153 m/s`；V2E validation XY P95 `51.6 mm`、速度 P95 `0.130 m/s`。这只说明各自训练域的留出 episode，不代表共同参考泛化。

两个新 checkpoint 读取完全相同、已查看过的九集 Direct-HDR 2 kHz 开发参考和 581 个接触前查询，100 Hz poll、10 ms observation latency。相同冻结 predictor 的到达/拦截差以查询 cutoff 的 GT 状态作为参考，**不是实测越过拦截平面的真值**。

| 速度 | 模型 | 可达覆盖 | XY 位置误差 mean/P95 | XY 速度误差 mean/P95 | 拦截误差 P95（可达子集） | 到达偏差 P95（可达子集） |
|---:|---|---:|---:|---:|---:|---:|
| 0.30 m/s | Ours HDR | 140/275 (50.9%) | 83.6/428.3 mm | 1.576/7.532 m/s | 137.0 mm | 764.9 ms |
| 0.30 m/s | V2E | 217/275 (78.9%) | 5.6/12.5 mm | 0.133/0.318 m/s | 64.8 mm | 554.4 ms |
| 0.40 m/s | Ours HDR | 87/174 (50.0%) | 90.0/409.0 mm | 2.271/8.353 m/s | 77.7 mm | 797.0 ms |
| 0.40 m/s | V2E | 165/174 (94.8%) | 4.3/11.2 mm | 0.119/0.279 m/s | 44.5 mm | 622.7 ms |
| 0.50 m/s | Ours HDR | 88/132 (66.7%) | 76.0/460.7 mm | 1.544/7.257 m/s | 72.6 mm | 354.1 ms |
| 0.50 m/s | V2E | 132/132 (100%) | 4.0/8.4 mm | 0.123/0.283 m/s | 33.6 mm | 576.2 ms |

汇总时两模型状态可用率均 `581/581`、age P95 `10 ms`。GT predictor 在此参考集的 581 个 query 全部可达，因此没有 GT-unreachable 负样本，无法估计正确拒绝率。Ours 全速度正侧边缘三集仍失效：`+55 mm` 的 0.30/0.40/0.50 m/s episode 可达覆盖分别 `34/87`、`21/58`、`21/46`，位置误差 P95 为 `545.2/537.4/494.0 mm`。因此**恢复正确 HDR 输入并在其上重训，并没有恢复 Ours 对共同 Direct-HDR 域的感知能力**。不能据此说仿真器全面劣于 V2E：本批响应/重建路径不同、仅为已见开发集，且不是正式匹配条件排名。

可复核评估：[`r1_common_direct_hdr_summary.json`](common_reference_eval/r1_common_direct_hdr_summary.json)、[`r1_common_direct_hdr_polls.csv`](common_reference_eval/r1_common_direct_hdr_polls.csv)；逐 episode/速度指标均在 summary 中。checkpoint SHA256：Ours `6ecea6a4520ae09b8cc99ebfd8da7a6a324363d2e57dbf73790e27ee2dbec8d2`，V2E `a7140b63f3b5042b0dc0ed7cc3aa2974c4e9859db919f318826358883f7ce8a0`。共同 reference manifest SHA256 `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`。native retrain summary SHA256 `0bccdaede5a932961a515d3456972f6baa8b875b40cdcf0ecf5d02315025c802`；common eval summary SHA256 `ae6db79403eb134ba79460225579a75ecd822e57b54f1be2b455ee8ed38aee4c`；poll CSV SHA256 `8189910cd1eb7ddb17240eb0ab58e3a35f1b5ca2bc6c7f99a50fc62b0e7fa4c2`。

## 成本、复跑与剩余检查

g1 运行代码 commit `4c5f5f29d4c46dd50401c0aa9079874f97afcbc6`，CPU only。native HDR event generation phase `2472.67 s`（逐 episode 插件调用累计 `2453.48 s`）；两组等预算 fit phase `1434.39 s`；共同 reference evaluation `13.74 s`；本进程端到端 `3920.80 s`，GNU time wall `1:05:23`，最大 RSS `5,114,596 KB`。该成本不含原始 physics/cache capture、已完成的 V2E event generation 和现成 reference capture。结果完整留存 g1：`/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-nativehdr-20261019a/benchmark/robot_closed_loop/results/r1_native_hdr_small9_retrain_20261019a/`；原生 Ours event payload 约 105 MB、复用的 V2E source 约 72 MB，未复制入 Git。g1 的原始 110 项 `SHA256SUMS` 已全项验证；本目录 `PORTABLE_SHA256SUMS` 校验仓库收录的结果文件（不包含 README 本身），manifest SHA256 `755b80a68d433fd9996392c98b00bb7c9639692cd35984beecdf2576391be143`。

复跑（须使用新 RUN_ID；launcher 拒绝覆盖结果）：

```bash
cd ~/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-nativehdr-20261019a
bash benchmark/robot_closed_loop/diagnostics/run_r1_native_hdr_development.sh r1_native_hdr_small9_retrain_<new_id>
```

生成链和定位错位仍未被因果拆分。下一项最有信息量的开发检查，是用固定的 +55 mm 和中心缓存状态对照源帧 seam burst：在不改正式 Direct-HDR 裁判的前提下，比较现有 HDR 重建与精确状态高频 direct-render 事件输入的 raw/五-bin/定位结果。如果这要求改 frozen plugin core，先提交复现、影响范围和建议后再决定；不把 seam 相关性写成已证实原因。正式比较需另留 episode，并先统一/声明 sensor response 边界。
