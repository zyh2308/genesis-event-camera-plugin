# R1.5 定位纠错：以像素误差验收

## 本轮结论

这轮不以“生成事件更像 Direct-HDR”判定修复，而以目标真实像素位置作为标签。在固定事件输入上，新候选的拦截前定位显著改善，并在训练之外的三条存量轨迹上复核通过。

修正方案是：让插件通过 `capture_observed` 接收实际、带时间戳的浮点线性 HDR 观测，不再把这些已观测图像强制经过原有稀疏中间图重建；用该入口生成相同九条物理轨迹的训练事件，再按原来的网络和预算重训。这一候选恢复了可用的观测输入路径。它没有把旧稀疏 warp/hold 路径整体修好，后者不能据此宣布通过。

## 1. 固定九条开发轨迹：581 个查询

| 模型 | 查询数 | 像素误差 mean / median / P95 | 超过 10 px 或无效 |
|---|---:|---:|---:|
| 旧 Ours | 581 | 24.98 / 8.18 / 106.72 px | 225/581 |
| 仅覆盖规则修正 | 581 | 36.99 / 29.09 / 99.97 px | 396/581 |
| 实际 HDR 观测入口候选 | 581 | 1.99 / 1.72 / 4.03 px | 1/581 |
| 固定 V2E 模型 | 581 | 2.39 / 2.15 / 5.23 px | 1/581 |

旧 Ours 与新候选都提供了 581/581 个有限定位；新候选不是通过拒绝困难查询降低误差。九条轨迹逐集的中位数和 P95 均低于旧 Ours。新旧评分的世界坐标位置、速度误差逐查询精确复现旧 evaluator，最大差值为 0；三个重复运行中的固定 V2E 对照也一致。

## 2. 冻结权重后的另三条存量轨迹：193 个查询

| 模型 | 查询数 | 像素误差 mean / median / P95 | 超过 10 px 或无效 |
|---|---:|---:|---:|
| 旧 Ours | 193 | 31.52 / 9.08 / 112.38 px | 87/193 |
| 仅覆盖规则修正 | 193 | 37.97 / 28.78 / 97.89 px | 124/193 |
| 实际 HDR 观测入口候选 | 193 | 1.99 / 1.82 / 4.24 px | 0/193 |
| 固定 V2E 模型 | 193 | 2.53 / 2.32 / 5.22 px | 1/193 |

这三条轨迹的 seed 为 `2026103900/2026103901/2026103902`，与本次全部训练/验证数据以及九条主开发轨迹均不重合。次级评估前记录了 checkpoint 哈希和验收规则，之后没有重训、调参或重新选 checkpoint。新候选在三条轨迹上逐集的中位数、P95 也均优于旧 Ours。

它们是历史归档数据，本次由这些固定权重首次评估；这是训练之外的补充确认，不冒称新采集的正式独立测试，也不冒称未见速度/落点条件。两批结果均为一个训练 seed 的开发验证，不据此宣称生成器排名。

## 3. 误差如何计算

- 原始相机分辨率 `320×240`。目标像素标签来自物理目标中心的固定相机投影，时刻为事件观测截止时刻。
- 所有模型读取相同的事件、相同 25 ms 因果五-bin 输入和查询时间，使用相同相机标定、平面高度约束和后处理。
- `pixel_euclidean_error_px = ||predicted_uv - physical_GT_uv||₂`，不是与另一个仿真器输出坐标的差。
- 2/5/10/20 px 阈值仅作诊断；缺失或无效定位计为失败，分母始终为全部查询。10 px 不是事后改写的抓取成功标准。
- 本地从 CSV 的预测/GT 像素坐标独立复算了所有 3,096 行，误差、均值、中位数、P95、阈值计数及各模型查询/GT 集合全部一致。

## 4. 为什么此次与“只修覆盖规则”不同

只修覆盖规则后的模型，主开发集像素误差中位数为 29.09 px，确实比旧模型 8.18 px 更差。这说明图像层的一项规则通过，不足以证明训练数据已经能支持定位。

本轮完整替换了这条任务的训练事件输入路径，并把重新训练后的定位结果作为验收。改善支持“旧中间图生成导致的输入差异是主要问题之一”，也表明现有网络、标定和标签可以实现准确定位。由于新路径绕过了整个稀疏重建环节，不能把改善全部归因于覆盖、阴影或某一条单独规则。

## 5. 当前可用方案与下一步

- 可作为下游定位开发候选：实际线性 HDR 观测入口、固定 0.5 ms 样本间隔、已验收 checkpoint。原有 calibrated `real_v2` 默认值未改，本轮延续之前明确标记的理想传感器诊断配置。
- 模型结构仍为 `heatmap_v2`，seed 606，40 epoch 上限，batch 32，训练 `[0,1,2,3,7,8]`、验证 `[4,5,6]`；新模型 best epoch 为 39，checkpoint 不由共同评估集选取。
- 原生 HDR 采样需要更多真实渲染；当前结论是定位恢复，不是稀疏重建效率改进。继续优化效率时也必须保住像素指标。
- 本轮是离线拦截前感知验证。没有将旧机器人闭环成功记到新模型名下；新的闭环抓取/反弹任务需要另跑。

## 6. 代码、证据与复现

- 代码分支：`exp/research-pixel-repair-20261101`。
- 像素评价代码 commit：`403426700ff279555ddbb060cec8c43094e693f2`。
- 观测入口实现继承：`1c6ce68244063c08977d1be0123eb12e3d65389a`。
- 主结果：`summary.json`、`all_models_pixel_polls.csv`；次级结果：`secondary_confirmation/summary.json`、`secondary_confirmation/all_models_pixel_polls.csv`。
- 指标图：`evidence/pixel_localization_results.png` / `.svg`。
- 全次级轨迹回放：`evidence/pixel_localization_secondary_replay.gif`，展示全部 193 个查询，绿色圈只用于事后标记 GT。
- checkpoint：`candidate_model/spatial_event_state.pt`，SHA256 `a11df9aedc49c9cff29554993159e9dc695d2e7ab6555669ff3bd5277c5212ac`。
- 全部 g1 原始结果：`/home/zhaoyuhan/Eventbased_WAM/runs/robot_r1_5_pixel_objective_20261101a/`。
- 原生 HDR/训练事件与旧训练过程：`/home/zhaoyuhan/Eventbased_WAM/runs/robot_r1_5_research_repair_demo_20261101a/`。
- 原生候选的有效浮点 HDR 输入在 v1/v2 入口下九集 raw events 均逐元素一致；这一检查用于版本追溯，不作为定位成功指标。

最小复跑命令（在仓库根目录运行，输出目录改成未存在的新路径）：

```bash
python benchmark/robot_closed_loop/diagnostics/check_pixel_localization_metrics.py
python -m benchmark.robot_closed_loop.diagnostics.evaluate_r1_common_direct_hdr \
  --scene-config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --control-config benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json \
  --reference-root /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016 \
  --ours-checkpoint /home/zhaoyuhan/Eventbased_WAM/runs/robot_r1_5_pixel_objective_20261101a/candidate_model/spatial_event_state.pt \
  --v2e-checkpoint /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018b/paired_r1/v2e_fit/spatial_event_state.pt \
  --output /absolute/new/output/path --deceleration 0.09803606718731761
```

次级复跑仅替换 `--reference-root` 为 `/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_7-heatmap-20261017/benchmark/robot_closed_loop/results/r0_7_heatmap_event_state_closed_loop_3seed_20261017`。训练数据与模型选择规则保持不变。
