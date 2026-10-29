# R1.5 方块定位失效定位：共同 Direct-HDR 事件上的热图解码诊断

## 结论先读

目前已把主要误差定位到**模型输出热图本身**：原生 HDR 训练的 Ours checkpoint 在共同 Direct-HDR 事件上，有相当一部分查询把最强热图峰放在方块之外。换成取最大峰或峰周围局部 soft-argmax，只能改善一部分“热图峰本来就在方块附近、但全图平均被拉偏”的窗口；对热图主峰本身就错的窗口基本无效。因此，当前最先坏掉的环节不是速度回归、控制器、陈旧输入或单纯的全图 soft-argmax。

**更深层原因还没有因果定位。** 现象符合事件来源/重建模式发生域偏移或模型学到了来源特有线索，但目前原生验证集和 Direct-HDR 轨迹不是同一物理运动，不能据此证明是哪一个生成环节导致失效。当前结果足以和研究会讨论下一项有信息量的实验，不足以宣称修复完成或给 Ours/V2E 排名。

## 本轮做了什么

- 在 g1 的 CPU 上读取既有 9 集共同 Direct-HDR 事件。两个 checkpoint 使用完全相同的 581 个接触前因果查询、事件文件、时间截点、五个 5 ms 时间 bin、模型结构与 GT 标签；分别离线解码网络热图为全图 soft-argmax、最大热图格点、最大峰周围 5×5 格点的局部 soft-argmax。
- 另分析原生 HDR 验证 episode 004/005/006 的 194 个窗口。它们与 Direct-HDR 只匹配速度和发射横向位置，**种子和物理轨迹不同**，仅作描述性域对照。
- 用 GT 方块投影和机器人 TCP/手部投影做了推理后的离线诊断；这些信息没有进入网络输入或事件张量。
- 没有运行 Genesis、renderer、物理 rollout、GPU 或训练；没有修改 frozen core、控制器、任务、checkpoint 或正式评测来源。

共同 Direct-HDR 查询均为 `[cutoff−25 ms, cutoff)` 的因果事件窗口，截止时间沿用公共评测 poll 减 10 ms 延迟；只保留首次真实接触前的查询。固定平面中心高度为 20 mm。GT 与投影 ROI 仅用于推理后的误差计算。

## 关键数字

| Checkpoint / 解码方式 | 查询数 | XY 误差均值 | 中位数 | P95 | 最大值 |
|---|---:|---:|---:|---:|---:|
| Ours 原生 HDR / 全图 soft-argmax（当前） | 581 | 83.8 mm | 14.4 mm | 444.9 mm | 818.4 mm |
| Ours / 最大热图格点 | 581 | 85.7 mm | 11.6 mm | 573.1 mm | 1164.7 mm |
| Ours / 峰周围局部 soft-argmax | 581 | 84.4 mm | 11.1 mm | 573.8 mm | 1090.4 mm |
| V2E checkpoint / 全图 soft-argmax | 581 | 4.84 mm | 4.09 mm | 11.26 mm | 31.23 mm |

这些是**不同训练来源的两个 checkpoint 在相同 Direct-HDR 输入上的迁移诊断**，不是生成器排名：两模型虽然结构相同，但各自训练数据来源不同，且不是正式独立测试集。

### 错误首先在哪里

- 581 个 Ours 查询中，95 个（16.4%）的最高热图峰离 GT 投影超过 20 px。
- 当前全图 soft-argmax 有 126/581 个查询误差超过 100 mm。其中 89 个同时属于“最高热图峰远离目标”；另有 37 个最高峰虽在 20 px 内，但整张热图的概率平均仍被其他区域拉偏。
- 在最高峰靠近目标的 486 个查询里，改为最大峰或局部 soft-argmax 后，超过 100 mm 的查询均为 0；在最高峰远离目标的 95 个查询里，两种替代解码仍分别有 88/87 个超过 100 mm。
- 峰远离目标时，目标附近 ROI 的热图概率质量中位数只有 0.077；峰靠近目标时为 0.998。这支持“热图把概率放错位置”，而非只是一点坐标解码偏差。
- 九集按 Ours 当前全图解码的 episode P95（mm）为：`298, 331, 545, 497, 89, 537, 241, 21, 494`，分别对应速度/横向发射位置 `.3/−55, .3/0, .3/+55, .4/−55, .4/0, .4/+55, .5/−55, .5/0, .5/+55`。失败不只发生在某一个速度档；+55 mm 三档都差，中心 .5 m/s 较好，其他位置有明显波动。
- V2E checkpoint 在相同 581 个查询上，全图 soft-argmax P95 为 11.26 mm。它说明这些共同输入对该 checkpoint 可定位，但不能单独说明 Ours 生成器整体弱于 V2E。

### 解码置信度能否先挡住错误

归一化熵、峰值概率、全图均值到最大峰的距离，在这批开发窗口上能识别错误（AUC 分别 0.959、0.951、0.927）。若把原生验证集熵 P95（0.3259）当拒绝阈值，在 Direct-HDR 查询中拒绝 49.7%，可挡住全部 126 个 >100 mm 错误；剩余 292 个接受查询的均值/P95 为 11.0/21.5 mm。

这是一个**离线拒答候选，不是定位修复**：阈值来自已参与 checkpoint 选择的原生验证数据，Direct-HDR 又是已看过的开发集，且 581 个窗口高度相关。还没有独立 episode 或闭环验证；约一半覆盖率也未必满足机器人控制需求。

### 机器人遮挡线索检查

95 个远离方块的错误主峰中，仅 6 个离 TCP 投影 25 px 内，7 个离手部 link 代表点 25 px 内。最简单的“网络只是跟着 TCP/手指点走”解释不太支持；但这**没有排除整条机械臂轮廓、机械臂阴影、桌面背景纹理或它们与训练域的交互**。

## 和此前事件回放问题的关系

较早的 2026-10-29d 检查确认，冷启动缓存回放会重置 DVS comparator 历史：同一 HDR 像素帧冷回放得到 5899 个事件，而在线历史状态得到 6179 个。用之前真实事件历史重建 comparator 后，可精确复现在线窗口的 6179 个事件。这是已证实的回放一致性缺陷；但在正确历史状态事件上，旧 checkpoint 仍然明显失准，所以该缺陷**不能解释全部定位失败**。

本轮进一步表明，在共同 Direct-HDR 输入上，Ours 网络的空间热图已会把主峰放错。下一步应先用同一条物理轨迹做事件来源交叉对照，才能分辨是事件生成/响应差异让模型失配，还是训练数据覆盖与模型泛化不足。之前报告过的显示 PNG→scene-linear HDR、comparator 历史等差异都应保留为待控变量，不能先认定某一项是根因。

## 当前认识：证据与假设分开

**已经证实：**

1. 历史缓存冷回放丢失 comparator 起始状态，会导致事件序列不等于在线同轨迹事件；该问题在历史重建状态后可复现修正。
2. 即使用正确的共同 Direct-HDR 输入，原生 HDR Ours checkpoint 的热图也在许多查询中定位错；仅换解码器不足以修复远峰错误。
3. V2E checkpoint 能在相同查询上给出较准结果，故该任务图像/事件中存在可利用的位置线索；但不证明 Ours 事件一定等价或更差。
4. 新鲜度/时间戳不是本轮共同查询的缺失项：581 个查询都有有效状态，模型使用相同窗口。

**仍未证实：**

- 原生 HDR 与 Direct-HDR 事件之间哪一种具体差异导致域偏移；需要同轨迹、相同 DVS 响应和 comparator 历史的来源交叉试验。
- 失败主峰具体对应哪种场景结构。当前只排查了 TCP 和一个手部点，没有检查完整机械臂、阴影、遮挡边界和背景。
- 是否通过统一事件生成配置、扩大训练覆盖或增加拒答状态即可恢复可用定位；没有做重训、独立新 episode 或闭环。
- 事件数量差异是否起因。计数分布确有变化，但 V2E checkpoint 在共同输入可用，单看事件数量不能建立因果关系。

## 建议的下一项实验

建议先做一组**同物理轨迹的事件来源交叉实验**，先选中心正常片段和 +55 mm 失败片段，复用缓存的完整物理状态、相机、时间戳、目标标签和 sensor response：

1. 从同一状态轨迹分别生成待比较的 Ours 原生 HDR 路径与直接参考事件；每一方独立、完整地维护同一初始化规则下的 comparator 历史，不以后处理删事件来凑一致。
2. 冻结当前两个 checkpoint，不改权重；对齐同一查询时刻，逐项保存事件、五-bin 张量、heatmap、位置误差和置信度。先查看事件和热图差异发生在哪一时刻/区域。
3. 将完整机械臂轮廓、阴影与目标区域做推理后可视化，定位错误峰落在哪类表面；GT/分割仍只用于离线诊断。
4. 若差异随事件来源变化，再最小化定位 HDR 输入、重建、采样、DVS 状态等环节；若两来源都失败在同类状态，再考虑小规模同预算重训和新 episode 验证。
5. 确认 GPU 资源有明确预约后再跑渲染；本轮没有启动渲染、训练或仿真。

这能区分“生成的事件让现有模型看不懂”和“现有训练/模型对这类运动泛化不足”，信息量高于立即调网络或继续改控制器。

## 复现与产物

全部文件与嵌套证据均由本目录 `SHA256SUMS` 覆盖（该清单不对自身计算哈希）。原始逐查询 CSV 保留 581 个 query × 6 个 checkpoint/decoder 组合，共 3486 行；摘要 JSON 保留整体、按速度、按 episode 统计。`native_validation_confidence/episode_004..006/` 保存三集 CSV/summary/原始归档，`scene_association/` 保存 TCP/手部关联检查。

在仓库根目录重新生成派生汇总：

```bash
python benchmark/robot_closed_loop/diagnostics/summarize_r1_5_decoder_failure_modes.py \
  --direct-csv benchmark/robot_closed_loop/results/r1_5_shared_direct_hdr_decoder_20261029f/result/shared_direct_hdr_decoder_queries.csv \
  --native-validation-dir benchmark/robot_closed_loop/results/r1_5_shared_direct_hdr_decoder_20261029f/native_validation_confidence \
  --scene-association-summary benchmark/robot_closed_loop/results/r1_5_shared_direct_hdr_decoder_20261029f/scene_association/summary.json \
  --output /tmp/r1_5_failure_mode_summary.json
```

逐查询推理由远端 g1 CPU 在 `4c5f5f29d4c46dd50401c0aa9079874f97afcbc6` 代码 checkout 上完成，Python 共享环境 `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python`，单 Torch 线程；推理 11.11 s，整个进程 wall time 14.07 s，最大 RSS 353,376 KiB。主诊断代码脚本 SHA256 为 `18fcb28bdab6ef7b3bc034aa98828d5d66209edef3aa0b71d29710ef39360b29`；空间解码 helper 为 `105e0a3f404bc3650766a8741fd8392690133aaabcec12f92ae3f76188c3d378`。可复核输入摘要、poll、scene、checkpoint、episode event/trace hashes 见 `result/summary.json` 与派生 `failure_mode_summary.json`。

本目录是 2026-10-29f 的只读开发分析快照；不代表独立测试、正式多 seed 结果、训练收益、事件生成器排名或机器人闭环成功。
