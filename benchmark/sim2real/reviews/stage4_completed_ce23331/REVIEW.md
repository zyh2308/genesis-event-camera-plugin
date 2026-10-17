# Stage 4 seed 0 完成核验与研究结论

核查日期：2026-10-17（Asia/Tokyo）。被核查提交：`ce2333128b5a203412b4e3aa80ca355fd669c07b`；分支：`exp/public-sim2real-v1`。

**结论：两组完整训练的完成声明属实；本轮物理正确性与研究结果验收不通过。** 独立检查发现共享数据的渲染分割编号与物理 link 编号错配，以及 STE-FlowNet 图像采样的坐标约定不一致。旧 EPE 可作为程序历史输出保留，不能作为有效的生成器质量比较。应先修复、重建受影响数据并短跑验收，再启动新的正式配对训练或增加 seeds。

这是研究会话的独立核验记录，不修改工程实现、不修改冻结 simulator core，也未开始新训练或真实数据评分。

## 1. 完成声明：通过

直接通过 SSH 读取 g1 原始日志、退出码及四个 checkpoint，执行 CPU 解包、计数和哈希核验；另对既有最佳模型做了每组 24 个固定合成窗口的有限 GPU 推理诊断。

| 项目 | 独立核查结果 |
| --- | --- |
| 每组 optimizer 更新 | 各 8,910；编号连续 1–8,910 |
| 每组训练 epoch / 验证记录 | 各 45；每轮 198 次更新 |
| 每轮训练覆盖 | 1,584 个唯一合成窗口；batch 8 |
| 全部配对样本顺序 | 8,910 次更新逐项一致 |
| 初始化 | 两组均 `acc6bbd11488a21868c5f147c7f397521480166c24914cc7988a6428d6ed71cf` |
| 更新日志 | 每组 8,955 行，含 8,910 次更新、45 次验证；所记录损失与梯度均有限 |
| 退出状态 | 两组 `train.exit_code.txt` 和总任务 `training_retry3.exit_code.txt` 均为 0 |
| checkpoint | 两组 best、last 四个文件 SHA-256 均与报告一致；模型参数有限、模型状态哈希匹配；last 的 optimizer step 均为 8,910 |
| 最佳 checkpoint 对应更新 | EventGenesis：epoch 16 / update 3,168；V2E：epoch 17 / update 3,366 |
| 任务结束时间 | 约 2026-10-17 13:03:48 JST；总耗时约 11 小时 29 分 |
| 活跃训练进程 | 核查时未检出 `train_steflow_stage4.py` |
| 仓库状态 | 核查时 WSL、g1 与 GitHub 分支均指向被核查提交；WSL/g1 工作树干净（本研究审查目录另行加入） |
| 冻结 core | `genesis_event_plugin/` 相对 `e79e656ac00f7847b98db9e12bf43b093f3afff1` 无差异 |

训练元数据记录的启动代码提交为 `1fab7e17e201943c77703aa8a1c0c6eaefbb8f5a`，完成报告提交为 `ce23331...`，两者不能混写为“从 ce23331 启动训练”。相关后续提交为报告和证据归档。此前登记的 739 项仓库证据文件哈希亦已核对。

CPU 载入核实了 checkpoint 内容，不等于修复了已知的 CUDA resume RNG 设备问题；本次完整训练没有依赖断点续训。工程方关于该问题不影响本次已完成运行的说明合理。

## 2. 数值复核：报告转录正确，但标签有误

单位均为 px / 30 ms，数值越低表示与当前保存标签的差异越小。

| 输出 | 最佳合成验证 EPE | 最佳 epoch | 第 45 轮 EPE |
| --- | ---: | ---: | ---: |
| EventGenesis | 0.9389129835 | 16 | 0.9466441114 |
| V2E | 0.9406893614 | 17 | 0.9438607459 |
| 恒定预测零光流 | 0.8241647044 | 不适用 | 不适用 |

零光流基线独立遍历全部 528 个验证窗口，使用与训练验证相同的 `flow_valid`、逐窗口均值和逐 rollout 等权汇总。它不是从 24 个诊断窗口外推得到的。两组最佳模型相对该零基线分别高约 13.92% / 14.14%，且 8 个验证 rollout 均高于对应零基线。

**上述数值全部针对旧的、有编号错配的 GT。** 零基线异常是进一步审查的线索，不能据此声称模型相对正确物理真值差 14%。先前中间更新中将其描述为模型质量问题，此处以新发现的标签错误修正解释。

即使暂不考虑下文错误，最佳值差 0.0017764 px（约 0.19%），末轮排名还发生反转；单 seed、没有真实域测试，也无法支持生成器优劣或统计显著性。现在发现基础数据错误，更不能用这张表判断两组“同等好”或 EventGenesis 略优。

训练损失确有下降：EventGenesis 首/末轮均值约 1214.63 → 1088.94，V2E 1209.22 → 1084.73。这只证明当前计算目标发生优化，不证明运动估计已经学对。

## 3. P0：渲染分割编号被当成物理 link 编号

### 原因与源码链

g1 实际运行的 Genesis 源码 `runtime/sim2real/genesis_hdr_v2/genesis/vis/rasterizer_context.py:18–28` 明确维护两种编号：

```python
self.idxc_map = {0: -1}
self.key_map = {-1: 0}
seg_idxc = self.key_map.setdefault(seg_key, len(self.key_map))
self.idxc_map[seg_idxc] = seg_key
```

`camera.render(... segmentation=True)` 返回渲染用 `seg_idxc`，不是保证等于物理 `link.idx` 的编号。`visualizer.py:318–322` 的 `segmentation_idx_dict` 提供其映射。此 bank 中背景/地面/三个盒子的渲染编号为 `0/1/2/3/4`，对应物理 link 为 `-1/0/1/2/3`。

`benchmark/sim2real/scripts/generate_stage4_flow_bank.py:167–182` 却直接将 raw `seg` 传给插件，位姿字典按 `link.idx` 建键；`250–254` 又用同一 raw segmentation 和 link 位姿计算共享 GT。渲染中地面的编号 1 因而匹配到盒子的 link 1，最后一个盒子的编号 4 则没有对应物体位姿。不能用“双方 GT 完全相同”替代“GT 物理正确”。

### 可独立复现的静态地面反例

在 `s1_06` 和 `s1_07`（静止相机、独立物体运动）各取两个现有窗口，直接读取保存的深度、分割和相机/物体位姿。四个窗口相机位姿变化为 0、地面 link 0 位姿变化也为 0，但旧标签给地面赋予了非零运动：

| rollout / 源帧 | 旧有效 mask 内地面像素数 | 旧 GT 地面平均位移 | 按正确编号映射重算位移 |
| --- | ---: | ---: | ---: |
| s1_06 / 99 | 63,603 | 0.504206 px | 0 |
| s1_06 / 195 | 16,582 | 2.252701 px | 0 |
| s1_07 / 99 | 63,261 | 0.553011 px | 0 |
| s1_07 / 195 | 29,090 | 3.023047 px | 约 3.2e-17 |

重算使用同一冻结投影函数、同一物理位姿和深度，仅修正输入编号。对本 bank 的检验用了 `raw>0 ? raw-1 : -1`，**该表达式只是当前数据的反事实诊断，不应作为所有场景通用映射实现**；工程应使用运行时实际映射，并明确背景语义。

### 影响范围

- 共享 flow GT、valid mask 与依赖这些标签的 synthetic-best 选择受影响。
- EventGenesis 插件在 `physics_plugin.py:319–325` 将这些分割与位姿传给物理插值器，Torch 后端按 `seg == object_id` 施加运动（`torch_physics.py:183–202`）；因此不只是报分标签出错，EventGenesis 合成过程的运动条件也错了。不能仅换 GT 重新打分就宣告修复。
- V2E 不消费这些物体编号，其输入 display frames 和原始事件是否可直接复用，应以依赖和哈希核查决定；不应无必要全部重渲染。
- 这里是 benchmark 提供者未满足冻结 core 输入契约。可先在 benchmark 的 frame-provider/GT 边界修正映射，无须改冻结 core 算法。

此前研究核验验证了配对一致性和接口运行，未发现这项语义映射错误；本次静态场景检查推翻了对旧标签物理正确性的验收。

## 4. P1：STE-FlowNet warp 坐标约定不一致

锁定源码 `STE-FlowNet/multiscaleloss.py:33–39` 用 `2*x/(W-1)-1`、`2*y/(H-1)-1` 归一化坐标，却未给 `grid_sample` 指定 `align_corners`；实际 PyTorch 运行环境默认 False。模型 `models/STEFlow.py` 的 `norm_grid/flow_warp` 也有同类组合。

真实训练调用链是 `train_steflow_stage4.py` → `run_steflow_stage2_pilot.objective_dt1` → 该锁定源码 `compute_photometric_loss`，并非闲置函数。此处应将坐标公式与采样约定配成一致，不能只屏蔽 warning。

独立测试提取锁定 `warp` 原函数，仅去除 CPU 测试不支持的 `.cuda()`；随机图像 32×40，舍弃 2 像素边界：

| 控制输入 | 当前实现 MAE | 坐标约定一致的对照 MAE |
| --- | ---: | ---: |
| 零光流，应保持原图 | 0.12265657 | 1.149e-7 |
| 水平 1 像素平移 | 0.12287799 | 1.083e-7 |

对照仅为这些基于 `W-1/H-1` 的调用明确 `align_corners=True`。这验证了数值契约错误，但没有证明它是模型全部异常的唯一原因。也不能机械地全仓库改 True：训练增强采用像素中心 `(x+0.5)/W` 配合显式 False，应独立核对并保留一致约定。建议使用 benchmark 的显式兼容补丁，保留第三方锁定版本与补丁哈希，双方采用同一实现。

## 5. 真实数据隔离与额外诊断范围

检查的训练入口、配置、日志和 checkpoint 元数据未发现真实数据进入训练、checkpoint 选择或生成器排序；优化器目标为合成图像的 photometric/smoothness loss，合成 GT 用于验证而非 optimizer 监督。该结论基于可见代码和记录，并非对所有服务器历史行为的绝对证明。本次审查没有打开新的真实域评分。

额外在空闲 GPU 0 上对每组最佳模型各运行 24 个固定合成窗口（每个验证 rollout 取 0、33、65），没有更新参数。该子集并非全验证集代表抽样，不得将其数字替代 528 窗口正式汇总；按各组事件支撑计算的 mask 也不同，不可跨生成器直接比较。本审查结论由源码、完整日志与独立受控检查支撑，不依赖该子集的排名。

## 6. 交给工程会话的下一步

目标仍是尽快得到可解释的完整管线；先完成两个明确错误的修复，避免把约 11.5 小时的有问题配置再扩成多个 seed。工程细节由执行会话自主决定。

1. 保留本次所有原始日志、checkpoint、报告及旧 bank；将本轮标为“完成训练、发现数据/warp 正确性缺陷，未通过研究验收”，不要覆盖旧版本或继续用旧 synthetic-best 作为正式结论。
2. 在 benchmark 边界通过 Genesis 实际 segmentation 映射统一渲染编号和 link 编号，显式处理背景；同时修复插件输入和共享 GT 路径。检查首帧是否已经处在预定义轨迹上，避免初始状态额外跳变。新增物理反例验收：静止相机/地面零 flow，物体运动只作用于对应物体，背景不绑定任意刚体。
3. 新建数据版本，重建共享 GT/valid 和受影响的 EventGenesis events、双方训练适配产物。逐项核定已有 radiance/depth/poses/display/V2E 原始 events 可复用范围，记录依赖与哈希，避免无必要重新渲染。修复必须覆盖训练和验证 rollout，不能只修某个报错样例。
4. 为确实使用了 `W-1/H-1` 的 loss/model warp 提供明确的兼容补丁；验证零位移、整数平移和反向传播。还应检查 E2VID 对应 warp 的坐标约定，但本审查未确认它是否存在同一缺陷。
5. 在新数据/新运行目录做有限的配对短跑，提供完整验证集零 flow 基线、GT warp 与零 flow 的图像残差对照（排除遮挡/光照不适用区域）、有限梯度、checkpoint 保存/恢复和双方一致性证据。不要在短跑前要求所有窗口都优于零基线；观察物理可解释性和学习趋势后再决定正式训练。支持真实域正式结论的指标与 mask 仍需按预先协议冻结。
6. 并行处理既有 resume RNG CPU/CUDA 设备问题及通用新运行入口；不要对已完成目录重跑恢复脚本。短跑核查通过后再提交启动新一轮完整 seed 0 配对训练，随后再扩 seeds。真实公共数据评分继续隔离，不能拿真实分数调这些修复。

无需为常规工程实现逐项来研究会话确认；若涉及改冻结 core、换模型/数据集/训练协议、使用真实域信息做选择，或修正后仍出现无法解释的物理错误，再汇报方向性问题。本记录不直接向 CLI 会话发送消息，也不启动其训练任务。

## 7. 证据入口

- `remote_verification.json`：全量更新/epoch/配对顺序、四个 checkpoint 内容与哈希、完整验证曲线、全 528 窗口零基线。
- `segmentation_verification.json`、`geometry_verification.json`：静态地面反例与原始几何检查。前者早期 renderer 扫描路径错误而 hits 为空；已在下一项用正确的项目 runtime 路径补查，几何重算结果不受该路径影响。
- `renderer_source_verification.json`：g1 实际 Genesis renderer 映射源码、返回值及 benchmark 调用处摘录。
- `local_source_excerpts.json`：冻结插件消费分割/位姿的路径摘录；`warp_coordinate_verification.json`：独立 warp 控制测试。
- `prediction_diagnostic.json`、`prediction_summary.json`：有限合成推理诊断及固定样本定义，不用于正式排名。
- 相邻 `audit_*.py` 与 `remote_*_program.py`：复现脚本；SSH 脚本只读 g1 数据，GPU 推理脚本仅运行既有模型，无训练。

原始完成报告：`benchmark/sim2real/STAGE4_SEED0_TRAINING_REPORT.md`。
g1 原始运行目录：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261016-stage4-flowdt1-seed0-v4/`。
