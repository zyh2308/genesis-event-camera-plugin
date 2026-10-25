# 光流与位姿问题定位：公共合成光流阳性对照（2026-10-26）

## 结论摘要

本次用一个公开、带事件和稠密光流标签的合成场景，对 STE-FlowNet 的输入—标签—优化路径做了限额检查。结果支持三点：

1. **STE-FlowNet 不是完全无法学习运动。** 使用正确方向的 eCARLA 标签做 20 次监督更新后，独立时间段验证中的光流误差下降；正常事件优于清零事件，也优于零光流基线的共同事件区域。
2. **公开数据原始光流方向有读取契约陷阱。** 该固定序列的原始 flow 与独立灰度跟踪近乎反向。按 eWiz 所依赖的 `inverse_optical_flow` 适配器转换后，方向、幅值和光度 warp 均与前向图像运动一致。原始标签不得不经适配直接训练。
3. **这还不能判定 EventGenesis/Genesis 仿真器有错。** 阳性对照换了数据集、事件生成器和监督目标，且只有一个场景、中心 128×128 crop 和 20 次更新。它排除了“STE-FlowNet/优化代码在任何合成事件上都学不了”的强解释，但没有证明 Genesis 事件质量、原自监督目标或全尺寸泛化正确。

真实数据评分没有运行，MVSEC/HQF 未参与本实验，真实评分门禁仍关闭。冻结 simulator core 未改。

## 公开数据与固定样本

- 数据集：[`eCARLA-scenes`](https://zenodo.org/records/14412251)，公开 CARLA 合成事件、灰度图和光流数据；官方仓库说明动态车辆/行人场景及 260×346 分辨率：[`CIRS-Girona/ecarla-scenes`](https://github.com/CIRS-Girona/ecarla-scenes)。
- 预先固定场景：`26_cars_town4_forward_cloudy-noon`。选择依据是公开动态车辆场景；选择发生在查看网络结果之前，没有按生成器事件量、模型分数或运动误差筛选。
- 只按 ZIP64 中央目录分段读取并提取该场景四个成员，没有下载整个 7.53 GB 动态归档，也没有验证整包 MD5。单成员 CRC 已核验。
- 提取文件在 WSL cache：`/home/科研/Eventbased_WAM/.cache/sim2real/ecarla-26-cars-20261026/`；文件 SHA256、字节数、事件总数、时间偏移和采样间隔见本目录 `ecarla_audit_v2.json`。
- 外部代码版本：eWiz `7e6f9132bcedb5ebaf49743f27a0679075017ad0`；STE-FlowNet `eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6`。

## 已确认的数据接口问题

### 事件时间窗边界

首版 v1 适配使用右侧插入点，32 个固定窗的事件计数均无法与 `flow_to_events` 对齐。这个结果保留在外部 cache 的 `audit-v1/`，并标记为失效尝试。核对 eWiz 写入代码后改为左侧插入点和 `[start,end)`：32/32 个固定 train/validation 窗事件数与官方索引差一致；同一采样时刻的 flow 与灰度 endpoint 时间戳完全相同，帧间隔为 39,999–40,001 μs。

### 原始光流方向

对预先登记的 8 个时间窗，用 OpenCV `goodFeaturesToTrack` 与金字塔 Lucas–Kanade 从 `gray[i]` 跟踪到 `gray[i+1]`，独立比较原始 flow 和转换 flow：

| 量 | 原始 flow | upstream inverse-flow 转换后 |
|---|---:|---:|
| 方向余弦中位数（8 窗的中位数） | −0.9982 | +0.9987 |
| 与图像跟踪的中位 EPE 中位数 | 8.28 px | 0.74 px |
| flow/跟踪位移幅值比中位数 | 0.926 | 1.001 |
| 正方向跟踪比例中位数 | 10.9% | 96.1% |

这是针对该实际 artifact 的实测契约，不据此推断整个 eCARLA 数据集都反向。训练和验证标签采用官方 `inverse_optical_flow.avg_method` 转换，并排除其 disocclusion mask、非有限像素及前向 endpoint 越界像素。转换算法输出的 604 个非有限通道值也被显式排除，没有当成有效 GT。

作为独立方向/warp 检查，8 个登记窗的纹理运动区域内，正确方向 GT 的平均光度 MAE 为 6.16，零流为 10.27，反向流为 12.56（8-bit 灰度单位）。事件活跃且纹理运动的区域更小，支持掩码、遮挡和量化会影响光度诊断；这些值不是网络指标。

## STE-FlowNet 固定小型监督拟合

配置在训练前固定：训练 flow indices `[100,104,108,112]`；独立时间段验证 indices `[600..615]`；中心 128×128 crop；5 个等时长事件 bins；相同转换后的前向 flow GT；四尺度鲁棒 EPE；Adam 原有配方；batch 1、seed 0、20 updates、无增强、无样本重排。验证正常输入、全零输入、反转 temporal bins；三个版本都在正常事件预先生成的同一 validity∩event-active mask 上评估。

- 设备：本地 WSL NVIDIA GeForce RTX 2050 4 GB；g1 当时 GPU 由其他用户进程占用，没有在服务器启动训练。
- 环境：Python 3.14.6、PyTorch 2.13.0+cu130、CUDA 13.0、h5py 3.16.0、NumPy 2.5.4、OpenCV 5.0.0；隔离 Python 包缓存中 `hdf5plugin 7.1.0`、`inverse_optical_flow 0.0.2`。
- 执行时间：2026-10-26 01:37:39–01:40:09 JST（UTC 16:37:39–16:40:09，约 158 秒）。CUDA 峰值 PyTorch 分配量 1.65 GB。
- 训练 objective 从第 1 步 11.60 降至第 20 步 9.34；每步都有有限且非零梯度、参数状态变动。
- 验证有效区域的零流基线为 22.65 px。20 步后正常输入 18.34 px，清零输入 20.47 px；固定事件活跃区域的初始 EPE 1.93 px、训练后正常输入 1.65 px、清零输入 2.33 px、零流 1.89 px。
- 活跃区域只占 crop 有效像素约 0.355%。正常预测在活跃 GT 运动上的平均方向余弦为 0.753；清零输入为 0.068。
- 反转时间 bins 后验证全区 EPE 18.32 px、活跃区 EPE 1.69 px，与正常输入非常接近。因此本小预算证明网络能利用部分事件支持，但**没有证明它学到了强时间顺序依赖**。
- 保存 554 MB checkpoint；加载后权重 hash 相同，普通 eval 下全部验证样本、三种输入的输出 hash 与保存前一致。

结果属于监督阳性对照，不是正式模型比较或泛化结论。训练与验证来自同一个场景的不同时间段，并非跨场景验证。

## 对三类原因的判断

| 假设 | 本次证据 | 当前判断 |
|---|---|---|
| STE 模型/优化完全不能学习 | 正确方向 GT 上 20 update loss 下降；验证 EPE 优于零流；清零事件变差；保存恢复一致 | **强版本已排除。** 仍未证明全分辨率、更多场景或正式自监督目标可靠。 |
| 标签/数据适配错误 | 原始 flow 与灰度前向跟踪方向相反；显式 inverse adapter 后独立 track 对齐；原始事件计数边界 v1/v2 不同 | **已确认存在读取契约问题，v2 已修正。** 原始 artifact 本身不改写。 |
| Genesis/EventGenesis 模拟器错误 | eCARLA 阳性对照不使用 Genesis；未运行新的 Genesis 高纹理场景；已有 Genesis bank 低事件支持但本实验没有重审 | **未确认，也未排除。** 不能用 eCARLA 结果声称仿真器正确或错误。 |
| 光流数据可观测性不足 | eCARLA 验证 crop 有效像素中共同活跃支持约 0.36%；反转时间影响小；原 Genesis pilot 此前更低 | **是当前训练诊断的重要限制。** 需优先补场景/支持覆盖证据，不按某生成器事件量挑样本。 |

## 位姿已确认问题（引用已有证据，本次未重复访问数据）

E-POSE 的现有单样本审计已发现 2D keypoint 投影标签疑似构造缺陷：正确逐点深度投影与 MAT keypoints 平均差 16.450 px、最大 32.744 px；复现导出器给所有角点误用首角深度的行为后，残差降至均值 0.166 px、最大 0.349 px。CAD 与目标木块匹配、坐标尺度也通过静态核查。结论是**数据标签构造路径具体可疑/与源码故障一致**，不是“模型没训练够”；还需用其他时间戳确认坐标/帧约定。依据见 [sim2real_status.md](../sim2real_status.md) 和 [STAGE6_THREE_LINE_HANDOFF_20261024.md](../STAGE6_THREE_LINE_HANDOFF_20261024.md)。本次未重新下载或评分 E-POSE。

## 复现

以下命令在 WSL 仓库根目录运行。数据成员已在 cache；`audit-v2/` 与 `network-fit-v2/` 是本次只写新目录，旧 v1 现场保留。

```bash
PYTHONPATH=/home/科研/Eventbased_WAM/.cache/sim2real/python_pkgs \
python benchmark/sim2real/scripts/audit_ecarla_flow_sequence.py \
  --data-root /home/科研/Eventbased_WAM/.cache/sim2real/ecarla-26-cars-20261026 \
  --config benchmark/sim2real/configs/ecarla_flow_diagnostic_20261026_v2.json \
  --output-root /home/科研/Eventbased_WAM/.cache/sim2real/ecarla-26-cars-20261026/audit-v2

PYTHONPATH=/home/科研/Eventbased_WAM/.cache/sim2real/python_pkgs \
python benchmark/sim2real/scripts/run_ecarla_steflow_network_fit.py \
  --data /home/科研/Eventbased_WAM/.cache/sim2real/ecarla-26-cars-20261026/audit-v2/ecarla_fixed_windows.npz \
  --model-root /home/科研/Eventbased_WAM/external_sources/STE-FlowNet \
  --config benchmark/sim2real/configs/ecarla_steflow_network_fit_20261026_v1.json \
  --output /home/科研/Eventbased_WAM/.cache/sim2real/ecarla-26-cars-20261026/network-fit-v2 \
  --device cuda:0
```

运行前核对 NVIDIA GPU 资源。首次 v2 adapter 训练脚本曾出现固定索引减法错误，留下空目录 `network-fit-v1/`；修正脚本后在 `network-fit-v2/` 完成。没有删除失败尝试或覆盖 v1 结果。

## 证据索引

- 本目录 `ecarla_audit_v2.json`、`network_fit_summary.json`、`updates.jsonl` 为小型可入库证据。
- 完整压缩窗口 NPZ：`/home/科研/Eventbased_WAM/.cache/sim2real/ecarla-26-cars-20261026/audit-v2/ecarla_fixed_windows.npz`，SHA256 `2b698ffa8fec3bdb8be189533750b3c406ffa22dca30b62e6a487a683d67ed0b`。
- checkpoint：`/home/科研/Eventbased_WAM/.cache/sim2real/ecarla-26-cars-20261026/network-fit-v2/checkpoint.pt`，SHA256 `01bfed84b7dce6afd6a4f0ae53a426d4fd99a6d68dc53c451acca022fe08262b`。
- 每个数据成员 SHA256/大小、配置和脚本 hash、模型 hash、退出状态见本目录 `evidence_index.json`。

## 最小下一步建议

暂不扩展训练预算。先用相同 STE 模型和正确已核查的输入接口，在一份新的、预登记的 Genesis 高纹理/持续相机运动共享 bank 上做固定样本 GT 监督拟合与独立 rollout 对照；再让原自监督损失跑有限配对短训。这样可以分别检验 Genesis 标签/事件接口与自监督目标，而不把两者混在一起。只有在共同事件支持达到事先约定的可观测门槛后，才考虑正式扩训。
