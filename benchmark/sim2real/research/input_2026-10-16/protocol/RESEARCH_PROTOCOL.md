# EventGenesis 公共真实数据 Sim-to-Real：配置决策与工程交接

日期：2026-10-16（Asia/Tokyo）。状态：研究配置决策稿 v1；不是已经通过运行验证的 formal freeze。

依据：用户指定的 `01_COMMON_RESEARCH_BASELINE.md`、`03_CHAT_SIM2REAL_HANDOFF.md`，以及本会话后续指令。核心仍固定为 `e79e656ac00f7847b98db9e12bf43b093f3afff1`。本文件不修改 simulator core，不采用 2026-09-30 方案。

## 1. 直接采用的实验组合

| 任务 | 主模型 | 真实测试 | 地位 | 主指标 |
|---|---|---|---|---|
| L1 重建 | E2VIDRecurrent，采用 E2VID+ 的公开合成训练路线 | **HQF 全部 14 序列** | 补充主证据 | LPIPS-Alex ↓ |
| L2 光流 | **STE-FlowNet**，官方自监督训练目标 | **MVSEC left：indoor_flying1/2/3、outdoor_day1** | 全分支主实验 | dt=1 四序列等权 AEE ↓ |
| L3 机器人相关物体状态 | **YOLO-6D / SingleShotPose multi-object**，13 类，keypoints + PnP | **E-POSE** | 工程可用性达标后进入 | ADD(-S) < 0.1d success ↑ |

L1 将原候选 HS-ERGB 改为 HQF：HQF 是专为事件重建建立的真实测试数据，EVREAL 有直接配置；HS-ERGB 原始任务是视频插值，需要额外定义重建采样和对齐。HS-ERGB 保留为扩展，不能影响 HQF 主协议。依据：[HQF/E2VID+ 论文](https://arxiv.org/abs/2003.09078)、[EVREAL HQF 配置](https://github.com/ercanburak/EVREAL/blob/398f03551b7f5d150ae10aa89e4e731557fce240/config/dataset/HQF.json)、[HS-ERGB 原仓库](https://github.com/uzh-rpg/rpg_timelens)。

STE-FlowNet 保留：已有 MVSEC 训练、GT 合成、评测与 MIT 代码。合成灰度帧只用于训练的 photometric supervision；推理输入仍只有事件。**不将其悄悄改成 supervised flow 网络**，flow GT 用于 synthetic validation 和诊断。ADMFlow 已审计但不加做第二模型：它另有多密度生成/损失要求，代码配置的 1,000,000 steps 与论文 150k iterations 也不一致。依据：[STE 官方代码](https://github.com/ruizhao26/STE-FlowNet)、[ADMFlow 论文](https://openaccess.thecvf.com/content/ICCV2023/html/Luo_Learning_Optical_Flow_from_Event_Camera_with_Rendered_Dataset_ICCV_2023_paper.html)。

L3 用 E-POSE 论文已采用的 YOLO-6D 模型家族，优先其官方 multi-object 实现，一组 generator/seed 只训一个 13 类模型。DOPE 虽有完整合成训练代码，但其官方建议按物体分别训练，增加本项目运行数量，不列为默认。E-POSE 的公开仓库是标注脚本和资产，并没有发布完整的 E-POSE 基线训练复现包；因此不能声称 L3 已可一键复现。依据：[E-POSE](https://doi.org/10.1038/s41597-025-04536-5)、[SingleShotPose](https://github.com/microsoft/singleshotpose)、[DOPE training](https://github.com/NVlabs/Deep_Object_Pose/tree/035f8e75d6ab20a33425c5364a8a3481a6c99bc6/train)。

## 2. 对旧候选配置的三项明确修订

1. **各任务沿用其成熟输入，组间严格一致**，不再要求所有任务都强制同一种 signed 5-bin voxel。L1 用 5-bin signed voxel；STE 用其 polarity-separated Gaussian-weighted event representation；L3 用 E-POSE 的三通道事件图。各任务之间的表示不作为消融变量。这是为遵循用户“成熟做法、减少工程投入”所作的研究协议修订。
2. L1 主数据集变为 HQF；HS-ERGB 扩展不作为必须完成项。
3. L2 原生训练为 synthetic events + synthetic grayscale supervision；不要求使用 flow GT 训练。保存同一套 flow GT 供验证，仍满足 shared physical labels 和 event-only inference。

## 3. 所有任务共用的控制条件

每个任务都先建立不含事件的 latent bank，再由两个 generator 导出。无需强迫三项任务使用同一个任务分布，但每项任务内部两组必须使用完全相同的 trajectory IDs、时间窗口、标签字节、划分和采样顺序。

- 训练 seeds 固定 **[0, 1, 2]**。同 seed 两组复制同一份初始模型权重、数据顺序和增强参数。
- 两组分别从该初值训练，不共享训练后的模型。L1/L2/L3 均不载入真实事件数据训练过的下游 checkpoint。
- 真实测试不参与 checkpoint、超参数、simulator、randomization 或子集选择；不进行 real-data fine-tuning、test-time adaptation 或更新 BatchNorm。
- 模型输入不含 RGB、depth、segmentation、真实/合成 GT mask、GT box、GT pose、robot state。训练 loss 使用合成标签合法；公开相机内参与 CAD 作为已知物体位姿系统的常规先验合法。
- 不用 GT 生成的 segmented events、GT crop 或提供的 masked image 作为 L3 推理输入。
- 固定物理时间窗，不按 generator 的 event count 改变窗口长度。不把两组事件数强行调成一样，不因某组稀疏删除样本。
- 空事件窗保留；输入为零。只有标签本身缺失/损坏可按预先列出的共同规则排除。loss 无有效标签的窗做显式 skip，并保证两组相同的优化更新次数。
- EventGenesis 的 adaptive render requests 合法，V2E 的插值合法；它们属于生成管线。记录渲染量与生成时间，不声称两组算力相同。

## 4. 合成语料：规模和生成设置

以下规模、轨迹分层及 rendering cadence 是**本项目的预算决策**，不是声称原论文的最佳设置；先用小 smoke 检验吞吐，不做真实测试排名驱动的 sweep。

### 4.1 L1/L2 共用 motion bank

- **160 个独立 rollout，每个 10 s**；144 train、16 synthetic val，按完整 scene/trajectory group 划分。
- 四层各 40 个：室内静态场景 + 相机运动；室外静态场景 + 相机运动；固定相机 + 多刚体运动；相机和刚体同时运动。每层 36 train、4 val。
- 相邻窗口/同一 rollout 不跨 split；同一基础场景仅改变时间偏移的派生版本也属于同组。
- 生成分辨率 **346×260**。物理标签每 **50 ms** 一组（20 Hz）；每 rollout 有 200 个 dt=1 区间，197 个 dt=4 起点。
- 因而 train 为 **28,800 dt=1 / 28,368 dt=4**；val 为 **3,200 / 3,152**。额外的启动状态与初始帧不计作独立样本。
- radiance 基础 endpoints **100 Hz**；标签时间落在该时钟上；物理积分步长、renderer 设置和 adaptive tolerance 单独写入生成 manifest。100 Hz 是本项目预设，不冒充文献统一标准。
- 每 rollout 固定灯光，跨 rollout 随机灯光、材质和曝光范围。L2 遵守 photometric supervision 的亮度一致性假设，不额外引入闪灯任务。
- 不重建真实测试场景，不复制真实测试相机轨迹。intrinsics/resolution 等公开设备元数据可用于确定成像几何，不能根据真实模型误差调整分布。
- 若 smoke 证明预算不合适，仅在未接触真实评分前整体修订 bank 规模/基础 cadence；两组同改并记录 changelog。160×10 s 不是已经生成或 hash 的数据。

### 4.2 V2E 与传感器配置

V2E 固定官方版本，完整 video-to-event 流程保留 SuperSloMo，目标插值步长 **1 ms**，禁用自动决定插值倍率，输入帧保持 100 Hz，空间分辨率不变。不要使用有损视频编码；按 image-sequence/export adapter 输入。固定色彩转换、曝光、tone mapping 和量化，不做每帧自动归一化。

V2E nominal 默认依据审计到的 `v2e_args.py`：ON/OFF threshold=0.2/0.2、sigma=0.03、cutoff=300 Hz、leak=0.01 Hz、shot noise=0.001 Hz、refractory=0.0005 s。它们只是起始的**nominal 配方**，并非 MVSEC/HQF/E-POSE 传感器实测值。正式前还必须记录 SuperSloMo checkpoint hash。依据：[V2E 参数源](https://github.com/SensorsINI/v2e/blob/cac638444557a224ba6ee5c29260916f152c00bd/v2ecore/v2e_args.py)。

EventGenesis 使用 frozen core 可配置的同名/可比名义参数，已有独立校准配置则完整记录来源；不为凑参数强改 frozen core。不支持的非理想项明确填 unsupported，不能声称 response model 完全匹配。**本实验比较完整生成管线**；要将优势进一步归因于某个单一 HDR 或 visibility 模块，需另行消融，本轮不新增。

光度导出默认决策：共享 scene-linear luminance Y，固定曝光 E=1，显示帧使用 `sRGB(Reinhard(E·Y))` 后量化为 8 bit；L1/L2 合成灰度监督来自同一导出，不分别使用 generator-specific GT。绝对 radiance 单位、E=1 的含义与 renderer 单位必须在 smoke 审计，若没有可解释的范围，先修订统一曝光定义再冻结；不允许真实测试后挑曝光。此映射是项目约定，不是声称 V2E 官方强制映射。

**不先做 contrast-threshold 大扫参**。论文已说明 threshold 分布会影响迁移，但本次先固定一套名义设定，结论限定于这套生成配置；不以真实 event-rate matching 来调训练数据。

## 5. L2：STE-FlowNet → MVSEC（先完成）

### 模型、输入与训练

| 项目 | 确定配置 | 依据/性质 |
|---|---|---|
| 架构 | 官方 `models/STEFlow.py`；不加模块 | 官方 |
| 初始化 | 随机；同 seed 两组相同；保留审计版本 initializer | 官方架构 + 配对控制 |
| 分辨率 | train 256×256 crop；test 中心裁剪 346×260 → 256×256，x=45、y=2 | 官方 |
| 时间设置 | dt=1、dt=4 分别训练模型；dt=1 主终点 | 官方双设置 + 项目主终点 |
| 表示 | 官方 Gaussian weighted、分极性；每基础区间输出 5 slices；dt=4 拼 20 slices | 官方；不是普通 voxel |
| loss | 四尺度 photometric Charbonnier + 10×二阶 smoothness；weights=[1,1,1,1] | 官方 |
| dt=4 loss | 对起点到 1/2/3/4 个区间末端的输出计算官方 loss，再取平均 | 官方 |
| optimizer | Adam，betas=(0.9,0.999)；weight decay=4e-4，bias decay=0 | 官方 |
| batch | 有效 batch=8；不足显存可累积，不能改变有效 batch | 官方 + 工程适配 |
| dt=1 | lr=4e-4；45 epochs；milestones=[5,10,20]；gamma=0.7 | 官方入口默认 |
| dt=4 | lr=2e-4；100 epochs；milestones=[5,10,20,30,40,50,70,90,110,130,150,170]；gamma=0.7 | README 推荐 lr；入口其余默认；>100 的 milestones 不触发 |
| augmentation | 两轴 flip 各 0.5；rotation ±30°；RandomResizedCrop 256，scale=(0.5,1)，ratio=(0.75,4/3) | 官方；所有事件通道和灰度帧同变换 |
| checkpoint | 每 epoch 保存；以全部 synthetic val 的 AEE 最小选；并保存最后一轮作为附表诊断 | 项目隔离规则 |
| 验证频率 | 每 5 epochs + 最后一轮；dt=4 同样每 5 轮 | 限定验证成本；只看 synthetic val |

按完整 batch（drop_last=True）计算：dt=1 每 epoch 3,600 updates，共 162,000；dt=4 每 epoch 3,546，共 354,600。drop_last 和同一 seed 的样本顺序必须两组一致。不能因为原训练脚本默认 test-set 指向真实 MVSEC，就在训练中执行它。

### 官方代码不能照搬的三个细节

1. `encoding/split_encoding_torch.py` 按事件数切片，并有 outdoor 13→5 子采样分支。这仍处在一个固定物理窗口内，和固定数量事件来改变物理窗口是两回事。**不改变公式/uint8 表示做新方法，但必须将分支显式化**：L2 合成主 bank 全部采用通用 5-slice 分支；真实 MVSEC 采用其原始序列分支。记录 outdoor adapter 的差异，不把其称作统一 equal-time voxel。少事件、空极性或相同时间戳时补零，避免除零；不借此删除样本。若这种表示在合成计数范围中溢出，S0/1 必须发现并统一确定 saturation/float policy 后再 freeze，不能静默 wrap。
2. 原脚本有无效边界填零、以事件是否非零跳过、timestamp 精度和 scheduler 调用顺序等旧实现问题。loader 用显式有效 index manifest，训练不按 generator event count 决定 skip，采样范围与优化步数一致；float64 原始时间戳先减去共同 origin。任何行为修订都记录补丁并对两组生效。
3. `outdoor_day1` loader 的布尔表达式与 `validate()` 内的筛选不一致。按 `validate()` 的有效范围预生成 manifest，禁止沿用错误表达式做正式样本选择。

若工程检查显示原生表示/旧 CUDA extension 无法在限定适配时间内稳定复现，则停止 STE 路线，依据工程原因另写一次模型切换记录；不拿真实分数挑 fallback。本稿不同时启动 ADMFlow 或 E-RAFT 训练。

### 真实测试和指标

- 使用官方 HDF5 与 flow GT；left camera；原生 346×260，不做非等比 resize。
- `indoor_flying1/2/3`：沿官方 evaluator 的边界条件 `i>20 && i+20<N`；每个 dt 另外验证所有组成区间和 GT 都存在。
- `outdoor_day1`：`i ∈ [9200,9600) ∪ [10500,10900)`，零基 image index；再与上述有效条件相交。裁剪后只用顶部 190 行，沿 STE evaluator 的既有约定。
- dt=1/4 指相隔 1/4 个 APS frame intervals，不硬编码成“恒定毫秒”；读取 `image_raw_ts` 合成对应 GT displacement。单位 **pixels per interval**，不是 px/s。
- events 使用 `(t_i,t_{i+dt}]`；输出时刻是窗口末端，估计该窗口的位移；不声称在起点时已经预测未来流。
- 主 AEE：官方有效非零 flow GT ∩ 原始真实 event support；同一 raw-real support/mask 给两组，不从网络预测或归一化 signed voxel 推 mask。按 frame 平均，再按 sequence 等权，最后 4 序列等权；dt=1 为唯一 confirmatory endpoint。
- 正式同时导出 **Outlier-3px&5%**：`EPE>3 && EPE>0.05*||GT||`。另存 **Legacy-STE-3px**：`EPE>=3`，供核对官方脚本；二者绝不能混名。
- 补充列出 all-valid-GT 的 AEE 与有效像素数量；称 all-valid，不能把稀疏 GT 说成 dense ground truth。
- 无有效 GT 的 frame 写为未定义并列入共同排除 manifest；不能将空 mask 的误差写为 0。
- 先只在 synthetic val 检查单位、resize 后向量缩放、GT 时间组合。真实 evaluator 的数据读取/数值核对在最终配置锁定后完成；若发现独立于方法的 bug，统一修复重跑。

证据：[训练入口](https://github.com/ruizhao26/STE-FlowNet/blob/eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6/main_steflow_dt1.py)、[dt=4](https://github.com/ruizhao26/STE-FlowNet/blob/eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6/main_steflow_dt4.py)、[encoding](https://github.com/ruizhao26/STE-FlowNet/blob/eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6/encoding/split_encoding_torch.py)、[loss/metric](https://github.com/ruizhao26/STE-FlowNet/blob/eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6/multiscaleloss.py)、[MVSEC](https://daniilidis-group.github.io/mvsec/)。

## 6. L1：E2VID → HQF

采用 E2VIDRecurrent 同一架构、E2VID+ 风格公开训练配置；论文命名写清“E2VID architecture retrained on each synthetic corpus”，不将我们的 checkpoint 冒充已发布 E2VID+ 权重。

| 项目 | 确定配置 |
|---|---|
| 网络 | 3 encoder、32 base channels、2 residual blocks、ConvLSTM、sum skip、upsample-conv、norm=none；5 input bins |
| 初始化 | 下游网络随机初始化；不加载 E2VID/E2VID+ pretrained downstream weights |
| 输入 | signed 5-bin voxel，按相邻 GT frame timestamps 划窗；不做 event count 截断；采用 E2VID+ 的不归一化输入 |
| 序列 | L=40；每次 clip 起点 state 清零；train non-overlap 40-window clips，144×5=720 clips/epoch |
| 裁剪 | 112×112 RandomCrop；clip 内共用同一 crop/flip；val 固定 center crop 160×160、完整 200-window rollout |
| optimizer | Adam lr=1e-4，betas=(0.9,0.999)，amsgrad=True，weight_decay=0 |
| budget | batch=2 clips，200 epochs，constant lr；360 updates/epoch，共72,000 updates |
| loss | LPIPS-VGG ×1 + temporal consistency ×2，L0=2；采用公开实现，不新增 loss |
| temporal supervision | 共享 Genesis GT flow/visibility 只进入训练 loss；核对公开 loss 要求的 flow 方向和位移尺度 |
| augmentation | RandomFlip；pause P(pause|run)=0.05，P(pause|pause)=0.9；暂停时图像、事件及 flow 一致冻结/置零 |
| 额外 tensor noise | 主比较关闭额外 Gaussian/hot-pixel augmentation，避免覆盖 generator 的噪声差异；传感器噪声由已冻结 generator 配置提供 |
| checkpoint | 每5 epochs，全 synthetic val 平均 LPIPS-Alex 最小；保存 last checkpoint |

loss 的 VGG 只是两组共用的冻结 perceptual-loss 网络；这项 ImageNet 先验要披露，不属于真实事件训练。公开训练 JSON 存在 network num_bins=10 而 loader num_bins=5 的不一致：本项目显式设为 **5 signed/combined channels**，train/val 完全一致；不能直接复制原 JSON 后声称是 5-bin。

**HQF 评测**：全 14 序列：bike_bay_hdr、boxes、desk_6k、desk_fast、desk_hand_only、desk_slow、engineering_posters、high_texture_plants、poster_pillar_1、poster_pillar_2、reflective_materials、slow_and_fast_desk、slow_hand、still_life。若原始下载文件的命名有差异，只做对应表，不自选子集。

- 使用 EVREAL `between_frames` 思路，输出时间与 GT 对齐；全序列连续递归，sequence 起点重置，sequence 内不重置。不把前后帧 RGB 作为 seed image。
- 原生真实分辨率评测；为网络整除要求只做 padding，输出移除 padding。无额外手工 ROI，无时间/几何误差最小化搜索。
- 主计分从第一个有完整事件区间的 GT frame 开始；不按模型状态再挑 warm-up。空窗照常推理。
- 参照 E2VID+ 的 `event_tensor_normalization=false`、`post_process_norm=none`；输出和 GT clip 到[0,1]，`histeq=none`，不做逐图 GT-fitted brightness/gamma/histogram matching。
- LPIPS-Alex 为主；SSIM、PSNR、MSE 为辅。灰度复制三通道后计算 LPIPS；GT 灰度转换固定 OpenCV 路径。SSIM：Gaussian window、sigma=1.5、population covariance、data_range=1；PSNR=10log10(1/MSE)。保存库和权重 hash。
- 两组使用同一个浮点 metric path，不能一组先存 8-bit 图再评分、另一组直接浮点评分。输出定量图像与可视化分开。
- frame → sequence → dataset 等权聚合。完整保留每帧分数、时间戳和每序列汇总。

证据：[公开训练配置](https://github.com/TimoStoff/event_cnn_minimal/blob/78aa471a66593f55e349303563997297fbc816e8/config/reconstruction.json)、[EVREAL E2VID+](https://github.com/ercanburak/EVREAL/blob/398f03551b7f5d150ae10aa89e4e731557fce240/config/method/E2VID%2B.json)、[EVREAL std](https://github.com/ercanburak/EVREAL/blob/398f03551b7f5d150ae10aa89e4e731557fce240/config/eval/std.json)、[metrics](https://github.com/ercanburak/EVREAL/blob/398f03551b7f5d150ae10aa89e4e731557fce240/utils/eval_metrics.py)。

### HS-ERGB 扩展预留

仅在 L1 主流程完成且无需新对齐方法时启用：使用原发布 close/test、far/test 全部可用序列，分别汇总；不采用其他去模糊论文为其任务选择的四序列划分。`events_aligned` 的 x,y 必须除以32，保留 remapped 浮点坐标，用固定空间双线性 splat；原发布 corrected images/时间戳为准。没有对应 event/timestamp 的额外 images 按官方说明忽略。输入只用目标时间之前的事件。

HS-ERGB 不存在 EVREAL 已发布的现成 std 数据配置，因此这里是**自定义 dataset adapter + 固定重建指标**，不能标作“官方标准 E2VID-HS-ERGB 协议”。不得用 TimeLens 插值出的帧当 real reconstruction GT。

## 7. L3：YOLO-6D → E-POSE

### 明确的取舍

采用 keypoint→PnP 这条成熟路线，13 类 multi-object 一个网络。输入整张事件图，网络预测类别和关键点；K、已知 CAD 为常规先验。无 GT ROI、无 depth/ICP refinement、无 GT 初始化 tracker。PoseCNN 的训练环境和 DGECN 不完整发布不作为并行方案。

E-POSE 原论文基线含真实迁移学习，本项目全部换为合成任务训练。为保持最严格的 synthetic-only 口径，下游从随机初始化开始；不加载其真实数据 finetuned checkpoints 或 LINEMOD init weights。若合成验证不能收敛，先判定 L3 配方未准备好，不拿真实测试选择预训练方案。

### 配置

- 输入原始 sensor 346×260；**10 ms 窗口，100 Hz，固定窗口右端为 label 时刻**，两组标签共享。原论文用 last-event timestamp；本项目改用固定右端是为了防止两个 generator 因末事件不同而得到不同 pose labels。这是协议改动，不伪称原协议逐字复现。
- 表示遵循 E-POSE 的三通道事件时间图定义，且只读事件。原脚本通道为 event timestamp accumulator、ON time surface、OFF time surface；它不等于三通道 RGB，也不是 GT segmented events。
- **正式前必须核对论文定义与脚本状态更新语义**：脚本时间累积/归一化跨窗口保留，不能擅自变成独立 10 ms histogram。选择一种原生语义后两组和真实测试共用并锁 hash；不做“哪个表示真实分高”的选择。若原发布三通道文件和 raw conversion 对不上，L3 暂不 formal。
- 13 类对象取 E-POSE 原发布资产和类别映射：wood block、sugar box、bleach cleanser、scissors、drill、hammer、wrench、potted meat can、cracker box、Rubik's cube、nine hole peg、timer、mustard bottle。使用其发布的 obj_files/xyz geometry，不默认把同名 YCB-V mesh 的坐标原点当成一致。
- pose bank：每类20条×2s，共260条；每类18 train/2 val，100 Hz 为46,800/5,200个窗口。每条包含目标及0–4个同资产 distractors，所有在视野中的对象均输出标签。完整 rollout 划分，train/val 背景及轨迹种子独立。该规模是本项目预算选择。
- 网络：官方 multi-object YOLO-6D，classes=13、coords=18、5 anchors，head channels=5×(18+1+13)=160；train/test固定416×416，原图和关键点同一坐标变换，PnP 前映射回原始 K 坐标。
- loss：原生 RegionLoss，object=5、noobject=0.1、class=1、coord=1；固定 pretrain_num_epochs=0。只适配类别、数据和张量 API，不设计新 head/loss。
- optimizer：沿官方 SGD 实现，nominal lr=0.001、momentum=0.9、decay=0.0005、effective batch=32；原实现实际 lr=nominal_lr/batch，实际 weight_decay=decay×batch，**必须保留这套 sum-loss 缩放关系**，不能直接把 lr=0.001 填入 mean-loss trainer。
- schedule：按原 multi cfg 的 processed batches，steps=[-1,100,20000,30000]，scales=[0.1,10,0.1,0.1]。预算 **80,200 optimizer updates**，显式执行上限；不要把读取但未执行的 max_batches 当作已生效，也不要同时再跑500 epochs。
- 有效 batch=32；训练随机 crop/scale 按官方几何增强并同步关键点；关闭 RGB HSV/曝光 jitter、RGB background compositing、multi-scale image-size schedule。背景随机化在 Genesis 场景完成，不借 GT mask 拼真实照片。
- checkpoint：每2,000 updates，在全部 synthetic val 上按最终 ADD(-S) success 选；若并列，用平均 keypoint reprojection error；所有检测阈值/PnP 参数仅在 synthetic val 确定后冻结。

### 测试与计分

E-POSE 全部公开且标签完整的306序列为目标测试池，按13对象、single/multi、lighting、speed、trajectory类型分别列出覆盖。数据只允许按文件完整性、有效 pose/K/资产对应关系共同排除；没有官方统一公开 split 的确证时，不编造“官方测试集”。全部真实数据只做测试，不做8:2随机训练划分。

- 主：ADD（非对称）/ADD-S（已声明对称）<0.1×物体直径 d 的 success rate。symmetry metadata 来自公开资产/协议，在真实评分前锁定；未知对称性不能按排名临时决定。
- 辅：ADD-S AUC，积分区间0–0.1m；reprojection error；translation error（cm）；rotation error（deg，带对称性说明）。
- 另列论文原表的 **ADD-S<0.1m** 作为 legacy 列；这不是 0.1d。
- 所有目标实例计入分母；漏检/类别错误/PnP失败是失败，AUC按0记。不能只在成功检出的帧算 pose accuracy。需要匹配时，用固定同类检测匹配规则，GT 仅在 evaluator 使用。
- sequence内所有实例先汇总；每对象等权，再13对象等权；同时报告single/multi分项。不要让长序列或易物体主导均值。
- 不使用原仓库导出的2D bbox/keypoint文件作为可信的最终 evaluator GT：源码多个顶点投影复用 point1 的 z 分母。使用经核验的 pose matrix + 对应CAD + K 重新投影，先做方法无关的坐标一致性检查。该审计不允许通过真实模型排名挑修正。

证据：[E-POSE paper](https://doi.org/10.1038/s41597-025-04536-5)、[标注源码](https://github.com/oussamaabdulhay/E_Pose/blob/b96ae783fcb6de84e1e9c602e91bce941cab0234/CreateSaveDataset.m)、[YOLO multi config](https://github.com/microsoft/singleshotpose/blob/2c282619101f9f31666dd18b42393375bea1600f/multi_obj_pose_estimation/cfg/yolo-pose-multi.cfg)、[multi trainer](https://github.com/microsoft/singleshotpose/blob/2c282619101f9f31666dd18b42393375bea1600f/multi_obj_pose_estimation/train_multi.py)。

## 8. 隔离、pilot 与统计

**最省事的 pilot 隔离：在全部配置锁定之前，不运行任何候选模型在真实数据上的评分。** S2 改为 synthetic train→synthetic val 的完整最小闭环。真实下载、schema、内参、时间单位、GT数值合法性检查可以先做；不查看候选模型在真实样本上的预测或排名，不据此调 simulator。

首次真实评测在模型/数据/指标/全部 postprocess config hashes 固定后运行所有预登记种子。若有方法无关 bug，保存原始记录，统一修复受影响 generator/seed 并重跑；不能只修输的一组。

- 最低3 seeds，每任务配对。总体主终点是 L2 dt=1 四序列等权 AEE；dt=4、L1、L3 为 supporting endpoints，不用它们替换未达到预期的主终点。
- 每 seed 先算固定测试池的汇总分数，再报告跨seed mean±sample std。不得把几万帧当成几万次独立训练重复。
- 改善量统一正向：误差指标Δ=V2E−EventGenesis，success指标Δ=EventGenesis−V2E。给出三个配对Δ及其均值。
- 可报告配对Δ的95% t区间：mean(Δ)±4.303×sd(Δ)/√3。该区间只反映训练seed变化，n=3很小，不包装成对所有场景/传感器泛化的高置信结论。不额外堆砌小样本p值。
- 每序列/对象完整表必须保留。无需另做复杂层级模型、多模型排行榜或全套显著性设计。
- 结论只覆盖所测模型、真实数据和生成配置；L3 只能称机器人相关状态估计，不能直接称闭环机器人成功率验证。

## 9. 工程会话的执行顺序与有限未决项

顺序：**先L2 dt=1 → 冻结L2双dt协议 → 完成L2六对训练 → L1 → L3**。L1可复用motion bank。主表计划：L2 12个runs、L1 6个runs、L3 6个runs，共24个；不是每个generator都做模型搜索。

只保留以下必须由实际文件/运行确认的项目，不再开放式研究：

| 必须确认 | 验收结果 | 未通过时 |
|---|---|---|
| generator export/曝光/输入域 | 两组同latent和GT hash；V2E插值checkpoint、光度adapter、EventGenesis完整config有hash；记录不匹配传感器项 | 在synthetic smoke内修订并锁定，不看real score |
| STE native encoder边界/计数精度、CUDA correlation | 合成loader空窗可读、无NaN/计数wrap、单位正确，固定全部补丁 | 限定1个工程工作日适配；超时记录工程失败后另行切换模型 |
| L1公开trainer的5-channel一致性和flow方向 | forward/backward+短合成训练可运行；loss不用真实GT | 限定1个工程工作日，避免重写整个训练框架 |
| E-POSE raw数据、资产坐标、三通道语义 | 下载文件/序列ID/hash齐全；CAD单位/原点/pose方向一致；合成验证收敛；原生representation确定 | 限定2个工程工作日；未通过则L3保持optional，不拖主实验 |
| 正式数据manifest | 真实/合成文件、精确index、排除理由、版本与license记录完整 | 无manifest不得称formal |

上述时间是研究投入上限的建议，不是硬件运行时间保证。已通过的决定不因真实排名不理想而重开。

代码和数据 license 各自单独记录；代码仓库没有LICENSE不等于可自由再分发。本次只完成公开资料/源码审计，未运行模型训练、未下载整个数据集、未证明GPU可运行。Figshare数据API在本机返回403，因此E-POSE原始数据可用性仍是明确的工程验收项。

## 10. 交付件

- `RESEARCH_PROTOCOL.md`：本文件，含依据、默认值、必要偏离与门槛。
- `protocol_candidate_v1.yaml`：机器可读研究配置。status不是formal；工程验收前禁止自动启动真实评分。
- `source-lock*.json`：审计到的公开repo commits；`sources/`是抽取的原始源码，非完整可运行checkout。
- `SOURCE_AUDIT.md`：关键事实、拒绝额外复杂度的原因、源码核对点。

正式执行时由工程会话在目标仓库建立 `benchmark/sim2real/protocol_v1.yaml`、data manifests、patches和`sim2real_status.md`，运行依赖检查后生成最终sha256。本研究文件的hash只用于标记交接版本，不能代替最终实验冻结。
