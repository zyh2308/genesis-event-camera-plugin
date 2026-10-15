# 官方资料与源码审计摘要

审计日期2026-10-16。研究只对公开资料与抽取源码作静态检查，不等于运行复现。可机读版本记录见 source-lock.json、source-lock-extra.json、source-lock-pose.json。

## 选型依据

| 对象 | 核实的事实 | 决策 |
|---|---|---|
| STE-FlowNet | 官方训练、MVSEC编码/评测可见，MIT；自监督photometric+smoothness，原生Gaussian权重分极性输入 | 保留L2，不将flow GT监督当作原生做法 |
| ADMFlow | 直接研究rendered→MVSEC，但多密度损失/数据额外复杂，公开config与论文预算不同 | 只作论证参考，不增加一套模型 |
| E2VID | 原repo主要提供推理；不能仅靠下载原repo就宣称可以训练 | 使用E2VID+公开训练配置，显式修正5-channel一致性 |
| E2VID+ / event_cnn_minimal | 含训练入口、序列40、crop112、Adam1e-4、200epochs、perceptual/temporal loss | 沿成熟路线训练相同E2VID架构 |
| EVREAL | 支持HQF、between_frames、MSE/SSIM/LPIPS；E2VID与E2VID+前后处理不同 | 锁E2VID+路径，不混用原E2VID robust normalization |
| HQF | 原论文全14序列评测，真实事件+同步图像，重建专用 | 替代HS-ERGB为L1主测试 |
| HS-ERGB | 原TimeLens数据，coordinates/32，已有aligned events和corrected images；非EVREAL原生配置 | 仅扩展，无必要新设计对齐 |
| E-POSE | 13对象、306序列、10ms原生采样；基线有真实迁移学习；公开repo为MATLAB标注而非完整模型训练包 | 用synthetic-only重训YOLO-6D家族，保留工程门槛 |
| SingleShotPose multi | 官方13类、多物体keypoint/PnP结构与训练代码，MIT；旧PyTorch接口 | 一个13类网络，不按物体训练13个模型 |
| DOPE | 完整train/inference/eval、支持synthetic；官方training建议每物体分别训练 | 不列默认，避免模型数量膨胀 |
| V2E | 公开nominal传感器默认、插值参数 | 固定完整pipeline与checkpoint，不用真实测试拟合 |

## 重要代码核对点

- STE `main_steflow_dt1.py` / `main_steflow_dt4.py`：训练默认real `--test-set` 必须拆出；原脚本每若干轮在真实序列验证会破坏隔离。dt4 README lr=2e-4，而parser默认4e-4，应显式写命令参数。
- STE `encoding/split_encoding_torch.py`：equal-count within interval、Gaussian weighted count、uint8中间存储、outdoor13→5分支；不能称为普通signed5-bin voxel，不能不检查计数溢出。
- STE `multiscaleloss.py`：`percent_AEE`统计`EE<3`；`1-percent_AEE`是3-pixel error rate，没有5%相对阈值。若报告3px&5%须显式新增通用metric列。
- STE `validate()`：outdoor_day1取9200:9600与10500:10900；loader附近另有不一致OR表达式。计分mask来自real events和有效GT，不来自预测。
- E2VID+ `config/reconstruction.json`：arch num_bins10 / loader5不一致，不是可直接复制的无歧义preset。固定signed5-channel，train/val统一。
- EVREAL `config/method/E2VID.json`是normalization=true/robust；`E2VID+.json`为false/none。`std.json`是between_frames/histeq=none；不能把三者混成一个“官方E2VID默认”。
- E-POSE `CreateSaveDataset.m`：相机pose插值与固定时间packet；原脚本的实际窗口端点可能是第一个超过nominal time的事件，不能让两组合成labels随末事件改变。正式用共同固定右端、相同label文件。
- E-POSE三通道图的状态在loop前初始化；存在跨窗累积和归一化语义，须对照论文和发布数据核实；不自行替换成普通RGB式histogram。
- E-POSE源码投影point2..point9时多处复用point1的z；由pose+CAD+K重投影校验，避免将导出的2D关键点直接当最终GT。此处是静态源码风险，不代表已证明所有发布标签损坏。
- E-POSE `segmented_image`/mask由GT生成，只能作训练监督/评估，不能输入模型。
- YOLO multi `train_multi.py`：默认加载LINEMOD init.weights、固定真实验证文件列表、timestamp种子、RGB背景合成；必须改成配对随机初始化/合成验证/固定种子/Genesis背景。
- YOLO multi SGD的lr和decay按batch缩放，loss是sum型；把yaml nominal值直接填进新的mean-loss优化器并不等价。
- YOLO multi cfg有max_batches80200，但循环按max_epochs500运行；研究预算要求显式停止80,200 updates，不仅设置字段。

## 版本与数据可用性边界

本机Git HTTPS clone返回connection reset，改用GitHub API与raw公开文件按commit获取。因此sources不是完整安装；不存在已经通过的环境测试。

E-POSE Figshare API读取返回403，未下载其全数据，不能提供真实data hash或保证现时下载入口可达。TimeLens的数据入口包含申请/邮件下载说明；本次没有提交任何表单或对外发送消息。

repo的license与dataset的license分开登记；STE/EVREAL/SingleShotPose/V2E的许可证可以在官方仓库检查；event_cnn_minimal、ADMFlow、E_Pose在审计tree未见明确顶层LICENSE，不作可自由再分发推断。MVSEC主页标示CC BY-SA 4.0。不将这些说明扩展成未进行的法律审计。

## 主要论文与官方实现

- [Reducing the Sim-to-Real Gap for Event Cameras, ECCV2020](https://arxiv.org/abs/2003.09078)：与本项目“固定网络、改变synthetic training data、真实域测试”最接近的成熟研究范式。
- [STE-FlowNet, AAAI2022](https://arxiv.org/abs/2109.04871)：所选L2网络及原生输入/训练。
- [ADMFlow/MDR, ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Luo_Learning_Optical_Flow_from_Event_Camera_with_Rendered_Dataset_ICCV_2023_paper.html)：rendered synthetic events→real MVSEC先例；不照搬其real validation做模型选择。
- [EVREAL, CVPRW2023 implementation](https://github.com/ercanburak/EVREAL)：L1评测实现。
- [TimeLens, CVPR2021 repository](https://github.com/uzh-rpg/rpg_timelens)：HS-ERGB原始数据格式。
- [E-POSE, Scientific Data2025](https://doi.org/10.1038/s41597-025-04536-5)：位姿数据与基线。
- [SingleShotPose, CVPR2018](https://github.com/microsoft/singleshotpose)：成熟keypoint/PnP位姿模型。
- [DOPE, CoRL2018](https://arxiv.org/abs/1809.10790)：synthetic-only位姿先例，非当前默认模型。

所有推荐中自定的语料规模、预算限制、统计主终点及隔离修订均为EventGenesis研究决策；不标为文献原设定。
