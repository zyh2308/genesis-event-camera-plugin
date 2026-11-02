# 新版仿真器三项 Sim-to-Real 实验部署

部署日：2026-11-03（JST）。目标交付日：2026-11-09。此目录记录部署事实、代码和检查证据，**不是已完成训练或生成器性能优劣报告**。

## 实验与边界

统一采用 frozen release `62da4f35ce3c5c96f04268bd1687a27657af9238`，明确启用 `coverage_time_log_linear`。旧 `e79e656…` 结果保留为历史记录，本轮不复用其训练权重。核心 17 个文件按冻结描述逐个校验，未修改核心。

| 任务 | 下游模型 | 每个方法、每个 seed 的预算 | 真实测试范围 |
|---|---|---|---|
| 重建 | E2VID+ / E2VIDRecurrent | 200 epoch，12,000 次更新；40 帧/clip，有效 batch 2 clips | HQF 全 14 序列，沿用冻结评价规则；属于此前已见数据的复测。HS-ERGB 是扩展项 |
| 光流 | STE-FlowNet dt1 | 45 epoch，8,910 次更新；有效 batch 8 | MVSEC indoor_flying1/2/3；outdoor_day1 须在正式评分前完成获取与审计。须明确报告实际范围 |
| 位姿 | SingleShotPose / YOLO-6D，共享中心分类适配 | 30,000 次更新；batch 8 | E-POSE 木块子集完整注册序列；不是全物体 E-POSE。已见 r_01_gl/r_05_gl 与其余序列须分开标注 |

五种管线为 EventGenesis、官方 SuperSloMo 路线 V2E、Vid2E（SSM + ESIM）、FILM + ESIM、RIFE + ESIM。每项任务使用训练 seed 0/1/2，总计 **45 个正式训练作业**。

同一任务共用物理数据、标签、划分、模型结构、有效 batch、优化器、更新预算和增强规则。逐 seed 独立生成初始化参考 hash，实际模型在第一步 optimizer 更新前必须匹配。所有下游模型从随机初始化开始；不使用旧短训的生成器专属 warm start。不匹配事件数量，不因某个生成器事件少而删窗。

本轮训练和合成选择阶段的真实评分开关关闭。最终真实评分、统计表与定量图表是后续阶段；45 次训练完成标记本身不代表真实评分完成。真实分数不得反馈到生成器、训练、调参或选权重。

## 共享数据

重建与光流共用 8 个真实 ReplicaCAD 家具布局的 32 条 2 秒物理轨迹（24 train / 8 validation），共 64 秒。训练与验证按完整布局隔离；源文件保存原生 100 Hz float HDR、深度、物理 link 分割、K、相机及物体 SE(3)。这是一套明确规模的受控训练 bank，不声称覆盖整个 ReplicaCAD。

四类运动为相机运动、静止相机下物体运动、相机与物体共同运动、含遮挡的混合运动。生成前的合成几何审计中，三类含相机运动场景的平均有效像素位移大于 0.5 px/30 ms 的比例约为 42.1%、76.5%、79.8%；物体独立运动组约为 1.5%。这是 GT 运动量口径，不能与旧报告的“事件活跃区比例”直接计算倍数。

光流采用直接 30 ms 位移。重建 TC 标签从相邻物理状态直接计算 10 ms 位移，并逐窗复核原 30 ms 标签；不将 30 ms flow 除以三。

位姿保存匹配 YCB `036_wood_block` 纹理 CAD 的 128 条 200 ms 轨迹（96 train / 32 validation），每条 21 个原生端点。跳过前 40 ms 后取 16 个 10 ms 窗口，共 1,536 train / 512 validation 窗口。物理相机方向在 rollout 内固定，平移形成实际像素运动；不跟踪物体而抵消运动。不向模型提供 GT crop、mask、box 或选点 cell。

两套物理 bank、划分及其 manifest 在服务器已设为只读。不同生成器的事件写入独立目录，不向公共物理 HDF5 追加事件。

## 成熟基线与适配说明

- Vid2E 使用官方 **2021 SSM 版本** `3954ede04bb00286269c40201140964251f4f61e`，避免把后来已切换 FILM 的主分支再次标成 SSM。保持其按预测运动量自适应插帧的路线。
- FILM 使用 Google 官方代码 `69f8708f08e62c2edf46a27616a4bfcf083e2076` 与官方 Style SavedModel；RIFE 使用作者代码 `5d8adbdd40e12c2c8f91930eff838aebe561c086` 与论文模型权重。两者采用成熟的递归中点插帧，固定 8 倍。
- 三种插帧基线共用固定版本 ESIM，C+/C−=0.2、refractory=0.5 ms。ESIM 原生无噪声；Ours/V2E 保留预注册的非理想传感器参数。比较的是完整原生生成管线，不能声称已隔离为“完全相同非理想传感器响应下的插值器消融”。
- 插帧网络只做步长所需 padding，生成图像恢复到共同原生视野；没有 GT 驱动的插帧、空间裁剪或场景筛选。物理端点与图像映射完全共用。
- ESIM 仅在私有副本中做 Torch 2.8 API 名称兼容，事件内核未改算法。正/负亮度斜坡与静止输入的独立数值检查通过。
- FILM 的 TensorFlow 与 ESIM 的 Torch 分进程运行，使用各自私有 CUDA 依赖；固定使用支持的私有 ptxas 12.6，未更改共享环境。

所有官方源码、权重下载来源、版本、大小及 SHA256 均在本目录 manifest 中。大权重、完整数据和日志保留在服务器，不纳入 Git。

## 训练前检查与本轮修正

已通过：三种模型在新版事件上的实际前向、反传和有限梯度；公共入口与冻结核心文件检查；首条丰富场景完整的光流及重建适配；640 个合成位姿帧的 CAD 投影/光栅包含关系与 GT PnP 回算；三个插帧基线的原生窗口测试；9 组任务/seed 的独立初始化参考。

发现并在正式训练前统一纠正了 E2VID 旧 temporal warp 的坐标约定：W−1/H−1 归一化的两个采样显式使用 `align_corners=True`，避免零流对齐相同图像仍产生误差。只在本轮公共兼容层生效，原上游文件和仿真器核心不改。保留上游负向 forward-flow 近似与图像差异可见性权重；不声称使用了精确 target-grid backward GT 或显式 flow_valid 屏蔽。

位姿共享中心 CE 适配和合成验证重投影选权重规则在正式训练前固定；不声称严格复现作者原始训练损失。普通 eval 下按预测中心置信度选点，不使用 GT cell。

**已完成的每模型 2 次更新属于小型运行诊断，不计作正式训练进度。** 实际每个正式 run 仍须先完成完整数据上的 20 步短检、独立进程恢复到第 21 步以及共用初始化/标签/样本序列检查，才会继续完整预算。CUDA 非确定性反传不要求整段训练轨迹逐 bit 相同。异常失败保留，受影响作业停止；不因某个生成器当前指标较差而删组或延长其训练。

## 后台执行与留卡

服务器工作根目录：

`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1`

正式任务共分配为 GPU3 13 个、GPU4 12 个、GPU5 20 个。光流仅使用两张相同型号 V100S 32GB；重建 seed1 的五种管线统一放在 TITAN RTX，其余使用 V100S；位姿统一使用 TITAN RTX。逐 run 记录实际 UUID。固定预算、batch 与精度保持一致，生成效率须另在共同硬件上计时，不能直接比较这些异卡运行日志。

队列采用独立进程、原子任务文件、UUID 级排他 flock、持久日志和退出码。三条线可并行，无需交互式 SSH 会话存活。已实际检查三个 GPU manager 的 PID 与排他锁，不依靠陈旧状态文件判断占用。

按照用户指示，**GPU3/4 正常完成队列后继续保留预约和低负载 CUDA 占位**。临时 GPU5 在全部 45 个训练作业完成后由结束标记释放。源程序见 `server_queue.py`、`finalize_training_campaign.py`。不运行空转计算，不停止其他用户或其他会话的任务，不访问已禁用的 g3。

服务器 `CAMPAIGN_LIVE_STATUS.json` 是机器状态；本目录 `DEPLOYMENT_VERIFIED_SNAPSHOT.json` 是带时间戳的不可变部署快照，不能当成未来实时进度。当前快照处于事件数据生成阶段，正式训练更新为 0。

## 审查入口

- `training_recipe.json`、`stage9_sensor_profile.json`：公共训练与新版传感器配置。
- `training_source_manifest.json`：服务器实际源文件指纹。
- `QUEUE_REBALANCE_COMPLETE.json`：完整 45 次训练的排程和留卡规则。
- `DEPLOYMENT_VERIFIED_SNAPSHOT.json`：实际进程、flock、数据阶段及原生检查结果。
- `physical_motion_audit.json`、`pose_physical_geometry_audit.json`、`early_data_adapter_audit.json`：合成数据与标签检查。
- `initialization_reference.json`：独立计算的全部任务/seed 初始化 hash。

后续论文表须报告真实测试每序列/每 seed 原始值、三 seed 均值与样本标准差、成对差异及按序列聚类的置信区间；失败与缺失必须保留。位姿失效 PnP 计入全窗口失败分母，不能仅报有效样本均值而隐去失败比例。不得将当前部署检查或合成验证值写作真实 Sim-to-Real 优势。

附记：额外基线的事件数据优先于长程位姿训练准备，避免数据等待阻塞其他两条线；见 DATA_GENERATION_PRIORITY_AMENDMENT.json。历史训练与本轮使用的物理 bank、公共兼容层不同，不能将新旧结果之差直接解释成 core 修复的独立消融。
