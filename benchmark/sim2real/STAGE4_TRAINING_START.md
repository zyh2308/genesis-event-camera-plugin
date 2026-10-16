# Stage 4：立即启动重建与光流的实际训练

工程执行会话：**搭建Sim-to-Real实验闭环**（VS Code → WSL Codex）。这是研究会话在 Stage 3 验收后的执行指令。请直接实施、开训和交付运行证据，不停在计划或重复 smoke。

## 1. 当前授权与优先级

用户要求尽快开始训练。重建和光流的管线已经验收通过，执行本文件不需要再向研究会话确认常规工程选择。先读取 `reviews/stage3_78f7161/REVIEW.md` 与本轮 `configs/stage4_training_launch_v1.json`，保留已有审查材料和运行记录。

优先启动 **STE-FlowNet dt1、seed 0、EventGenesis/V2E 两组配对训练**；随后利用可用资源并行启动 E2VID 两组 seed 0。其他种子按同一配置排队。dt4 和位姿不阻塞这两项开训。

本轮为预先固定的 synthetic-only 训练 v1。它是实际训练，不是重复单步接口测试；它仍不是已证明充分的数据规模，也不自动产生正式 Sim-to-Real 结论。真实评分继续关闭，完整 test manifest/最终评价版本由研究侧后续冻结。这不阻碍现在开始合成训练。

## 2. 数据准备要直接服务于开训

不要在 Stage 2 的六条 0.3 s 落方块轨迹上反复延长 epochs 充当规模化训练。按 JSON 生成第一版紧凑而有运动/场景变化的共享 bank：32 条 × 2 s，4 类运动，各 8 条；每类 6 train / 2 synthetic-validation。分组、种子、轨迹 ID 先确定，不按 generator 输出挑样本。

场景采用现有 Genesis 官方示例、简单程序化几何/纹理即可，工程实现自行选择。至少覆盖相机运动、独立刚体运动、二者组合，以及有遮挡的组合运动。持续运动、有效纹理和深度变化应贯穿片段，避免落地后大部分时间静止。validation 使用独立 scene/trajectory group，不只对同一 scene 做邻接窗口切分。

保存同一份 radiance/depth/seg/pose/K/时间和物理标签，再分别导出 EventGenesis/V2E。使用已经验证的 nominal generator 设置及 V2E 25 μs 配置，禁止退回曾失败的 1 ms 配置，禁止据真实数据分数调 generator。输入映射保持已公开的 linear-radiance 与固定 display 映射方案，结论范围为完整生成管线比较，不宣称已校准到真实传感器。

流的基本训练窗口采用相邻三个 100 Hz 物理端点间隔，即 30 ms，接近 MVSEC 实际 APS 间隔；dt4 对应四个这样的窗口，即 120 ms。不是把旧 10 ms GT 只改单位。事件、灰度端点和位移标签必须从同一时间对直接构造，标签来自物理投影；不要用事件数或单纯求和位移替代。

重建使用同一 bank 的 10 ms 基本事件窗。任务窗口可以不同，同一任务的两个 generator 必须完全相同。

生成前估算一小段吞吐和空间即可，不再运行独立整套 readiness。生成完成后自动完成两组标签、时间窗、split、输入契约核对，冻结实际文件 manifest/hash，**同一作业链立即进入训练**，不要结束会话等待批准。bank 不完整或配对错误时应阻止开训并明确报错。

如果语料生成耗时，使用现有渲染/缓存/并行能力加速，常规优化自主处理；记录实际进度和预计完成时间。不将旧微型 bank 偷换成训练 v1，也不擅自降低已登记的分辨率/数量/窗口。遇到确实影响开训的资源或接口问题，给出具体证据和最小替代方案，勿停在泛泛的“需要进一步确认”。

## 3. 训练执行

JSON 中的模型、损失和优化器沿用已锁定官方源码与已验收适配。它是研究配置契约，由你接入现有训练入口，不假定旧 smoke 脚本能直接解析该文件。实现 batch、采样、augmentation、resume 和运行器是本阶段正常工程工作。

- **光流 dt1**：原 STE 多尺度 photometric + 10× smoothness；Adam 4e-4，effective batch 8，45 epochs，官方 milestone/gamma；全部合格训练窗口参与，不再每条轨迹只取头尾两个窗口。选择 synthetic-validation EPE 最优 checkpoint。
- **重建**：固定 E2VIDRecurrent 五个 signed bins；长度 40、112×112 训练 crop、effective batch 2；Adam 1e-4/AMSGrad，200 epochs，LPIPS-VGG + 2× temporal consistency。选择 synthetic-validation LPIPS-VGG 最优 checkpoint，TC 另报。本轮先关闭额外 tensor noise/hot pixels/pause augmentation，保留同步 crop/flip。
- 种子为 0、1、2，每个种子内两组初始化、采样次序、augmentation 和优化预算配对。先把 seed 0 两组实际跑起来，再按资源排队其他种子，不要求六个/十二个作业同时占卡。
- 允许 microbatch + gradient accumulation 达到登记 effective batch；日志区分 microstep 与 optimizer update。不能只改 batch 却不记录损失归一化和有效预算。
- random crop/flip 必须让事件、图像及所用 flow/mask 采用同一变换，flow 分量翻转符号正确。现有无 augmentation 的 E2VID adapter 不能直接套上各自随机 crop；用一个独立几何例子验证后直接进入训练，不另起大规模测试项目。
- 可保留已通过数值核验的相关算子；性能优化不得默默改变数学定义。若 OOM，先调 microbatch/accumulation，记录吞吐；不通过缩小两组不同输入或减少某组样本解决。

不要为了尽快开训恢复 Stage 2/3 的 smoke 权重当正式随机初始化。首次从本轮登记 seed 初始化；中断恢复则使用本轮 last checkpoint，并恢复 optimizer/scheduler/RNG/采样状态。

## 4. 阶段进度要让研究会话直接核实

先创建并持续更新 `TRAINING_LIVE_STATUS.md`。内容保持简短且可核验：

1. bank 生成/两组导出进度、数据 manifest/hash；
2. 训练状态：任务、generator、seed、GPU、PID、开始时间、run 目录、代码提交；
3. 实际 optimizer update、loss 分项、有限梯度、有效 batch、吞吐、显存、最近 checkpoint 和文件更新时间；
4. 失败及恢复情况。

**第一次反馈不要等训练结束。** 两组光流 seed 0 均完成至少 20 个真实 optimizer updates，并有可恢复 checkpoint 后，立即交付首份运行状态。明确哪几项已经运行、哪几项在排队；数据生成开始和进程创建不等于训练已经开始。

如果暂时只起了一组，先如实记录其状态，同时继续启动另一组。之后至少在前 100、500 updates 和每次 synthetic validation/checkpoint 时保留证据。不得用打印的预期 step 数或旧 checkpoint 冒充新训练进度。

## 5. 资源及停止条件

研究会话核查时 g1 GPU0/1 基本空闲，其他空闲卡可按实际条件安排；这不是资源预留。启动前重新检查所有者/显存，GPU3 有其他用户作业，禁止停止或修改。同步导出/训练合理分配资源，勿无控制占满服务器。

遇到 NaN/Inf、错误标签/时间、两组样本不一致或不可恢复 checkpoint 时停止受影响作业，修复后统一处理受影响两组，保留旧记录。正常验证波动或 V2E 暂时更好不是停止理由。

本轮不进行新真实推理或评分，不使用已暴露 MVSEC pilot 误差选择超参数、训练时长、场景或权重；不使用 GT crop/mask/pose 作为网络推理输入。E-POSE/BN 模型问题独立报告，不阻塞开训。

## 6. 版本与交付

开训前将本轮研究配置、实际 corpus/split manifest、模型及导出配置、工程实现提交，记录完整 hash；更新已有证据索引并同步 WSL/g1。GitHub 若有网络问题可先沿用已验证的 bundle 同步办法，不让上传文档阻塞合成训练；如实报告未推送状态。

最终提供 run 入口和复现命令、`TRAINING_LIVE_STATUS.md`、日志/checkpoint 路径、代码提交及数据 hash。研究会话会直接从仓库和 g1 核查，不需要用户搬运文件。

**请直接执行到真实训练开始，并先汇报首批 optimizer 更新证据。**
