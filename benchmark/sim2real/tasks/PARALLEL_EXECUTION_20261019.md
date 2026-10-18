# 三条 Sim-to-Real 实验并行推进指令

你继续负责 Genesis Event Camera / EventGenesis 的 Sim-to-Real 工程执行。本阶段必须同时推进重建、光流和物体六自由度位姿，各线独立排期、独立门禁、独立交付。不要再组织成“光流全部结束后才开始另外两条线”。本次优先交付是启动重建正式配对训练，同时完成光流收尾与测试准备，推进位姿的数据和模型问题。

工作仓库：/home/科研/Eventbased_WAM/repositories/gec-sim2real
分支：exp/public-sim2real-v1
核查基准提交：b4f671b85ba1c2cf672c75afca71d108b4aa51a2
frozen simulator core：e79e656ac00f7847b98db9e12bf43b093f3afff1

优先阅读：
- 01_COMMON_RESEARCH_BASELINE.md 与 03_CHAT_SIM2REAL_HANDOFF.md 的入仓版本。
- benchmark/sim2real/configs/stage4_corrected_v2_flowdt1_seed0.json
- benchmark/sim2real/reviews/stage4_motion_b4f671b_20261019/REVIEW.md
- benchmark/sim2real/tasks/reconstruction_e2vid_hqf.md
- benchmark/sim2real/tasks/pose_yolo6d_epose.md

保留研究会话已新增的审查目录，不清理或覆盖它们；开工先刷新实际运行状态。下列进度已核查，不要把旧文档的“仍在训练”当成当前状态。

## 已核实的起点

L1 重建：两组只完成旧版 E2VIDRecurrent 的 2 epoch / 224 updates 试训；旧权重完成 HQF bike_bay_hdr 的 2431 时刻推理，没有真实评分。正式 Stage 4 重建训练尚未启动，原定 HS-ERGB 尚无完整验证链路。

L2 光流：修正版本 STE-FlowNet dt1 seed 0 的 EventGenesis/V2E 均完成 45 epoch、8910 updates、退出码 0。synthetic-best 为 EventGenesis epoch 33、EPE 0.2832538118；V2E epoch 20、EPE 0.2828732346。真实 MVSEC 最终评分尚未进行。seed 1/2 尚未有本轮正式交付。

L3 位姿：仅有通用仿真方块上的 8 train / 4 validation 窗口、16 epoch / 128 updates 试训；不是 E-POSE 实验。事件时间窗已修复，但模型 eval 模式的 BN 状态与置信度选点问题仍未解决；尚未取得可核验的 E-POSE 原始样本和匹配 CAD，没有正式训练或真实测试。

## L1 重建 先把正式训练启动

1. 复用已修正的 32-rollout 共享物理 bank、配对原始事件和既定 split，建立重建专用的不可变 adapter 与 manifest。不要用旧六条微型轨迹或旧 smoke 权重代替正式语料和初始化。

2. 研究会话抽查确认，该 bank 有 100 Hz 图像端点、10 ms 事件信息，但显式保存的 flow 是 30 ms。请从相同 depth/pose/K/可见性状态补算 temporal consistency 所需的精确 10 ms 位移和有效区。不能直接套用 30 ms 标签或简单除以三。新增为有版本的派生产物，保留已完成光流所用文件及哈希。

3. 沿用当前 JSON 的 reconstruction 配置：固定 E2VIDRecurrent、5 signed bins、40 帧训练序列、112×112 crop、effective batch 2、Adam 1e-4/AMSGrad、200 epoch，LPIPS-VGG + 2× temporal consistency；synthetic-validation LPIPS-VGG 选 checkpoint，TC 单独记录。先启动 seed 0 两组；microbatch、累积、加载和资源安排自行处理。若实际样本数与预期预算不同，开训前写清原因和计算，不静默改变预算。

4. 做一次直接服务于开训的短预检：时间窗和监督目标、TC 位移方向及坐标约定、共同 crop/flip、循环状态的延续与重置、损失分项、梯度、checkpoint 恢复和下一批次。特别确认 TC 实际参与优化。通过后自动接上完整训练，不停在单步 smoke 或等待常规批准。

5. 两组各完成至少 20 个真实 optimizer updates，并保存/恢复核验 checkpoint 后，立即交付首个开训证据，后台继续原定预算。seed 0 稳定后按资源推进已登记的其他种子。

6. 并行推进真实测试准备：当前实际接通的是 HQF，可先整理完整序列清单和 EVREAL 接口；同时写清原定 HS-ERGB 的目标时刻与重建对齐可行性。不得将 HQF 已跑通写成 HS-ERGB 已完成，也不要自行删除 HS-ERGB。最终真实测试集角色及评分协议作为明确研究决策提交，不阻塞现在的合成训练。训练选点 LPIPS-VGG 与最终报告 LPIPS 的网络版本分别登记；真实评分前冻结 SSIM/MSE/LPIPS、光度处理和跨序列汇总方式。

## L2 光流 完成交付并准备真实评测

1. 立即更新 TRAINING_LIVE_STATUS.md 和总状态表为训练已完成。整理包含 pilot 第 1 轮在内的完整 45 轮曲线、两组退出码、best/final 权重、配置/数据哈希与复现命令。沿用原定 synthetic-best，不按真实分数换权重。

2. 将既有固定 64 窗口诊断应用于最终选定的两组 synthetic-best，报告共同事件活跃区域与既有分层结果；不要把 epoch 23 的诊断冒充最终模型结果。不同最佳 epoch 是既定选点规则的结果，诊断中明确记录。

3. 修正“必须学会全图稠密光流才能测试”的表述。STE-FlowNet 正式评测关注有效 GT 与事件活跃像素的交集；无事件区预测接近零不能单独判定失败。全有效区指标保留为诊断，不临时替换本轮选点规则，也不因 V2E 数值更低而调参或重训。

4. 对照已锁定的官方实现准备可直接执行的 MVSEC dt1 测试配置：序列及样本范围、APS 端点、事件窗、GT 时间传播、无效 GT、裁剪与坐标、共同真实事件掩码、空有效窗口、AEE 和 Outlier、逐序列汇总。Outlier 使用 EPE 同时大于 3 px 和 GT 幅值 5% 的定义，不把旧 helper 的单独 3 px 比例直接当正式指标。dt4 独立列为补充，不混入 dt1。

5. 用解析/合成样例完成评测器契约检查，将最终测试 manifest、评价配置与固定权重索引提交研究核验。沿用当前真实评分关闭状态，待本线评测协议核验；该门禁不阻塞重建训练、位姿排障或光流其他种子的合成训练。种子 1/2 可按已冻结协议和资源排队，是否补种子不由真实分数决定。

## L3 位姿 同时解决数据与模型两个问题

1. 重新核查 E-POSE 的公开下载入口、官方文件列表及可验证的公开镜像，优先取得最小原始事件/位姿样本及精确匹配 CAD，不必先下载全部数据。区分以前的访问失败与本次实际访问结果。未经用户指示不要联系作者或向外发消息。

2. 记录原始时间单位、相机 K、位姿变换方向、坐标轴和尺度、同步、物体 ID/CAD 对应及对称性。真实样本用于读取和标注契约审计，不用于模型/生成器调参或按误差选择方案。通用仿真方块仍只能用于单元诊断，不能冒充 E-POSE 类别。

3. 与数据获取并行，排查 SingleShotPose 的普通 model.eval() 推理稳定性和置信度学习。使用合理的合成训练批次、训练预算及仅由合成数据建立的统计，检查持久化/恢复后能否复现正常的关键点和 PnP。保留成熟实现，不做新颖方法设计。不能用 train-mode 推理、GT-cell 选点、GT crop/mask/box 或分割事件绕过问题。

4. 如果现有成熟模型能用常规工程处理修复，记录策略并对两组一致实施；如果必须更换主模型，提交一个推荐替代及最小合成验证依据，交研究侧确认方向，同时继续数据/CAD工作。不要在多个新模型上展开无边界探索。

5. 数据/CAD与正常推理门禁通过后，固定该任务的模型、输入、匹配物体资产、训练预算与合成 split，生成同一物理轨迹/标签的两组事件，先启动 seed 0 配对训练。位姿任务可使用自身匹配 CAD 的场景，但同任务两生成器必须共享场景与标签；不直接把无匹配类别的光流 bank 当 E-POSE 训练集。先提交实际开训和恢复证据，再扩展预算与种子。若仍受阻，给出准确失败证据、已经完成的独立工作和下一项最小需求。

## 并行安排与反馈

三线分别维护待办、运行目录、日志和 checkpoint。资源上优先让重建 seed 0 两组启动并持续运行；光流收尾/评测准备、位姿下载/模型诊断同时推进。启动前检查当前 GPU 占用，不把历史空闲状态当预留，不停止他人作业。某线等待数据、资源或研究决策时，其余线照常推进。

工程实现、下载转换、兼容性、批量与累积、作业托管和例行修复自行解决。涉及任务/模型/最终数据集变更、修改 frozen core、真实数据反馈，或确认当前训练无效的重大问题，再向研究会话汇报。发现确定的标签/配对/非有限数值或恢复错误时，保留证据并暂停受影响任务；不要因指标弱或对手更好停训。

所有训练与选点仍为 synthetic-only，网络推理保持 event-only；禁止 privileged information leakage。候选 Vid2E/FILM/RIFE 暂不扩展到本轮训练，以先闭合 EventGenesis/V2E 三任务配对线。

交付一个三线并排状态表：最近核查时间、已完成的真实动作、实际进程/update、当前阻塞、下一交付。代码、报告、小型证据和索引保存在共享仓库；大数据和权重留在 g1 并登记路径与 SHA256。注明本地/GitHub/g1 同步状态，避免将所有路线压成一个状态页上的单条光流进度。

第一次反馈优先提交重建真实开训证据；同时交付光流最终归档/评测准备进展和位姿数据/模型两条子任务的实际结果。不要等三条线全部结束才汇报，也不要只提交下一步计划。
