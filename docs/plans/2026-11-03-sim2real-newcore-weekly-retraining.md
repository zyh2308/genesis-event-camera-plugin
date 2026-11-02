# 新版 Sim-to-Real 三线重训与定量结果 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> 本环境该技能对应已提供的 executing-plans；使用其分批执行、留证据原则。用户已授权执行，不必再询问执行方式或日常工程选择。

**Goal:** 使用新版冻结 EventGenesis 完成重建、光流、位姿的配对正式训练和公共真实测试，在2026年11月9日交付全部可核验数据表与定量图表。

**Architecture:** 固定核心发布提交和显式重建配置，复用现有骨干、真实数据、只读缓存与已验证适配器。三项任务各有独立数据/训练队列，训练仅用合成数据；研究协议和合成验证先冻结，真实测试在权重锁定后执行。代码在独立工程 worktree，结果、任务状态和外部工件索引回到共享仓库。

**Tech Stack:** Genesis、固定 NumPy/Torch EventGenesis core、官方 V2E/ESIM与插帧器、PyTorch E2VID+/STE-FlowNet/SingleShotPose、HDF5、既有HQF/MVSEC/E-POSE评价器、CSV/XLSX/LaTeX与Matplotlib矢量图。

---

用户最新范围：**只要数据图表**。本轮不制作Demo视频、定性拼图、架构图或PPT样式。截止时间是下周一11月9日，不是此前周五/周日候选。

科学规格的机器入口是 `benchmark/sim2real/protocol/stage9_newcore_campaign_v1.json`。其中训练更新预算是吞吐量预估候选；必须在第一步正式 optimizer 更新之前落最终协议和数据内容哈希。可以根据纯合成诊断与实测吞吐量统一定预算，不许看真实分数后改训练。

## Task 1: 固定源码、资源和完成矩阵

**Files:**
- Read: `benchmark/sim2real/reviews/frozen_simulator_20261103/README.md`
- Read: `benchmark/sim2real/protocol/stage9_newcore_campaign_v1.json`
- Create: `benchmark/sim2real/reviews/newcore_week_20261103/EXECUTION_STATUS.json`
- Create: `benchmark/sim2real/reviews/newcore_week_20261103/BASELINE_AVAILABILITY.json`
- Create: `benchmark/sim2real/reviews/newcore_week_20261103/RESOURCE_AND_ETA.json`

1. 阅读最新 /home/科研/AGENTS.md。g3仍禁用。原有共享checkout不切分支/reset/覆盖。
2. 固定完整发布62da4f35ce3c5c96f04268bd1687a27657af9238；显式coverage_time_log_linear、coverage0.995、无高频补帧。源码17项哈希核对真实导入路径。
3. GPU3/4及新预约GPU5的UUID、原占位PID、启动时间和run目录见机器协议。原预约由研究侧拥有、明确授权给本轮。部署时安全退出对应占位进程并原子移交任务预约；空闲时保持低负载占位，不占GPU0/1/2。
4. 核对5条生成流水线的官方源码与checkpoint：EventGenesis、V2E、Vid2E(SuperSloMo+ESIM)、FILM+ESIM、RIFE+ESIM。后两者是插帧器，不能单独当事件仿真器。
5. 45个目标训练单元=3任务×5方法×3seed，全部登记 planned/data/preflight/training/evaluating/complete/failed/blocked。缺少源码/模型须立即登记，并继续其他项，不能用零值或替代方法填数。
6. 每条线先测小段真实吞吐量，算到11月8日锁表的GPU小时，不作“肯定按时全完”的空承诺。优先完成每条线Ours/V2E的3seed，再补齐其余比较与必要消融。

## Task 2: 三线数据与正确性预检并行

**Files:**
- Reuse: `scripts/generate_stage4_flow_bank.py`、`build_e2vid_stage4_10ms_data.py`
- Reuse: Stage8 ReplicaCAD生成/构建脚本
- Reuse: `reviews/epose_diverse_short_20261101/launched_motion/sources/`
- Create: `configs/stage9_*.json`、`manifests/stage9_*.json`
- Create: `reviews/newcore_week_20261103/PREFLIGHT_RECEIPTS.json`

1. 光流/重建：优先复用已有完整floatHDR/depth/seg/K/SE3，不重复渲染；丰富ReplicaCAD场景超出旧两rollout，并覆盖多个运动幅值、方向与独立刚体运动。整布局/轨迹划分，不能相邻窗口随机拆分。场景筛选基于预定几何与可观测性，不能按生成器表现选样。
2. 同一新物理bank生成全部方法的事件；保留空窗，禁止为追平事件数调阈值。显示灰度只给明确需要它的基线，不冒充Ours原生HDR。
3. E-POSE：旧R10没有HDR/depth/seg缓存，补采真实Genesis渲染并保存完整字段。匹配CAD frame/纹理/米制尺寸；按通用全画面/多尺度/方向/截断和运动随机化扩大合成覆盖，不按真实测试GT分布拟合。
4. 先完成已核验wood-block的完整定义，评价范围明确为E-POSE wood-block子集，不称整个多物体数据集。额外对象在确认CAD与预算后扩展，不能阻塞三条主线。
5. 入口/时间/单位/K/标签/PnP/输出裁剪与损失oracle沿用已验证机制；重建补查range、voxel、recurrent reset与网格伪影；光流单窗口拟合不能代替独立rollout检查。
6. 每条通过后即可进入训练，不等另外两条全部完成。不要建立必须位级复现CUDA grid_sample整条训练轨迹的假门禁；核对checkpoint边界、独立next-batch oracle和实际optimizer计数即可。

## Task 3: 完成注册训练队列并固化权重

**Files:**
- Reuse: `scripts/train_steflow_stage4.py`、`train_e2vid_stage4.py`、已验证pose训练入口
- Create: 每任务实际冻结 `stage9_*_final.json`
- Create: `reviews/newcore_week_20261103/TRAINING_MATRIX.csv`

1. 每任务/seed采用相同随机初始化和样本/增强流；旧R6/R10各生成器专属checkpoint不得充当正式起点。
2. 同任务全部方法固定同有效batch、目标、更新预算与schedule。不要用自监督STE-FlowNet与GT监督短训互相替代。
3. 验证仅合成；记录每次实际optimizer更新、梯度、验证及resume。按协议选synthetic-best，失败全部保留。不因V2E更好而停训或加训。
4. 预检通过后启动可靠tmux/后台队列，保存launcher/子进程PID、日志、退出码、权重hash。CLI回合结束不应让训练队列停止。
5. 核验各任务初始20步交付，然后训练继续。首批真实证据是实际更新和可恢复checkpoint，不是只跑前向或生成空状态文件。

## Task 4: 锁真实评测范围与正式分数

**Files:**
- Reuse: HQF14、MVSEC、E-POSE已有评价器与协议
- Create: `protocol/stage9_real_eval_frozen.json`
- Create: `manifests/stage9_real_test_files.json`
- Create: `results/stage9/per_sequence_seed_metrics.csv`

1. 用户本轮已授权真实最终测试，无须在协议/权重/数据门禁通过后重复请求人为许可。未满足门禁不得开评分；不要直接绕掉旧评分防护。
2. HQF14完整序列；明确此前已见，为新版已见复评。HS-ERGB仍扩展，不把额外下载阻塞主重建。
3. MVSEC按事先固定的完整序列与有效APS窗口评分，至少明确可用indoor_flying1/2/3；outdoor获取和hash契约在评分前定。下载受阻立即登记，不能看完分数再决定去留。
4. E-POSE完成注册wood-block全序列的旋转/平移、速度与照明分层，不止200窗。r01_gl/r05_gl已见；其他序列在权重hash与协议锁定后开分。原始全帧事件、普通eval、confidence选点、固定PnP，无GT crop/mask/box/cell。
5. ADD-S阈值/AUC范围、无效PnP失败分母、MVSEC外点规则、HQF预处理均在开分前冻结。

## Task 5: 数值、统计与图表交付

**Files:**
- Create: `results/stage9/main_results.csv`、`per_sequence_seed_metrics.csv`
- Create: `results/stage9/ablation.csv`、`generation_efficiency.csv`
- Create: `results/stage9/tables.xlsx`、`tables.tex`、`plots/*.pdf|svg`
- Create: `reviews/newcore_week_20261103/FINAL_NUMERIC_AUDIT.md`

1. 3seed均值±样本标准差，保留全部seed；统计单位是完整序列/轨迹，不把邻近帧当独立样本。配对差异与95%cluster CI清楚说明重采样单位和小样本限制。
2. 主表至少重建LPIPS/SSIM/PSNR，光流AEE和外点，位姿ADD-S成功率与AUC；MSE、位姿cm/deg和2Dpx辅助指标、无效数与序列分母随附。
3. 时间效率用相同设备/负载/输入，固定warmup和CUDA同步，RTF=物理时长/生成耗时；共用渲染、冷启动与生成各段范围注明。不能以CPU/GPU差异或其他任务竞争制造速度优势。
4. 每张图从锁定CSV由脚本产生，矢量PDF/SVG与可编辑XLSX/LaTeX一起交付。禁止手填改数、选好看片段或隐藏失败。
5. 11月8日锁表，11月9日交付。完整矩阵若有缺项，给出确切缺项/已完成部分/资源量，不冒充45项全完成。

首轮执行交付必须包括：实际运行中的三线任务、首个通过门禁的训练进程及20-update证据、实测吞吐量与截止前ETA、阻塞基线清单。不要只回一份计划。
