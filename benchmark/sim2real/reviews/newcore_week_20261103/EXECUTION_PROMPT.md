你是已有的“搭建 Sim-to-Real 实验闭环”工程执行会话。研究侧依据用户的新指示续接你，并授权继续实际工程与服务器执行。用户要求：三条线都使用新版仿真器重新训练，按可用于CVPR论文实验的规格交付；最终截止是2026年11月9日（日本时间），本轮只要数据表与定量图表，暂不做demo视频、定性拼图、架构图和PPT设计。

请立刻读取并执行共享仓库中的正式任务入口：
1. docs/plans/2026-11-03-sim2real-newcore-weekly-retraining.md
2. benchmark/sim2real/protocol/stage9_newcore_campaign_v1.json
3. benchmark/sim2real/reviews/frozen_simulator_20261103/README.md
以及 /home/科研/AGENTS.md 最新服务器规则。

你的执行worktree是 /home/科研/Eventbased_WAM/repositories/gec-sim2real-newcore-week-20261103 ，分支 exp/sim2real-newcore-week-20261103。原共享主研究仓库保持可读，不切换/reset它。所有本轮实现、计划最终化、机器状态、数据指纹、报告和索引都在执行分支提交，g1大数据/权重保持外部只读输入加唯一run输出。普通工程决策和兼容问题自行处理，重大研究改变/缺失依赖/超过预算及时报告并继续未受阻线。

新核心的完整发布提交是62da4f35ce3c5c96f04268bd1687a27657af9238，核心来自6ac417c7a03e125f40ebf967796bf28191d55859；显式coverage_time_log_linear、coverage门槛0.995，原生linear HDR，不依靠高频补渲染。构造器默认没有自动改成这个模式，旧脚本sys.path还可能优先导入旧run源码，必须按真实__file__和17文件SHA核对。机器人理想无噪声验收profile原件不改；正式任务保留已登记非理想传感器参数时另存任务profile并验证，老/新核心消融不得同时改变噪声/采样率。

资源已经由本研究会话预约并明确授权移交本轮，启动前仍实时核验。GPU3 V100S32GB 默认光流，holder PID4088994；GPU4 V100S32GB 默认重建，holder PID4089019；两者run是/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261102-sim2real-gpu34-060016。GPU5 TITAN RTX24GB默认位姿，holder PID325653，run=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-sim2real-weekly-gpu5-175055。各run内有安全release_gpu.py；先校验用户/PID启动时间/命令/UUID，仅退出对应本轮占位，原子转登记实际任务；不能碰GPU0/1/2或其他任务。空闲时按用户要求保持低负载占位。g3仍然禁止连接。

已有三线下游骨干、HQF14完整数据和E-POSE完整下载均复用。三线并行准备/预检/训练，不等光流完成再开其余。重点优先补足光流丰富场景与独立rollout，先固定短合成正确性诊断再长训；重建验证训练/推理一致性和网格伪影；位姿不再用通用立方体/旧R6暖启动冒充正式木块训练。R10 HDF5没有保存float HDR/depth/seg，必须补采并存好这些中间输入。r01_gl/r05_gl已见，不能称全新测试。

比较目标为EventGenesis、官方V2E、Vid2E(SuperSloMo+ESIM)、FILM+ESIM、RIFE+ESIM，训练seed0/1/2。先保证三线Ours/V2E完整3seed，其他官方基线的源码/权重查验和准备同步做，缺项立即列证据，绝不偷换成线性插值。预算机器协议为预估，按纯合成吞吐/诊断统一冻结最终规格后再正式更新。不能看真实分数调参或扩预算。真实最终评分在所有协议/权重/输入哈希门禁通过后已获用户授权，不必再等研究侧手工盖章，但不得绕掉门禁。

这次不要只读资料和输出计划。请先完成资源移交、三线实际数据/预检作业，并启动首个通过预检的正式训练。至少交付实际20次optimizer更新、可恢复checkpoint和独立核验；随后让可靠后台队列继续完整预算。请落地EXECUTION_STATUS.json、TRAINING_MATRIX.csv、RESOURCE_AND_ETA.json及明确的第三方baseline缺失清单，状态包含PID/日志/退出/哈希。训练与数据可在g1后台继续，CLI回合结束不应停止它们。不要循环空等待；有一个阻塞先推进另外两线。首批数据图表只需真实数值，所有数字从原始日志/指标文件复算。
