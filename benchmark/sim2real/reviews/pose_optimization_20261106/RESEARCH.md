# 位姿优化调研与建议（2026-11-06）

目标仍为比较事件生成管线的 synthetic-only Sim-to-Real 能力。本文是新候选协议的研究建议，不修改当前Stage9冻结训练、检查点或真实评分。

## 已知事实与可疑环节

当前真实木块主统计中，EventGenesis成功率4.05%、V2E1.71%，所有已完成样本PnP返回有效解。有效解不代表正确位姿，不能只把PnP换成另一求解器就预期解决问题。

核对本地归档的 recovery_20261103/train_pose_stage9.py、pose_physical_worker.py、pose_bank_protocol.json、training_recipe.json：

- 96条训练轨迹、32条验证轨迹，每条0.2秒。训练仅1529个窗口，且轨迹内高度相关；30000次batch8更新约等于对这个小bank抽样157遍，不能把更新数当作视角多样性。
- 位置已经覆盖画面约10%—90%，距离0.38—1.15米，旋转有随机化，不能再声称所有物体只在图像中央。
- 场景仍为固定地面、四个固定盒子、固定环境光，训练增强主要为缩放和平移。外观、背景和光照的覆盖明显弱于位姿范围的覆盖。
- 输入是10ms的ON/OFF/total三通道计数，经log1p/固定log9缩放并截到1，再resize至416。它没有保留事件先后顺序；原生像素计数达到8即饱和。信息损失是确定事实，对性能贡献仍需消融验证。
- 当前head为13x13中心网格；坐标回归只监督GT中心cell，最终按channel18最高logit选cell。损失采用坐标MSE+全网格中心CE，是工程适配，不是官方SingleShotPose原样训练流程。
- 官方region_loss.py还包含与角点预测质量关联的confidence监督，并先训练坐标再加入confidence。这一差异值得优先对照，不能未经验证直接宣布现有CE错误。

## 先定位，再比较优化候选

只在合成独立rollout上比较：

1. 正常预测cell的中心误差、角点误差、完整ADD-S。
2. 仅诊断用GT cell解码角点，再PnP，观察相对正常路径改善。禁止将此oracle路径用于真实主测试或最终成绩。
3. 已知正确合成投影点加固定像素扰动，再PnP，量化0/1/2/5/10px误差的几何影响；只诊断求解器敏感性。
4. 对完整合成验证执行同一CAD/PnP/ADD-S评分，补齐关键点EPE与最终机器人状态指标之间的联系。原Stage9选点保持原规则，诊断不换其权重。

若oracle cell大幅改善，先修定位和confidence；若oracle cell仍很差，先处理角点回归、截断、尺度及数据覆盖；若合成已好而真实差，再优先研究域随机化与表示鲁棒性。PnP都有解并不能单独排除标定/标签/几何误差。

## 候选顺序与学术依据

### 第一优先：恢复成熟训练目标的对照

独立版本对照当前MSE+CE与官方坐标/质量confidence及分阶段训练，保持同bank、同初始化、相同预算，先EventGenesis/V2E配对短检。不能只给EventGenesis更好的head。若明确因优化目标失配，才推广共同的新head。

官方代码：https://github.com/microsoft/singleshotpose/blob/master/region_loss.py

### 第二优先：更多独立物理场景，保留原始事件规律

扩充独立背景、材质/纹理、照明、距离、运动方向/速度、遮挡和截断组合，均在Genesis物理场景层生成，所有事件管线复用同一物理缓存和标签。参数范围由公开规格/合成设计预先登记，不根据E-POSE分数或GT分布拟合；不为提高分数只修改我们的噪声/阈值，不强行匹配各方法事件数。

优先增加独立轨迹/场景，而非只把96条现有轨迹切得更密。事件阈值/背景活动等传感器随机化应另列后续消融，并明确共同物理参数映射，不能把它混入第一轮场景随机化。

AEE事件位姿工作采用synthetic-only与域随机化、模板编码和检测框辅助平移估计；可借鉴其形状鲁棒性思路，但其模板输入不直接对应我们的完整事件生成实验。

论文：https://doi.org/10.1109/LRA.2025.3644151
开放原文：https://sure.sunderland.ac.uk/id/eprint/19801/1/Learning_6D_Object_Pose_Estimation_With_Event_Cameras_Using_Synthetic_Data_and_Domain_Randomization.pdf

### 第三优先：时序输入的轻量对照

同一原始事件bank，比较当前count图与固定时间分箱voxel grid；不必为此重新生成事件。选一个成熟固定分箱方案作对照，先不引入复杂可学习表示。首层通道变化与预处理参数全方法相同，checkpoint选择只用合成验证。饱和率统计可作为诊断，但不能据真实测试逐方法调整归一化。

方法依据与官方代码：
https://openaccess.thecvf.com/content_ICCV_2019/html/Gehrig_End-to-End_Learning_of_Representations_for_Asynchronous_Event-Based_Data_ICCV_2019_paper.html
https://github.com/uzh-rpg/rpg_event_representation_learning

### 后备：更稳健的关键点定位模型

若前三项无法改善合成几何，则采用PVNet式密集投票/不确定性关键点方案，或EPOS式对称性对应关系。它们是成熟RGB位姿方法的可迁移结构思路，并非已验证可直接提升本项目事件结果。先考察复现成本；不把RGB预训练权重直接带入严格synthetic-only协议。

PVNet：https://openaccess.thecvf.com/content_CVPR_2019/html/Peng_PVNet_Pixel-Wise_Voting_Network_for_6DoF_Pose_Estimation_CVPR_2019_paper.html
EPOS：https://openaccess.thecvf.com/content_CVPR_2020/html/Hodan_EPOS_Estimating_6D_Pose_of_Objects_With_Symmetries_CVPR_2020_paper.html

长方体外观可能产生关键点歧义，但公开CAD并非可未经核验的完美对称长方体。对称集合必须由CAD/公开约定确定，不可按预测误差临时扩大。

## 文献分数不能直接照搬的原因

- AEE分开报告rotation-only和完整变换ADD-S，rotation-only会代入GT平移，是模块诊断，不是完整6DoF成绩。
- E-POSE官方数据另提供由GT几何生成的mask/segmented事件图；不能将这些文件直接当作无特权输入。官方论文验证任务与本项目synthetic-only任务不同。
- Event6D是事件+深度的跟踪框架，不符合当前event-only单帧位姿协议，不能直接替换主线。
- 正常预测检测框后裁剪在原则上可行，但检测器也必须满足合成训练限制；GT框、GT mask、GT初始位姿均不进入正式推理。

E-POSE原文：https://pmc.ncbi.nlm.nih.gov/articles/PMC11822054/
Event6D：https://openaccess.thecvf.com/content/CVPR2026/html/Kang_Event6D_Event-based_Novel_Object_6D_Pose_Tracking_CVPR_2026_paper.html

## 独立版本与测试隔离

当前Stage9完整五方法评价继续作为固定基线。候选改进另登记版本，先合成独立rollout筛选，不能看真实分数反复选方案。最终所有生成器共享选定模型、数据分布、输入编码、预算和seed。

木块真实结果已查看，今后的重复评价必须称已见基准复测，不能重新标为盲测。若要保留独立最终验证，需要事先划定尚未查看结果的真实对象/数据范围，并让相应CAD和训练标签对所有管线一致。

最有希望的低成本路径：合成oracle分解 → 官方训练目标对照 → 增加独立场景 → 时间分箱。暂不启动新一轮长训，不承诺未经实验的提升幅度。
