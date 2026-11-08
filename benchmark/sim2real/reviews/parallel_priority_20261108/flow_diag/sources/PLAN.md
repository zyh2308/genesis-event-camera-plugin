# 光流优先：先用已有合成数据定位，不读新MVSEC成绩

固定核心仍为62da4f35…；正式15组45epoch/8910步、已选权重及正在GPU2/5执行的统一MVSEC矩阵全部保持原样。本目录只写新的只读诊断工具，不自动启动训练。

## 已经从冻结源码确认的事实

1. 当前正式bank是8个ReplicaCAD布局、32个完整物理rollout，24train/8val；训练1584、验证528个30ms窗口，四种运动分层各自6train/2val，不是早期简单场景。
2. 正式训练是随机初始化的STE-FlowNet，batch8、45轮、Adam lr4e-4，光度+10×二阶平滑；GT flow只用于合成验证/诊断。训练/验证按完整布局分离。
3. 锁定multiscaleloss的Charbonnier为像素SUM，再平均四个尺度；smooth是二阶差分MEAN，尺度权重1,.5,.25,.125。因此“平滑权重10”不等于平滑梯度占主导。先量梯度，不能盲目降权重。
4. 光度函数的event_images参数实际没有使用；它遍历全图，不使用物理可见性或事件活跃mask。越界warp图被置0，但残差仍与prev灰度比较。这是原作者目标的行为，不是本次发现的新代码损坏；对静态背景、弱运动、遮挡区域的训练贡献需要量化。

精确来源：

- g1 R/sources/benchmark/sim2real/scripts/train_steflow_stage9.py：list_split、apply_joint_transform、val_input、evaluate_validation。
- R/sources/benchmark/sim2real/scripts/run_steflow_stage2_pilot.py：objective_dt1、参考相关性。
- P/external/sources/downstream/STE-FlowNet/multiscaleloss.py：charbonnier_loss、compute_photometric_loss、smooth_loss。
- R/sources/benchmark/sim2real/compat/steflow_align_corners.py：训练内部warp与光度warp兼容规则。
- 本地官方源码位于WSL /home/科研/Eventbased_WAM/external_sources/STE-FlowNet/。

R=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1；P=/home/zhaoyuhan/Eventbased_WAM。

## 两个最小诊断作业

`flow_synthetic_diagnostic.py bank --contract diagnostic_contract.json --out <new>`：CPU读取原32条合成rollout的两生成器NPZ，逐文件/标签/灰度/时窗核对。按0–.1、.1–.5、.5–1、1–2、2–5、≥5px/30ms记录运动分布、共同事件活跃/不活跃、有效像素和零流基线。全量2112个物理窗口只统计一次，不读取模型/真实分数。invalid geometry/visibility不能直接叫“遮挡”：原adapter已经把无效GT置0，不能在它上面伪造occluded EPE。

`... probe ...`：仅使用g1 GPU3固定UUID GPU-253ab9b8-a0e5-4f12-0579-261818cf31dd，复用既有队列/租约，不触碰GPU2/5。读取原seed0 synthetic_best（两组固定hash），每个验证rollout固定0/22/43/65，共32个窗口/方法，记录预测幅值、零流/EPE、运动bins、共同活动区及无效区计数；这是诊断子集，不替代528窗正式指标。

再对每方法一个固定、共享增强的真实合成TRAIN batch8（每运动分层首条train轨迹的11/44号窗）执行一次模型前向。模型无BN，输出detach后只求四尺度光流输出上的光度/10×平滑梯度、能量支持与方向；不求或更新网络参数，0 optimizer更新。GT仅参与EPE梯度oracle和可见区亮度一致性对照，不传入模型，不作为optimizer监督。

额外比较同一灰度端点在零流、预测、正确物理GT投影下的可见区Charbonnier均值，区分“监督目标本身不偏好正确运动”与“模型尚未学到”。不改变冻结损失或mask。源模型状态和CP完整SHA前后检查。脚本总预算600秒；调用队列另设有界phase超时，不影响原队列parent或其他作业。

梯度空间是flow输出，不是70组网络参数，不能将局部诊断写成完整参数优化结论。官方CUDA grid_sample反传可有微小重复漂移，工具不宣称轨迹逐位相同。

## 诊断后的最小候选

优先只改变**共享物理训练窗口采样**，保留模型、表示、损失、阈值、噪声与标签。仅当CPU/预测分布确认多数训练贡献集中于近零背景、而明显运动窗口不足时，才登记一个独立2方法×2臂短训：uniform control与motion-balanced sampler，各固定64更新、相同源权重/Adam初态和预算，最终固定权重评估原完整528合成窗口；无长训、无真实反馈。

可预登记的简单采样式：p_i=0.5/N+0.5*f_i/sum(f)，f_i为原训练有效GT中≥1px运动像素占比；物理元数据只决定两生成器共同的训练样本序列。无窗被删除，GT不进入优化损失，验证窗统计不参与拟合采样权重。within-control/candidate的样本序列有意不同；相同候选下两生成器样本与增强必须逐项相同。该候选尚未执行，也不承诺有效。

若光度GT oracle反而不优于零流，先定位光度假设/可见性/缩放再训练；若真实测得平滑梯度确实主导且与GT EPE方向冲突，才改用单变量lambda对照。若平滑梯度很小，就排除“平滑过强”解释，不继续扫系数。诊断报告决定哪一条假设值得短检，不同时铺开多模型多参数。

新短训将是独立研究消融，不能覆盖正式矩阵、复用正式resume标记冒充原训练续跑，或用新MVSEC分数选采样阈值、权重与模型。

## 本地验证与边界

3项小型CPU夹具覆盖：运动bins/共同活动/无效区域定义、ON/OFF活动图轴、四尺度梯度与GT oracle管线有限且不修改输入/不产生参数grad。未在本会话加载服务器模型，实际GPU阶段是首个有界执行探针，不能把CPU夹具通过写成GPU已完成。

所有输出新建，失败写独立failure.json。没有SSH、训练、评分、远程队列写入或仿真器修改。
