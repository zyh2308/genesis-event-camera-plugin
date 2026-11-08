# 前景支持、注意力与跨光照尺度：零更新诊断登记

用户已要求先确定具体原因，不再盲目试训练方案。此前三项配对短训均未通过；本轮先完成以下诊断，不自动启动训练或真实评分。

## 固定范围与隔离

- 冻结 core `62da4f35ce3c5c96f04268bd1687a27657af9238`、传感器/噪声/遮挡规则、原始事件、模型权重及数据划分。只使用合成数据；真实评分数据不用于诊断和选择。
- 全部既定独立 VAL `validation_000/031/063/095`、k=4..19的64窗，正常/半亮/1/8光照，同一物理窗口严格配对。EventGenesis/V2E，seed0/1/2，C0/C1，中心与深度两个独立模型。
- RAW与原固定2ms BAF仅作冻结模型输入响应；BAF不重新训练，深度BAF响应不是前轮“只改中心输入”候选的正式结果。0 optimizer更新；不修改decoder。
- 原整体30+8门槛和真实E-POSE隔离保持。GT分割、CAD、位姿及深度只用于事后诊断/误差，不进入模型输入、裁剪、过滤选择或attention重权。无GT选择性输入消融。

## 区域与时序口径

实际HDF5含21帧、346×260分割/深度、六物体位姿/segID表。每文件动态用target_pose与object_to_world唯一匹配target，并独立核mask面积等于target_visible_pixels。已查四个普通光照样本均为object1/segID1，但不得写死。三照度的几何、分割和时间应一致，先核再解释结果。

终点实例mask是该渲染时刻的可见目标区域；事件覆盖整10ms，不能把该区域外事件都叫背景噪声。分列start/end及endpoint_union_approx，后者不是真实连续扫掠可见mask，也不能给每个事件标真实来源。四轨迹每窗最大角点位移约1.8/2.4/2.9/3.25px，必须按运动幅度核对。保留空窗/空mask，零分母记null与计数。

报告事件数、空间占用/覆盖、RAW→BAF保留率、mask面积、GT anchor与可见区域/质心关系，以及投影3D bbox凸包相对可见mask的几何sanity；bbox凸包为近似，面积比不称精确遮挡率。只报告不改mask或筛窗。

## 注意力、尺度与路径控制

用只读hooks取得原forward features与sizehead输入，正常原forward输出必须与上轮导出逐字段一致；UV归一化、log分量、Z米的maxabs容差预设2e-6。原生346×260→cv2 linear416→native CUDA area128；mask用同映射到16×16的fractional occupancy，点命中用实际model.grid。大RF可能包含区域外上下文，不能用mask mass单独判注意力正确。

两独立模型分别报告attention mass/富集比/峰值、anchor误差及特征统计；center监督的原GT软heatmap可作事后参考，不改前向。深度模型attention是size feature pooling，不直接当最终定位输出。

同窗跨照度核log_size/log_shape/log_z的signed误差、MAE、scatter和补偿，检查Δlogz=Δlogshape−Δlogsize。额外仅在RAW depth做预定2×2路径诊断：normal/relit features和normal/relit attention交叉，context随feature来源，重建local后调用原frozen sizehead。两个对角须复原原forward≤2e-6。交叉利用了另一照度的配对事件，是非部署hybrid机制诊断，不能作模型成绩、生成器排名、训练门禁或真实推理方法。

先合并逐窗事件支持、注意力、尺度、运动与遮挡/截断，再按全部轨迹和seed汇总。统计相关只支持后续机制假设，单轨迹或单seed获益不能开训。若标签/区域映射不一致，先停止机制判断修诊断；若过滤删有效前景，暂停该BAF路线；若前景充分但注意力漂移，才设计单项预测可得的定位/表征验证；若中心稳定但尺度漂移，先核原log_size/log-depth一致性项实际是否已启用，避免重试旧方案。

## 执行与证据

只用g1现有GPU4预约做冻结前向（启动前实时核UUID/归属，其他进程出现则fail closed），CPU2线程、cv2单线程；不干扰重建和MVSEC。独立新run输出；锁源码、实际CPU小检查、输入与模型哈希、原输出parity通过后才完整矩阵。物理H5 probe和真实source facts已保存，源及结果按哈希索引归档共享仓库；旧review封存字节不改。若未锁定具体机制，不启动新的配对短训。
