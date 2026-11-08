# 固定 BAF 输入的中心分支短训：更新前登记

登记依据为已经完成的固定模型零更新诊断，不使用新真实评分。EG 暗光中心 EPE 29.28→20.85 px，但正常/半亮度 8.21→9.90 / 9.99→11.05，TRAIN 三种光照均退步。BAF 不可直接部署；检验重新训练中心分支是否能适应共同固定输入处理。

- 冻结 core `62da4f35ce3c5c96f04268bd1687a27657af9238`、原始事件、物理轨迹/标签、噪声/遮挡规则；只改下游 center 输入。BAF 为固定 2 ms、8 邻域、无极性、零初始历史、拒绝事件仍更新输入时间表。浮点时间戳保留，按整 episode 因果处理，不按窗口清空历史；不声称分数微秒与官方整数实现逐位相同。
- EventGenesis/V2E，seed 0/1/2。每组从原 C1 的 source 初始化，六轮、batch 32、288 optimizer 更新。共六个新 center 分支、1,728 次真实更新，不在已训 C1 上额外续训。复用 C1 RAW 控制，明确不冒称本轮重跑控制。
- 保持 C1 half 采样概率 .75、AdamW LR .00015 / weight decay .01、clip 5、正常 teacher 权重 2。teacher 看到同一滤波正常视图；不用配对一致损失，不改 depth/rotation 权重。
- 任意训练前独立重建原六组全部 288 步的样本、标签、增强/抽样并核原日志；验证不消耗正式训练 RNG。普通光照严格保留原 NPZ 顺序和标签，滤波图按 ID 对齐；aux 使用原 fresh 标签。不得以新投影浮点值替换原 NPZ 标签。
- 缓存逐 episode 编码后用 native CUDA area resize 到 128，绑定输入/source/合同 SHA；smoke 与 study 复用相同缓存以避免重复预处理。不得以 GT 改滤波或挑掉空事件窗口。
- smoke 两种方法 seed 0 各一次中心更新，单独 checkpoint/Adam/RNG/cursor 读回；study 原六组各 288 更新。两阶段安全上限各 3,600 秒，不改变训练预算；只在 g1 GPU4 执行，不影响光流和重建。
- 中心阶段 36 项检查：每 method/seed 的 clean384、half64 中心 EPE 不高于 RAW C1，dark64 严格降低；若完整 pose 评价已执行，clean384 成功率最多降 2 个百分点、half64/dark64 不降低。正常 matched64 与 clean384 分开列出，不混总体。depth 始终使用 RAW C1；rotation 始终 RAW 输入、预测中心裁剪和预测 ray frame。
- 原整体 30+8 门槛不放宽；未实际完成不得写 pass。本试验单独记录 center-stage gate，不能替代整体门槛，不能开启真实 E-POSE 评分。已知 C1 半亮度深度仍退步，中心改进不能自动消除此问题。

机器合同 SHA256：`60db65c84cba5c6045542eca90640017f4e863e5e8ad9cab774dae778deee762`。合同、源码及真实执行证据将在研究核验目录归档。


# 首次更新前勘误：检查点核验范围

先前登记中的“单独 checkpoint/Adam/RNG/cursor 读回”表述超过现有 frozen BoundedFitter 实际能力。此次短训只宣称模型与 Adam 状态独立重建、参数/计数一致；保存 RNG 快照不等于已验证中断续训的下一批或整段轨迹等价。

smoke 和 study 都从原初始化一次性执行，study 只复用 smoke 的固定输入缓存，不从 smoke 权重续训。`resume_supported=False`；不开放恢复入口，不冒称完整恢复通过。全部 288 步样本/标签/增强对原 C1 的更新前重放及训练后日志核对仍保留。

此勘误在首次候选 optimizer 更新前记录；不改变训练目标、预算、数据角色和阶段门槛。
