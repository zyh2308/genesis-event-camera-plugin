# 首次更新前勘误：检查点核验范围

先前登记中的“单独 checkpoint/Adam/RNG/cursor 读回”表述超过现有 frozen BoundedFitter 实际能力。此次短训只宣称模型与 Adam 状态独立重建、参数/计数一致；保存 RNG 快照不等于已验证中断续训的下一批或整段轨迹等价。

smoke 和 study 都从原初始化一次性执行，study 只复用 smoke 的固定输入缓存，不从 smoke 权重续训。`resume_supported=False`；不开放恢复入口，不冒称完整恢复通过。全部 288 步样本/标签/增强对原 C1 的更新前重放及训练后日志核对仍保留。

此勘误在首次候选 optimizer 更新前记录；不改变训练目标、预算、数据角色和阶段门槛。
