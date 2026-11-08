# 原始输入教师 / BAF学生：首次更新前登记

前一轮 BAF 双方配对短训已完成六组各288更新，中心阶段20/36通过，整体失败，不采用。独立source-init teacher零更新检查表明：正常TRAIN的EG旧模型误差从RAW2.72变成BAF8.36px，V2E4.26变成17.11px。正常保留项正让学生模仿过滤后偏移的旧输出。此为具体可检验机制，不能视作唯一根因。

本轮仅把正常光teacher输入改为对应RAW视图；student仍BAF，teacher与student使用native matched_affine共享完全相同几何增强、标签和单次RNG进度。所有损失系数、AdamW、clip、采样比例、batch32、288步、初始source模型不变；六组方法×三seed新增1728个center更新，另有两次smoke。旧RAW C1和旧BAF+filtered-teacher控制均复用并声明，不在旧C1上额外训练。

复用已核SHA的旧BAF128缓存；新增RAW正常128严格从原1529个NPZ、原顺序与原标签构建。全部六组288步label/draw/augmentation预先重放并在训练后核日志。深度始终RAW C1，旋转头RAW输入按预测中心裁剪，参数冻结；不得按光照切有利模型、输入GT框或GT中心。

先actual CPU tests、两方法各一次smoke及模型/Adam独立读回，再完整短训。不提供中断续训，不宣称完整RNG/cursor恢复。各阶段安全上限3600s，仅g1 GPU4。原36个中心阶段检查不放宽，原global30+8未执行不得写pass；新真实E-POSE评分关闭，无自动长训。现有重建和MVSEC评分不干扰。

本轮结束后先如实记录结果。若仍失败，不继续扫权重或增加训练步数，应先查过滤对物体前景事件保留、几何预测与位姿成功阈值的差距。

机器合同SHA256：`d682813400a1709c90b47b4dcc73a83c8f7ea3fe830e9d837316555dea11ca41`。
