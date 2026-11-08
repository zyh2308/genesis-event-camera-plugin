# 位姿特征冻结短训：服务器CPU 4/4及GPU 12次更新smoke通过；正式study已启动、尚未完成

本轮只新增 2 个生成器 × 3 个 seed × 2 个分支 × 288 次更新，即 12 个 fit、3,456 次 optimizer 更新；C0/C1 使用上一轮已冻结的可靠模型与训练记录，无新控制组更新。候选与原 C1 使用相同初始化、物理光照采样、标签、增强轨迹和损失，唯一差别为 features 权重及缓冲被冻结，仅训练头部。half KD 实际权重为 0。

原 30 项光照门槛和 8 项原确认检查保持不变；另按原 C0 指标复核 30 项门槛，真实评分关闭。旧 native 的 paired_consistency 槽仅用于接口兼容，其 canonical 名称为 frozen_features，不表示使用一致性损失。

当前为服务器CPU 4/4及GPU 12次更新smoke通过；正式study已启动、尚未完成：本地 CPU 4/4 测试通过，包括真实微更新、特征及 BN 缓冲冻结、checkpoint/optimizer 恢复、标签和增强一致性与旧控制组不可变验证。此处不声明服务器训练已开始、已完成或候选已经改善。

sources 保存锁定包的 11 个文件及 source_manifest.json 原字节，manifest SHA256 为 6aa9a9e96419838ecd8be2d300100cae824ffde0abc2a8bee2d46fc9210ffafc。未修改已锁定源文件、旧实验结果或根状态页。
