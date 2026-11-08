# 合成重建光度与细节配对优化及三线阶段结果

最新结果、限制与恢复入口见 [CURRENT_STATUS.md](CURRENT_STATUS.md)。

- **重建**：300步×3臂×2生成器已部署，最后核验控制臂289/300和300/300；完整合成效果尚未取回。SSH重置后唯一重试超时，当前任务状态未能刷新；不能宣称已经改善。
- **光流**：15/15正式训练完成；合成EPE有用，暂未稳定优于V2E；室外GT断点内容不匹配，自动停止并保留，真实评分未做。
- **位姿**：短训完成，暗光深度改善；其他条件退步，14/30项稳定性门槛通过，新真实评分关闭。

PROTOCOL.md登记本轮预算和门槛；LITERATURE.md列出论文/作者代码依据；CODE_REVIEW.md为独立静态审查。CONTEXT_ANALYSIS.md、context_analysis.json及context_status.json保存已核实的光流/位姿分数。context_figures/提供PNG与PDF数据图；mvsec_resume_evidence/保留实际传输失败收据。

scripts/为独立实验快照及只读收集/分析入口，没有改写冻结simulator或正式训练源。完整新结果仍等待服务器访问恢复后核验入库；本目录没有把阶段状态伪装成优化完成。
