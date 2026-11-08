# 合成重建短训：完整结果与三线状态

**2026-11-08恢复核验：六个短分支均已完成。** 最新结论见[FIRST_SHORT_TRAIN_RESULT.md](FIRST_SHORT_TRAIN_RESULT.md)；完整数值见[results_analysis/REPORT.md](results_analysis/REPORT.md)，独立重算见[INDEPENDENT_RESULT_REVIEW.md](INDEPENDENT_RESULT_REVIEW.md)。

- EventGenesis加L1使MSE下降15.48%、PSNR增加0.932dB，感知和细节均改善，9/9门槛通过。
- V2E光度改善，但VGG和梯度误差退步；共同候选未通过，不启动长训或新真实评分。
- 下一轮拟仅检验共同较弱L1=0.25，所有门槛不变；不在本报告预写运行结果。
- 光流15/15正式训练完成；正确MVSEC室外GT重传中，不匹配未完成文件已完整保留，待全文件/数组核验后统一固定真实测试。
- 位姿光照门槛14/30，新E-POSE评分仍关闭。

result_summary.json为完整服务器导出（含源码/日志/CP哈希），results_analysis/提供主/次CSV、逐轨迹表、门槛及PNG/PDF。PROTOCOL.md、LITERATURE.md、CODE_REVIEW.md保留原登记与证据边界。CURRENT_STATUS.md与CONTEXT_ANALYSIS.md为此前时间快照，旧SSH不通及结果待取回的状态以本次恢复核验更新为准。
