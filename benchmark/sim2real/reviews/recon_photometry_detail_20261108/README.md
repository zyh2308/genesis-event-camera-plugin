# 光度与细节配对短训：运行前协议及阶段快照

2026-11-08：两生成器各三臂、各额外300步的合成短训运行中；本提交不包含尚未产出的最终重建分数，不表示优化已通过。没有启动长训或本轮真实评分。

- PROTOCOL.md：结果观察前登记的预算、损失、采用门槛与预检修订。
- LITERATURE.md：作者论文/代码依据及借鉴边界。
- CODE_REVIEW.md：独立静态审查，不能替代实际运行收据。
- CONTEXT_ANALYSIS.md / context_analysis.json：当前光流和位姿的已完成成果；光流15/15训练收齐；位姿光照门槛14/30，真实评分关闭。
- scripts/：独立运行快照和离线收集/分析脚本，不覆盖冻结 simulator 或正式训练源文件。

g1 短训根：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-recon-photometry-detail-{eventgenesis,v2e}-v3/`。v1/v2 预检失败原样保留；v3 使用原 Trainer 函数，不宣称整段 CUDA 梯度逐位复现。

MVSEC 室外数据正在从经核验的本地投影续传；训练完成不代表真实测试完成。全部新的合成结果与采用判断在训练完成后另行入库。
