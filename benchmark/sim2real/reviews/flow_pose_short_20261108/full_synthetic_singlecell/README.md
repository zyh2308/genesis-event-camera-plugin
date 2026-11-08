# 完整合成光流分解评估：单cell限时补齐

原完整五方法 × 三seed评估采用冻结正式 synthetic_best，每组完整528个合成验证窗。原seed01父任务在60分钟硬时限结束，结合已完成的seed2分片共保留14个完整cell；缺失项仅为rife_esim / seed1。原父进程、原14项结果与失败/限时记录保持原样。

缺失cell在新的独立root `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-flow-full-synthetic-readonly-rife-seed1-v1` 使用GPU1、20分钟上限完成，只读评估退出码为0。显式选择 `--seeds 1 --methods rife_esim`，未冒用seed2身份，未训练、更新模型、重新选择checkpoint或读取真实数据。

CPU检查11/11通过；补跑保留原完整528窗、原模型/数据/15个权重绑定、固定EG∩V2E活跃区、原指标及历史验证parity。原科学函数的源码摘要保持一致。资源父进程采用原子global与UUID锁、实际进程核查及自身进程组清理，补跑结束释放资源。

CPU `merge_matrix.py` 已核验原14项与补齐1项并完成15cell合并，输出新root的 `merged_matrix.json`。合并核对每项方法/seed/checkpoint SHA、528窗、原验证parity及源manifest，不覆盖原结果。本目录保存最终锁源；结果分析与完整三seed分场景结论由本轮独立报告记录，不以单seed或单类总体误差宣称运动追踪优势。

`sources/source_manifest.json` 的SHA256为 `3baf525d3276f874311899977aefa4c2df6abe90af5cf1b661cfb2ee0db54189`。含manifest共13文件，均原字节复制并逐项核对SHA。此处未修改根状态页，也未执行Git提交或推送。