# 原稀疏路径定位修复实测

[完整结果与复跑](../../benchmark/robot_closed_loop/results/r1_5_sparse_time_repair_20261101a/README.md)。

生成只使用原有 100Hz HDR 端点与几何信息；没有高频补图或 capture_observed。相同模型与 40 轮预算，主评估中位数／P95 从 8.18／106.72 px 降到 2.33／5.77 px；补充复查从 9.08／112.38 降到 2.20／5.54 px。全部 12 条轨迹的中位数与 P95 均改善。大于 10px 或缺失：225/581 到 3/581，87/193 到 0/193。

已验收范围是 NumPy／CPU 的 coverage_time_log_linear 稀疏重建候选；CUDA 新模式与闭环抓取复跑尚未完成，主分支及旧冻结标签未移动。此前高频 HDR 阳性对照不能计作本项修复。
