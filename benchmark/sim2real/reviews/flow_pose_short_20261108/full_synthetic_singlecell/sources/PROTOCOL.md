# 条件式补齐RIFE seed1完整合成评估

仅在原GPU0父任务已停止并released，且原rife_esim/seed1完整cell仍缺失/不完整时部署。原任务若自行完成则不部署。本程序不停止、覆盖、修改原parent或任何原结果；不冒用seed2身份。

新独立root为20261108-flow-full-synthetic-readonly-rife-seed1-v1。固定`--seeds 1 --methods rife_esim --out ROOT/rife_seed1`，GPU1 UUID0262d545-e9bd-7fcb-66e5-2ef81465c460（RTX3090），原子global+UUID非阻塞锁、所有registry无活动lease、实际无其他compute context后才执行；结束/失败只清理自身进程组并释放，硬上限1200秒。父进程不创建CUDA占位。

保留原全部15个synthetic_best与数据/模型/源/标签pin，单cell只评原528窗及固定EG∩V2E活跃区、四strata、motion bins；原metric/聚合/历史parity1e-5函数原样复用。0optimizer、不读取真实事件/结果、不重新选checkpoint。单cell通过后CPU merge_matrix.py逐项验证原14+补1的15个权重绑定、源manifest、每cell528及parity，输出新merged_matrix.json；不覆盖旧矩阵。

先运行CPUtest_scientific.py与test_runtime.py。root核验后才发布DEPLOYMENT_READY.json：schema=flow_full_synthetic_cell_ready_v1，passed=true、task_root、source_manifest_sha256、worker_contract_sha256、read_only_model=true、optimizer_updates=0、real_scores_computed=false。build_command.py只封存stdin command，不申请GPU或启动worker。root用lease_single_cell.py --task-root ROOT --gpu-index1 --contract-sha256 CMD_SHA接sealed stdin才实际原子准入。
