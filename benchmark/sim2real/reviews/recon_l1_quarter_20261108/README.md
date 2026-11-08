# 重建共同L1=0.25：有限配对短训运行中

上一轮L1=1改善光度，但V2E感知/梯度退步，双方共同门槛未过。本轮仅检验较小共同权重0.25；不预写成功结果，不放宽九项门槛，不长训或真实评分。

源CP4800、Adam/RNG、标签与增强、LPIPS/TC、8条200帧验证及300额外更新/臂不变。共两方法×control/L1两臂，1200主更新+4smoke。

原GPU0/1提交因他会话有效租约被拒绝，未进行更新或抢占。改为GPU2同卡串行，两个smoke及跨方法实际样本哈希通过后才开始主训；当前快照见launch_status.json，完整结果待收集。

运行根：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-recon-l1-quarter-paired-gpu2-v1/`；只读结果收集器独立等待，产物根`.../20261108-recon-l1-quarter-results-v1/`。总任务2h上限，结束释放GPU，无空占卡。

worker_source/为实际冻结运行源码及manifest；analysis_scripts/含严格完整性检查与原九项采用条件。单seed结果仅作合成开发诊断；不得用不同训练轮次或不同硬件轮次的绝对差替代本轮同期对照。
