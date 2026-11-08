# job172成功后自动执行固定RAW C1确认

本包独立于已锁preparation/evaluation包；没有修改其源码、合同、轨迹或统计方案。本launcher不训练、不生成事件、不调整模型，也不初始化CUDA上下文，避免父launcher成为评价器GPU排他检查中的额外进程。

根会话在服务器核源字节并实际运行本包10项CPU测试和原eval包5项CPU测试后，可把本包放于N/advantage_continuation_sources_v1，向**已有GPU4队列**提交一个依赖job172的新任务：

```text
<原env/python> -B -u N/advantage_continuation_sources_v1/continue_after_events.py --source-manifest-sha256 <本包source_manifest.json实际SHA>
```

queue spec必须含`after: ["172_pose_advantage_events_20261109"]`；ready_files至少含N/advantage_confirmation/event_completion.json、N/advantage_evaluation_sources_v1/evaluation_source_manifest.json和本包source_manifest.json。根自行设置合法output_dir/stdout路径，勿预建N/advantage_continuation_v1或N/advantage_confirmation/evaluation，这两个是launcher与child分别独占的新输出。

launcher自己还独立核验R/gpu4/job172.done.json的complete/exit0及其.running_spec完全对应；实际原producer argv必须使用相同env Python、events子命令、既定合同SHA和physical SHA，producer脚本位于N内且真实SHA3a443...。不是只信queue的依赖文件存在。

固定链：

- N=`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261109-pose-advantage-and-failure-v1`。
- R=`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1`。
- PREPARATION_CONTRACT SHA `8acbe3f76b841e908814b8c1d533c2c6ebabafa8e6dd78a5a69f8729d6e3540ef`。
- physical_completion SHA `a8a3f0c3bc04657d38196e0575072116284da028e8bf3f8af18902afcd4580fd`。
- evaluation_source_manifest SHA `8a285c7343edc3be18f20d4650fb6e7cc2bc94489f08b73663da2711298f255d`。
- GPU4 UUID `GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3`、原队列父PID335186；launcher必须由该父进程直接启动，CVD严格为该UUID。

待event_completion写完后才读取其**实际SHA**：核registration及physical收据、全32有序物理文件、三光照×两方法192事件流和六manifest、事件/物理文件真实字节、冻结source/core17/profile及eval包四文件。随后先写admission.json，将实际eventSHA放入原评价器准确CLI：

```text
<原env/python> -B -u N/advantage_evaluation_sources_v1/evaluate_confirmation.py --registration N/PREPARATION_CONTRACT.json --registration-sha256 8acbe3f76b841e908814b8c1d533c2c6ebabafa8e6dd78a5a69f8729d6e3540ef --event-receipt-sha256 <刚完成收据的实际SHA> --out N/advantage_confirmation/evaluation
```

原评价器仍执行自己的GPU4 guard、旧18cell原RAW C1完整pose parity、全部新32轨迹/三照/三seed、97.5%两primary区间和始末源/模型/input SHA。没有新参数或GT推理输入。最暗结果和失败条件全部保留，质量失败也只是结果，不自动触发训练、真实评分或aux调参。

资源：admission上限600秒，child等待上限3600秒，launcher总alarm4200秒。child另建自己PG，收到中断/超时只清理该child PG，绝不对原GPU4队列、job172或其他任务发信号。最终completion区分任务成功与conditional_advantage_passed；失败保留failure.json，不自动重试或覆盖。

本地10项CPU真实通过；只验证完成依赖/argv、物理及六事件bank库存/hash拒绝、固定评价CLI、只清理自己child PG等，不声称实际服务器job172或GPU确认已经完成。原包仍须根会话独立服务器准入。本包没有提交队列。
