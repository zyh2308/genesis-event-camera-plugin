# 32轨迹固定评价续接 v2

原job174在准备合同校验处退出。v1源码将实际SHA片段`d53c2`写成`d533c2`，造成65位摘要。v2按实际归档字节校验准备合同、物理收据和job172运行配置，并绑定已完成的事件收据与job172成功退出收据。父队列使用GPU4，校验PID335186及实际启动时钟53273702。

将v2包解压到N/advantage_continuation_sources_v2。先在服务器运行`python -B -m unittest -v test_continuation.py`。服务器测试直接读取已有准备合同、物理收据、事件收据和job172原始退出记录，不申请GPU。N为`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261109-pose-advantage-and-failure-v1`。

通过后由现有GPU4队列执行：

```text
/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python -B -u N/advantage_continuation_sources_v2/continue_after_events.py --source-manifest-sha256 <v2 source_manifest.json SHA256>
```

队列依赖为`172_pose_advantage_events_20261109`。沿用原job174的CUDA UUID和环境变量，在`source_sha256`中固定v2文件、已核验的评价器文件、实际证据和v2 CPU准入收据。队列stdout目录使用独立`evaluation_v2`，launcher收据目录为`N/advantage_continuation_v2`，由程序新建。

评价器仍来自`N/advantage_evaluation_sources_v1/evaluate_confirmation.py`，原评价器参数为`--registration N/PREPARATION_CONTRACT.json --registration-sha256 <实际准备合同SHA> --event-receipt-sha256 <实际完成事件收据SHA> --out N/advantage_confirmation/evaluation`。确认最后两个新输出目录尚未建立。原job174退出及v1文件保留。

续接执行同一固定RAW C1评估。它覆盖32条共享轨迹、两种生成器、三个种子及三档光照。正常和半亮的预登记区间为97.5%，最暗条件同表报告。launcher在600秒内完成准入，评价器上限3600秒。超时清理范围为本次创建的子进程组。
