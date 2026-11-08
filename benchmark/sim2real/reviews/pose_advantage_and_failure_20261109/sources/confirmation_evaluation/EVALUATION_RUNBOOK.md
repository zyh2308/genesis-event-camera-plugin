# 独立32轨迹固定RAW C1评价（后继独立包）

本包包含与physical/events v1**原字节相同**的prepare_confirmation.py（SHA3a44394f510182bd3950cac011248692a11a67fc94ed7406e470536527d435f4），没有修改已锁生成包。

在32条CPU生成和全部三光照×两生成器重放完成后，根会话核实际event_completion.json与所有manifest/源字节、GPU4资源准入，再向现有受控队列提交：

```text
python -B -u evaluate_confirmation.py --registration N/PREPARATION_CONTRACT.json --registration-sha256 <实际合同SHA> --event-receipt-sha256 <实际event_completionSHA> --out N/advantage_confirmation/evaluation
```

N是20261109-pose-advantage-and-failure-v1。根资源parent供单worker最长3600秒边界，脚本也在每seed检查3600秒；只清理自身进程组，不创建或抢占GPU租约。完整设备guard沿原pds E.check_gpu/原GPU4及现有queue父进程，不能换卡绕过。

流程：

1. 同一已签操作合同绑定原科学prereg、core17文件、原库存和source；先核actual全部event artifacts，原adapter不改。
2. 各method×seed先在旧normal/half/eighth全部64窗重建原RAW C1完整pose，全部7metric的summary及by_rollout与既存native_global control分数按原rtol1e-6/atol1e-8检查；不是只比UV。
3. 对全32新trajectory、原k4..19和原几何eligibility做RAW C1前向；编码、global softargmax、独立center/depth、预测中心/尺度crop与rotation/rayframe、score沿原函数。两方法三光照的有序ID与K/T/keys/uv完全同一；所有被原几何规则排除的candidateID明确记录，无事件/预测依赖删窗。最多512窗/condition、9216完整pose case。
4. 旧baseline18cells与新18cells完全分开。新完整prediction JSONL保留UV/z/size/R/标签供独立CPU重算；GT仅评分和记录，不入网络/crop。每组center/depth/R状态SHA始末一致，source/事件/core前后重核。
5. primary为normal与half完整ADD-S success，两条件各three-seed均值正及97.5% trajectory-paired bootstrap CI下界>0，10000次，bootstrapseed20261109137。先每trajectory平均三seed差再抽32轨迹。最暗全表强制保留，次要指标不能替primary过关。只给独立**合成条件优势**；不改变原formal30+8，不允许real/long。

输出：predictions.jsonl；scores.json（18cell含完整by_rollout和window_ids）；eligibility.json；baseline_parity.json（18cell）；conditional_advantage.json；completion.json及源/模型/输入SHA。进程完成不等于primary通过；失败或无优势均不得换seed/光照/model、扫描读出或自动训练。

本地WSL五项科学CPU测试实际通过，覆盖配对seed与Bonferroni区间、单seed退步均值不能掩盖、缺cell/重复/缺rid拒绝、dark保留且不偷换为primary、原baseline指标/字段改变拒绝。它们没有执行实际32轨迹GPU评价；实际输入/模型必须根会话服务器准入。
