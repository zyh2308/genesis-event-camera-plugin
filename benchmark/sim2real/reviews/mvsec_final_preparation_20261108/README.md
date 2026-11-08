# MVSEC四序列正式测试：输入修复、准备完成并启动固定队列

VPN恢复后已使用同一长SSH连接完成核验。原不匹配GT完整保留，正确GT及标定重传完成；三文件SHA、六required数组、ZIP CRC全部通过。原7700室内窗口保持逐行完全相同，新增原定800室外窗口，共8500；15个正式synthetic_best权重、配方/初始化/数据和模型源树均已核验。

独立研究审核及完整15权重×两合成输入的预测数值检查通过后，按用户既有授权冻结协议/manifest及统计实现，并绑定独立授权SHA；固定GPU5单卡15项真实测试队列已启动。启动快照见launch_evidence/launch_status.json；完整真实成绩仍待全部任务完成，当前没有发布最终排名或四序列成绩表。

- 运行根：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-mvsec-frozen-matrix-gpu5-v1/`。
- 控制器：`.../20261108-mvsec-frozen-control-v1/`，同一GPU5 UUID，15固定任务，18h总体/2h单任务，失败停止，结束释放；不重训或依分数调参。
- 完整范围：indoor_flying1/2/3、outdoor_day1；原室内2165/2624/2911窗口和室外固定800窗口。
- 模型输入：同一原始事件表示，260×346传感器、pad352；GT仅用于评价。主ROI为全原生画幅，副表保留固定legacy ROI。
- 指标：AEE(px/实际APS区间)、3px且5% outlier、零流AEE；不能将真实分数额外写成px/30ms。
- 统计：窗口→序列、四序列等权，3seed均值/样本SD；配对sequence-cluster bootstrap固定10000次、RNG20261108，四序列簇、条件于所见3seed，不把窗口当独立重复。kernel及contract在评分前锁定。

所有冻结内容、授权与数值收据的SHA链已验证并保存于launch_evidence/。RESEARCH_PROTOCOL_REVIEW.md为数值probe运行前的独立审核，后续完整probe收据证实其最后条件已通过。source_snapshot.zip只含锁定评价源及metadata，不含事件数据或权重。

训练后端披露：seed0/2为original_pytorch；seed1五方法一致使用reference-forward/ordered-CUDA-backward。真实推理15组统一为原PyTorch前向、不反传；不是逐位等同声明。完整模型probe已在固定合成输入确认误差低于预设0.001px最大绝对/0.0001px RMS界限，未用真实分数选模型。
