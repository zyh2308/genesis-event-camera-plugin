# 宽分布初始化失败：只读审查

本次只读检查现有benchmark与旧收尾，没有修改源、运行物理或读取新blind结果。

## 现有runner会怎样处理

`SparseEventSensor.__init__`先实际渲染一帧RGB-D并保存`initialization.json`。检测`valid=False`时设置`initialization_failed=True`，随后在启动事件worker、建立HDR HDF5和首次capture之前直接return，不raise。见`benchmark/mature_catch/sparse_event_sensor.py`约40–63行。

`run_original.py`约159–164行把这类回合记录为`success=False,steps=0,sim_time=0,failure_reason='single RGB initialization failed'`并继续。最终正常exit0；`finally env.close()`会令sensor写出空raw/packet/sample/window文件。没有worker/HDR文件是没有开始事件生成的正常结果。

所以任务执行器不能把这类失败换seed、GT初始化或单独忽略。其自然含义是固定机位和单帧初始化协议下的任务失败。

## 旧收尾不能直接照搬

旧`fast_paired_20261108/analyze_cadence.py`对V2E无条件读取`worker_start.json`（约59–61行），但初始化失败时worker从未启动；也在约73行直接访问`result['first_touch']`。旧early-failure result缺少`first_touch/initial/hand_contact_steps`且没有trace文件，因此可能被收尾误判为基础设施错误。

旧postprocessor还可能无条件要求窗口HDF/实际控制trace。不能通过删掉这类回合让旧审计“通过”。

## 给新fair收尾的最小处理建议

1. 最好在新benchmark runner的零步结果中保留已经计算好的实际`initial`，加上零接触、`first_touch=None`、零control poll、`trace=[]`和显式`failure_class='initialization_failure'`。原生物理终止标志可保持False；这是未开始控制的任务失败，不应捏造物理碰撞终止。
2. 结果记录仍必须包含同一原权重、源文件回执、首帧实际RGB-D预算与实际检测失败原因。
3. 此明确分支应核查`initialization.valid=False`、success=False、零实际控制/物理推进、零事件/packet/sample；不要求从未建立的worker/HDF/window产生正常路径文件。外部native V2E预检仍独立核验，不能把它虚构为该回合已经启动worker。
4. 正常初始化路径继续执行完整原始事件/因果/窗口/冻结源审计，不能因此全局放松要求。
5. 若同block同一首次RGB-D一方初始化失败、另一方成功，应核对实际图像/检测与reset是否一致；不静默补GT或选择更好初始化。

本地没有做新场景几何/FOV投影或真实初始化测试，不能断言高位样本必然不可见。原7°机位保持；若出现不可见样本，如实计数并保留，而不是看过结果后修改bank。
