# 冻结 C0/C1 的事件支持、注意力与跨光照尺度诊断

V2只修复CPU依赖导入的设备边界。原V1源码/ZIP和实际GPU准入失败证据保留；该失败发生于模型前向和optimizer更新之前。冻结BAF模块顶层会清空本进程CVD，因此V2在导入前验证原GPU4分配、确认CUDA未初始化，导入后仍须未初始化，再恢复同一原分配；不猜设备、不修复已初始化context，随后仍执行原E.check_gpu的PID/UUID检查。prepare与forward另断言张量和模型设备一致，不隐式搬移修补。其余数据、forward、指标、范围和baseline/full门禁与V1不变。

本包只做零更新诊断，不修改仿真器、噪声、事件、模型、损失或旧结果；不打开真实评分，不提出新训练配方。原30+8门槛保持。主模型为原 `clean_control=C0`、`physical_supervised=C1`，两个生成器、三个 seed；独立中心和深度分支分别观察。后续同名 legacy 槽不混入。

数据完整固定为四条独立 VAL、每条16窗、normal/half/eighth 同物理时间和标签；RAW 与已锁的2ms BAF 都只是冻结模型的输入诊断，不代表采用过滤器。原 adapter/encoder 的416输入、原 CUDA area128 和原模型 forward 不变。采用实际 `7d024570…` 的 CenterScale，在 features 和 size-input 节点注册只读 hook；原 forward 只执行一次，记录后原样移除 hook。捕获的 local/context 必须逐位重现原 size-head 输入，softmax与原 uv 解码也须逐位一致。depth 内部 attention 与独立中心 attention 分开；中心分支未用于最终位姿的 size-head 输出不作为深度结果。

RAW 多字段须与已锁的旧导出逐窗对齐：两分支各自 UV、深度的 log_size/log_shape/anchor z。预先固定绝对误差各≤2e-6；UV为归一化坐标、两个log项无量纲、z为米。任何失败停止，保留新 failure，不调阈值。baseline 阶段仅两个生成器、seed0、两臂两分支、三光照各前8窗，共192条工程一致性记录；不是质量子集。full 必须绑定同一源码、数据契约和设备的成功 baseline 收据。

GT instance/depth 区域只在原 forward 后评价，不改变输入、裁剪、decoder或筛窗。公共 event_support 模块每个 H5 动态核验唯一 target actor 及21帧分割计数，不硬编码segID。endpoint/startpoint精确帧可见区域与 endpoint_union_approx 分列；区域中的事件不等于有目标来源GT的事件。映射沿原 LINEAR416→CPU AREA128→AREA16 作空间权重参考，不解释成卷积感受野纯前景。报告绝对注意力质量、区域面积和富集、峰/UV命中及CAD anchor误差。空事件窗保留；空区域质量为null而非伪造零。

原中心监督本来是 `softmax(-.5*distance²*16²)` 的软目标，可能合理地跨越物体边界。仅事后计算该原名义目标的区域质量、解码UV偏差和可表达范围，不能因mask内质量低就断定attention错误。depth原监督还包含size/shape/z，多目标作用亦须保留解释。

跨照度按相同模型与physical ID配对，记录local/context向量、cos/L2、signed size/shape/log-z及物理深度误差、bias/MAE/centered scatter和误差乘积；验证 `e_logz=e_shape−e_size`。零范数cos为null。轨迹内描述性统计后等权宏平均，不把窗口当独立统计样本；区域支持、运动和轮廓字段用于事后对应，相关性不作因果结论。

更强的机制检查仅用 RAW depth 分支：对于 normal→half/eighth，同窗 feature 来源与 attention 来源作2×2组合，context跟随feature来源，local和原size-head公式不变。两个对角 log_size/log_shape/z 必须复现原输出≤2e-6。两交叉需要同一场景在两个光照下的输入，因此是不可部署的机制诊断，不是生成器成绩、定位改进或真实时可用方案；不输入GT mask。保留全部组合和全部seed。共6144逐窗组合、96个逐seed宏单元和32行三seed均值表。

full输出9216条分支/输入逐窗观测、384条无重复seed的事件支持，以及6144条RAW路径组合；大文件留服务器并在完成收据记录SHA，不要求进入Git。模型状态与源码/权重/事件/H5字节前后SHA核验，0 optimizer更新，GT未改forward，运行上限3600秒。

CPU控制使用真实原类、随机定值输入和已知解析样例，不虚称服务器数据已通过。服务器运行测试时设 `POSE_NATIVE_SCALE_SOURCE` 指向实际SHA `7d0245708c02e13df316f81c89af82f4083844d19b01963fc23a172364607866` 的原文件。

```text
PY -B test_attention_scale.py
PY -B probe_attention_scale.py baseline --contract FACTOR/sources/contract.json --diagnostic-sources FACTOR/sources --baf-sources BAF/sources --support-sources SUPPORT/sources --support-manifest-sha256 SUPPORT_SHA --source-manifest-sha256 THIS_MANIFEST_SHA --out NEW_BASELINE
PY -B probe_attention_scale.py full [相同参数] --baseline-receipt NEW_BASELINE/completion.json --out NEW_FULL
```

默认GPU4，仅调用既有E.load_context的UUID/进程实检；根会话负责CPU预检、已有队列和资源准入，不新建SSH或抢占其他任务。CPU两线程、cv2单线程、TF32关闭。任一源、库存、标签、原生/对角parity或状态不可变检查失败即停止，不自动重跑、训练或真实测试。
