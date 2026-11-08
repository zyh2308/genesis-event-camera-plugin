# MVSEC正式真实测试：独立研究核验

核验依据：服务器导出research_review_payload.json（56304bytes，SHA256 fdb9bc51a71e7d03f91b9695ab1fa98c790810d98aa9f06ac88accf39948c7bb）；导出时间2026-11-08T06:25:34.077078+00:00。本复核只读本地证据与固定源码，没有SSH、训练、真实预测或评分。

## 结论与唯一剩余启动条件

**四序列、15权重、真实输入及统计合同均具备冻结条件。** 可沿既有用户授权推进统一真实测试，不需要新增用户审批；不得改指标、窗口、生成器名单或检查点选择。

**当前还不能直接启动真实评分：先完成根会话已登记的15权重synthetic-only完整预测数值对照。** 全部通过后，根研究会话把本报告、统计合同及实现SHA纳入冻结记录，生成与最终protocol/manifest精确SHA绑定的授权，再允许队列启动。若对照失败，保存失败证据、停止真实评分，不提高阈值或改选权重。

这一条件是对既有训练/评价前向数值一致性的工程核验，不涉及真实数据调参或新研究方向。当前候选协议/manifest和完成收据均保持real_scores_authorized=false、real_scores_computed=false；prepared、complete_unscored或文件名“frozen”不能代替内容hash绑定的评分授权。

## 数据范围与隔离

|序列|登记窗口数|规则|
|---|---:|---|
|indoor_flying1|2165|保留原登记行|
|indoor_flying2|2624|保留原登记行|
|indoor_flying3|2911|保留原登记行|
|outdoor_day1|800|APS[9200,9600)与[10500,10900)|
|合计|8500|无按分数筛选|

服务器已直接对新注册前7700行与原注册进行列表exact equality断言，导出收据为true；我独立核对该断言、原注册SHA、四序列计数及完整800个室外APS起点。没有窗口选择、重切室内范围或删掉室外。当前导出是核验收据而非整份8500行原文，因此不能写成我在本地重新逐字节读取了全部窗口。

原注册SHA：46e0141bde84b8c3c3cc96615cec247f44bb8488ae33bdaf626d4f23d8dca02f。新注册SHA：6bc2318b9ea49a61c14d42675596088fa7972d2dd4149a3057de8cf68fe6e5a1。

室外新receipt于2026-11-08T06:18:00.539666+00:00完成：data/GT/标定ZIP三文件完整SHA、四个data数组与两个GT数组shape/dtype/SHA、ZIP CRC全部通过，使用原writer独占锁，且模型未加载、分数未计算。其精确绑定服务器left-only实际容器，未把full-native原件的容器SHA冒充投影SHA。坏GT被保留但不在登记输入中。原室内六文件完整SHA收据也全部通过。

来源应如实写为固定EI-Nexus镜像原生HDF5的无损左目投影及官方标定对照，不扩大声称已获得官方室外全文件的独立字节级认证。

## 15个检查点的公平绑定

5方法×3seed全矩阵，全部45epoch/8910更新；仅接受对应Stage9运行目录的synthetic_best.pt。selected/last、completion和训练数据manifest均绑定SHA，prepare已实际复核文件。每个seed的五个方法初始化相同；同方法三个seed的数据manifest相同，跨方法数据SHA不同符合event generator作为变量的设计。配方统一为51b7c723c9c4f1022483136f244d1970496b74b6886ad08b2f09be7ee1f221da。

本复核检查服务器导出绑定和固定prepare代码，不在本地重新加载15份大权重。正式evaluate还会再次检查权重及checkpoint内部的generator/seed/config/data/initial-state字段，禁止旧Stage4、last、预检或中断权重替代。

## 后端混合：训练反向不同，真实推理统一

|seed|五种方法共同的训练后端|数量|
|---|---|---:|
|0|original_pytorch|5|
|1|reference_forward_ordered_cuda_backward|5|
|2|original_pytorch|5|

不是某个生成器偷偷使用另一数学模型。seed1的ordered_adapter.forward直接调用冻结_MemoryEfficientSpatialCorrelation.forward，仅backward使用有序CUDA kernel；五方法共同切换，不与生成器身份混淆。被拒绝的官方全CUDA前向方案未进入这些检查点。种子之间的训练反向实现差异必须在复现材料披露，不能声称完整训练轨迹逐位相同。

既有评价adapter对所有15组统一使用纯PyTorch pad+stack相关性前向，inference_mode不反传，因此不需在评价时加载seed1的CUDA反向扩展。

**数值边界必须说清楚：**训练参考前向使用裁剪+sum，现有adapter使用pad+stack+sum，公式相同但归约布局不同。我的额外CPU原源码探针在FP32实际首尺度1×64×176×176得到最大绝对差4.76837e-6，非逐位相同；临时尝试的rtol1e-6/atol1e-7检查失败。该自选严格容差不是先前研究门槛，不把它换阈值后改记通过，也不能仅凭单算子微小差异推断最终像素误差。

根会话已在看真实分数前明确正式启动对照：同一固定15权重、两个固定合成输入、seed20261108、native260×346事件中心pad352、FP32/TF32off/eval/inference_mode；原训练reference与原adapter两种完整预测均需maxabs≤1e-3 px/interval且RMS≤1e-4。两种输入使用合法非负ON/OFF计数通道，绝不把重建的signed voxel错误送入STE。额外小tensorFP64只核数学公式。不得读取真实events/GT/分数。所有15模型都过，才可冻结并启动；GPU租约和≤10分钟任务由根会话管理，结果另附SHA。

## 模型与指标合同

固定STE-FlowNet eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6：14源码文件、205744bytes及整体tree SHA通过。固定adapter0247d1d3…、metrics06b4265b…、compat0f7290aa…均与本地锁定字节一致，依赖布局含scripts、相邻manifests及verifier。评价在任一前向前调用与训练相同的steflow_align_corners.patch_ste，模型内部warp保留align_corners=True。

- 输入仅公共真实events：原生260×346、每APS区间5等事件数Gaussian ON/OFF切片，中心pad到352，预测裁回原传感器，非任意缩放。
- 位移单位px_per_actual_APS_interval；事件窗(start,end]，端点及GT传播逐窗按注册SHA复核，不换算成固定30ms速度。
- 主ROI为[0,260,0,346]；共有mask为有限非零GT向量∩同一真实事件活跃像素。每模型使用同一注册event/GT/mask SHA，不能各挑有利区域。
- 副ROI固定：室内[2,258,45,301]，室外[2,192,45,301]，始终报告，不按分数挑ROI。
- AEE严格使用u/v两个分量；outlier严格为EPE>3px且EPE>0.05×|GT|，不是OR或单独≥3px。零流走同一mask。
- 空support保留登记行和分母统计、指标undefined而非零分，序列均值仅使用scored窗口并明确报告empty计数。

## 统计实现：在真实分数产生前锁定

既有runner先写各序列指标；本轮补上独立数学kernel，不修改原adapter/metrics。每method/seed四序列等权均值，再对3seed求mean与样本SD(ddof=1)。AEE主指标和outlier百分比都使用同样顺序，不能按有效像素数或窗口数加权。

4个EventGenesis−baseline比较全部报告。每序列先平均三组paired-seed差，四序列簇有放回抽四个，B10000、numpy.default_rng(seed20261108)、分位数method=linear。所有比较/指标使用同一抽样索引；种子全部留在簇内，不把窗口当独立样本。区间条件于已见3seed，且只有4簇；不加显著性星号，不掩盖四组比较，不把跨零与否包装成普遍泛化证据。

三个有意义数值夹具已通过：常量paired差的CI精确为常量、3seed sampleSD明确ddof1、序列均值不按像素/窗口加权；另验缺seed、非有限、零流不一致均拒绝全表。未读取任何真实分数。

|锁定文件|SHA256|
|---|---|
|aggregation_contract.json|d1f18ac76ad09fdbfc7b39635c9366d389a848a29f2d5e707f55f07c2d61662e|
|aggregation_kernel.py|160144b79ff01ef17728294f751625e28e908b307400fa80a5bb0366dd6100c7|

根会话冻结时在最终protocol加入aggregation_implementation={path,sha256,contract_sha256}并重算最终protocol/manifest SHA。上游candidate SHA仅用于证据核对，不得当成修改后的最终授权SHA。

## 允许的下一步

1. 完成并保留前述15模型synthetic-only完整预测对照，失败则停止不改门槛。
2. 通过后冻结既定协议、manifest、聚合代码/合同及预检SHA，生成与最终内容精确绑定的授权记录。队列同时检查预检passed与SHA，不能只看文件名。
3. 安全取得单卡UUID租约，以相同输入、源代码与全部15个已选权重完成统一四序列真实测试；禁止真实分数反馈选权重或调整配方。

这份审查批准研究合同的科学范围与冻结流程，**没有把未完成的模型数值对照写成通过，也没有在本轮开启真实评分。**

