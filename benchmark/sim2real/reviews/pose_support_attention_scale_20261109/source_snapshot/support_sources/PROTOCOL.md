# 前景支持、可见区域与注意力公共诊断模块

本模块只做冻结合成数据的事后诊断，不训练、不访问真实事件数据，不修改模型、冻结 simulator core、噪声、遮挡、事件、训练或评价阈值。GT 区域绝不进入网络输入、filter、裁剪或推理选点。RAW 与既有固定 BAF 的完整事件流都保留；区域划分只能解释空间支持，不能把区域内事件都叫目标信号、区域外事件都叫噪声。

## 实际来源与目标身份

已读取根会话实际取得的 `outputs/pose_support_attention_scale_20261108/source_facts`：原 scale 源 SHA `7d024570…`、native `85951e6d…`、fresh adapter `d1b16345…`，以及 actual HDF5 schema / 四条 VAL 的 target identity probe。本模块每 episode 自行重复两项证明，不硬编码 target slot/segID=1：

1. `target_pose[21]` 必须逐元素精确等于 `object_to_world[:,slot]`，且只匹配一个 slot；从 `object_segmentation_link_ids[slot]` 取得目标 ID。
2. 该 ID 在 `segmentation_link_idx` 的**全部21帧**像素数必须与 `target_visible_pixels` 完全一致。仅核验这些可见像素的 depth finite/positive，不按 depth 重新删改 mask。

四条实际轨迹目前唯一匹配 slot1/segID1；mini fixture 故意使用 slot3/segID37，防止代码误写成固定1或六对象并集。存在精确物理 instance mask，因此不使用 CAD 投影替代目标区域；若字段/身份/计数失效，保留失败窗口记录，不能悄悄改用粗 mask 声称成功。

## 窗口与区域

全部原始4个独立 VAL rollout×16窗口，sample ID=`rid/k:03d`，k=4…19，窗口 `(ts[k],ts[k+1]]`，label/endpoint 为k+1。normal / half / eighth 使用同物理窗口，事件支持表按 method+condition+sample ID 加 outer key，不带 seed，所有 C0/C1/三个 seed 共享。

- `startpoint`：k帧 renderer 可见实例 mask，精确于该帧。
- `endpoint`：k+1帧 renderer 可见实例 mask，精确于该帧，注意力与尺度主要参照。
- `endpoint_union_approx`：两端 mask 的并集。**不是连续 swept mask**，可能遗漏中间平移/旋转扫过的区域，不是整10ms每事件可见 GT。

两个精确端点 mask 也不能精确决定整窗口每个事件的物理来源。模块报告最大八角点投影位移和 anchor 位移，分别列端点/并集支持，避免把运动拖迹误判为背景噪声。不按事件数、可见面积或模型成败删除窗口。

`verify_matched_regions` 前置检查全部21帧 K、camera/target/allobject poses、keypoints、timestamps、segmentation、visible counts 的字节绑定，另核窗口时间与两端 mask。跨照度/生成器若物理内容改变即失败；空或不可用字段不当成“相等已验证”。

## 几何 sanity，不改变 renderer mask

额外计算投影3D bbox八角的2D凸包，明确名为 `projected_bbox_hull_approx`，不是精确 CAD silhouette。栅格用 OpenCV凸包、1/256 pixel定点顶点和LINE_8；另列1 pixel Chebyshev膨胀容差。报告 renderer可见mask与该 hull 的覆盖/外溢、两面积及比值，但**面积比不叫遮挡率**，投影只作 sanity、不替代 instance mask、不筛窗。

另列 GT anchor 在 renderer可见mask/投影凸包内的状态、可见mask质心、anchor−质心向量和像素距离。注意力落在物体内不等于 anchor 预测准确。可见mask bbox为整数像素单元、右/下边界max-exclusive；其几何尺度不是原CAD全顶点尺度标签。边界接触只叫 `visible_mask_touches_border`，不自动认定截断或遮挡。

## 346×260 →416→128→16

区域权重按原图表示的同一算子链：native bool mask转float32，OpenCV `INTER_LINEAR` 到416²，Torch CPU `area` 到128²，再 `area` 到实际16² attention网格。提供 fractional occupancy 和 native/grid 两种面积占比，不能以 nearest粗缩替代。

这是**空间位置权重代理**，不代表 feature receptive field只含前景。原模型有大感受野、GroupNorm及global context；FG attention mass不能直接归因背景或前景特征。统一提供 mass、grid面积占比和 enrichment=mass/area；uniform attention应为 enrichment≈1。空/不可用区域的质量值为null，不当作模型得0分。

点归属遵循原模型 grid `(i+.5)/16` 输出坐标：normalized UV×native宽高，按floor查原mask像素单元；UV=1不在frame内。它与fractional面积权重分列，也不解释为卷积RF中心。GT anchor误差由上层单独计算。`summarize_attention(...,pyramid=...)` 可缓存同一region的映射；缓存的gridshape/mask必须匹配。

## RAW/BAF 事件支持

每个完整 episode 只读一次 RAW NPY，核对原event manifest SHA。时间因果邻域计算与mask完全分离：8邻居排除自身、不分极性、严格 `<2000µs`、零初始化、拒绝事件也更新历史。原始float秒只乘1e6为float64微秒，不round；早期零初始化保留与真实历史支持分开统计。独立算法的keep必须逐事件等于已冻结BAF，差异即停止，不据此换阈值。统计期间不改变给网络的任何输入。

每个start/end/union空间区域及其补集分别报告 RAW/BAF计数、occupied pixels、占用率、每区域像素事件数和保留率。时距使用**过去RAW输入**，包含窗口之前的历史；给8邻居最近时距和same-pixel间隔分位数，以及无历史比例。整窗空间8邻居占用同时给“全窗口邻居”与“仅同region邻居”，前者可以含未来窗口内事件，明确不称causal支持；后者受region边界影响。量化统计只用有限历史，未定义分位数/零分母均为null。

空事件窗保留，已有面积上的occupancy=0是观测值；retention/邻域质量在0事件时为null。空mask也保留，区域分母0时ratio=null。失败mask以unavailable记录保留，不伪造可见区域。所有输出字典兼容`json.dumps(...,allow_nan=False)`。

## 公共 API

```text
regions_for_rows(bank, rows, strict=False) -> {sample_id: region_record}
verify_matched_regions(reference,current) -> passed/checks（不一致抛错）
region_geometry(region) -> 可见bbox/面积/尺度代理
region_pyramid(region) -> attention16 + encoder/input fractional weights
summarize_attention(region,attention16,pred_uv,peak_uv=None,pyramid=None)
window_support_for_rows(bank,rows,method,frozen_baf_module,regions=None)
    -> (support_by_sample_id, actual_event_source_records)
```

mini CPU测试使用人工HDF5/事件/attention，不读取项目bank、不跑模型、不初始化GPU、不训练。正式数据诊断由根会话调用并按完整库存/源SHA闭合；本模块不授权任何后续短训。
