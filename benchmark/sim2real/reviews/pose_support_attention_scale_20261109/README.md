# 位姿：事件支持、注意力与尺度诊断

本轮只有冻结模型的合成诊断：固定4条轨迹、64个物理窗、三光照、两生成器、C0/C1与全部3个seed；9,216模型记录、6,144非部署hybrid、384份去重事件支持。**0次训练更新，无新增真实E-POSE评分，也不宣称通过原稳定性门槛。**下述模型比较聚焦当前C1。

## 1. 过滤有条件收益，不能作为统一修复

| 方法 | 光照 | 端点目标区域/区域外事件保留率 | 中心误差 RAW→BAF（px） |
|---|---|---:|---:|
| EventGenesis | 正常 | 88.16% / 86.56% | 8.21→9.90 |
| EventGenesis | 半亮 | 86.84% / 78.53% | 9.99→11.05 |
| EventGenesis | 最暗 | 76.84% / 55.17% | 29.28→20.85 |
| V2E | 正常 | 73.74% / 68.36% | 14.61→14.84 |
| V2E | 半亮 | 78.08% / 72.65% | 15.78→15.06 |
| V2E | 最暗 | 80.90% / 79.76% | 15.75→16.03 |

保留率是四轨迹事件计数池化的比值，**不是窗口比率或轨迹比率的平均**；中心误差先平均每轨迹16窗，再对轨迹与seed等权。EventGenesis在正常/半亮定位更好，最暗较差；BAF减轻其暗光误差，却损害正常/半亮，且最暗000轨迹仍退步。它同时删除约23%的最暗目标区域事件，不能直接叫“仅去噪”。端点mask只在该帧精确；窗内约1.8–3.3px运动会造成区域归属误差，区域外也可能是有效场景事件。并集近似的分列结果保留，不替换为真正连续可见性。

## 2. 峰落在物体上，平均位置仍可被尾部拉偏

预定k12、C1独立中心、最暗RAW中，EventGenesis有11/12个“seed×轨迹”组合的峰落在端点目标区域，这11个组合仍有正向区域外尾部投影。EventGenesis四轨迹的尾部投影三seed均值依次约 **8.68、41.68、14.67、1.67px**；峰命中不足以说明anchor定位准确。

这是已核验attention map的加权平均代数分解，不是删除背景后的模型成绩，不证明背景事件是唯一根因；也不能把k12推广成全部64窗概率图的独立核验。feature感受野、GroupNorm及global context仍可能跨越区域。

## 3. 暗光尺度漂移与定位相伴，单换attention解释不完

EventGenesis半亮的log-shape/log-size平均偏差为−0.0191/−0.0149，相减后log-depth偏差仅−0.0043；最暗变为−0.0534/+0.0093，深度偏差累积至−0.0627。两分量的偏差与配对关系都重要，不能只看某项MAE较低就断言深度更准。

固定feature/context，仅换配对正常光attention，EventGenesis深度MAE半亮 **3.594→3.350cm**、最暗 **6.375→5.673cm**；分别9/12、8/12组合改善，但最暗组合变化可从恶化2.90cm到改善3.48cm。V2E半亮反而3.641→3.731cm。改换正常光feature/context也有明显场景差异，未得到统一单路径解释。

正常光不是GT或更优oracle；这64窗的正常光C1深度MAE为EventGenesis4.232cm、V2E4.694cm，不能与旧clean384混用。hybrid借用了另一光照输入，只是机制探针，不是可部署方法或完整6DoF成功率。

## 可信范围与后续

V1中CPU helper清空GPU变量，原设备guard正确拦截；V2仅修复设备路由，实际CPU16项、baseline/full及4,608条RAW native parity通过。预定576张k12 map另经独立核验；tail是随后数学分析。原始173MB表留服务器，紧凑表和SHA链已归档；统计重算不冒称独立GPU验证。

当前同时观察到输入支持变化、空间权重尾部及尺度偏差三类现象，主导训练原因尚未确认。下一步仅针对这些固定场景做有明确假设的合成机制验证，先闭合逐轨迹/三seed证据，不继续盲扫loss、加步数、开长训或真实评分。16相邻窗及3模型seed不当作独立场景样本，不给p值。

依据：[独立审计](AUDIT.md)、[完整逐轨迹三seed表](evidence/joint_results/per_rid_three_seed_ranges.csv)、[四cell逐轨迹表](evidence/joint_results/all_hybrid_cells_by_rid_seed.csv) 与 [全部紧凑汇总](research_evidence_lite.zip)、[k12尾部192行表](evidence/attention_tail/three_seed_per_rid.csv)。

归档说明：紧凑ZIP包含65份已校验文件，内部原始manifest另外列出8份只保留g1的大文件。完整研究副本位于 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-pose-support-attention-scale-v1/research_evidence_full.zip`（SHA256 `8cee42248d2b713d83733421110863df40326ca646912ece814f91b6342c5840`）；173,373,709 bytes原逐窗JSONL仍在同根 `full_v2/per_window.jsonl`，其SHA见完成收据。实际CPU日志及失败/通过收据分别见 `actual_cpu_test_logs.json`、`actual_cpu_and_queue_checks.json` 和 `evidence/`；39份新诊断源码及fixture原字节存于 `source_snapshot/`。独立CSV重聚合与producer原口径有最大约3.09e-9 m数值差，细节和实际逐例核验范围保留在AUDIT，不宣称逐位等价。
