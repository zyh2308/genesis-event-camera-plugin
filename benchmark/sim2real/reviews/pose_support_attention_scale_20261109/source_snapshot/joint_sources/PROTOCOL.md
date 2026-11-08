# 冻结合成诊断的只读联合后处理

只读 completed **full** 的 V1 schema（V2仅设备修复）：`per_window.jsonl` 9,216行、`event_support.jsonl`384行、`raw_feature_attention_hybrids.jsonl`6,144行。拒绝baseline/partial、未知view、重复ID、缺配对、重复support和跨model/view改变的共享物理信息。SHA绑定 root提供的 full/completion，以及全部输入artifact。纯CPU/NumPy，不读权重、不模型forward、不用GPU、不训练、不真实评分，不产生自动训练决策。

实际M/S字段由已锁 producer 源码核对；统计依赖 `attention_scale_metrics.py` 原字节cc71eb…不改。本模块直接读已有field，不重设计指标。新脚本不修改旧源码、模型、原始事件、GTmask或推理规则。

## 配对、分层与N口径

- 9,216模型行形成4,608个同method/seed/arm/branch/light/rid/window的 RAW→BAF配对；endpoint与endpoint_union_approx分列，得576个rid×seed组，每组原16窗全保留。
- Δ定义固定为BAF−RAW。列中心EPE、FG attention mass/富集比、peak/UV命中、signed scale/logdepth/物理depth误差及绝对误差。只有双方均定义时计算该指标配对值，同时保存available/mismatch计数，不填0、不删整窗。
- support只从384行独立表聚合，按method/light/rid/region得到48组，**不乘模型seed、arm或branch**。计数求和；保留率为BAF事件数总和/RAW事件数总和；occupancy及稀疏性作窗口平均，报告有效mask/非空事件窗数。空间region内外不是物理事件来源，endpoint_union_approx不是连续swept可见性。
- CSV192行保留所有method/arm/branch/light/rid/region，给三模型seed均值/最小/最大及有效seed数，并通过`support_unique_key`标明support重复展示而非新样本。16相邻窗、3模型seed、不同光照和生成器都不冒称独立数据样本。真正物理轨迹是4条，物理时间窗64个；不做p值/置信区间。

## 尺度与四cell

depth分支单列288组within-light、192组normal→half/eighth matched drift；signed bias、MAE、population scatter/RMSE及size/shape误差乘积、中心化协方差。`e_logz=e_shape−e_size`与平方项恒等式核验；正误差乘积只表示代数抵消，不当作因果贡献。depth UV只是该分支内部attention中心，不能混作最终独立localizer或pose成功率。

完整2×2 RAW feature/context与attention交叉保留384个rid×seed cell表；每个cell保留16窗，不筛赢家、不选部署hybrid。feature/context来源联动由原producer定义。原32个三seed均值**原值保留**；用原M重算核对1e-10容差，明确这是统计重算，不是独立GPU/模型验证。

## 文件与紧凑导出

- `joint_summary.json`：完整576模型配对组、48 unique support组、尺度within/drift表、原32hybrid均值、分层null计数与N说明。
- `per_rid_three_seed_ranges.csv`：192行三seed范围。
- `all_hybrid_cells_by_rid_seed.csv`：384行全部四cell，含误差bias/scatter。
- 固定sentinel **k12**：`sentinel_k12_models.jsonl`576个全部模型payload；共享GT geometry/nominal reference/support按method/light/window去重为24份 `sentinel_k12_shared.json`；另留384条sentinel hybrid。固定k12在读取实际结果前登记，不后验挑窗口。
- 原始大表全部留服务器，仅导出 `raw_server_artifact_index.json`。输入artifact和原M源码前后重hash；原checkpoint/数据源验证引用producer收据，不伪称后处理重新读取了所有权重。completion记录新计算代码和输出SHA，失败不覆盖原结果。

输入没有保存完整attention16概率图，因此本导出不能让本地凭sentinel重新执行mask×attention；可以独立复算所存预测/尺度误差与联合聚合。原图/模型forward的验证仍来自已冻结producer/native parity，不额外声称可验证未导出的张量。

## 入口与测试

```text
python3 -B -m unittest -v test_joint_analysis
python3 -B analyze_joint.py --input-root <SUP/full> --completion-sha256 <实际完成收据SHA> --out <新joint目录>
```

CPU4项fixture包括完整9,216/384/6,144 schema端到端、SHA/库存、重复/错配拒绝、support不乘seed、补偿恒等式、四cell保留和k12库存；不读取项目模型、事件或真实数据。所有输出仅作本轮机制诊断，不授权训练、换阈值或挑最优模型。
