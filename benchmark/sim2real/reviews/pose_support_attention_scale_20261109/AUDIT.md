# 本轮零更新合成诊断：独立只读审查

核查对象为 `research_evidence_lite`，压缩包 SHA256：`443065ab65181fd5fb3c516b0c3b9bb1a6e2373136d05ac354188f9aa8491c9b`。本地65项工件的大小/SHA重新核对通过；manifest另列8项只保存在g1的原始大文件，不是缺失损坏。未SSH、未修改仓库/冻结源码，未执行新前向、训练或真实评分。

完整V2收据记录9216条观测、6144条hybrid及384条唯一支持统计；独立verifier记录576/576预定k12图通过、9216输入按288个batch32前向、模型/源码不变。注意：本地lite没有576张原map、全9216逐窗结果和尾部分解逐例明细，故这里不声称本地重放它们；本地审的是源代码、机器收据、384个hybrid轨迹/seed单元和192条尾部聚合。

## 最有证据的三点

### 1. 部分定位偏差确实包含“峰在物体区域内，区域外注意力仍牵引均值”的成分，但不同轨迹原因不同

下表仅为C1、RAW、独立center分支、最暗条件的固定k12窗口；每格是三个seed等权的有符号投影，单位px。foreground项与outside-tail项沿总error方向相加；负值表示抵消，不是负误差率。

| 方法/轨迹 | endpoint foreground投影 | endpoint outside-tail投影 | 峰在endpoint区域内 |
|---|---:|---:|---:|
| EG / 000 | 3.999 | 8.683 | 3/3 |
| EG / 031 | -2.654 | 41.680 | 3/3 |
| EG / 063 | 18.177 | 14.673 | 2/3 |
| EG / 095 | 4.921 | 1.670 | 3/3 |
| V2E / 031 | 4.358 | 7.401 | 3/3 |
| V2E / 063 | 18.416 | 21.966 | 2/3 |

031的EG线索尤其强：峰仍在物体区域内，但区域外尾部正投影41.680px，foreground项反而抵消2.654px。同窗endpoint-union近似下仍为-3.036/42.061px，方向稳定。063却是两方法foreground内部偏差本身都约18px，不能统一归因于尾部。095的尾部较小，union下V2E甚至为-0.036px，必须保留抵消和异质性。

EG四轨迹k12等权平均为foreground 6.111px、tail 16.676px；11/12个峰在endpoint区域内，这11个均有正tail投影。这个k12分解的总长度约22.787px，**不是**完整64窗平均定位EPE29.278px；031的k12约39.026px也不能与该轨迹的全时段均值混用。

这只是既存softmax分布的事后GT区域代数，不能表述为“截掉背景就能达到的精度”、已部署预测或物理噪声成因。原Gaussian监督本就可能对物体外网格赋权；lite未提供足够原始参考来判定这些尾部全部属于异常或无用信息。

### 2. 固定BAF的区域选择性与光照有关，并没有统一保住所有有用信息

以下是完整64窗支持统计：先按真实事件数量求和计算保留率，支持计数没有乘seed或模型；center EPE先轨迹等权、再三个seed等权。

| EG C1条件 | endpoint区域事件保留 | 区域外事件保留 | RAW→冻结模型BAF输入 center EPE，px |
|---|---:|---:|---:|
| 正常 | 88.16% | 86.56% | 8.215→9.898 |
| 半亮 | 86.84% | 78.53% | 9.993→11.054 |
| 最暗 | 76.84% | 55.17% | 29.278→20.850 |

最暗条件中，过滤对区域外事件抑制更强，attention endpoint mass由0.4601增至0.5114、峰命中率由75.52%增至86.46%；正常/半亮却退步。这支持“作用随条件变化”的观察，不能推出前景支持普遍缺失，或被删区域外事件全部是噪声。这里的事件归类只按其XY落在哪个端点区域，不能当作每事件物理来源；endpoint-union也只是两端并集，不是整窗可见性。

### 3. 深度的feature/context与attention存在耦合，恢复正常光路径没有跨轨迹一致获益

EG C1最暗深度MAE（cm），完整64窗/全部三seed：

| feature map + global context来源 | attention来源 | MAE |
|---|---|---:|
| 最暗 | 最暗 | 6.375 |
| 最暗 | 正常 | 5.673 |
| 正常 | 最暗 | 7.651 |
| 正常 | 正常 | 4.232 |

仅替换正常attention平均改善0.702cm，但12个轨迹×seed单元中只有8个改善；000三个seed均改善，031为[-0.253,+0.152,-2.899]cm，仍有明显反向。替换正常feature+context平均反而退步1.276cm，只有7/12改善；031三个seed均退步，分别约19.002、5.539、4.839cm。半亮仅换attention平均改善0.243cm（9/12），换正常feature+context平均退步0.831cm（4/12改善）。因此不能把一个分支叫作全局根因，也不能把正常光参考当作总是更优。

hybrid依赖同物理窗另一光照的事件参考，不是可部署推理；feature因子包含同来源global context，不能把feature、context与非线性head分别认成唯一因果。12个单元来自四条物理轨迹和三个模型seed，不是12个独立真实场景。本轮没有p值或总体显著性结论。

## 聚合与数值核对说明

- joint库存/语义与源码一致：576个配对的逐seed/rid/region单元、48个唯一支持单元、192行三seed范围表、288个within-light尺度单元、192个cross-light单元、384个hybrid单元。signed bias与窗口内population scatter分开；size/shape补偿是代数描述，不是因果分配。center分支的4608个`scale=None`是预期语义。
- 本地192条tail JSON与CSV逐数字比较最大差为0。三seed范围和可用数量保留，零mass/零EPE不以0冒充未定义投影。
- 从384个hybrid CSV轨迹/seed单元独立聚合32个producer均值时，发现最大 **3.0893493319084087e-9 m** 的逐值分歧，发生在EG C1、normal feature/eighth attention的depth MAE：CSV聚合0.07650899017850558，producer均值0.07650899326785492。它不影响上述展示精度，但不能宣称二者逐位等价或全部误差≤1e-10。
- 源码路径清楚说明两个口径：`sources_v2/probe_attention_scale.py:234`把`z_m=float(values['z'][i])`写成Python float；第237行先以NumPy FP32标量做`values['z'][i]-gt_z`再转float保存`signed_depth_error_m`。`sources_v2/attention_scale_metrics.py:139`则由JSON中的Python `z_m-gt_z_m`重新构造FP64误差；joint CSV采用预存signed字段。对lite实际包含的384条k12 hybrid逐行独立复核，全部384条预存signed值均匹配IEEE FP32减法，和Python差值的最大差为2.6113874840305584e-8m。其余5760条原始hybrid未在本地读取，不能冒称逐行复核。
- producer内部“复用原M重新计算其32均值”通过1e-10一致性检查，与上面跨两种误差口径的微差不矛盾。未修改任何原收据、科学源码或门槛。
- 有序配对依赖已核：冻结BAF2 `D.prepare:97–99`比较native `identity`的有序list；native85951第15–16行包含ID及K/T/keys/UV字节。再由support核全21帧物理/时间及端点mask，不是仅比较ID集合。

本轮仅完成零更新诊断，没有新训练、新真实指标、已通过的原30+8稳定性门槛或可采用的新方案。V1设备导入被guard拒绝的失败仍保留；V2完整运行和独立sentinel通过不能抹掉该历史。
