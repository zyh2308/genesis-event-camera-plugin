# 固定soft-focus读出：实际工件独立审计

结论：机制门槛 **27/36通过、9项失败**，不采用候选。不开展独立confirmation、长训、真实评分或sigma扫描。原formal30+8未重测、阈值未改。

核验输入为`outputs/pose_readout_control_20261109/research_evidence_lite`。ZIP SHA256：`3808c5f78e10ab159c7492603b8df87d36a95ba41eb46f215628549e7449c840`。本地按ARCHIVE_MANIFEST重新核对58/58项大小/SHA，全部一致。未SSH、未执行模型前向或训练，未修改仓库。

## 直接从原生scores重算

从`full/scores.json`的36个评分cell，逐method/seed/光照按固定criterion运算符重算全部36项，和`mechanism_advance_result.json`逐项一致。六行三seed均值的center、success、depth、AUC也与`result_summary.json`一致，差值小于1e-10。逐seed失败如下；success以百分比显示，门槛使用原0至1数值且不加epsilon。

| 方法 | seed | 光照 | 失败指标及要求 | RAW C1→候选 |
|---|---:|---|---|---:|
| EG | 0 | 最暗 | center严格降低 | 30.332972→34.816901 px |
| EG | 1 | 最暗 | center严格降低 | 27.188788→27.216215 px |
| EG | 1 | 最暗 | pose success严格提高 | 25.0000%→25.0000% |
| EG | 2 | 正常 | center不升 | 8.714751→9.191540 px |
| EG | 2 | 最暗 | center严格降低 | 30.311410→30.820908 px |
| V2E | 0 | 最暗 | pose success不降 | 28.1250%→26.5625% |
| V2E | 1 | 最暗 | center不升 | 17.613448→17.989475 px |
| V2E | 2 | 半亮 | pose success不降 | 48.4375%→45.3125% |
| V2E | 2 | 最暗 | center不升 | 13.598110→14.980845 px |

因此平均正常/半亮指标改善不等于稳定性通过；暗光EG均值pose提升，也不能冲销三个seed中心退步或seed1 pose没有严格提高。每个cell保留原64 ID，control/candidate ID序列一致，没有用删除窗口换取结果。

## 来源和独立复算层次

- registration固定RAW C1、sigma1/16、两gen/三seed/三光照/四轨迹k4–19、0更新，不使用GT读出或参数扫描。其SHA `cb0a02ee6f8a9f8e089d81ecc534a712f7a8ef5e7d4e924841a0fb8ef24dccca`与full收据一致。
- criterion SHA仍为预登记`db9803873fc4da10b983477e34baa10d1188f515b133905be590808abebdd6cd`，原36定义没有放宽。
- full completion SHA `77a06dbedd0c8c63fe46f4c39d7c4c83bf53681931cae1be1f3cee6da053c154`；实际完成1152个配对窗口、36评分cell、0更新，原18个cell控制parity、全部模型状态及来源哈希通过。depth/size共用冻结输出；两臂rotation均按实际各自中心重跑crop/rayframe。
- g1独立CPU AUDIT SHA `4b90db37306af96cb709e586d77a762423f0b9a2ad9d003479e661066dd5091d`，报告重新计算1152份概率、2304份pose、36cell×5个summary/rid层级×7指标=1260个标量；success精确一致、七项指标maxabs均0，invalid pose两臂均0。
- 本地逐值比较`independent_verification/scores_recomputed.json`与原`full/scores.json`的全部1260个标量，确实全部相等，并核对full/registration/criterion/独立AUDIT/独立verifier源码哈希链。
- 这不等于本地重新读取大map/paired_cases/vertices或重跑1152/2304原始几何。它们保留于g1，由`raw_server_artifact_index.json`和上述实际独立执行记录关联。独立AUDIT还明确没有重新载入框架模型状态，只核原模型字节及始末state收据。

## 解释边界

此前k12区域分解显示某些窗口有明显区域外尾部牵引；本次固定预测峰Gaussian权重并不等于精确删掉背景。峰在物体内不保证等于CAD anchor，区域内支持和完整pose的rotation裁剪也可能改变。当前负结果只表明这条固定读出未满足稳定性门槛，不能倒推唯一新根因，也不能证明所有局部读出或背景抑制均无效。

所有数值均是已见四条合成development轨迹；不能作为真实E-POSE泛化结果或独立确认。root没有因失败继续扫描、生成新确认集或启动长训。
