# Stage-1 独立研究验收：88b0d14

核验日期：2026-10-16（Asia/Tokyo）。提交：`88b0d14ae955f50185b4cd1e1f4aea51cd14212a`，分支`exp/public-sim2real-v1`。

**结论：第一阶段部分通过。重建与光流的单步工程smoke有实物证据；重建的时间一致性损失未被本次smoke触发，完整目标尚未验收；位姿路线未完成。可以继续并行做合成pilot和数据接入，不满足直接开启formal真实测试/正式比较的条件。**

本核验独立读取了GitHub分支/提交、WSL checkout及g1服务器文件，复算哈希、检查样例和checkpoint的优化器状态，并对相关算子做了独立CPU数值/梯度检查。没有重新运行GPU训练、重放Genesis渲染或执行真实数据评分。

## 1. 已核实的交付事实

| 检查项 | 核验结果 |
|---|---|
| GitHub发布 | 分支ref确实指向88b0d14；不是仅存在于本地 |
| WSL checkout | `/home/科研/Eventbased_WAM/repositories/gec-sim2real`，HEAD一致，核验开始时工作区干净 |
| g1 checkout | `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real`，HEAD一致，工作区干净 |
| frozen core | `genesis_event_plugin/`相对e79e656无差异；HDR integration仅README改变，实现文件无差异 |
| 归档证据 | `artifact_index.json`中137个文件，大小与SHA256全部一致 |
| 下游第三方源码 | WSL中5个repo的Git HEAD均匹配登记的pin且无tracked修改；WSL与g1的686个源文件都匹配清单 |
| V2E | g1上107个文件匹配source pin；SuperSloMo权重hash匹配登记值 |
| g1数据 | 清单中15个数据/转换清单文件全部存在、大小及SHA256一致 |
| g1 checkpoint | 4个文件的SHA256匹配，安全CPU加载成功，参数均有限；所有已创建optimizer state的step都是1，seed为71，存储的loss与运行result一致 |
| 退出状态 | 最终E2VID、STE、V2E配对运行exit code为0；失败尝试保留；轨迹生成没有exit-code记录，报告已如实说明 |

这些证据相互一致，支持“确实保存了两组下游的一次优化更新与合成验证产物”。哈希及优化器状态不是完整重新训练的替代，本核验不宣称独立复现了截图中的全部GPU数值。

## 2. 样例配对、标签和输入

两个rollout各0.1 s、11个160×128 radiance endpoints、10个10 ms flow区间。EventGenesis事件数为4068/4832；接受的25μs V2E为2084/2571。时间有序，转换后的区间事件数守恒。

STE两组的`gray_prev`、`gray_next`、`flow_gt`、`flow_valid`及padding信息逐元素相同，事件表示同为`[10,2,256,256,5]`。

E2VID两组的目标图像与shared bank右端图像完全相同；按实际loader的dt恢复后，flow与shared物理位移的最大绝对误差约`5.96e-8 px`。**这证明当前两段小样例中的标签配对成立，不代表所有稀疏/空事件情况都成立。**

检查的模型入口只读取合成事件张量，合成图像和flow用于训练loss/验证；没有发现depth/segmentation/pose作为额外模型输入。真实HQF/MVSEC/E-POSE没有出现在所检查runner的训练/验证输入中。`real_data_accessed=false`本身是写入的声明字段，隔离结论依据实际代码及路径，而不是该字段单独提供访问审计。

## 3. 必须补验的问题

### R1：E2VID时间一致性损失在本次smoke中没有计算

`run_e2vid_stage0_smoke.py:124`使用`sequence_length=2`，第166行配置`temporal_consistency_loss(L0=2)`。官方`Trainer.forward_sequence`以零基索引枚举时间步；官方loss只在`i>=L0`时返回数值。因此本次时间步只有0和1，时间一致性项始终返回None，总loss实际只有LPIPS项。

研究会话抽取并执行了准确pin中的loss class，用独立计数stub确认：长度2时底层时间loss调用0次；长度3时调用1次。不是根据日志缺失猜测。

影响：重建的forward、LPIPS/backward、optimizer update和合成validation可以验收，但“LPIPS+temporal consistency完整目标跑通”不成立。

下一步要求：延长小型合成序列，使时间一致性项实际执行；保存各loss分量、有效调用/样本数量及正常反传的证据，核对flow方向。具体序列长度和实现方式由工程会话决定，无需重做整个框架。

证据：[runner](https://github.com/zyh2308/genesis-event-camera-plugin/blob/88b0d14ae955f50185b4cd1e1f4aea51cd14212a/benchmark/sim2real/scripts/run_e2vid_stage0_smoke.py#L123)、[官方loss](https://github.com/TimoStoff/event_cnn_minimal/blob/78aa471a66593f55e349303563997297fbc816e8/model/loss.py#L126)。

### R2：E2VID flow标签恢复目前依赖事件是否足够，扩大数据前需修正契约

`prepare_e2vid_stage0.py:127–134`在事件数不足3或observed event span为0时，将存储的flow设为0。正常区间用物理位移除以observed span，再由stock loader乘回dt，因此当前样例的恢复正确；但稀疏、空或同时间戳区间会把物理GT变成0。两个generator的事件数量不同，就可能得到不同的有效标签。

本次样例没有发现这种标签不一致，所以不推翻已验证的配对数据；这是从现有执行分支直接确认的扩展边界问题。

下一步要求：物理位移及共同validity必须由共享物理时钟/标签决定，不由某generator事件数量或observed span决定；补测空窗、极稀疏窗及tied timestamps。修正loader或存储适配方式由工程会话选择。不要通过丢弃某组稀疏样本绕过。

证据：[转换脚本](https://github.com/zyh2308/genesis-event-camera-plugin/blob/88b0d14ae955f50185b4cd1e1f4aea51cd14212a/benchmark/sim2real/scripts/prepare_e2vid_stage0.py#L123)。

### R3：光流相关算子的核验范围

替代实现的`F.unfold`相关算子，与独立显式81位移参考在CPU float64上的forward最大误差为0；对两个输入的梯度最大误差约`2.0e-15`和`3.6e-15`。其数学操作通过本次核验。

本次没有安装或直接对照原生CUDA extension，也没有验证所有尺寸、精度和显存行为。工程会话可以保留经过明确测试的替代实现，或使用原生extension，不要求为了“原生”二字额外重写。正式配置需锁定所选实现，覆盖目标尺寸/精度并实测吞吐和有效batch策略。

### R4：样例归档应补显式K及完整运行完成状态

shared HDF5保存了depth、seg、camera/object pose及flow，但没有显式保存生成时实际使用的camera intrinsics矩阵。config有FOV和分辨率，但完整GT复算不应依靠重新猜K。下一批语料请保存实际K、坐标/时间单位及geometry contract；同时统一记录exit code与开始/结束时间。缺失历史exit code不要事后编造。

## 4. 三条路线的验收状态

| 路线 | 状态 | 能支持的结论 |
|---|---|---|
| E2VID+ → HQF | **条件通过** | 合成loader、递归网络、LPIPS单步优化与合成验证有证据；需R1/R2才能声称完整训练目标可用 |
| STE-FlowNet → MVSEC | **通过单步smoke** | 同标签配对输入、原生网络/photometric优化和synthetic EPE计算有证据；相关算子数学检查通过，尚无正式训练/真实测试 |
| YOLO-6D → E-POSE | **未完成** | 只有源码/公开入口审计；没有raw sample契约、pose loader、forward或training证据 |

当前合成validation区间上的零光流EPE为`0.488651 px/interval`，低于两个单步网络的`0.708796 / 0.654945`。这很正常：一次更新既不证明收敛，也不足以比较generator优劣。截图已明确不排名，这一表述是正确的。

真实数据集尚未纳入验证产物，当前没有public-real Sim-to-Real结果；三个方向的论文证据均未完成。样例radiance最大值约0.209，也不是HDR/clipping优势测试，因此不应将该smoke用于支持HDR保真claim。

## 5. 下一阶段建议：继续并行，不启动formal真实排名

- **重建**：补齐R1/R2，使用小规模多步合成pilot证明完整loss与标签契约；同时准备HQF数据manifest和固定评测接口。
- **光流**：使用多个物理rollout做短合成训练/验证，覆盖主协议时间窗及dt=4接口；确定相关算子与实际batch/显存策略，同时准备MVSEC真实文件、GT及排除规则清单。
- **位姿**：优先获得至少一个raw event/pose/CAD/K样本，写清全帧event-only输入和坐标/单位契约；能接成熟模型则做最小合成闭环，不能则提交明确的替代建议，不无限耗在旧RGB loader上。
- **共同交付**：只需更新同一仓库的状态、配置和证据，不要求用户搬文件。formal前再冻结配置、数据manifest和评价版本；本次不据单步loss调simulator。

工程问题自行处理；仅涉及模型/数据替换、任务定义变化、frozen core或真实测试隔离等方向性问题再交研究会话决策。

## 6. 本次独立证据

- `local_verification.json`：137归档文件、共享样例、标签恢复、temporal实际调用、相关算子forward/backward检查。
- `source_git_pins.json`：5个本地官方checkout的HEAD与干净状态。
- `remote_audit_stdout.json`：g1仓库、686源文件、15数据文件、4checkpoint及optimizer step检查。
- `remote_extra.json`：V2E 107源文件、SuperSloMo权重与HDR runtime manifest存在性。
- `audit_local.py`、`audit_remote_client.py`、`audit_remote_extra.py`：独立核验方法记录；只做read-only审查及CPU检查。

这些研究验收材料另外保存到共享WSL仓库的`benchmark/sim2real/reviews/stage1_88b0d14/`供工程会话读取。本次不修改实现，不重新训练，不对远端Git仓库执行push。被审查提交仍为88b0d14；新增review材料不是原交付的一部分。
