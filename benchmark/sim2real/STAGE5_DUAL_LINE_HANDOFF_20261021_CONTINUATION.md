# Stage 5 双线执行交接（续）— 2026-10-21

本页续接前一份 [Stage 5 双线交接](STAGE5_DUAL_LINE_HANDOFF_20261021.md)，保留旧页作历史快照。最新综合观测为 **2026-10-20 21:00:30 UTC / 2026-10-21 06:00:30 JST**：g1在20:57:04重读重建summary，两组均200 epoch、12,000/12,000 updates、exit 0；20:59:39光流bank仍是3/8，`s5q_01_val`到80/201 endpoints，runner/worker `2229637/2229727`存活；20:59:56 GPU0本任务9,360 MiB、26%，GPU2空闲，MVSEC候选两权重文件大小与SHA均复核正确，g1可用空间约12 TiB。g1复核的EventGenesis/V2E synthetic-best checkpoint分别为580,038,858 B、SHA256 `a4e6c7e1d10a4d76562a046c364a5e17de6452d5b648fc8ccbaa3bd43c824a8e`和`11041863685b417306f534b8e12a1177b1337fc2818d255ec7b76fa6e20cd474`，与候选绑定一致。WSL在21:00:30读取下载清单，六个官方文件hash验证完成，正在下载`indoor_flying3_gt.hdf5`，之后仍有outdoor_day1两文件和室外标定。实际MVSEC `indoor_flying1` APS index 21窗口审计通过：194个事件、31.859 ms、GT位移单位正确，未加载模型或计分。精确GT端点传播适配已修复，官方helper对照、8项适配和合成指标入口检查通过。位姿按用户要求暂停，未探测E-POSE；训练/选点仍为synthetic-only，真实评分关闭，frozen core `e79e656ac00f7847b98db9e12bf43b093f3afff1`未改。机器快照见[21:00:30状态JSON](evidence/S5/20261021-recon-flow-localization/live_observation_20261020T210030Z.json)，早前状态仍保留。

## 当前状态

| 线路 | 已实际完成 | 当前运行/未完成 | 下一步 |
|---|---|---|---|
| **重建 / E2VID+** | 修正 TC 后 seed 0 两组各完成 200 epoch / 12,000 updates，run summary 为 `complete`，best/final 权重及完整 JSONL 曲线都在 g1。HQF 14/14 文件已转换并校验；HQF 主测、HS-ERGB 扩展、LPIPS-Alex 主指标和序列等权规则已按研究 review 登记，real score 关闭。两组候选模型都已在固定合成物理样本上完成事件输入→递归推理→200张输出→指标→独立 manifest/hash 复核。 | 未在 HQF 上运行候选 checkpoint 或计算真实分数。 | 归档验证回执和完整训练曲线摘要；保持真实评分门关闭。 |
| **光流 / STE-FlowNet dt1** | 既有固定训练窗GT监督记忆诊断两组各200 updates均通过；一个新v3共享场景及光度warp审计通过。MVSEC协议候选已修正checkpoint选择来源并新增固定样本真实文件窗口登记。合成指标入口synthetic fixture与实际STE GT传播对照共8项通过。 | g1 20:59:39 UTC时，八rollout bank 3/8完成，`s5q_01_val` 20/201；当前活跃进程`2229637/2229727`在GPU0，其他用户作业未触碰。V2E配对bank及新版原自监督短训尚未开始。MVSEC真实文件WSL串行下载六项完成，正在下载`indoor_flying3_gt.hdf5`。 | 等八rollout物理bank完成；资源允许时生成配对V2E事件、统计active-union覆盖；随后跑固定预算STE自监督短诊断与独立rollout验证。MVSEC完整文件传至g1后生成全窗口hash绑定manifest，研究冻结前不评分。 |
| **位姿** | 无新增动作，按用户要求暂缓。 | E-POSE访问与下载探测暂停。 | 等用户通知后恢复。 |

## 重建：既有完整训练和新合成端到端验证

### 已完成训练（历史权重保留）

g1 根目录：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0/`。EventGenesis 为 12,000/12,000 updates、200 epochs，best epoch 170，synthetic LPIPS-VGG `0.04116681637`；synthetic-best SHA256 `506e81bd558f1bcca2b67d34d4c1004836fc7cce65393084eb52a378dfd554bc`，final SHA256 `fe7eacf5865f486b7355e4f60202368de34b56d3dbcde1412ee67bfbfc29cbaf`，退出码为 0。V2E 同预算，best epoch 195、synthetic LPIPS-VGG `0.08367991168`；best SHA256 `598b1e7837b4b02b50a7f390889c7737ed797840043a3f23121f54dc99d050d8`，final SHA256 `af861ca93da680fd83b58c6644376e99ba2c9007f379b92a29221031159726db`，退出码为 0。两组初始模型 SHA256 相同：`48426d94f4159592155eda873d4e71e2f06439c10eade990de5979231ed07a0f`。

完整 `training_metrics.jsonl` 分别保留在 g1 两个 run 子目录，刷新时大小10,245,043 / 10,132,565 bytes，SHA256 分别为 `c8116cf82e2587de19bf854b0767ae6289f32d858a3f56e2ed509b7874f0ffad` 和 `39bbf9ff62be766567977dff7751177d39a626db1eec87f5543f625ebbc81d2d`；每份12,040行。两组均正常退出（退出码0）。checkpoint、summary、曲线大小与哈希已收录在[重建训练产物清单](evidence/S5/20261021-recon-flow-localization/reconstruction_training_artifact_manifest.json)。训练结果只用于记录合成验证选点，不代表真实重建表现。

### synthetic fixture 事件输入到指标入口

固定 bank 是 v3 smoke rollout `s5q_02_train`，g1 路径 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-stage5-flow-coverage-v3-smoke-r3/shared_trajectory.h5`，SHA256 `5271c7e683deb09437985c53af071d05a1fe2c3b41420acaa2f1dada3a8b4a1f`。对每个模型均只输入 events：t0 只重置递归状态，随后精确推理200个 `(t_{i-1}, t_i]` 窗口；不把参考图像像素传给模型，不做真实数据评分。模型配置显式来自仓库 `configs/stage4_corrected_v2_flowdt1_seed0.json`，SHA256 `6fc6d28673689868fa18b76be4c1f4af1d84da0cad287cbf596801cb7ca6610d`，避免训练 checkpoint 缺少 `model_args` 时误加载 EVREAL 公共权重。

EventGenesis 完成200帧原生 `260×346` 推理、一次递归 reset、float32/PNG 保存及事件端点复核。旧 PerceptualSimilarity 依赖调用已移除的 `skimage.measure.compare_ssim` 导致首次组合入口失败；在 benchmark 兼容层映射到现代 `skimage.metrics.structural_similarity` 后，**同一固定推理产物**经指标脚本复跑通过。指标输出状态明确为 `synthetic_plumbing_pass_not_model_quality_or_generator_result`；LPIPS-Alex `0.36155`、MSE `0.03611`、PSNR `14.4236 dB`、SSIM `0.61583` 只验证接口和数值域，不作模型质量或生成器结论。

V2E 同样已产出200张 float32 重建和指标文件（run：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-stage5-recon-synth-fixture-v2e-v1/`；驱动退出码0）。两组随后均通过独立 verifier：验证器核对 checkpoint SHA、共享 bank hash、输入事件和目标帧文件 hash、图像与 float32 输出、200个有序评测行，以及 fixture/inference/metrics 的绑定。最终回执 SHA256：EventGenesis `6c5be8f5eb78c138623b9a452147c9d86ccbf1d0154e5e5ebc325a64569654ae`；V2E `8841ca8d60a9d2200479d95f5b469ea8ce51ef385dd788ad8c056d9b1437a12d`。首次 EventGenesis 组合脚本曾因旧 `compare_ssim` 导入退出1，兼容补丁后同一推理产物完成指标复跑。最初验证器把两组都误标成“兼容复跑后通过”；现已保留原文件，另生成用词修正的 `verification_v2.json`，明确只有 EventGenesis 需要兼容复跑，V2E 原评测驱动一次退出0。逐组记录见 [pair summary](evidence/S5/20261021-recon-flow-localization/reconstruction_fixture/reconstruction_fixture_pair_summary.json) 与 per-arm `verification_v2.json`。指标只属入口接线样例，不比较两模型质量。

本轮真实评测候选为 HQF 全14序列。原始包总大小2,902,695,576 bytes；转换及 raw/converted 文件 manifest 校验已完成，manifest SHA256 `76e19db296be055135fa7446a4da68b704a1c573dd7f2b8f06ce599ddc1a3add`，状态 `ready_unscored`。已登记原生分辨率 padding/unpadding、序列级递归状态重置、图像端点 `(previous,current]` 事件窗、source index 528 的固定坏时间戳排除规则、LPIPS-Alex/SSIM/PSNR/MSE、帧均值后14序列等权平均和不做亮度/直方图拟合。检查发现 HQF protocol v3 中模型参数来源文件的 SHA 曾误复制为另一个同参数配置的 hash；按实际 `stage4_corrected_v2_flowdt1_seed0.json` 更新为 `6fc6d286…6610d`，并增加回归检查防止再次错配。HQF 主测 / HS-ERGB 扩展是研究 review 已定角色，不再作为阻塞；真实评分授权和开关均仍为 false。

## 光流：网络记忆证据与运动可观测性

### 实际 STE 网络固定窗拟合

此前对 `s2_00/dt1_144` 的独立 GT 监督诊断两组各200 optimizer updates，关闭随机增强和样本轮换，模型 seed 0 配对初始化。样本由物理 bank ID 预先固定，数据清单 SHA256 为 `1f9aff5164c71301a7406c9981c31ef0d5df381df81641cf2a1d397e8df246b1`。零流基线是 full `0.54151`、共同事件活跃区 `0.32972` px/30ms。两个实际 STE 网络都有真实参数变化，且 EPE 显著下降：

| arm | optimizer updates | full EPE @0 → @100 → @200 | common-active EPE @0 → @100 → @200 | parameter ΔL2 @200 |
|---|---:|---:|---:|---:|
| EventGenesis | 200 | `0.56700 → 0.10373 → 0.06259` | `1.68166 → 0.16163 → 0.08011` | `100.608` |
| V2E | 200 | `0.54841 → 0.12264 → 0.04702` | `0.97759 → 0.14988 → 0.07829` | `108.777` |

所以两组实际 STE 网络和优化路径都能拟合一个已登记样本，训练窗口误差也低于同一支持上的零流基线；这只证明单样本记忆能力，不证明独立 rollout 泛化，也不涉及原自监督正式目标。checkpoint 边界恢复、恢复后普通 `eval()` 推理复现、权重哈希和逐100步日志见 [`flow_paired_fixed_sample_fit_summary.json`](evidence/S5/20261021-recon-flow-localization/flow_paired_fixed_sample_fit_summary.json) 及 per-arm 证据目录。

### 已完成的 synthetic-best 输入扰动诊断

复用了2026-10-19已在同一 corrected-v2合成验证集完成的固定窗口结果；checkpoint SHA 与45 epoch seed0 synthetic-best相符：EventGenesis `a4e6c7e1…c824a8e`、V2E `11041863…cd474`。样本是四个运动层各2个验证 rollout 的 `dt1_000`，共8窗。扰动发生在STE实际读取的 **5-bin signed event tensor** 上，不是raw event list：正常、全零、每极性内部反转时间bin。全零和反转张量均实际改变；时间bin反转保留每极性空间支持和事件计数。两模型、所有变体共用从正常两组输入的事件活跃并集冻结出的掩码，覆盖4,122/718,212个有效像素（0.574%）。

全有效区域 EPE：EventGenesis normal `0.10873`、zero `0.10614`、reverse `0.11169`；V2E normal `0.10604`、zero `0.10614`、reverse `0.10604`，共同零流 `0.10608` px/30ms。固定 active-union 区：EventGenesis normal `0.16211`、zero `0.09307`、reverse `0.22439`；V2E normal `0.09128`、zero `0.09321`、reverse `0.09156`，共同零流 `0.09308`。扰动影响了模型输出，EventGenesis反转响应明显；V2E响应较小。**扰动敏感不等于运动预测正确**：两组全图几乎贴近零流；EventGenesis active区更差，V2E仅略优于零流且改善很小。8窗共记录17,916个 EventGenesis、1,037个 V2E tensor event counts；没有据此筛样本或调整阈值。这是既有synthetic-best诊断，不追溯改变旧checkpoint选择。

四层验证窗口的 active-union EPE（每层2窗样本宏平均，px/30ms）：

| 既有运动层 | EventGenesis normal / zero / reversed | V2E normal / zero / reversed |
|---|---:|---:|
| 相机运动、静态场景 | `0.13546 / 0.09054 / 0.20410` | `0.09336 / 0.09054 / 0.09358` |
| 静态相机、独立刚体运动 | `0.16325 / 0.01473 / 0.17374` | `0.02690 / 0.01473 / 0.02694` |
| 相机与刚体合并运动 | `0.17032 / 0.13454 / 0.26180` | `0.11565 / 0.13454 / 0.11664` |
| 合并运动与遮挡 | `0.17940 / 0.13251 / 0.25793` | `0.12923 / 0.13251 / 0.12909` |

这显示 EventGenesis 输出对时间反转更敏感但 active区域 EPE 都高于零流；V2E normal 与零流接近，只有部分层略优。不同层 GT 幅度不相同，不能单凭层间 EPE 推断事件表达好坏。

机器证据：[`flow_input_perturbation_results.json`](evidence/S5/20261019-parallel-diagnostics/flow_input_perturbation_results.json)，SHA256 `7e627c6d583322ce71c8df9ec7af53786a891828b460359b9c368dc7e0c9c2ed`；扰动定义和边界见同目录 `README.md`。

### 新 v3 smoke：共同事件支持与真值 warp

固定测试区在运行前登记：active = 两组正常输入 Gaussian-count 表示非零像素并集；motion = 共享 GT 位移至少 `0.5 px/30ms`；texture = linearized display luma Sobel 梯度 `>1/255`。两个 generator 的共同掩码再与共享有效 GT 相交，不按事件数量筛窗。v3 单 rollout 有66个30ms窗口，平均共同活跃有效覆盖率 `0.3702%`；共同活跃且运动纹理像素平均 `9.86`，60/66窗非空。事件活跃动态纹理确实存在，但绝对支持仍稀疏。

在这60个非空窗（总计651个像素）上，linearized display proxy 光度 MAE 宏平均为：GT warp `0.002193`、zero `0.004346`、reverse `0.005548`、错时 `+8 windows` `0.004956`。全有效区是静态像素主导（5,913,219像素），GT与zero仅 `7.25e-6` / `7.93e-6`，不可用于判定运动方向。独立解析平移 `(2.25,-1.5)` 的 GT warp MAE `0.000409`，zero `0.048684`，reverse `0.094059`，用于校验诊断坐标约定。该 proxy 来自 Genesis uint8 display RGB 的 gamma-2.2 解码，包含8-bit量化/裁剪，非 HDR、未作绝对辐射校准。以上只能说明一个 synthetic smoke 中指定动态活跃区有可观测信号、光度方向合理；不说明 generator 质量、网络已学会运动或真实域泛化。

### 八 rollout 扩展仍在生成

g1 20:59:56 UTC复查确认tmux `s5_flow_v3_full_bank_eg_20261021` 的runner/worker PID `2229637/2229727`仍存活，输出目录`/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-stage5-flow-coverage-v3-eight-rollouts/`，run目录`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-stage5-flow-coverage-v3-eight-rollouts-eg/`。源码快照commit `7c55e3e5c2b253a13ebae5e2ba99e6f158809bc9`、配置SHA256 `a64ca5000cbc40e8059450d4c6819b3206ffb0fca6715f17547733fdc4e4febe`。最新progress `2026-10-20T20:59:39.189097Z`：3/8 rollout完成，当前`s5q_01_val`第80/201 endpoint，近期速率0.335 endpoint/s；20:59:56 GPU0本任务9,360 MiB/26%，GPU2 1 MiB/0%，其他GPU进程保持原样。每条固定201个100Hz render endpoint、66个30ms标签/事件窗，预登记4 train/4 validation。全bank仍在生成；V2E事件配对与STE自监督短训未开。资源观察以各自时间戳为准，不将短期速率外推为完成时间。

首个共享场景在同一配置、同一物理轨迹上已经完成实际双生成器检查：固定 `s5q_02_train`、66个30ms窗口，8×8地面格纹和4×4物体顶面纹理；两组均生成事件，GT形状 `[66,2,260,346]`、平均有效率0.99897。共同事件活跃有效区域平均覆盖0.3702%，每窗事件活跃且有纹理、运动的像素平均14.1；该面积仍稀疏。线性化 display proxy 中，在活跃运动纹理子集上真值warp MAE `0.00278`，零流 `0.01282`、反向 `0.01258`、错时 `0.01278`（宏平均相关像素数很少）；独立解析平移 `(2.25,-1.5)` 的GT warp MAE `0.000409`，零流 `0.048684`、反向 `0.094059`。证据 [`flow_smoke_audit_v3_regionfix.json`](evidence/S5/20261021-recon-flow-localization/flow_scene_smoke_v3/flow_smoke_audit_v3_regionfix.json)，shared HDF5 仍在 g1，SHA256 `5271c7e683deb09437985c53af071d05a1fe2c3b41420acaa2f1dada3a8b4a1f`。这证明当前标签/warp约定在一个固定可见纹理场景上相容，不证明模型泛化或生成器质量。


## MVSEC dt1 协议与真实文件准备

协议候选v3覆盖`indoor_flying1/2/3`与`outdoor_day1`，真实评分继续关闭。主候选使用原生`260×346`画幅，另把STE历史中心crop列为兼容诊断；AEE、`EPE>3px AND EPE>5%|GT|` Outlier、空窗口与四序列等权规则见[dt1协议候选v3](protocol/mvsec_dt1_protocol_candidate_20261021_v3.json)。v2与更早候选作为历史保留。两模型仍通过`checkpoint_binding`绑定原45 epoch合成验证选出的synthetic-best（EG epoch33、V2E epoch20），不据新诊断改选旧权重。新增加的v3全窗口注册器会在加载权重前固定所有合格APS index、事件起止及数量、GT传播地图、共同mask计数、数据/GT/K文件hash和逐窗数组hash；零事件和空support行仍保留。候选完整清单模板见[manifest v3](manifests/mvsec_dt1_test_manifest_candidate_20261021_v3.json)。

g1旧Stage-3目录仍仅有`indoor_flying1`既有文件，Stage-5完整文件通过WSL串行下载后再复制到独立g1根目录`/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-public-v2/`。截至21:00:30 UTC，WSL manifest完成6项：indoor_flying1 data 1,700,819,167 B / `9e831540…c5160eab`，GT 3,370,770,337 B / `cc3c6bc8…d7015d`；indoor_flying2 data 2,532,552,285 B / `a547babf…3d608`，GT 4,284,255,701 B / `a7603861…e5dcfb`；室内标定 1,294,736 B / `2bc61565…d1f94`。当前在下载`indoor_flying3_gt.hdf5`；完整十项后同步g1并逐文件复核SHA。官方下载索引见[MVSEC HDF5 source index](evidence/S5/20261021-recon-flow-localization/mvsec_official_hdf5_source_index_20261021.json)，g1直连Drive的ConnectTimeout失败现场继续保留。

### MVSEC真实文件窗口与评价器检查（未评分）

用已校验完整的`indoor_flying1`公共文件在WSL固定审计APS index 21：数据文件SHA `9e831540…c5160eab`、1,700,819,167 B；GT SHA `cc3c6bc8…d7015d`、3,370,770,337 B；APS/GT timestamp数为2206/1399。事件窗为`(1504645178.1181343, 1504645178.1499934] s`，31.859 ms，独立按时间戳二分查找得到194个事件，5-bin事件表示shape `[2,260,346,5]`；与官方APS event index两端差值均为1，符合锁定loader兼容校验。GT来自时间戳1504645178.1108832至1504645178.160897 s的第8张flow map，输出单位`px_per_interval`；全图有效GT 89,960像素，其中事件活跃且GT非零交集35像素。这个计数仅核查接口，不计算模型误差、不排名、不调参。可复现JSON：[APS 21 real-file window audit](evidence/S5/20261021-recon-flow-localization/mvsec_real_file_window_audit_v1/indoor_flying1_aps000021_unscored.json)。

在构造端点测试时发现本项目GT传播适配按APS end截取GT时间戳会在右端恰与GT timestamp相等时缺少控制所需的相邻区间。已修为按实际相交区间逐段传播；仅在锁定STE helper自身会执行的短窗口零长度尾步时读取后续flow map，并对数据末端跳过上游helper无法处理的越界no-op。固定STE commit `eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6` 上，普通非对齐窗、精确内部端点、单个精确GT区间和最终GT时间边界检查通过；真实窗APS 21的解析也通过。8项MVSEC adapter检查、合成artifact→manifest→checkpoint hash→指标入口self-test均通过，机器证据见[MVSEC v6检查记录](evidence/S5/20261021-recon-flow-localization/mvsec_metric_entry_selftest_v6/)。调试期v3失败记录保留；最终v6退出码0。

完整测试协议/脚本入口仍需研究侧冻结后才能启用真实分数；当前版本只完成一个实际文件窗审计和synthetic-only evaluator检查。全序列hash绑定清单尚待g1完整数据到齐后生成。

更新 [`download_mvsec_candidate_hdf5.py`](scripts/download_mvsec_candidate_hdf5.py)：按文件串行下载、原子更新路径/大小/SHA256清单、校验HDF5字段与时间戳；支持从已验证的本地Stage-3目录复制已有官方文件，失败时将顶层manifest标为failed。新增 `scripts/run_mvsec_candidate_download_local.sh` 用于隔离的WSL数据暂存和日志记录；`scripts/run_mvsec_candidate_download_server.sh` 保留作为服务器直连入口。当前g1失败目录和日志保留。WSL本地隔离run已启动，20:44:26 UTC manifest记录复用文件：indoor_flying1 data 1,700,819,167 bytes / `9e831540…c5160eab`，GT 3,370,770,337 bytes / `cc3c6bc8…d7015d`，indoor calibration 1,294,736 bytes / `2bc61565…d1f94`；都与原 Stage-3校验一致。剩余官方HDF5/室外标定下载完成后，会部署到g1新目录 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-public-v2/`，保留失败现场。下载、校验和传输都不加载模型、不评分。

全入口合成检查已在本次再次运行：`python3 benchmark/sim2real/scripts/stage4_mvsec_dt1_metrics.py --self-test` 退出0，手算 AEE `2.5 px`、Outlier `25%`，确认空公共支持返回 undefined、outdoor legacy ROI 边界、候选权重哈希错误拒绝、manifest 到 checkpoint SHA 绑定以及未授权真实评分 fail-closed。回执位于 `evidence/S5/20261021-recon-flow-localization/mvsec_candidate_full_entry_selftest_20261021/`；不读取真实输入或模型权重。

设计固定为4个训练/4个独立验证物理 rollout，包含静态场景、低速相机运动、中速刚体运动、较快合并运动与遮挡；共享8×8地面纹理、4×4物体纹理、K、轨迹、flow标签与 valid。两生成器使用相同bank；不改阈值、不强制匹配事件数量、不按任一生成器事件筛样本。当前光流 checkpoint-selection 规则不会追溯改变；新的 synthetic-best 共同 active-union∩valid 主区规则只适用于独立配置的新正式版本。

## 命令与复现入口

- 重建 synthetic fixture：`benchmark/sim2real/scripts/run_stage5_synthetic_reconstruction_fixture.sh`；对应模型推理、指标与交叉核验分别由 `infer_hqf_e2vid_candidate.py`、`evaluate_synthetic_reconstruction_fixture.py`、`verify_stage5_synthetic_reconstruction_fixture.py` 完成。
- MVSEC本地暂存：WSL tmux `mvsec_wsl_dl_20261021`，PID78581；命令与日志在`/home/科研/Eventbased_WAM/runs/sim2real/20261021-mvsec-local-stage-v2/`，运行清单在`/home/科研/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-download-v2/public_download_manifest.json`。目前6/10文件校验完成，逐个串行下载。单窗检查命令：`python3 benchmark/sim2real/scripts/audit_mvsec_real_window_contract.py --data-root /home/科研/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-download-v2 --sequence indoor_flying1 --aps-start-index 21 --output benchmark/sim2real/evidence/S5/20261021-recon-flow-localization/mvsec_real_file_window_audit_v1/indoor_flying1_aps000021_unscored.json`。完整注册入口：`python3 benchmark/sim2real/scripts/register_mvsec_dt1_test_windows.py --base-manifest benchmark/sim2real/manifests/mvsec_dt1_test_manifest_candidate_20261021_v3.json --protocol benchmark/sim2real/protocol/mvsec_dt1_protocol_candidate_20261021_v3.json --data-root /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-public-v2 --reported-root-on-g1 /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-public-v2 --output /tmp/mvsec_dt1_test_manifest_v3_registered.json`（完整注册输出先另存临时路径，审核后再冻结副本；不会覆盖模板）。
- 光流新 bank：g1 detached 命令保存在 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-stage5-flow-coverage-v3-eight-rollouts-eg/command.txt`；实时 stdout/progress 在该目录。
- 光流 paired adapter 与短训练入口已实现于 `scripts/prepare_stage5_flow_diagnostic_bank.py`、`scripts/train_stage5_flow_selfsup_diagnostic.py`，但必须等八条 bank 与 V2E artifact 完成，并在配置中登记确切 bank/manifest/code hash 后运行。
- MVSEC数值/协议入口检查：`STE_FLOWNET_ROOT=/home/科研/Eventbased_WAM/external_sources/STE-FlowNet python3 -m unittest benchmark.sim2real.tests.test_mvsec_adapter -v`（8/8通过）；`python3 benchmark/sim2real/scripts/stage4_mvsec_dt1_metrics.py --self-test`（手算AEE 2.5px、Outlier 25%，错checkpoint/window index/缺失窗口metadata拒绝，真实评分保持关闭）。运行证据见`evidence/S5/20261021-recon-flow-localization/mvsec_metric_entry_selftest_v6/`。

本次变更基于`exp/public-sim2real-v1`、基线HEAD `9c7de5e3a7a3097694b742abc86f3b6fabe0691f`。g1仍运行的bank使用隔离源码快照`7c55e3e5c2b253a13ebae5e2ba99e6f158809bc9`；WSL下载器启动时用`9c7de5e3a7a3097694b742abc86f3b6fabe0691f`。本次MVSEC适配和报告提交不改正在运行进程已导入的代码。

## 已确认、未确认与影响

- **已确认：** 重建两组12,000次更新完整结束；两组STE真实网络在一个固定GT监督窗口可记忆拟合；MVSEC实际文件 APS21 的event/APS/GT时间及表示接口审计通过；精确GT时间端点的适配切片已修正，并与锁定官方helper和全合成评价入口验证通过。
- **尚待验证：** 八rollout bank完整生成和标签/支持统计、V2E成对输出、跨rollout事件活跃覆盖、STE原自监督目标的新短训是否超过零流及独立rollout泛化；MVSEC全量文件上传和全窗口清单待完成；真实评分等待研究核验与显式授权。
- **对实验有效性的影响：** 已完成的45 epoch光流权重/选择结果未更改。GT监督固定窗只证明训练链路有记忆能力；单场景和单条MVSEC数据窗都不代表测试排名。新bank需配对V2E并验证独立rollout原自监督训练后，才决定是否建议正式扩训。

机器可读实时记录、双 generator reconstruction verifier 结果及八 rollout bank 后续报告保存在本页与 `evidence/S5/20261021-recon-flow-localization/`。大数据/权重仍在 g1，仓库只纳入必要证据与 hash。
