# Stage 5 双线执行交接（续）— 实时覆盖更新

本次续接更新于 **2026-10-22 16:04 UTC / 2026-10-23 01:04 JST**。它覆盖下方较早快照中的“当前状态”，旧运行记录仍保留作证据。仓库基于 `b0f05e7ebc34ab9b2c96f728a3823a15b6ad220f`；本更新尚待提交。冻结核心 `e79e656ac00f7847b98db9e12bf43b093f3afff1` 未修改。真实数据只做文件/接口核验，**真实模型评分仍关闭**；位姿依用户要求暂停。

## 最新三线状态

| 线路 | 已核实完成 | 当前状态与阻塞 | 下一步 |
|---|---|---|---|
| **重建 / E2VID+** | g1 最近一次可读 summary（2026-10-20 20:57:04 UTC）记录 EventGenesis/V2E seed 0 各完成 200 epoch、12,000 optimizer updates、退出码 0；合成验证 best、final、完整曲线均有路径和 SHA。两组也完成固定合成 rollout 的 event-only 递归推理及独立产物核对。HQF 14 序列数据 manifest 已校验，HQF 主测、HS-ERGB 扩展角色已由研究审查确定。 | 没有 HQF 真实推理或评分。g1 本轮无法刷新；训练“完成”根据上述带时间戳的 summary，不宣称此次重新读取。 | 保持训练归档；真实评测须等待审查确认冻结后才打开。 |
| **光流 / STE-FlowNet dt1** | 原配对 seed 0 训练 45 epoch / 8,910 updates 已结束；固定合成窗两组各 200 次 GT 监督更新能拟合单样本，证明网络/优化可记忆该窗；不证明独立泛化。另有共享场景 warp 诊断。MVSEC 的 indoor_flying1/2/3 各自固定 APS index 21 文件窗解析均通过。 | g1 八 rollout bank 的**当前进度未知**：最后确认于 2026-10-20 21:09:41 UTC，4/8 完整，活动 rollout `s5q_02_train` endpoint 60/201。2026-10-22 两次串行 SSH 均 ConnectTimeout（每次15秒，间隔至少60秒），按服务器保护规则停止远程操作。自监督新短训尚无可核实记录。 | 网络恢复后只做一次低频状态刷新；若 bank 已完整，先审计标签和配对 V2E，再跑登记的短自监督/独立 rollout 检查；证据不足前不扩成长训。 |
| **位姿** | 无新增执行。 | 按用户要求暂停，未进行 E-POSE 探测或下载。 | 等用户通知。 |

## 本次新增/确认的 MVSEC 文件审计

审计在运行前固定了每序列 APS start index 21，不依据模型输出或事件数选窗。indoor_flying1/2/3 窗口时长均约 31.859 ms，事件数分别 194/176/133；事件表示均为 `[2,260,346,5]`，GT 单位按锁定 STE 接口为 `px_per_interval`。有效 GT 像素均为 89,960；共同事件活跃、有效且非零 GT 的交集分别为 35/31/26 像素。三份结果均明确 `candidate_weights_loaded=false`、`predictions_computed=false`、`real_scores_computed=false`。复现证据：[indoor_flying1](evidence/S5/20261021-recon-flow-localization/mvsec_real_file_window_audit_v1/indoor_flying1_aps000021_unscored.json)、[indoor_flying2/3 计划与结果](evidence/S5/20261021-recon-flow-localization/mvsec_real_file_window_audit_v2/)。各数据/GT 大小、SHA、时间戳和窗口端点写在 JSON 中；v2 运行 SHA 清单在该目录 `run/SHA256SUMS.txt`。

WSL 的 MVSEC manifest 仍是 `failed`：室内三序列的 data/GT 六个 HDF5 和室内标定 ZIP 共 7 个文件已完成并校验；缺 `outdoor_day1_data.hdf5`、GT 和 outdoor calibration。首次 Drive 请求失败后，2026-10-22 16:02 UTC 做过一次间隔超过 24 小时的官方 HDF5 重试：退出码 1、下载 0 bytes，错误仍是 Google Drive “Too many users…”配额。详见[下载来源与重试审计](evidence/S5/20261021-recon-flow-localization/mvsec_download_route_audit_v1.json)及[隔离重试日志](</home/科研/Eventbased_WAM/runs/sim2real/20261023-mvsec-outdoor-official-hdf5-retry-v1/>). 官方页面也提供 ROS bag 镜像，但它不同于当前 HDF5 loader 契约，本轮未替换或转换；参见[MVSEC 官方下载页](https://daniilidis-group.github.io/mvsec/download/)。

## g1 状态与同步边界

本次续接只进行 WSL 文件/证据核验与报告整理，未占用 GPU、未启动训练；官方 HDF5 重试约3秒内因配额退出，下载0 bytes。

本次两次 SSH 到 `zhaoyuhan@10.129.163.68` 均连接超时，第二次是在至少 60 秒退避后；未再尝试连接。因此不能声称 g1 当前训练/进程状态、远端 Git 状态或本次提交同步状态。最近的 g1 观测仍是下方记录的 2026-10-20 时间点。WSL 本地报告/证据可提交并推 GitHub；g1 同步要等连接恢复后验证。机器快照：[2026-10-22 16:04 UTC JSON](evidence/S5/20261021-recon-flow-localization/live_observation_20261022T160417Z.json)，SHA256 sidecar 同目录。

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

### 八 rollout 扩展：保留最后一次可确认状态

**以下为 2026-10-20 的历史观测，当前远端进度未知。** g1 21:06:49 UTC 复查确认 tmux `s5_flow_v3_full_bank_eg_20261021` 的 runner/worker PID `2229637/2229727` 仍存活，输出目录 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-stage5-flow-coverage-v3-eight-rollouts/`，run 目录 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-stage5-flow-coverage-v3-eight-rollouts-eg/`。源码快照 commit `7c55e3e5c2b253a13ebae5e2ba99e6f158809bc9`、配置 SHA256 `a64ca5000cbc40e8059450d4c6819b3206ffb0fca6715f17547733fdc4e4febe`。生成日志显示 4/8 rollout 已完整写入，刚开始 `s5q_02_train`；21:06:49 UTC 的 GPU0 为 9,704 MiB/92%，GPU2 1 MiB/0%。每条固定 201 个 100Hz render endpoint、66 个 30ms 标签/事件窗，预登记 4 train/4 validation。当时全 bank 仍在生成；V2E 事件配对与 STE 自监督短训未开。2026-10-22 两次 SSH ConnectTimeout 后没有刷新该状态；请以本页开头的最新状态表为准，不将此历史进度写成当前进度。

首个共享场景在同一配置、同一物理轨迹上已经完成实际双生成器检查：固定 `s5q_02_train`、66 个 30ms 窗口，8×8 地面格纹和 4×4 物体顶面纹理；两组均生成事件，GT 形状 `[66,2,260,346]`、平均有效率 0.99897。共同事件活跃有效区域平均覆盖 0.3702%；交集“活跃且 GT 位移≥0.5 px 且有纹理”的像素平均 9.86 个、60/66 窗非空。未与事件活跃区相交的全部有效运动纹理像素平均 14.1 个、62/66 窗非空。两种口径都显示绝对支持稀疏。线性化 display proxy 中，在活跃运动纹理子集上真值 warp MAE `0.00278`，零流 `0.01282`、反向 `0.01258`、错时 `0.01278`（宏平均相关像素数很少）；独立解析平移 `(2.25,-1.5)` 的 GT warp MAE `0.000409`，零流 `0.048684`、反向 `0.094059`。证据 [`flow_smoke_audit_v3_regionfix.json`](evidence/S5/20261021-recon-flow-localization/flow_scene_smoke_v3/flow_smoke_audit_v3_regionfix.json)，shared HDF5 仍在 g1，SHA256 `5271c7e683deb09437985c53af071d05a1fe2c3b41420acaa2f1dada3a8b4a1f`。这证明当前标签/warp约定在一个固定可见纹理场景上相容，不证明模型泛化或生成器质量。


## MVSEC dt1 协议与真实文件准备

协议候选v3覆盖`indoor_flying1/2/3`与`outdoor_day1`，真实评分继续关闭。主候选使用原生`260×346`画幅，另把STE历史中心crop列为兼容诊断；AEE、`EPE>3px AND EPE>5%|GT|` Outlier、空窗口与四序列等权规则见[dt1协议候选v3](protocol/mvsec_dt1_protocol_candidate_20261021_v3.json)。v2与更早候选作为历史保留。两模型仍通过`checkpoint_binding`绑定原45 epoch合成验证选出的synthetic-best（EG epoch33、V2E epoch20），不据新诊断改选旧权重。新增加的v3全窗口注册器会在加载权重前固定所有合格APS index、事件起止及数量、GT传播地图、共同mask计数、数据/GT/K文件hash和逐窗数组hash；零事件和空support行仍保留。候选完整清单模板见[manifest v3](manifests/mvsec_dt1_test_manifest_candidate_20261021_v3.json)。

截至 2026-10-20 的计划是将 Stage-5 文件复制到独立 g1 根目录 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-public-v2/`。这个段落后续由本页开头的 2026-10-22 更新覆盖：本地 manifest 最终有 7/8 项完成，indoor_flying3 GT 已完成校验，失败项为 outdoor_day1 data；间隔超过24小时的单次重试仍触发Drive配额错误。当前没有可确认的 g1 文件传输或全序列清单。官方下载索引见 [MVSEC HDF5 source index](evidence/S5/20261021-recon-flow-localization/mvsec_official_hdf5_source_index_20261021.json)，g1 直连 Drive 的 ConnectTimeout 失败现场继续保留。

### MVSEC真实文件窗口与评价器检查（未评分）

用已校验完整的`indoor_flying1`公共文件在WSL固定审计APS index 21：数据文件SHA `9e831540…c5160eab`、1,700,819,167 B；GT SHA `cc3c6bc8…d7015d`、3,370,770,337 B；APS/GT timestamp数为2206/1399。事件窗为`(1504645178.1181343, 1504645178.1499934] s`，31.859 ms，独立按时间戳二分查找得到194个事件，5-bin事件表示shape `[2,260,346,5]`；与官方APS event index两端差值均为1，符合锁定loader兼容校验。GT来自时间戳1504645178.1108832至1504645178.160897 s的第8张flow map，输出单位`px_per_interval`；全图有效GT 89,960像素，其中事件活跃且GT非零交集35像素。这个计数仅核查接口，不计算模型误差、不排名、不调参。可复现JSON：[APS 21 real-file window audit](evidence/S5/20261021-recon-flow-localization/mvsec_real_file_window_audit_v1/indoor_flying1_aps000021_unscored.json)。

在构造端点测试时发现本项目GT传播适配按APS end截取GT时间戳会在右端恰与GT timestamp相等时缺少控制所需的相邻区间。已修为按实际相交区间逐段传播；仅在锁定STE helper自身会执行的短窗口零长度尾步时读取后续flow map，并对数据末端跳过上游helper无法处理的越界no-op。固定STE commit `eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6` 上，普通非对齐窗、精确内部端点、单个精确GT区间和最终GT时间边界检查通过；真实窗APS 21的解析也通过。8项MVSEC adapter检查、合成artifact→manifest→checkpoint hash→指标入口self-test均通过，机器证据见[MVSEC v6检查记录](evidence/S5/20261021-recon-flow-localization/mvsec_metric_entry_selftest_v6/)。调试期v3失败记录保留；最终v6退出码0。

完整测试协议/脚本入口仍需研究侧冻结后才能启用真实分数；当前版本只完成一个实际文件窗审计和synthetic-only evaluator检查。全序列hash绑定清单尚待g1完整数据到齐后生成。

更新 [`download_mvsec_candidate_hdf5.py`](scripts/download_mvsec_candidate_hdf5.py)：按文件串行下载、原子更新路径/大小/SHA256清单、校验HDF5字段与时间戳；支持从已验证的本地Stage-3目录复制已有官方文件，失败时将顶层manifest标为failed。新增 `scripts/run_mvsec_candidate_download_local.sh` 用于隔离的WSL数据暂存和日志记录；`scripts/run_mvsec_candidate_download_server.sh` 保留作为服务器直连入口。当前g1失败目录和日志保留。WSL本地隔离run已启动，20:44:26 UTC manifest记录复用文件：indoor_flying1 data 1,700,819,167 bytes / `9e831540…c5160eab`，GT 3,370,770,337 bytes / `cc3c6bc8…d7015d`，indoor calibration 1,294,736 bytes / `2bc61565…d1f94`；都与原 Stage-3校验一致。剩余官方HDF5/室外标定下载完成后，会部署到g1新目录 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-public-v2/`，保留失败现场。下载、校验和传输都不加载模型、不评分。

全入口合成检查已在本次再次运行：`python3 benchmark/sim2real/scripts/stage4_mvsec_dt1_metrics.py --self-test` 退出0，手算 AEE `2.5 px`、Outlier `25%`，确认空公共支持返回 undefined、outdoor legacy ROI 边界、候选权重哈希错误拒绝、manifest 到 checkpoint SHA 绑定以及未授权真实评分 fail-closed。回执位于 `evidence/S5/20261021-recon-flow-localization/mvsec_candidate_full_entry_selftest_20261021/`；不读取真实输入或模型权重。

设计固定为4个训练/4个独立验证物理 rollout，包含静态场景、低速相机运动、中速刚体运动、较快合并运动与遮挡；共享8×8地面纹理、4×4物体纹理、K、轨迹、flow标签与 valid。两生成器使用相同bank；不改阈值、不强制匹配事件数量、不按任一生成器事件筛样本。当前光流 checkpoint-selection 规则不会追溯改变；新的 synthetic-best 共同 active-union∩valid 主区规则只适用于独立配置的新正式版本。

## 命令与复现入口

- 重建 synthetic fixture：`benchmark/sim2real/scripts/run_stage5_synthetic_reconstruction_fixture.sh`；对应模型推理、指标与交叉核验分别由 `infer_hqf_e2vid_candidate.py`、`evaluate_synthetic_reconstruction_fixture.py`、`verify_stage5_synthetic_reconstruction_fixture.py` 完成。
- MVSEC本地暂存（历史运行，已失败退出）：命令和日志在`/home/科研/Eventbased_WAM/runs/sim2real/20261021-mvsec-local-stage-v2/`，manifest 在`/home/科研/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-download-v2/public_download_manifest.json`；7/8项完成，失败在outdoor_day1 data。后续配额重试见本页开头和机器证据。单窗检查命令：`python3 benchmark/sim2real/scripts/audit_mvsec_real_window_contract.py --data-root /home/科研/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-download-v2 --sequence indoor_flying1 --aps-start-index 21 --output benchmark/sim2real/evidence/S5/20261021-recon-flow-localization/mvsec_real_file_window_audit_v1/indoor_flying1_aps000021_unscored.json`。完整注册入口：`python3 benchmark/sim2real/scripts/register_mvsec_dt1_test_windows.py --base-manifest benchmark/sim2real/manifests/mvsec_dt1_test_manifest_candidate_20261021_v3.json --protocol benchmark/sim2real/protocol/mvsec_dt1_protocol_candidate_20261021_v3.json --data-root /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-public-v2 --reported-root-on-g1 /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-public-v2 --output /tmp/mvsec_dt1_test_manifest_v3_registered.json`（完整注册输出先另存临时路径，审核后再冻结副本；不会覆盖模板）。
- 光流新 bank：g1 detached 命令保存在 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-stage5-flow-coverage-v3-eight-rollouts-eg/command.txt`；实时 stdout/progress 在该目录。
- 光流 paired adapter 与短训练入口已实现于 `scripts/prepare_stage5_flow_diagnostic_bank.py`、`scripts/train_stage5_flow_selfsup_diagnostic.py`，但必须等八条 bank 与 V2E artifact 完成，并在配置中登记确切 bank/manifest/code hash 后运行。
- MVSEC数值/协议入口检查：`STE_FLOWNET_ROOT=/home/科研/Eventbased_WAM/external_sources/STE-FlowNet python3 -m unittest benchmark.sim2real.tests.test_mvsec_adapter -v`（8/8通过）；`python3 benchmark/sim2real/scripts/stage4_mvsec_dt1_metrics.py --self-test`（手算AEE 2.5px、Outlier 25%，错checkpoint/window index/缺失窗口metadata拒绝，真实评分保持关闭）。运行证据见`evidence/S5/20261021-recon-flow-localization/mvsec_metric_entry_selftest_v6/`。

本次变更基于`exp/public-sim2real-v1`、基线HEAD `9c7de5e3a7a3097694b742abc86f3b6fabe0691f`。g1仍运行的bank使用隔离源码快照`7c55e3e5c2b253a13ebae5e2ba99e6f158809bc9`；WSL下载器启动时用`9c7de5e3a7a3097694b742abc86f3b6fabe0691f`。本次MVSEC适配和报告提交不改正在运行进程已导入的代码。

## 已确认、未确认与影响

- **已确认：** 重建两组12,000次更新完整结束；两组STE真实网络在一个固定GT监督窗口可记忆拟合；MVSEC实际文件 APS21 的event/APS/GT时间及表示接口审计通过；精确GT时间端点的适配切片已修正，并与锁定官方helper和全合成评价入口验证通过。
- **尚待验证：** 八rollout bank完整生成和标签/支持统计、V2E成对输出、跨rollout事件活跃覆盖、STE原自监督目标的新短训是否超过零流及独立rollout泛化；MVSEC全量文件上传和全窗口清单待完成；真实评分等待研究核验与显式授权。
- **对实验有效性的影响：** 已完成的45 epoch光流权重/选择结果未更改。GT监督固定窗只证明训练链路有记忆能力；单场景和单条MVSEC数据窗都不代表测试排名。新bank需配对V2E并验证独立rollout原自监督训练后，才决定是否建议正式扩训。

本次续接证据索引见 [`continuation_evidence_index_20261022.json`](evidence/S5/20261021-recon-flow-localization/continuation_evidence_index_20261022.json)。机器可读实时记录、双 generator reconstruction verifier 结果及八 rollout bank 后续报告保存在本页与 `evidence/S5/20261021-recon-flow-localization/`。大数据/权重仍在 g1，仓库只纳入必要证据与 hash。
