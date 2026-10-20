# Stage 5 双线执行交接（续）— 2026-10-21

本页续接前一份 [Stage 5 双线交接](STAGE5_DUAL_LINE_HANDOFF_20261021.md)，保留旧页作为历史快照。g1 最近一次资源/进程刷新为 **2026-10-20 20:23:46 UTC / 2026-10-21 05:23:46 JST**；光流 bank 的 progress 文件更新于20:22:25 UTC，记录1/8完成、第二条 rollout 到60/201 endpoints。重建两组 fixture verifier 的更正回执在20:20:31 UTC完成。当前范围只含重建和光流；位姿按用户要求暂缓，没有执行 E-POSE 访问。训练、checkpoint 选择只使用合成数据，真实评分保持关闭。冻结核心 `e79e656ac00f7847b98db9e12bf43b093f3afff1` 未改。实时记录见 [20:23:46 状态 JSON](evidence/S5/20261021-recon-flow-localization/live_observation_20261020T202346Z.json)。

## 当前状态

| 线路 | 已实际完成 | 当前运行/未完成 | 下一步 |
|---|---|---|---|
| **重建 / E2VID+** | 修正 TC 后 seed 0 两组各完成 200 epoch / 12,000 updates，run summary 为 `complete`，best/final 权重及完整 JSONL 曲线都在 g1。HQF 14/14 文件已转换并校验；HQF 主测、HS-ERGB 扩展、LPIPS-Alex 主指标和序列等权规则已按研究 review 登记，real score 关闭。两组候选模型都已在固定合成物理样本上完成事件输入→递归推理→200张输出→指标→独立 manifest/hash 复核。 | 未在 HQF 上运行候选 checkpoint 或计算真实分数。 | 归档验证回执和完整训练曲线摘要；保持真实评分门关闭。 |
| **光流 / STE-FlowNet dt1** | 既有固定训练窗的监督记忆诊断两组各200 updates均通过，active-union/full EPE 均降至零流基线以下。一个新 v3 共享场景已完成 paired event、GT 和线性化 display proxy 审计，真值 warp 的事件活跃运动纹理区域误差优于零流/反向/错时对照。解析 warp fixture `(2.25,-1.5)` 通过；既有synthetic-best输入置零/时间反转诊断结果已复核并纳入报告。 | g1 当前运行新的八 rollout 物理 bank；20:23:46 UTC进程仍存活，GPU0 worker为本任务 PID `2229727`；progress文件最近更新时间20:22:25 UTC，`s5q_00_train` 完成、`s5q_00_val` 到60/201 endpoints。GPU0本任务占用7,920 MiB、利用率92%；GPU2空闲，其他用户GPU进程未触碰。估算余下约1.2–1.8小时，实际时间取决于后续场景渲染速度。V2E八 rollout配对生成和新版STE自监督短训尚未开始。 | 不按事件数筛样本；完成同一物理 bank 后，在新鲜 GPU 检查通过时并行配对生成 V2E，统计共同覆盖，再启动固定100 updates/arm 的 STE 原自监督目标短诊断与独立 rollout 验证。 |
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

此前对 `s2_00/dt1_144` 的独立 GT 监督诊断两组各200 optimizer updates，关闭随机增强和样本轮换，模型 seed 0 配对初始化。EventGenesis full valid EPE 从 `0.56700` 降至 `0.06259` px/30ms、共同事件活跃 EPE 从 `1.68166` 降至 `0.08011`；V2E 分别从 `0.54841` 降至 `0.04702`、从 `0.97759` 降至 `0.07829`。零流基线是 full `0.54151`、active `0.32972`。所以两组实际 STE 网络和优化路径能拟合一个已登记样本；这不证明独立 rollout 泛化，也不涉及原自监督正式目标。checkpoint 边界恢复和权重哈希见 `flow_paired_fixed_sample_fit_summary.json` 及 per-arm 证据目录。

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


## MVSEC dt1 协议与真实文件准备

协议候选明确为 `indoor_flying1/2/3` 与 `outdoor_day1`，真实评分继续关闭。主候选使用四条序列的原生 `260×346` 画幅；已并列记录 STE 官方旧代码的中心裁剪（室内 `y=[2,258), x=[45,301)`；outdoor 在该裁剪后仅取前190行）作为兼容诊断，论文定义并没有规定这个矩形 crop，需研究审查冻结。事件窗使用 APS 时间戳 `(t_i,t_{i+1}]`；GT 是到下一 GT 时间戳的位移，按锁定的 STE `estimate_corresponding_gt_flow` 语义传播；共同支持为有限非零 GT 与两个候选共用的事件活跃区。AEE 和 `EPE>3px AND EPE>5%|GT|` 的 Outlier 定义、空窗口与四序列等权汇总都写入 [dt1 协议候选](protocol/mvsec_dt1_protocol_candidate_20261019.json)，两模型权重通过 `checkpoint_binding` 绑定。

真实文件盘点发现：g1 的旧 Stage-3 目录目前只存有 `indoor_flying1` 的1秒数据/GT切片和室内标定，完整四序列文件为0/4；WSL 本地留有完整 `indoor_flying1` HDF5 1.70 GB/3.37 GB，其余三条不在本地或 g1。官方 [MVSEC 下载页](https://daniilidis-group.github.io/mvsec/download/)公开提供 HDF5 文件目录；已从该目录解析本候选八个 HDF5 文件及室内/室外两份标定 ZIP 的 Drive ID，机器索引见 [官方 HDF5 文件索引](evidence/S5/20261021-recon-flow-localization/mvsec_official_hdf5_source_index_20261021.json)。新版本候选清单将其下载目标隔离到 g1 `data/sim2real/stage5/real/MVSEC/`，并绑定 seed0 synthetic-best checkpoint：EG epoch33 `a4e6c7e1…c824a8e`、V2E epoch20 `11041863…cd474`；索引见 [dt1 candidate manifest v2](manifests/mvsec_dt1_test_manifest_candidate_20261021_v2.json)。清单的窗口暂时为空，完整 HDF5 到达后仍要按注册规则枚举所有时间合格窗口、时间戳和 event index，不读取模型输出，不因空事件区丢窗口。

新增 [`download_mvsec_candidate_hdf5.py`](scripts/download_mvsec_candidate_hdf5.py)：按文件串行下载、原子更新路径/大小/SHA256清单，校验 HDF5 必需字段、传感器形状和时间戳，保留旧 Stage-3 文件、不覆盖已有未登记文件；不加载模型、不计算分数。下载启动/完成状态将由独立机器日志记录。当前 g1 仍以“1秒固定片段”作为唯一实际测试数据；协议准备不等于完整四序列数据已在 g1，也不等于正式评分通过。

全入口合成检查已在本次再次运行：`python3 benchmark/sim2real/scripts/stage4_mvsec_dt1_metrics.py --self-test` 退出0，手算 AEE `2.5 px`、Outlier `25%`，确认空公共支持返回 undefined、outdoor legacy ROI 边界、候选权重哈希错误拒绝、manifest 到 checkpoint SHA 绑定以及未授权真实评分 fail-closed。回执位于 `evidence/S5/20261021-recon-flow-localization/mvsec_candidate_full_entry_selftest_20261021/`；不读取真实输入或模型权重。

设计固定为4个训练/4个独立验证物理 rollout，包含静态场景、低速相机运动、中速刚体运动、较快合并运动与遮挡；共享8×8地面纹理、4×4物体纹理、K、轨迹、flow标签与 valid。两生成器使用相同bank；不改阈值、不强制匹配事件数量、不按任一生成器事件筛样本。当前光流 checkpoint-selection 规则不会追溯改变；新的 synthetic-best 共同 active-union∩valid 主区规则只适用于独立配置的新正式版本。

## 命令与复现入口

- 重建 synthetic fixture：`benchmark/sim2real/scripts/run_stage5_synthetic_reconstruction_fixture.sh`；对应模型推理、指标与交叉核验分别由 `infer_hqf_e2vid_candidate.py`、`evaluate_synthetic_reconstruction_fixture.py`、`verify_stage5_synthetic_reconstruction_fixture.py` 完成。
- 光流新 bank：g1 detached 命令保存在 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-stage5-flow-coverage-v3-eight-rollouts-eg/command.txt`；实时 stdout/progress 在该目录。
- 光流 paired adapter 与短训练入口已实现于 `scripts/prepare_stage5_flow_diagnostic_bank.py`、`scripts/train_stage5_flow_selfsup_diagnostic.py`，但必须等八条 bank 与 V2E artifact 完成，并在配置中登记确切 bank/manifest/code hash 后运行。
- 当前本地运行 `python3 -m unittest -v benchmark.sim2real.tests.test_reconstruction_metric_contract benchmark.sim2real.tests.test_stage5_flow_renderer_transfer benchmark.sim2real.tests.test_hqf_time_index_contract`：12项中10项通过、2项因本机缺 scikit-image 而跳过。g1 固定 synthetic fixture 的实际 LPIPS/SSIM/MSE/PSNR 入口已成功运行；不是 HQF real score。

## 已确认、未确认与影响

- **已确认：** 修正 TC 训练预算完成；两个重建 candidate 都能在同一类型 synthetic artifact 上执行完整 event-only 重建及指标脚本；perceptual dependency 有一个兼容缺口，已在 benchmark 层修复；STE 网络可监督拟合固定物理窗口；v3 synthetic smoke 的事件活跃动态纹理子集上 GT warp 优于固定对照。
- **尚待验证：** 八 rollout 全量统计、V2E 配对生成、跨 rollout 支持、原自监督 STE 短训是否超过零流并对输入扰动有合理响应；HQF 真实推理适配虽有完整候选配置，但研究 review 冻结前不评分。
- **对实验有效性的影响：** 旧45 epoch光流训练和 checkpoint 结果未改；固定样本监督拟合仅是网络能力诊断。单场景可观测性仍稀疏，因此不能据当前证据认定网络在独立轨迹上学到运动。只有配对完整bank、独立 rollout 自监督短训和冻结 active-union 合成验证后，才决定是否建议正式扩训。

机器可读实时记录、双 generator reconstruction verifier 结果及八 rollout bank 后续报告保存在本页与 `evidence/S5/20261021-recon-flow-localization/`。大数据/权重仍在 g1，仓库只纳入必要证据与 hash。
