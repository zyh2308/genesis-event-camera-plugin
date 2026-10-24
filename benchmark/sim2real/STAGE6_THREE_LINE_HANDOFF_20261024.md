# 三线并行执行阶段性交接（2026-10-24）

**范围：** 光流场景升级/入口修复、HQF 全 14 序列真实测试准备、E-POSE 样本与 CAD 契约审计。位姿已按本轮用户指令恢复推进。Frozen simulator core `e79e656ac00f7847b98db9e12bf43b093f3afff1` 未修改。真实数据没有用于训练、checkpoint 选择或调参。

**最近服务器观测：** 2026-10-24 12:00:45 UTC（21:00:45 JST），来自 `zhaoyuhan@g1.pkuml.org` 的只读 SSH 会话：六张卡利用率为 68–94%；11:49:42 UTC 的前一轮 ownership-aware 快照显示六卡均有其他用户任务。未启动远程训练、推理或数据生成，也没有停止或修改任何进程。该次连接另核对了两份 E-POSE CAD 文件哈希和远程传输回执。远端共享工作树 11:49 时为 HEAD `b4f671b85ba1c2cf672c75afca71d108b4aa51a2` 且有其他未提交文件；本轮没有覆盖它。WSL 工作树起始 HEAD 为 `30503672019d87a7a85f7fbbbe9e557376d0c9db`。

## 当前判断

| 线路 | 已实际完成 | 尚未完成 / 阻塞 | 下一步 |
|---|---|---|---|
| 光流 | 修正两个诊断入口的源码指纹路径：脚本本身明确指向脚本文件，第三方模型指纹使用真实文件名 `models/STEFlow.py`。CPU 隔离验证实际 STE-FlowNet 的四尺度前向、一轮反传/Adam 更新、保存/恢复后普通 `eval()` 输出逐位一致；70/70 个梯度张量有限且非零，输出差 `0.0`。另从 g1 保存的既有 pilot 做出固定 `(0,30 ms]` 预览，含 Genesis RGB proxy、深度、GT flow、两组事件和相机路径，输出 SHA 见机器清单。 | 可视化的是历史单 rollout 玩具场景，不是 ReplicaCAD/新丰富场景，也不代表新训练。它对应的单 smoke 66 个窗口平均共同事件活跃有效覆盖率为 0.370%；共同事件支持、GT 位移≥0.5 px、纹理三者交集平均仅 14.1 像素（62/66 窗口非空）。Genesis 输入为 gamma 逆变换后的 8-bit display RGB proxy，不是 native HDR radiance。CPU 检查不是 CUDA 验证、完整数据注册或 100-update 配对训练。全新自监督训练和独立丰富场景验证未运行。官方 ReplicaCAD 下载从 WSL 查询元数据时 `Connection refused`，本轮未导入新房间资产。 | 恢复可信 ReplicaCAD 下载路径并在 g1 导入单一完整布局，登记场景/轨迹后重新渲染预览；再审核深度、flow、事件和增强后支持。只有信号门槛预先达标、GPU 空闲且完整配对数据通过后，才运行两组相同预算的短训。若 Genesis 无法取得基线要求的原生 radiance，作为数据语义限制单独提交研究决策。 |
| 重建 | 既有 Stage5 修正 TC 训练已完成：EventGenesis/V2E seed 0 各 200 epoch、12,000 updates。HQF v3 已冻结为独立执行协议，锁定全部 14 序列、时间戳规则、event-only、原生分辨率、LPIPS-Alex 主指标、等序列权重及两个 synthetic-best 权重/代码/manifest hash。56 帧合成夹具通过，四类契约反例被拒；漏掉 `--enable-real-scoring` 时 scorer/runner 均在读真实数据前以 exit 2 拒绝。 | 本轮尚未执行真实推理或计算 HQF 分数。冻结协议在 g1 的实际数据、权重和 pinned source checkout 尚未再次逐哈希 preflight。当前 GPU 仍处高利用率，完整约 1.5 万帧推理不可安全挤占。HS-ERGB 仍是扩展准备，不替代 HQF 主测。 | 在 g1 完成只读实际文件/权重指纹核验，并找到共享 GPU 安全空档后，按固定的全部 14 序列清单运行两组 seed 0；归档完整推理日志、逐序列及等序汇总。分数只报告，不用于改配置或挑权重。 |
| 位姿 | E-POSE 导入收据已验证 18/18 原始包、2,377,982 文件、48,610,498,574 解压字节。样本 `block/r_01_gl/0001_data.mat` 与原始 HDF5 精确对齐：捕获的 g1 审计报告 717 events 完全匹配；本地脚本独立复核窗口/标签端点算术。找到与 wood block 同一 CAD 网格的本地资产，8,194 顶点，尺度约 0.104×0.103×0.206 m；两个 CAD 文件已传到 g1 独立目录并复核 SHA256。 | 正确逐点投影与 MAT 2D keypoints 的均值差 16.450 px、最大 32.744 px；复刻官方错误共享第一角深度并去畸变后残差均值 0.166 px、最大 0.349 px。源码确认 2D 标签构建错误。`cameraPose` exporter 固定取第 2 帧，但本次首个样本内部仍自洽；后续时刻/记录的具体误差待再核。尚未跑成对 CAD 合成训练、独立 rollout eval/PnP 或恢复检查。 | 使用 CAD 与 6D pose 重投影构造一致的 2D 几何标签；真实 MAT keypoints 只留作原始数据审计。验证模型正常 `eval()` 下置信度选点与 PnP，不使用 GT crop/mask/box/选点。待共享 GPU 空闲后跑最小配对训练与独立验证，并向研究会话报告该数据集标签语义限制。 |

## 证据解释与边界

1. **场景支持不足是已证实事实，不是模型诊断结论。** 现有 66 窗 smoke 的共同事件活跃像素占共同有效区约 0.370%；满足事件支持、GT 运动≥0.5 px/30 ms、纹理梯度阈值的像素平均约 14.1。正确 GT 在全部共同运动纹理像素上的光度 MAE 没有稳定优于零流，像素数也太少；这只能说明当前物理 pilot 提供的可观测性弱，不能归因于任一事件生成器。
2. **renderer 信号来源必须如实标记。** 审计记录 Genesis 1.2.2 rasterizer 的 uint8 display RGB，经固定 gamma 2.2 逆变换后作为事件输入。其有 8-bit 量化、clamp 且无辐射标定，不得称为 native HDR/linear radiance。正式扩训前应优先验证 renderer 的原生辐照度接口；若不能获得，要报告其对研究基线的影响。
3. **ReplicaCAD 当前是访问/部署阻塞，不是“数据集不存在”。** 官方页面与 Hugging Face 数据卡列出 Interactive 公寓场景和 CC BY 4.0。WSL 官方 `hf download` 在查询元数据时返回 `Connection refused`；单次 no-proxy 请求又在 5 秒后 TLS 超时；此前 g1 访问返回 `Network is unreachable`。本机也没有 Genesis 环境。本轮未下载或导入，也未把 RGB 视频冒充 Genesis bank。恢复下载后记录 revision/hash，再做单布局导入；若有结构性不兼容，再评估 HSSD。
4. **HQF 真实测试未运行。** 本轮没有真实 LPIPS/SSIM/PSNR/MSE 分数，不存在根据 HQF 选择 checkpoint、挑序列或改变配置的动作。研究已接受 HQF 为主测；本轮工程目标是冻结和接通全量管线，但实际执行受共享 GPU 占用阻塞。
5. **E-POSE 审计边界。** 行事件匹配数字来自本轮捕获的 g1 摘要；本机没有原始 H5/MAT 数组，独立脚本只重算时间算术、CAD 对应、pose 几何和投影，不宣称本机逐事件重算。MAT `cameraPose` 源码固定索引是已确认问题；但当前首个样本的 cameraPose 与 object pose 仍自洽，需另一时刻确认实际字段偏差。
6. **旧结果原样保留。** 旧 Stage5 重建权重/分数、旧 45 epoch 光流结果、旧八 rollout 方块 bank 和本次可观测性失败样例均保留，不将 CPU shim 或单场景描述统计包装成新训练成果。

## HQF 冻结门禁与历史场景拼图

- 冻结协议：[hqf_14_reconstruction_test_frozen_20261024_v1.json](protocol/hqf_14_reconstruction_test_frozen_20261024_v1.json)，SHA256 `46656bbf0b02e3cddc79b9957cde9092b33bba0ecca48a7687c0e733cab97494`。原始 v3 candidate 保持不变。该 freeze 只锁定执行协议，不表示已运行模型或产生真实指标。
- HQF 本地合成自检：[local_validation_record.json](evidence/20261024-hqf-eval-entry/local_validation_record.json)，exit 0；synthetic scorer contract 覆盖 14/14 序列、56 帧，阻止缺一序列、错时间窗、预测 artifact hash mismatch、checkpoint 绑定错误；省略真实评分显式 flag 的 runner/scorer 各 exit 2。
- g1 旧场景拼图：[s5q_02_train_dt1_000.png](evidence/20261024-flow-scene-preview/repro_run1/s5q_02_train_dt1_000.png)。这是固定的历史单 rollout 首窗；生成器原始事件分别为 EventGenesis 9,972、V2E 600。该窗口预注册选择，不按模型或 generator 分数选择。相机在本 rollout 中静止，不能满足新相机运动场景要求。
- 拼图的 g1 可复现日志、命令、环境与退出码：[repro_run1](evidence/20261024-flow-scene-preview/repro_run1/)。bank SHA `5271c7e6…3a8b4a1f`（16,957,544 bytes），V2E raw SHA `13d42a37…d8537b9cbc86`（111,456 bytes）。
- ReplicaCAD 下载尝试：[机器记录](evidence/20261024-replica-access/attempt.json)。本机运行官方 HF CLI 到仓库元数据请求即失败，未传输任何数据，也未重试。官网给出的 Interactive 资产包约 132 MB；Hugging Face 数据卡显示许可证 CC BY 4.0，并说明包含完整室内场景与交互资产。
- E-POSE CAD g1 传输：[CAD manifest](manifests/epose_wood_block_cad_asset_manifest_20261024.json)、[远端回执副本](evidence/20261024-execution-refresh/epose_cad_transfer_receipt.json)、[12:00:45 UTC 远端 GPU/哈希回执](evidence/20261024-execution-refresh/g1_gpu_refresh_20261024T120045Z.json)。只传输两份小型匹配 CAD 文件；RAR 和原始 E-POSE 数据未传输或重解压。

## 可复查入口

- 本轮文件与哈希总索引：[INDEX.md](evidence/20261024-stage6/INDEX.md)、[机器可读清单](evidence/20261024-stage6/index.json)。
- 研究核查：[restart_three_lines_20261024/REVIEW.md](reviews/restart_three_lines_20261024/REVIEW.md)
- HQF 协议：[hqf_14_reconstruction_test_candidate_20261021_v3.json](protocol/hqf_14_reconstruction_test_candidate_20261021_v3.json)
- HQF 历史数据准备：[20261021 HQF 14 序列清单与转换证据](evidence/S5/20261021-recon-flow-localization/20261021-hqf14-manifest-validation-v3/)
- Flow 入口 CPU 功能检查：[fingerprint_cpu_shim.json](evidence/20261024-flow-entrypoint/fingerprint_cpu_shim.json)
- Flow 场景 smoke 与事件/几何/覆盖统计：[flow_scene_smoke_v3_regionfix.json](evidence/S5/20261021-recon-flow-localization/flow_scene_smoke_v3/flow_smoke_audit_v3_regionfix.json)
- g1 最新只读状态：[GPU/repo 快照](evidence/20261024-execution-refresh/g1_live_resource_snapshot.json)、[12:00:45 UTC GPU 读数](evidence/20261024-execution-refresh/g1_gpu_refresh_20261024T120045Z.json)、[E-POSE 样本审计](evidence/20261024-execution-refresh/g1_sample_audit_stdout.txt)；此前较宽的目录扫描在 SSH 中断时被 Ctrl-C，退出码 255，只保留为不完整尝试。
- E-POSE 导入收据：[ingestion_receipt.json](evidence/20261024-epose-ingestion/ingestion_receipt.json)
- E-POSE 文件/样本核验：[g1_sample_audit_stdout.txt](evidence/20261024-execution-refresh/g1_sample_audit_stdout.txt)
- E-POSE 可复现语义审计：[epose_block_r01gl_semantic_audit_v1.md](evidence/20261024-execution-refresh/epose_block_r01gl_semantic_audit_v1.md)、[机器结果](evidence/20261024-execution-refresh/epose_block_r01gl_semantic_audit_v1.json)
- HQF freeze 与 synthetic scorer contract：[protocol](protocol/hqf_14_reconstruction_test_frozen_20261024_v1.json)、[本地验证记录](evidence/20261024-hqf-eval-entry/local_validation_record.json)
- g1 历史场景拼图及复现记录：[image](evidence/20261024-flow-scene-preview/repro_run1/s5q_02_train_dt1_000.png)、[preview manifest](evidence/20261024-flow-scene-preview/repro_run1/preview_manifest.json)

## 执行状态

- 本轮未在服务器生成新数据、训练或运行 HQF 推理；没有改变任何远程进程。远端仅进行只读 HDF5/资源检查、旧结果 CPU 预览，以及两份小型 E-POSE CAD 资产的独立路径传输和哈希核验。
- WSL 修正仅涉及 Stage5 flow 诊断入口的源文件指纹定位。Frozen core、正式模型、训练目标和历史产物均未更改。
- 研究会话需要决策的两个实质问题：Genesis 原生 HDR/radiance 接口能否满足基线；E-POSE 官方 2D keypoint/cameraPose 标签错误是否按 CAD+pose 重建监督与评估标签。
- 建议后续优先顺序：ReplicaCAD 导入与场景预览 → 共享运动可观测性门槛 → 新场景配对光流短训；共享 GPU 空闲后并行运行 HQF 固定全量评分与 E-POSE 配对小试。
