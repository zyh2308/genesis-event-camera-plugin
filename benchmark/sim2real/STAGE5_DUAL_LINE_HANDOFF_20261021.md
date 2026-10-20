# Stage 5 双线执行交接 — 2026-10-21

更新到 **2026-10-20 18:32:33 UTC / 2026-10-21 03:32:33 JST**。g1 进程与产物最后一次直接核对为 **18:27:54 UTC**。范围仅含重建与光流；位姿**按用户要求暂缓**，未重新访问或下载 E-POSE。冻结 simulator core `e79e656ac00f7847b98db9e12bf43b093f3afff1` 未改；所有训练和模型选择只用 synthetic 数据，真实评分关闭。

## 状态一览

| 线路 | 已完成并核实 | 当前阻塞/边界 | 下一步 |
|---|---|---|---|
| 重建 / E2VID+ | corrected-TC seed 0 两组均已完成 200 epochs / 12,000 updates，退出码 0。HQF 14/14 原始包已传至 g1 并与本地逐文件 SHA256 一致。已检查一条原始 bag 的事件和图像话题；合成 E2VID event-only 推理适配自检通过。 | HQF 转换首次运行在第 5 条序列失败，根因是端点偏移验证把 end-exclusive offset 当作被包含事件的索引；同时发现锁定 EVREAL 旧转换器也在 `[prev,end)` 切片中漏掉端点事件。修正已做，尚未在 g1 重跑 14 序列。没有任何真实模型评分。 | 部署修正版到新的干净 g1 checkout，在新输出目录转换 14 条；完成文件、时间和端点 manifest；真实评分仍等待研究核验。 |
| 光流 / STE-FlowNet dt1 | EventGenesis 和 V2E 均在同一固定合成样本 `s2_00/dt1_144`、同一共享 GT/active-union mask、同一配对初始化下，各完成 200 次多尺度监督拟合；预登记拟合阈值和 checkpoint 恢复检查全部通过。 | 这只证明固定窗口记忆，不是独立 rollout 泛化。新场景 smoke 在 Genesis 初始化后失败：已安装 Genesis 1.2.2 的 `Camera.render()` 不支持 `radiance=True`。首次物理 capture 未完成，因此目前没有新场景的 events/GT 可观测性结果；现有 8 rollout bank 尚未启动。 | 用 benchmark 适配层读取实际 uint8 display RGB、按锁定 shader 的 gamma 2.2 解码为明确标注的 linearized-display proxy，保留饱和/量化限制并记录 Genesis 源码 hash；在干净 checkout 重跑一条共享场景，再依据其审计决定 8 rollout 配置。 |
| 位姿 | 无新增动作。 | **按用户要求暂缓**。 | 等用户通知后恢复。 |

## 光流：实际网络固定窗口拟合

这是独立 GT 监督诊断，不改变正式自监督目标、原有 45 epoch 结果或 synthetic-best 选择规则。固定数据和预算：`configs/stage5_flow_fixed_sample_fit_v1.json`，样本 `s2_00/dt1_144`，无随机增强/窗口轮换。事件活跃区在运行前由正常输入 EventGenesis 与 V2E 的共同并集构造，再与共享有效 GT 相交；不按任一生成器事件数选样本。

| 生成器 | 200步监督目标 / 初始 | full valid EPE 初始→最终（px/30ms） | active-union EPE 初始→最终（px/30ms） | 零流 full / active | checkpoint |
|---|---:|---:|---:|---:|---|
| EventGenesis | 0.69489 → 0.03098（0.0446×） | 0.56700 → 0.06259（0.1104×） | 1.68166 → 0.08011（0.0476×） | 0.54151 / 0.32972 | g1 `.../20261021-stage5-flow-fixed-sample-fit-v4/results/eventgenesis_last.pt`，580,024,698 bytes，SHA256 `c1a9d6426962f44db59d2a94c0ead7ffe1eb35c45c057e1cbc044b4c6f15241f` |
| V2E | 0.48137 → 0.02255（0.0468×） | 0.54841 → 0.04702（0.0857×） | 0.97759 → 0.07829（0.0801×） | 0.54151 / 0.32972 | g1 `.../20261021-stage5-flow-fixed-sample-fit-v5/results/v2e_last.pt`，580,022,034 bytes，SHA256 `2b6f70c55ab170bc79b4da4269c6f8634c7824b1875f9b5435fff80c27e0c965` |

两组参数均发生明显变化，有限非零梯度正常；V2E 的 paired initial-state hash 与 EventGenesis 初始化相同。EventGenesis restore 后 optimizer step 为 200，state SHA 一致，重复 eval 最大差 `3.5763e-7`（登记浮点容差 `1e-6`）；V2E restore 后 optimizer step 为 200，state 完全相同。两组最终 EPE 都低于零流基线，说明网络、输入和优化路径至少能够记住这个标签；不说明新 rollout 泛化、不说明事件包含充分运动信息，也不比较生成器优劣。

机器汇总：`evidence/S5/20261021-recon-flow-localization/flow_paired_fixed_sample_fit_summary.json`。逐步小日志和恢复证据：`flow_fixed_sample_fit_v4_eventgenesis/`、`flow_fixed_sample_fit_v5_v2e/`。checkpoint 保留在 g1；两组源数据 manifest SHA 为 `1f9aff5164c71301a7406c9981c31ef0d5df381df81641cf2a1d397e8df246b1`。运行的干净代码快照是 `836fe01db81d541de3b056fb748f6f289c81596d`。

先前的正式 45 epoch / 8,910 updates 配对训练、synthetic-best 与旧曲线均保留，未因这次诊断重挑 checkpoint。原训练 checkpoint 选择与指标不变。

## 光流：物理场景 smoke 与 renderer 信号契约

已登记试跑配置 `configs/stage5_flow_scene_coverage_pilot_v1.json` 的 `s5p_00_train`。g1 GPU2 在启动前直接检查为空闲，其他用户的 GPU/作业未碰。首次执行路径：

- run：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-stage5-flow-coverage-v1-smoke-r2/`
- bank：`/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-stage5-flow-coverage-v1-smoke-r2/`
- 18:22:30 UTC 开始，18:25:53 UTC 退出码 1；Genesis 初始化和场景构造成功，但首个 capture 调用报 `TypeError: Camera.render() got an unexpected keyword argument 'radiance'`。未生成完整 trajectory，不开始 V2E，也没有可观测性结论。失败日志与 `.partial` 文件保留。Taichi cache lock 告警不是退出原因；未清理共享缓存。

g1 只读源码核对：`Camera.render` 签名不含 `radiance`，普通 pyrender `mesh.frag` 用 `pow(color,1/2.2)` 编码并 clamp，`Renderer.render` 返回 uint8。由此只能在 benchmark/provider 层对真实返回 RGB 做 `(rgb_uint8/255)^2.2` 解码，作为 **linearized display RGB proxy**。该信号仍有 8-bit 量化、renderer clipping，且无绝对辐射标定；绝不记作 native HDR radiance。新的 bank schema 为 v2，逐帧保存饱和通道比例，并记录 Genesis `camera.py`、`rasterizer.py`、`mesh.frag` SHA256。EventGenesis 和 V2E 都从同一 proxy 派生输入；阈值、噪声、共同轨迹/标签未变。

修正代码与数学回归检查目前通过本地小测试，但截至本报告时间尚未部署或实际重跑；renderer API 与源文件 hash 的远端证据是 `flow_renderer_contract_probe.json`；因此场景输入契约仍未通过端到端验证，尚无新的运动可观测性数据。

## 重建与 HQF

已完成的训练：g1 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0/`。EventGenesis 12,000 updates，exit 0，synthetic-best epoch170 LPIPS-VGG `0.04116681637`，last SHA256 `fe7eacf5865f486b7355e4f60202368de34b56d3dbcde1412ee67bfbfc29cbaf`；V2E 同预算，exit 0，synthetic-best epoch195 LPIPS-VGG `0.08367991168`，last SHA256 `af861ca93da680fd83b58c6644376e99ba2c9007f379b92a29221031159726db`。共同初始化 SHA256 `48426d94f4159592155eda873d4e71e2f06439c10eade990de5979231ed07a0f`。选择继续只用 synthetic LPIPS-VGG。

研究侧 review `reviews/recon_flow_localization_20261019/REVIEW.md` 已明确 HQF 全14序列为重建主测试，HS-ERGB 仅扩展。HQF 候选协议为 LPIPS-AlexNet 主指标，SSIM/PSNR/MSE 辅助；先序列内帧均值，再14序列等权平均；不做 real-GT 亮度/直方图拟合。协议文件 `protocol/hqf_14_reconstruction_test_candidate_20261021.json`。转换第一次失败摘要在 `evidence/S5/20261021-recon-flow-localization/hqf_conversion_attempt01/`；按 timestamp 的独立小例子覆盖首窗口、同戳、空窗和未来事件，锁定 EVREAL MemMap 是 end-exclusive slice。修正实现已更新 converter/builder/inference validator，并通过本地索引边界回归；HQF 合成模型适配入口曾在 fabricated MemMap 上通过递归 reset、event-only 输入、padding/unpadding 与最终 endpoint 自检。新修正版本还需重新实际运行完整14条转换。合成推理自检本轮清洁源码重跑通过，证据 `hqf_adapter_selftest_v6/selftest_outputs/inference_manifest.json`；先前失败/不完整的适配尝试见 `hqf_adapter_selftest_attempts.json`。

HQF 原始数据 g1 路径 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-hqf14-raw/`，14 包总计 2,902,695,576 bytes。WSL 下载清单 `evidence/S5/20261021-recon-flow-localization/hqf_raw_download_manifest.json`，g1 SHA 输出 `hqf_g1_sha256sum.txt`，14/14 与本地相符。`desk_hand_only.bag` 已读到 `/dvs/events` (`dvs_msgs/msg/EventArray`) 和 `/dvs/image_raw` (`sensor_msgs/msg/Image`)；完整 MemMap manifest 仍待修正转换。

## MVSEC 与真实数据隔离

MVSEC 机器清单 `evidence/S5/20261021-recon-flow-localization/mvsec_manifest_g1/mvsec_dt1_file_manifest_v2.json` SHA256 `2c4f7e6b2fe6014f098416b464893f7f7b694da5125ec07237fd05d297c5a7cc`，当前登记到一个 `indoor_flying1_registered_1s` 配对片段，完整 dt1 序列 0/4。评价器 artifact→manifest→checkpoint hash 绑定及全合成数据入口自检通过（`mvsec_evaluator_self_test.json`）；还未开放真实评分。后续 MVSEC 时间窗、GT/掩码裁剪和指标配置继续按候选协议锁定，真实 GT 分数不得进入任何训练/选点或配置修改。

## 复现入口与环境

- g1 环境：`Eventbased_WAM`，Python 3.11.15，PyTorch 2.5.1+cu121，Genesis 1.2.2；V2E 使用既有 pinned checkout。隔离 `rosbags==0.11.7` 路径为 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-hqf-convert-deps/pydeps`。
- 单样本两组拟合命令分别保存在 g1 `.../20261021-stage5-flow-fixed-sample-fit-v4/command.txt` 与 `...fit-v5/command.txt`；配置和汇总在仓库上述路径。
- HQF 全量转换命令：`bash benchmark/sim2real/scripts/convert_hqf_14_dataset.sh <raw_bag_root> <new_memmap_root> <new_manifest.json>`，必须使用 `PYTHONPATH=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-hqf-convert-deps/pydeps` 与 `PYTHON_BIN=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python`；需在修正代码快照上执行，输出使用新目录。
- 新 renderer adapter 单测：`python3 benchmark/sim2real/tests/test_stage5_flow_renderer_transfer.py`；HQF 时间索引单测：`python3 benchmark/sim2real/tests/test_hqf_time_index_contract.py`。本报告时间两者均本地通过；不表示远端场景 smoke 已通过。

当前已发布 GitHub 版本为 `836fe01db81d541de3b056fb748f6f289c81596d`；renderer 适配、HQF index 修正和最新文档尚未提交/部署。大权重和真实数据留 g1，仓库仅保存机器证据、小型产物、路径和 hash。**未得到新场景完整 bank 前不启动光流长训或其他 seeds；任何真实评分仍关闭。**
