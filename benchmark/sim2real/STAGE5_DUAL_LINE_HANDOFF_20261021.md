# Stage 5 双线执行交接 — 2026-10-21

更新到 **2026-10-20 19:40:41 UTC / 2026-10-21 04:40:41 JST**；重建 checkpoint 元数据、HQF manifest、光流 v3 bank 和 V2E 作业均在本轮直接刷新。范围仅含重建与光流；位姿**按用户要求暂缓**，未重新访问或下载 E-POSE。冻结 simulator core `e79e656ac00f7847b98db9e12bf43b093f3afff1` 未改；训练和 checkpoint 选择仍只用 synthetic 数据，真实评分关闭。

## 状态一览

| 线路 | 已完成并核实 | 当前阻塞/边界 | 下一步 |
|---|---|---|---|
| 重建 / E2VID+ | corrected-TC seed 0 两组均完成 200 epochs / 12,000 updates；run summary 为 `complete`、exit 0。synthetic-best 是 EventGenesis epoch170 SHA `506e81bd…f554bc`、V2E epoch195 SHA `598b1e78…d050d8`；final checkpoint SHA 已登记。HQF 全14条 `ready_unscored`，HQF 主测/HS-ERGB 扩展，LPIPS-Alex 主指标、SSIM/PSNR/MSE 辅助。 | 尚未用实际训练权重通过合成 artifact 跑完整推理/指标入口；未进行 HQF 候选评分。`engineering_posters` source 图索引528按固定规则排除，事件流不改。 | 用 synthetic validation artifact 核验两个训练权重、事件窗、递归状态、原生分辨率 padding、指标输入/聚合；保持真实评分门关闭。 |
| 光流 / STE-FlowNet dt1 | 固定样本 GT 拟合两组各200步，通过参数变化、EPE和恢复检查。v3 `s5q_02_train` 物理 capture 完成：201端点、2秒、EG 28,020 events、66个 flow 标签，bank SHA `5271c7e6…a8b4a1f`、exit0；V2E 于19:38:06 UTC 在GPU2启动配对生成。 | 固定样本拟合只证明记忆，不证明独立 rollout 泛化。v2共同事件活跃有效区约0.58%、moving-texture交集为0，故不进新训练。v3 V2E和共同覆盖/真值warp审计未完成；八条 bank与新版自监督短训未启动。renderer输入仍是未校准显示代理。 | 完成 smoke 配对事件与光度审计；通过后生成唯一登记的八 rollout bank，再看独立 rollout 上 STE 原自监督目标。 |
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

v1 renderer API 错误、v2 支持不足和 v3 两次部署前失败都保留原始目录。v3 中等物体运动 rollout `s5q_02_train` 完成物理 capture：201端点、2秒、EG 28,020 events、66个30ms flow标签，平均有效率0.99897，首帧物体位姿误差 `6.17e-9 m`，K 与 renderer source hash 写入 bank manifest；SHA256 `5271c7e683deb09437985c53af071d05a1fe2c3b41420acaa2f1dada3a8b4a1f`。物理 capture exit 0，但不等于 paired smoke pass。V2E 于19:38:06 UTC 在 tmux `s5_flow_v3_smoke_v2e_20261021` 启动；19:40:41 UTC 实际 PID2203951，GPU2用显存2391 MiB/32768 MiB、利用率44%。事件时序/支持和 GT warp 审计仍待完成。实时证据见 `evidence/S5/20261021-recon-flow-localization/live_observation_20261020T194041Z.json`。

- run：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-stage5-flow-coverage-v1-smoke-r2/`
- bank：`/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-stage5-flow-coverage-v1-smoke-r2/`
- 18:22:30 UTC 开始，18:25:53 UTC 退出码 1；Genesis 初始化和场景构造成功，但首个 capture 调用报 `TypeError: Camera.render() got an unexpected keyword argument 'radiance'`。未生成完整 trajectory，不开始 V2E，也没有可观测性结论。失败日志与 `.partial` 文件保留。Taichi cache lock 告警不是退出原因；未清理共享缓存。

g1 只读源码核对：`Camera.render` 签名不含 `radiance`，普通 pyrender `mesh.frag` 用 `pow(color,1/2.2)` 编码并 clamp，`Renderer.render` 返回 uint8。由此只能在 benchmark/provider 层对真实返回 RGB 做 `(rgb_uint8/255)^2.2` 解码，作为 **linearized display RGB proxy**。该信号仍有 8-bit 量化、renderer clipping，且无绝对辐射标定；绝不记作 native HDR radiance。新的 bank schema 为 v2，逐帧保存饱和通道比例，并记录 Genesis `camera.py`、`rasterizer.py`、`mesh.frag` SHA256。EventGenesis 和 V2E 都从同一 proxy 派生输入；阈值、噪声、共同轨迹/标签未变。

v2 单 rollout `s5p_00_train` 原始产物在 g1 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-stage5-flow-coverage-v2-smoke/`，完整 66 个 30 ms 标签；修正后的 audit 报共同事件活跃有效区均值 `0.0057627`，GT 位移≥0.5 px 的有效像素均值 2306.8，但其中共同活跃仅 0.258 像素，且所有窗 `moving_texture` 像素为0。故 v2 只留作排障证据，不进入新训练。全图光度 warp 被静态像素主导，不能据此判断动态 GT 对齐。

独立解析平移 fixture `(u,v)=(2.25,-1.5)` 通过：GT warp 的 MAE `0.000409`，零流 `0.04868`，反向流 `0.09406`，验证了诊断的 warp 方向。v3 设计增加 8×8 地面纹理、4×4 物体纹理、静止/低速/中速/较快运动及遮挡分层；不改两生成器参数、阈值或事件数量。v3 使用同一代理信号给 EG/V2E：Genesis 显示 RGB 反 gamma 得到 linearized display proxy，明确包含量化/裁剪且无绝对辐射校准。

v3 首次尝试因压缩源快照缺少 `SOURCE_FILE_HASHES.json`、第二次因缺少插件 `PYTHONPATH` 均在物理 capture 前失败，未产生任何样本。失败现场仍在 g1 两个独立 `v3-smoke` 目录；核验后的快照 commit `7c55e3e5c2b253a13ebae5e2ba99e6f158809bc9` 包含1324个文件 hash；运行清洁快照与 config/source hash 清单见 `evidence/S5/20261021-recon-flow-localization/flow_v3_source_snapshot/` 和 `flow_v3_launch_attempts.json`。v3 目前仍在 capture，尚未宣称通过。

## 重建与 HQF

已完成的训练：g1 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0/`。EventGenesis 12,000 updates、exit0；synthetic-best epoch170/update10200、LPIPS-VGG `0.04116681637`、size171,428,482、SHA256 `506e81bd558f1bcca2b67d34d4c1004836fc7cce65393084eb52a378dfd554bc`；final `last.pt` SHA256 `fe7eacf5865f486b7355e4f60202368de34b56d3dbcde1412ee67bfbfc29cbaf`。V2E 同预算、exit0；synthetic-best epoch195/update11700、LPIPS-VGG `0.08367991168`、size171,428,482、SHA256 `598b1e7837b4b02b50a7f390889c7737ed797840043a3f23121f54dc99d050d8`；final `last.pt` SHA256 `af861ca93da680fd83b58c6644376e99ba2c9007f379b92a29221031159726db`。共同初始化 SHA256 `48426d94f4159592155eda873d4e71e2f06439c10eade990de5979231ed07a0f`。选择仅看 synthetic LPIPS-VGG。

研究侧 review `reviews/recon_flow_localization_20261019/REVIEW.md` 已明确 HQF 全14序列为重建主测试，HS-ERGB 仅扩展。指标候选为 LPIPS-AlexNet 主、SSIM/PSNR/MSE 辅助；序列内帧均值后对14序列等权平均；不拟合 real-GT 亮度或直方图。HQF all14 转换及 hash validator 已实际运行，状态 `ready_unscored`，manifest hash `76e19db296be055135fa7446a4da68b704a1c573dd7f2b8f06ce599ddc1a3add`。研究 review 已选择 HQF 主测角色；该项不是待决议。真实指标尚未计算，候选评分开关仍 false。

旧转换 v2 在 `engineering_posters` 拒绝一个明显错误的 header 时间戳（源图像序号528，倒退 2147.396356 秒）；该序列 bag record 顺序与事件批次时间连续，坏图像内容与相邻帧不同。14 条 raw bag 每帧审计证据在 `evidence/S5/20261021-recon-flow-localization/20261021-hqf14-timing-audit-v2/`，汇总为 `hqf_source_timing_audit_summary.json`。13/14 序列 header 时间严格递增；没有排序或时间插值。

benchmark converter 在 `bb71f7b1ddc586269352cbdba1b0704e8364d4df` 增加整数纳秒坏帧检查：按 bag 顺序保留单调端点，不改事件流；对非递增帧记录 source index、header/bag timestamp 和 payload SHA，由下一个合法端点消费间隔内所有事件。`engineering_posters` 的 source index528 被排除，1266/1267 参考图保留；其余13序列严格时间递增。具体完整文件/尺寸/hash 见 `20261021-hqf14-manifest-validation-v3/hqf_14_file_manifest_v3.json`。HQF protocol 的 scoring switch 仍 false，不作任何真实模型排名。

HQF 原始数据 g1 路径 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-hqf14-raw/`，14 包总计 2,902,695,576 bytes。WSL 下载清单 `hqf_raw_download_manifest.json`，g1 SHA 输出 `hqf_g1_sha256sum.txt`，14/14 与本地相符。全14 MemMap 在 g1 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-hqf14-memmap-v3/`，均 `ready_unscored`；conversion stdout、command、source hash 和 exit marker 归档于 `20261021-hqf14-conversion-v3/`。

## MVSEC 与真实数据隔离

MVSEC 机器清单 `evidence/S5/20261021-recon-flow-localization/mvsec_manifest_g1/mvsec_dt1_file_manifest_v2.json` SHA256 `2c4f7e6b2fe6014f098416b464893f7f7b694da5125ec07237fd05d297c5a7cc`，当前登记到一个 `indoor_flying1_registered_1s` 配对片段，完整 dt1 序列 0/4。评价器 artifact→manifest→checkpoint hash 绑定及全合成数据入口自检通过（`mvsec_evaluator_self_test.json`）；还未开放真实评分。后续 MVSEC 时间窗、GT/掩码裁剪和指标配置继续按候选协议锁定，真实 GT 分数不得进入任何训练/选点或配置修改。

## 复现入口与环境

- g1 环境：`Eventbased_WAM`，Python 3.11.15，PyTorch 2.5.1+cu121，Genesis 1.2.2；V2E 使用既有 pinned checkout。隔离 `rosbags==0.11.7` 路径为 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-hqf-convert-deps/pydeps`。
- 单样本两组拟合命令分别保存在 g1 `.../20261021-stage5-flow-fixed-sample-fit-v4/command.txt` 与 `...fit-v5/command.txt`；配置和汇总在仓库上述路径。
- HQF 修正版 v3 命令保存在 g1 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261021-hqf14-conversion-v3/command.txt`，driver、环境和代码 hash 在同目录；脚本使用干净源码快照 `gec-sim2real-exec-bb71f7b1`。
- 新 renderer adapter 单测：`python3 benchmark/sim2real/tests/test_stage5_flow_renderer_transfer.py`；HQF 时间索引单测：`python3 benchmark/sim2real/tests/test_hqf_time_index_contract.py`。本报告时间两者均本地通过；不表示远端场景 smoke 已通过。

当前 WSL 与 GitHub `origin/exp/public-sim2real-v1` 在 `bb71f7b1ddc586269352cbdba1b0704e8364d4df` 同步。HQF timing audit/conversion 使用 g1 干净快照 `gec-sim2real-exec-bb71f7b1`；仍在运行的物理 scene smoke 固定使用启动时快照 `gec-sim2real-exec-990674d6`。服务器原共享工作树、旧训练和失败转换输出均保留。大数据/权重留 g1，仓库纳入代码、协议、小证据和 hash。新共享 bank 与独立 rollout 自监督检查完成前不启动光流长训或其他 seeds；真实评分仍关闭。
