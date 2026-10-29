# Stage 7 execution update — 2026-10-29

## 状态摘要

本次刷新使用 Windows OpenSSH 连接 `zhaoyuhan@g1.pkuml.org` 和 `zhaoyuhan@g3.pkuml.org`。远端观测时间为 2026-10-29 06:02 UTC（15:02 JST）。冻结 simulator core `e79e656ac00f7847b98db9e12bf43b093f3afff1` 未修改；本次没有用真实分数改训练、选 checkpoint 或挑数据。

| 线路 | 已确认的实际进度 | 当前判断 / 下一步 |
|---|---|---|
| 重建 | E2VID+ seed 0 两组候选已完成 HQF 全 14 序列推理和固定协议评分。 | 数据测试入口已实际跑通；这是单 seed 真实测试结果，只作报告。HS-ERGB 仍是扩展项。 |
| 光流 | 在同一固定合成样本上，EventGenesis/V2E 各做了 200 次 STE-FlowNet 多尺度 GT 监督更新；随后在另一物理 rollout 的 66 个窗口上验证。 | 两组都能降低固定样本的监督目标，但独立验证没有显示可靠的运动泛化。未确认 generator 或模型代码存在新的标签方向/warp 错误。独立进程恢复检查还需修复启动命令并在 CUDA 可用时补做。 |
| 位姿 | E-POSE 已有 18/18 包导入。相邻两个 MAT 窗口分别与 raw HDF5 的 717/717、729/729 个事件精确对应；匹配 CAD 与 K 可用。 | 确认官方 2D keypoint 构造使用了共享深度，和逐点几何投影相差约 16.7 px 均值；不能直接用原 2D keypoints 训练/评价。还没有正式 pose 训练、独立 rollout/PnP 验证或真实评分。 |

## A. 重建：HQF 全量真实测试已经完成

固定 HQF v3 协议下，对 14 个序列分别运行两个 synthetic-best checkpoint，并作先帧、后序列等权汇总。`paired_score.json` 记录 `real_hqf14_paired_scoring_completed`、`real_scores_computed=true`、`checkpoint_selection_allowed=false`。

| seed 0 synthetic-best | LPIPS-Alex ↓ | SSIM ↑ | PSNR (dB) ↑ | MSE ↓ |
|---|---:|---:|---:|---:|
| EventGenesis | 0.570738 | 0.255283 | 11.6348 | 0.086147 |
| V2E | 0.582220 | 0.250035 | 12.0742 | 0.080667 |

这些指标呈现指标间的取舍；单 seed、单训练版本不足以支持稳定的生成器优劣结论。真实分数没有用于调整训练或 checkpoint。HS-ERGB 未在本次执行。

远端 run：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261025-hqf14-e2vid-seed0-actual-r2/`。评分 JSON SHA256 为 `4fb0679ddcde1f4af3b2cbccf1af99fc0e193c9c51e809b171947b2dd36a3692`；数据清单 SHA256 `76e19db296be055135fa7446a4da68b704a1c573dd7f2b8f06ce599ddc1a3add`；run manifest SHA256 `a8041212a2c73a4ed0aa67fc4baaf59b3a7cfe735751f1d64a80ac8a511bf67d`。两组 checkpoint SHA256 见证据索引。

## B. 光流：GT 拟合能记住固定样本，独立 rollout 仍接近零流

使用八 rollout 共享 bank 中预先按共同 GT 运动与纹理选择的 `s5q_02_train/dt1_000`，两组共用物理标签和初始化、STE-FlowNet、Adam、200 更新预算；随机增强关闭。事件活跃 mask 是共同有效 GT 与正常 EventGenesis/V2E 事件支持并集。flow 单位为每 30 ms 像素位移。独立验证使用 `s5q_02_val` 全 66 个窗口，没有用于 checkpoint 选择。

| 固定训练样本拟合，update 0 → 200 | 四尺度监督目标 | active EPE (px/30 ms) | full-valid EPE | update 200 零流 active/full |
|---|---:|---:|---:|---:|
| EventGenesis | 2.50648 → 0.01747 | 1.93879 → 0.10344 | 0.69894 → 0.03802 | 0.01385 / 0.01507 |
| V2E | 0.43460 → 0.00361 | 0.26345 → 0.01290 | 0.09924 → 0.00705 | 0.01385 / 0.01507 |

V2E 在固定样本上接近零流，EventGenesis 的 active EPE 仍高于零流。两组都通过预登记的多尺度目标拟合门槛，参数确实变化；这证明优化路径能拟合这个窗口，不等于模型能泛化。

| 独立验证 66 窗口，pixel-weighted EPE (px/30 ms) | EventGenesis | V2E | 同一 mask 的零流 |
|---|---:|---:|---:|
| Full valid | 0.03699 | 0.04776 | 0.03104 |
| 两组正常事件活跃 union | 0.62181 | 0.72091 | 0.56027 |
| GT 位移 ≥ 0.5 px | 1.40731 | 1.39464 | 1.40708 |

独立验证中，两组全有效区和事件活跃区都比零流差；动态区上 V2E 仅改善约 0.0124 px，幅度很小，EventGenesis 略差于零流。共同活跃支持覆盖 74,541 / 5,928,597 个有效像素，约 1.26%。因此目前不能说模型学到了可泛化的运动，也不能把这次结果归因于 generator 优劣；独立 rollout 的可观测动态支持仍弱。

checkpoint 在训练进程内加载后，状态一致、optimizer step 为 200、预测最大差为 0.0。单独新进程的恢复/续更验证没有通过：正式 driver 把 shell 字符串 `\${arm}_last.pt` 原样传给验证脚本，导致文件名含字面量 `${arm}` 并以 FileNotFoundError 结束，故根 run `exit_code=1`。之后单独 GPU 验证遇到 PyTorch CUDA driver 初始化失败；CPU 备选又被锁定模型 `STEFlowNet.py` 内部的 `.cuda()` 调用阻断。训练和独立验证主产物已完整保存，但不能将进程内 reload 说成独立进程续训已验证。保留失败目录，不重训；下一次只需修正验证命令变量展开，在有可确认 CUDA 资源时单独完成恢复后的一步 optimizer 更新。

run 根目录：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261029-stage5-hightexture-fixed-gtfit-g3-v2/`。摘要、66 窗逐窗评估、两个更新日志以及 driver 命令副本位于仓库 `evidence/20261029-execution-refresh/flow/`；原始 driver 日志留在 g1。作业于 2026-10-28 16:20:43 UTC 在 g3 启动、17:04:21 UTC 结束，run state 为 `failed`、exit 1（原因见上）。确实运行过的代码 commit 为 `d0e36796f04826a14a0931878405abed943a49a3`，现已快进合并到 `exp/public-sim2real-v1`。运行环境为 Python 3.11.15、PyTorch 2.5.1+cu121、h5py 3.14.0、NumPy 2.4.6、OpenCV 5.0.0；训练时 CUDA 可用，设备日志显示 RTX 4090。新加入的配对适配器、独立验证脚本和配置均在主分支。

## C. 位姿：已确认标签几何问题，训练仍未启动

数据目录：`/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1/`，导入回执显示 18/18 archives、2,377,982 个最终解压文件。此次只读样本为 `block/r_01_gl/0001_data.mat` 和 `0002_data.mat`，及对应原始 `data.h5`；没有读取 `.extracting` 临时目录，没有重新下载或解压，也没有做模型真实评分。

MAT 时间窗分别是 0.0613679886–0.0713729858 s 和 0.0713729858–0.0814039707 s。将原始事件时间按来源代码的时间原点变换后，各自与 HDF5 窗口事件逐项匹配：717/717 和 729/729，位置、极性及时间均一致。匹配 wood-block CAD 有 8,194 个顶点，尺寸约 0.103917 × 0.103464 × 0.206003 m；K 为 fx=292.52、fy=293.19、cx=172.61、cy=130.125。CAD、位姿、K 的相机坐标可自洽，最大坐标差约 1.7e-5 m。

但按每个关键点自身深度进行正确投影，和 MAT 保存的 2D 点均值差约 16.7 px、最大约 47.5–47.8 px；复刻 MATLAB 源码的共享第一点深度后，均值差降至约 0.109 px、最大约 0.325 px。这具体定位到官方 2D keypoint 标签共享深度错误。训练/评测必须从匹配 CAD + 6D pose + K 重新生成几何关键点，不能把原 MAT 2D keypoints 当正确监督。

`CreateSaveDataset.m` 第 308 行把同一个第二帧 `cameraPose` 写入每条样本。两个 MAT 的相机矩阵完全相同，但原始轨迹按时间插值后，样本 1 与该值吻合到约 5.6 μm / 0.00039°，样本 2 偏差约 0.82 mm / 0.0567°。这是实际时间字段不一致的证据，误差小于 2D 标签错误对当前几何的影响；后续应以 raw pose trajectory 和明确时间原点重建相机位姿。E-POSE 原始数据只用于契约审计，不作真实评分或模型选择。

机器结果：`camera_pose_trajectory_v3.json` SHA256 `36f634899f2314eb520a17ba027e9fec4633313c9e1aaed5b4ac9ac27f862ace`；样本审计 `result.json` SHA256 `2116dc6272aabc5dda1b7e189dffca47d53e135d044cd37bd0fd42fc0922d468`。完整路径与 MAT/H5/CAD 哈希见 `evidence/20261029-execution-refresh/evidence_index.json`；对应样本审计 JSON、时间检查 JSON 与脚本副本也在同目录下的 `epose/`。

## 资源、复现与下一步

- g1 实时快照（06:02:30 UTC）：GPU 0 上有两个 gec-robot 进程且约 316 MiB 显存；GPU 1–5 都有其他运行进程。没有安全的训练卡，本次没有提交新训练。
- g3 实时快照（06:02:51 UTC）：`nvidia-smi` 对 GPU `0000:56:00.0` 报 `Unknown Error`，无法完成设备/进程归属核对；因此不能据此分配 GPU。该节点当前没有可确认资源，本次不启动任务。不要以训练完成时的资源快照代替现在的分配检查。
- 完整 HQF per-frame 文件、原始数据和大权重继续留在 g1；光流 driver 脚本、逐步更新日志、训练摘要与独立验证 JSON，以及 E-POSE 两份样本审计 JSON 和时间检查脚本已复制到 Git。含整段 nvidia-smi 输出的原始 driver 日志仅留在 g1，其路径、大小和哈希见机器索引。
- 真实数据评测：HQF 已按预定固定协议完成；MVSEC 与 E-POSE 本次没有评分。未来训练和选择仍 synthetic-only，HQF 结果不用于再调参。

复现入口：运行脚本保存在 g1 光流 run 的 `driver.sh`，日志在同目录 `driver.log`；它包含共享 bank / split / V2E 与 SuperSloMo 哈希、CUDA/Python 版本、训练与评估调用。对应配置为主分支 `benchmark/sim2real/configs/stage5_genesis_high_texture_fixed_sample_gt_fit_v1.json`。HQF 的冻结协议为 `protocol/hqf_14_reconstruction_test_frozen_20261024_v1.json`。E-POSE 本地可复核脚本为 `scripts/audit_epose_block_sample_semantics_local.py`；远端 v3 时间检查脚本和输出按索引所列保存。

**最小下一步：** 不重跑 HQF、不重复 200-update 拟合；先等资源协调确认后，只运行修正过的独立 checkpoint restore+one-update 检查。光流的下一实验应转向独立运动场景/更强共享可观测性验证，而不是延长当前静态相机单场景。位姿先按正确几何重建标签，再做 synthetic train/independent rollout 门禁；不把错误 MAT keypoints 喂给模型。
