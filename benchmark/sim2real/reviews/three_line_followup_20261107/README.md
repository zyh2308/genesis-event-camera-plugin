# 三线并行推进：评分完成、光照覆盖与 MVSEC 室外恢复

核验快照：2026-11-07 11:24 UTC / 20:24 JST。此页是固定证据快照，不是实时监控页。

| 路线 | 本轮可核验进展 | 仍缺的部分 |
|---|---|---|
| 重建 | 5 方法 × 3 seed 全部完成训练；HQF 14 序列、210 次推理及评分完成，退出码 0；CSV、LaTeX、PNG/PDF 已生成 | 五方法 seed 2 共同出现近黑退化；须先解决合成训练稳定性，当前表不能支撑仿真器优势结论 |
| 光流 | 12/15 正式训练已完成；剩余 Vid2E、FILM+ESIM、RIFE+ESIM 的 seed 2 正在 GPU 3、4、2 并行运行 | 剩余训练、室内外数据服务器传输及注册、15 checkpoint 冻结后的 MVSEC 统一评分 |
| 位姿 | 原正式训练 15/15；完整原训练轨迹的配对物理光照数据正在 GPU 5 补算；条件短训及评分链已部署 | 光照补算尚未完成；新双分支短训尚未启动，须通过全部原稳定性门槛后才真实评分 |

冻结核心仍为 `62da4f35ce3c5c96f04268bd1687a27657af9238`，训练配方 SHA256 为 `51b7c723c9c4f1022483136f244d1970496b74b6886ad08b2f09be7ee1f221da`。本轮没有改动仿真器、噪声、遮挡规则或正在运行的正式训练。新增试验均用独立输出目录。GPU 预约有原子锁和作业结束释放机制；等待数据的位姿管理进程不占 GPU。

## 重建：完整结果与共同稳定性问题

[全部结果表](reconstruction/tables/README.md) · [逐序列逐 seed CSV](reconstruction/tables/hqf14_all_seeds_per_sequence.csv) · [均值和样本标准差](reconstruction/tables/hqf14_mean_sample_sd.csv) · [结果图 PDF](reconstruction/tables/hqf14_all_seed_metrics.pdf)

每个 checkpoint 评分 15,498 帧，合计 232,470 次帧评分；先在序列内平均，再等权平均 14 序列，最后跨三个 seed 求均值与样本标准差。没有使用 GT 亮度拟合、直方图匹配或 GT 初始化。权重仅按合成验证选取；不得用本轮 HQF 分数重新挑选。

| 方法 | LPIPS-Alex ↓ | SSIM ↑ | PSNR ↑ | MSE ↓ |
|---|---:|---:|---:|---:|
| EventGenesis | 0.6969 ± 0.2030 | 0.2526 ± 0.1835 | 10.304 ± 2.531 | 0.1168 ± 0.0759 |
| V2E | 0.6593 ± 0.2355 | 0.2520 ± 0.1837 | 10.919 ± 3.078 | 0.1082 ± 0.0835 |
| Vid2E | 0.7133 ± 0.1927 | 0.2351 ± 0.1701 | 10.239 ± 2.569 | 0.1215 ± 0.0733 |
| FILM+ESIM | 0.7475 ± 0.1603 | 0.2263 ± 0.1610 | 10.021 ± 2.304 | 0.1266 ± 0.0677 |
| RIFE+ESIM | 0.7373 ± 0.1713 | 0.2156 ± 0.1543 | 9.855 ± 2.207 | 0.1299 ± 0.0654 |

EventGenesis 的主指标目前未优于 V2E。配对 LPIPS 差值为 +0.03759，按 seed 和序列交叉重采样的描述性 95% 区间约为 [-0.00037, 0.06929]；只有三个 seed，不据此作显著性或胜出声明。所有 seed 2 均完整保留。

为了区分真实域差距与共同训练问题，本轮只读加载了 15 个正式权重，以及 EventGenesis/V2E seed 2 的初始化和 update-20 状态，在同一个固定合成验证 rollout 的 40 个连续窗口做了 19 组检查，没有 optimizer 更新。所有正式权重 SHA 与 HQF 绑定一致，采用原训练 loader。

在这些合成窗口中，seed 2 的输出进入负值，按原协议截到 [0,1] 后接近全黑；两组初始化及 update-20 也出现负输出。这说明退化能在合成路径复现，不能简单归因于真实数据、某个生成器或 HQF 评分错误。它还不是唯一根因的证明。下一轮重建修复应先审计线性输出、感知损失与时序损失的数值关系，并在固定合成窗口上完成短程配对检查；不得删除 seed 2、针对真实分数修改后处理或直接发起另一轮长训。

[合成输出诊断](reconstruction/synthetic_range_audit.json) · [40 窗口汇总 CSV](reconstruction/synthetic_range_summary.csv)

## 光流：三张卡继续训练，真实评分仍关闭

11:24 UTC 快照中，三个剩余 seed 2 作业的追加日志为 Vid2E 3,366/8,910、FILM+ESIM 6,148/8,910、RIFE+ESIM 792/8,910；各自日志时间戳记录在 [服务器快照](evidence/server_snapshot.json)。最近日志均有 70 个有限梯度张量。RIFE 任务已从原串行队列转为独立 GPU 2 作业，训练参数、数据、初始化和预算不变；短检与恢复后继续正式训练。

最终评分仍须遵守既有 15-checkpoint、四序列协议。原室内注册的 7,700 个窗口保持不变；室外样本只按既有时间定义补充注册。数据到齐不等于允许跳过协议检查或提前比较权重。

## MVSEC：可靠替代渠道已下载，服务器传输尚未完成

原官方 Google Drive 大文件仍遇到配额限制。查到 [EI-Nexus 作者仓库](https://github.com/ZhonghuaYi/EI-Nexus_official/blob/main/README.md)明确链接的 [Hugging Face MVSEC_RPE 数据归档](https://huggingface.co/datasets/Zhonghua/MVSEC_RPE/tree/362d64dca06043e4afc124bb8052360ca9ce8d2c)。本轮固定该 revision，仅提取原生 HDF5 成员，没有使用它额外的整流/预处理结果，也没有改变事件时间网格。

- `outdoor_day1_data.hdf5`：12,136,423,279 bytes，SHA256 `9628041e8af6eb7176499ddf54a40239c6e3c348ca3c3f8c99087edd1f3b5406`。
- `outdoor_day1_gt.hdf5`：6,267,453,264 bytes，SHA256 `52dfffdb4c65497494f5f48c131134cff01e3a2b0d3fa59048d7189d25c8809a`。
- 两文件 ZIP 成员 CRC、解压长度和原生 HDF5 结构检查通过。额外取出的整个 indoor_flying1 数据文件 SHA 与本项目已注册的官方文件一致；五个室外标定成员与本轮从官方取得的 ZIP 逐字节一致。

需明确一个来源限制：本轮未取得官方室外全文件的独立校验值，因此不宣称室外镜像与官方全文件逐字节一致。官方与镜像 GT 的前 4,096 bytes 一致；data 的相同范围只在偏移 20 的一个字节不同。两者均为 HDF5 superblock v0，该位置属于旧版 file consistency flags；[HDF5 官方规范](https://portal.hdfgroup.org/documentation/hdf5/latest/_f_m_t3.html)说明 v0/v1 的该字段未使用。原件没有打补丁，也没有关闭 HDF5 文件锁来强行读取；此检查不能替代全文件独立认证。

为缩短传输，只复制评测使用的完整 `davis/left` 组及属性，省略不用的其他传感器组。逐数据集核对了 shape、dtype 与数组字节 SHA；事件没有抽样，时间、坐标、精度和 GT 有效区没有变。新容器文件 SHA 与原件不同，来源关系完整保留。源原件仍在 WSL，未覆盖。

服务器目标为 `/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-EINexus-left-20261107-v1`。完整左目 data/GT/标定正采用可恢复、限速传输。只有三个文件 SHA、HDF5 数组校验和标定 ZIP 检查通过，才会写 `verified_input_receipt.json`。本快照该收据尚不存在，不能声称 g1 输入已就绪。

当前无需用户手动操作。备用渠道包括 [MVSEC 官方 ROS bag](https://daniilidis-group.github.io/mvsec/download/)及 [E-RAFT 作者发布的预处理集](https://github.com/uzh-rpg/E-RAFT/blob/main/README.md)；前者需要格式转换，后者改变了时间组织，因此没有静默替换到当前 HDF5 协议。

[原生文件下载收据](evidence/mvsec_native/completion.json) · [来源对照](evidence/mvsec_native/mirror_reference_checks.json) · [官方文件头对照](evidence/official_prefix_audit.json) · [无损左目投影清单](evidence/mvsec_projection/input_registration.json) · [下载及传输源码](mvsec_sources)

## 位姿：先补光照覆盖，再按原门槛决定是否真实测试

上一轮固定定位器的独立深度短训已完成，但未过门槛；不能只看最暗条件的深度改善就采用。最暗条件下定位中心平均误差仍约为 EventGenesis 30.2 px、V2E 15.6 px，深度分支本身无法纠正这个横向偏差。

本轮在原有 96 条训练轨迹上，以完全相同物理轨迹为双方补齐 0.5 和 0.125 倍辐照度事件。倍率不是标定 lux。GPU 5 正在单 worker 生成；快照中半亮度 EventGenesis 已生成 56/96 条，不等于整个 384 个生成器/轨迹/亮度组合已完成。原正常光照和验证轨迹隔离不变。

条件管理进程已实际部署，PID 20425，当前仅等待数据，不预约 GPU。执行顺序为：

1. 等上游准备退出码 0，核对来源哈希和全部数据；对已有 8 条轨迹、两生成器、两亮度的 32 个重叠事件数组核对，与早期光照试验语义一致。
2. 在安全取得 GPU 5 后先跑双方 smoke，再做 2 方法 × 3 seed × 3 训练臂的固定短训。三个臂为正常光照控制、物理变光监督、配对一致性。每个模型包训练独立深度与定位分支各 288 次更新，共 576 次；双方预算、窗口、增强一致，固定最终权重。
3. 保留深度单独修复、仅平移融合和完整预测位姿三个合成消融，避免把定位或预测裁剪改善误认成深度改善。旋转头权重不变，实际裁剪和射线使用预测中心/尺度；没有 GT crop、GT 深度或 GT cell 输入。
4. 完整保留 30 项光照检查及此前对原始 PnP、固定中位深度的 8 项稳定性比较。光照检查仍用原四窗口子集；原八项确认单独使用原完整 96 轨迹候选窗，不能拿不同子集比较。全部通过才允许真实评分，失败则保存失败项并释放 GPU，不放宽阈值。
5. 通过后只运行预先指定的正常控制及配对一致性两臂，对双方三 seed 的全部 20,916 个已注册真实样本评分。此前这些 E-POSE 木块序列已经看过，因此结果必须标为开发诊断，不再称盲测最终结果；真实标签只在预测完成后计算指标。

其他三条 simulator 基线若纳入新位姿结果表，也必须使用同一最终下游配方，不能将旧网络结果与两组新网络结果拼成不公平排名。新部署尚未完成短训，更没有新的真实 E-POSE 成功率。

[协议](pose_sources/POSE_TWO_BRANCH_PROTOCOL.md) · [实际部署源码及 SHA](pose_sources/source_manifest.json) · [分支和恢复程序预检](pose_sources/two_branch_pose_preflight.json) · [重叠事件检查预检](pose_sources/pose_overlap_preflight.json)

## 可复核性与边界

本目录源码与 g1 实际隔离运行快照逐项 SHA 校验一致。发布时最终真实评价器在本地生成文件名为 `pose_two_branch_real.py`，运行目录中固定名称为 `pose_coverage_real.py`；此处保存运行时名称与原始 SHA，避免混淆早期版本。测试源码在 `test_sources` 保留执行时依赖路径，须从原工作目录及前序依赖执行，不能当成独立 CI 包。

CPU 预检覆盖独立分支的一步 Adam 更新、冻结头、教师不变、CPU 保存恢复、预测分支接线、原稳定性门槛保留、NPY 事件重叠验证，以及 ZIP64、CRC、断点来源和无损 HDF5 投影。预检不是新模型精度或正式 GPU 训练完成的证明。固定部署链还会执行实际 GPU smoke。

原先两个等待器均在获得 GPU、启动任何训练之前被本会话替换，其取消收据保留在服务器快照中；没有终止别人的任务。服务器原共享 Git 工作树没有被重置。GitHub 提交记录与 g1 源码快照通过 SHA 关联，不声称服务器共享 HEAD 已同步。

[打包核验](evidence/package_verification.json) · [服务器状态证据](evidence/server_snapshot.json) · [本目录文件索引](artifact_index.json)
