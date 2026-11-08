# 重建五方法、三种子完整训练：独立部署

本包仅部署训练。共同配方为原生 sigmoid、LPIPS-VGG 1、TC 2，不采用未通过共同门槛的 L1。冻结 core、噪声、遮挡、事件 bank、24/8 整轨迹划分、学习率/优化器和 200 epoch / 12000 次更新全部不变。

g3 每张 RTX4090 固定一个 seed 的完整五方法，保证同 seed 同硬件。seed0 的 EventGenesis/V2E 从旧 safely-stopped last.pt 的 5264/5306 更新恢复，模型、Adam、RNG、epoch/batch cursor 精确核对，复制旧合成最优 checkpoint 及原训练日志至新目录后延续最优历史。其余十三组从相同 seed 初始参数开始。旧运行目录和权重不写入、不覆盖。

原训练器和修复 recipe 按字节复用，维持原有严格 resume 指纹；新改动只在运行门禁/控制器。旧 RTX3090 的两组前缀与 g3 续训的硬件沿革会保留，不能声称全程 CUDA 逐位复现。

完整稳定性门禁包含五方法×三 seed：复用已经固定的 EventGenesis/V2E 六份 300 步合成证据，核对其 source manifest、完成记录、实际 checkpoint SHA 和配对关系；Vid2E、FILM+ESIM、RIFE+ESIM 各三 seed 补跑 300 步、8 条完整 200 帧验证。全十五项稳定性门禁通过后才进入正式长训。任何失败全局停止，不删除 seed 或放宽阈值。门禁权重不作为十三组正式训练的初始权重。

三个 stdlib CPU 父管理器先取得 GPU UUID 原子租约，preparing 最多 30 分钟，不创建 CUDA 占位进程。root 在 CPU admission 通过并核对包之后，最后单独发布 DEPLOYMENT_READY.json。父管理器检查 source/admission/plan 三个 SHA 后运行同 seed 工作队列。整体 48 小时，每个完整训练进程最多 8 小时，异常只清理自己进程组。早完成 lane 立即释放；最后一个 lane 在全部十五组完成后原子冻结各自 synthetic_best，随后也释放。

真实评分此包默认关闭。后续独立 evaluator 只能对预先指定的 boxes、reflective_materials、still_life 做完整序列评分，明确标注 posthoc_selected_hqf3_development；不能称独立盲测或用这三条代表 HQF14。三条在已见结果后选择，旧全14和失败结果保留。真实分数不参与训练、checkpoint 选择、阈值或后处理。

## 部署接口

1. 上传 source_manifest.json 所列文件到共享新 ROOT/sources，保持字节。
2. 已授权的父管理器可先运行 `reserve_and_run_lane.py --seed 0/1/2` 取得 metadata 租约，等待 staging。
3. 使用共享 conda Python 运行 `sources/prepare_selected.py --root ROOT`。它只做 CPU admission，不写 DEPLOYMENT_READY，不启动训练或真实评分。
4. root 复核后以独占创建写 ready：schema=recon_selected_deployment_ready_v1，passed=true，source_manifest_sha256=sha(ROOT/sources/source_manifest.json)，cpu_admission_sha256=sha(ROOT/cpu_admission.json)，registered_plan_sha256=sha(ROOT/sources/registered_plan.json)。
5. 完成路径为 ROOT/training/seed{0,1,2}/{method}/{completion.json,run_summary.json,synthetic_best.pt,synthetic_best.json,last.pt}。最后冻结 ROOT/frozen_checkpoint_matrix.json；ROOT/completion.json status=full15_training_complete_selected_eval_pending。

本包不会自行发布 ready、改变原代码仓库或申请其他显卡。
