# Stage 4 固定合成运动学习诊断预注册

登记时间：2026-10-18（本地工作时区 Asia/Tokyo；提交记录为 UTC）。本文件和执行脚本先于读取诊断指标提交。

## 范围与固定检查集

- 只读取本轮 Stage 4 合成验证 split，不读取训练样本或真实数据。
- EventGenesis 与 V2E 使用同一 seed 0、optimizer update 4554、epoch 23 验证完成后的 checkpoint；二者配置、初始化、验证 adapter manifest 和 scheduler epoch 必须匹配。缺少 post-validation checkpoint 时不计算配对结果。
- 使用全部 8 条登记验证 rollout。每条固定取 adapter 行号 `0, 9, 18, 27, 36, 45, 54, 65`，共 64 个配对窗口；adapter 的 ID 使用物理端点偏移，因此对应 `dt1_000, dt1_027, dt1_054, dt1_081, dt1_108, dt1_135, dt1_162, dt1_195`。不按模型预测或误差筛选样本。
- 四个场景层为 `camera_motion_static_scene`、`independent_rigid_motion_static_camera`、`camera_and_independent_rigid_motion`、`combined_motion_with_occlusion`。每层固定两个验证 rollout。

## 固定区域和阈值

1. `full_valid`：共享物理 `flow_valid`。
2. `shared_event_active_union`：`flow_valid` 与 EventGenesis/V2E 五分箱事件表示任一时空/极性通道在该像素非零的空间并集。两模型共用同一张掩码；同时单独报告各组及并集的覆盖率。该表示经过固定 Gaussian count 编码，活动区域可能包含编码器的空间支持范围。
3. `gt_motion_ge_0_5`：`flow_valid` 且 GT 二维位移模长至少 0.5 px/30 ms。
4. `gt_motion_ge_1_0`：`flow_valid` 且 GT 二维位移模长至少 1.0 px/30 ms。

预测“近零”固定定义为模长小于 0.1 px/30 ms，并另报小于 0.5 px/30 ms 的比例。运动方向在两个固定 GT 动态区域中评估；预测模长小于 0.1 px/30 ms 的方向没有可解释性，因此报告其覆盖比例，并只在其余像素上汇总余弦、角度和 30/60 度内比例。

## 时间、单位和解码检查

- adapter 文件 SHA256 必须与固定 `training_bank_manifest.json` 一致；同一 sample ID 的两组 GT、有效 mask、灰度端点、K、起止时间和 label hash 必须匹配。再按登记的共享 bank hash，从原始 HDF5 读取所选 rollout 的对应 endpoint 灰度、时间戳、GT、mask 和 K，核对 adapter 内容。
- 事件区间是共享 100 Hz 物理 endpoint 上的 `(t_i, t_i+3]`，持续 30 ms；标签时间等于右端点。GT 来源字段为 `flow_forward_30ms_px`，方向从左端物理时刻前向投影到右端。灰度前后端点分别对应这两个 endpoint。
- GT 和预测单位均为每个 30 ms endpoint 区间的二维像素位移 `[u,v]`。STE-FlowNet 只接收事件体；模型输出须为 `[1,2,352,352]`，去除已登记 padding `y=46:306, x=3:349` 后得到 `[2,260,346]`。解码保留两个分量、方向和像素位移尺度，不额外重缩放。
- 在每个掩码报告预测 EPE 与零光流 EPE、幅值分位数、近零率和覆盖率；动态掩码另报方向和预测/GT 幅值比。诊断按固定窗口池化像素，不能替代原有所有 528 窗口、等 rollout 权重的验证 EPE，也不改变 checkpoint 选择规则。

## 产物

- 执行脚本：`benchmark/sim2real/scripts/stage4_motion_learning_diagnostic.py`
- 固定合成结果：`benchmark/sim2real/evidence/S4/20261018-motion-diagnostic/motion_diagnostic.json`
- 完整日志、两个大 checkpoint 副本及命令/环境记录保留在 g1 的独立 `motion_diagnostic_epoch23_update4554` 目录；仓库只记录路径、大小和哈希。
