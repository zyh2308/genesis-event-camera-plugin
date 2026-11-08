# 固定 BAF 输入：中心模型等预算配对短训

候选 `fixed_baf_center` 仅改变中心分支的事件输入。原始 C1 的中心训练在每个生成器、每个 seed 各运行 288 步。本轮从**同一个原始 source localizer 初始化**重新运行 288 步，绝不在已经训练 288 步的 C1 上继续加训。两生成器 × 三 seed 共六个新中心分支，**1,728 次新 optimizer 更新**。原 C1 全部固定；原 depth 模型与 rotation 模型不训练。

已有零更新诊断提示：BAF 对 EventGenesis 暗光独立验证的中心误差有改善，对正常/半亮和 TRAIN 有明显退步；因此本轮检验“中心网络重新适应过滤输入，能否恢复原正常/半亮度定位，同时保留暗光收益”。这是假设，不是预先宣布已解决。BAF2 默认阈值不根据本轮指标再调节，不启动真实评分。

## 唯一处理与不变训练项

- 固定 DV 风格 BAF：8 邻居、排除自身、不分极性、严格 `<2,000 µs`、零初始化、每个输入（包括拒绝事件）更新时间面。整 episode 因果过滤，再取原 `(start,end]` 窗口。原浮点秒仅乘 1e6 为 float64 微秒做决策，不 round、不改原文件/时间戳。这是连续时间扩展，不宣称分数微秒下与 DV 整数实现逐位一致。规则和测试复制自已锁 BAF v2，源 SHA 必须相同。
- [DV 官方规则源码 bf8c0ee8](https://gitlab.com/inivation/dv/dv-processing/-/raw/bf8c0ee8/include/dv-processing/noise/background_activity_noise_filter.hpp)；[TimeSurface 零初始化](https://gitlab.com/inivation/dv/dv-processing/-/raw/bf8c0ee8/include/dv-processing/core/core.hpp)。保留相同启动语义；不做阈值网格。
- 直接调用原字节 `pose_priority_policy.py` 的 `BoundedFitter.fit_center`（SHA `e1883b68715ecf067a9f4dccbd7a61af5f5dc26b557e061859f4066e7de8d38c`）。监督公式、normal teacher 权重 2、half:eighth 3:1、batch32、6 epochs、AdamW lr=1.5e-4 / weight_decay=.01、clip5、原 size-head freeze、增强、RNG 都不改。pair_weight=0，没有半亮度 KD 或新特征项。
- normal 与 aux 都使用相同固定 BAF。原 frozen teacher 看同一个增强后的 **BAF normal** 图，不做 raw→filtered 蒸馏。原始标量监督/KD 与 e188 完全相同。
- 原 `physical_supervised` 仅作为 `BoundedFitter` 内部 legacy 标签；外层候选名称 `fixed_baf_center`，表格短键 `baf_center` 明确映射。旧 C1 不覆盖、不重新选择权重。

## 原标签、原顺序和内存约束

normal 的 1,529 个窗口由实际 SHA 绑定的 `pilot.helpers(causal_tests_v2.py, R)['load'](method,'train')` 读取，保留 **NPZ 原标签和原 ID 顺序**；不替换为 fresh 的近似相等 GT。释放 NPZ 大图后，用 fresh 按相同 ID 对齐新的 BAF 图。aux 按原 `prep['selected']` 96 轨迹顺序，先 half，再 eighth，保留 fresh 原标签与原顺序。aux_clean 是同几何正常光 BAF 图的两个相同副本。

每次只由 fresh adapter 编码一个 episode。416 图经原 `shared.tensors(...,CUDA)` 变成 area128，再拷回 CPU 缓存并释放 416 原图及滤波图。不得持续保留全部 416 图映射。smoke 保存 CPU128 缓存及整个文件 SHA；study 只复用由相同合同、相同源 manifest、已通过 smoke 绑定的缓存，不再重复全 bank 过滤。

缓存另含实际输入 `input_lineage`：原96个 NPZ、radiance_train completion、各已存在 bank_manifest、必须的每个 event manifest，以及所有实际使用的 physical HDF5 和 raw event NPY 的完整文件 SHA/字节数。relit 的 bank_manifest 若没有发布不伪造，但其事件 manifest 与全部实际 HDF5/NPY 仍必须锁定。生成 cache 前再次核验输入，study 读取 cache 时重新核验真实输入字节；这些输入记录也进入最终 source_records 并在阶段结束复核。源码 SHA 不充当输入数据 SHA。

**任何训练更新前**，两个生成器 × 三 seed 全部六组都完整重放原 288 步的 normal indices、aux indices、half mask 和增强后标签累积 SHA。逐项匹配原 C1 日志与标签 hash；任一不符停止。每次 admission 用独立新建 NumPy RNG，原 Fitter 训练入口重新设置 `71070+seed` / `90710+seed`，不会因预检消耗实际训练 RNG。实际候选日志再次核对全部 draw 和标签 SHA。此检查证明固定数据/增强序列一致，不声称非确定性 CUDA 整段优化轨迹逐位一致。

## 阶段、验证与门槛

`--phase smoke`：也先做完整六组 288-step admission，然后只对两个生成器 seed0 各做一个真实中心更新（共 2）。原 Fitter 保存并独立读取模型与 Adam 状态，重建 optimizer 核验；其余 seed 不训练。`--phase study`：必须提供实际 smoke completion 路径和 SHA，完整重放再核验后做六组各288。两阶段最长各3,600秒仅是安全上限，不增加预算。根会话在已授权 GPU4 queue 下部署；此脚本不抢卡、不 kill 他人进程、没有自动真实评分。

本次是一次性有界短训，**不支持中断续训**。checkpoint 检查只证明 model/Adam 保存、独立读取和重建，没有从已保存 RNG/cursor 恢复并执行下一步，不能声称完整 continuation 或整段轨迹等价。receipt 明确 `checkpoint_verification_scope=model_optimizer_readback_only_no_resume_claim`、`resume_supported=false`；失败保留现有输出，后续若获授权只能使用新 root 从相同 source-init 重新开始，不能悄悄续训或覆盖结果。

验证固定三臂：原 C1 看 RAW；冻结原 C1 中心看 BAF（零更新诊断臂）；新中心看 BAF。正常光 `clean384` 与同 4 个 relit rollout 的 `matched_normal64` 单列，另有 half64 / eighth64，不混用。原始 C1 的 clean384/half/eighth 完整位姿 summary 必须和既有 native C1 一致；matched64 原 C1 的中心 UV 必须和已锁 GPU 导出相差 ≤2e-6 normalized。深度始终使用原 C1 对 RAW 图的 z/size；rotation 始终用 RAW 图及各臂预测中心做原预测 crop/rayframe，绝不使用 GT crop。

中心阶段逐 method/seed **36 项**：clean384 中心 EPE 不升；half 中心 EPE 不升；dark 中心 EPE 严格下降；clean384 pose success 下降不超过2pp；half/dark pose success 不降。每个 seed 单独执行，不用均值掩盖失败。中心 EPE 按 rollout 等权。

这不是原全局30项照明门槛，也不是原8项 full1536 确认。两者在本轮明确写 **not_evaluated**，阈值不放松；depth 原问题未修稳，不能仅凭中心阶段通过开启真实 E-POSE。所有源文件、原模型及原日志前后 SHA 复核，新增模型单独保存，0 real-data scoring。

## 入口

```text
python3 -B run_baf_center.py --contract contract.json --phase smoke --out <new-smoke>
python3 -B run_baf_center.py --contract contract.json --phase study --out <new-study> --smoke-receipt <smoke/completion.json> --smoke-receipt-sha256 <actual SHA>
```

合同 SHA：`60db65c84cba5c6045542eca90640017f4e863e5e8ad9cab774dae778deee762`。本地 CPU fixture 不读取项目 bank、真实数据或 GPU，实际做两次小模型 optimizer 更新和独立 Adam 恢复；正式 GPU smoke 由根会话另外执行。
