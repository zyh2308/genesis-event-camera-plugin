# Soft-focus readout：科学门禁与来源审查

仅研究零更新、RAW输入、独立center读出。sigma固定1/16归一化坐标；不改原attention、depth内部attention、depth/size输出或任何权重。rotation权重/RAW输入固定，但必须按候选中心实际重跑原预测crop/rayframe；不能把固定权重误写成预测R也不变。GT只进入原评分，不能用于窗口/温度/ROI/光照规则选择。此文档不启动服务器任务、不修改旧repo。

## 1. 本轮机制advance：相对不可变RAW C1，独立于formal30+8

`ADVANCE_CRITERION.json` SHA256：`db9803873fc4da10b983477e34baa10d1188f515b133905be590808abebdd6cd`。36项必须全部通过，无质量epsilon：

- 两生成器×三个seed×正常/半亮：`candidate.summary.center_px <= control.summary.center_px`，且 `candidate.summary.success >= control.summary.success`，共24项。
- EG三个seed最暗：中心严格降低，完整pose success严格提高，共6项。
- V2E三个seed最暗：中心不升，完整pose success不降，共6项，避免通过损伤对照制造优势。
- 每method/seed独立计算，沿用原四轨迹等权平均；不以所有seed/方法总平均掩盖退步。每rid负变化完整报告，但没有另设每个rid或每个窗口必须单调改善的新门槛。

完整64原窗口、全部预测有限、固定源码/权重/输入、原RAW C1完整pose parity是前置必要条件。缺窗/重复/非finite不产生通过结论。`assess_advance`为纯JSON科学质量判断；模型/源码/CP/基线parity仍由driver完整性admission独立证明，不能用一个quality JSON替代它们。

全部通过仅允许进入真正未使用的共享合成机制确认，不能称overall30+8通过、已胜V2E、可真实评分或允许长训。任一失败即停止该候选，不扫sigma、不按光照切换读出。

可执行入口：`assess_driver_outputs(scores,case_rows)`适配driver的`full36 scores.json`与`1152 paired_cases.jsonl`。只读取原`score.summary.center_px`/`success`；另核两臂每窗UV/R和共同z/size有限、正深度/尺度、原64 ID。CLI：

```text
python3 -B assess_advance.py --scores <scores.json> --cases <paired_cases.jsonl> --out <fresh_result.json>
```

纯stdlib三项CPU测试已实际通过，覆盖完整36精确通过、EG暗光持平必须失败、V2E损伤拒绝、flat schema端到端、R NaN及缺case/window拒绝。

## 2. 原formal30：阈值、参考组与数据保持原样

原源码：repo `benchmark/sim2real/reviews/three_line_followup_20261107/pose_sources/pose_two_branch_coverage.py`，SHA `85951e6d43e59140b81d7303370a6b252fec902dc065ff17bb30faf8e4fc6186`，217–224行。

相同定义的可调用函数：同目录`pose_coverage_real.py:gate_rows`（16–24行），SHA `d9e5c0cb1a677999883c77630c20613f362b53ed40604e5f2360dc3c7adc2621`。

每method/seed比较`run['arms']['paired_consistency']`与`run['arms']['clean_control']`（即C0；legacy槽本轮若映射候选须明确canonical名称）：

1. `clean`完整pose success候选−C0 ≥−0.02。
2. `half`深度MAE严格小于C0，且完整pose success不低于C0。
3. `eighth`同上。每run五项×六run=30项。

`clean`是96条原fresh轨迹各取4个候选窗的clean384；不是固定四条轨迹matched-normal64。half/eighth是四条固定relit轨迹、每条原16窗。完整pose沿原ADD-S严格`< 0.1*CAD diameter`，不能用UV、AUC、depth-only或冻结旧R的诊断替代。

本轮只改center、depth保持C1，因此原30中已有四个半亮depth失败不能由readout消除：EG seed0/1/2和V2E seed1。数据来源为本轮完整固定64 RAW对角的`full_v2/raw_hybrid_summaries.json` per-seed表；C0→C1深度MAE（m）分别为：

| 方法/seed | C0 half | C1 half | 原严格depth项 |
|---|---:|---:|---|
| EG0 | 0.0340938506814 | 0.0458681888252 | fail |
| EG1 | 0.0184419151379 | 0.0248942503795 | fail |
| EG2 | 0.0354929487443 | 0.0370550789242 | fail |
| V2E0 | 0.0360114593921 | 0.0356662412575 | pass |
| V2E1 | 0.0427849468478 | 0.0461424823166 | fail |
| V2E2 | 0.0275546121263 | 0.0274360145219 | pass |

这不阻止相对RAW C1的读出机制对照，但禁止把formal参考组改成C1来消除这些失败。

## 3. 原confirmation8：已见历史集，不能冒充本轮独立holdout

入口`pose_coverage_real.py:original_confirmation`（26–39行）：从候选`clean_full.by_rollout`取96条轨迹，保留同物理轨迹的`parents.by_rollout.original`和`median`；转换成`learned.success/depth_mae`后调用原fresh gate。

具体统计函数：repo `reviews/pose_rotation_pilot_20261106/pose_fresh_confirm.py:gate`（124–142行），实际服务器对应`R/recovery_20261105_mainline/pose_fresh_confirm_20261106_v3.py`，SHA `d1b16345bbf174f535453fc4c8e9657afe2d154f4e98a11e20d7940e8463446c`。

两method×original/median基线×depth/success=8项。每项要求三个seed各自平均改进均严格>0，并且按轨迹bootstrap的95% percentile区间下界严格>0：固定rng50855、4000次抽样、N=96，先平均三个seed的每轨迹delta再重采样轨迹。不能把窗口或seed当独立场景，不改sign、顺序或阈值。

原数据：`R/recovery_20261105_mainline/jobs/119_pose_fresh_confirm_paired_20261106/results/pose_bank`，physical base seed `20261106017`，96轨迹，每轨迹16个原候选窗、沿不变visibility规则（最大1536）。clean384与当前四条relit轨迹都来自该已反复使用的96轨迹集；历史confirmation8以及后续多个候选已经看过，不能再叫本轮全新confirmation。旧smoke base seed是`20261106017+400000`，也已经使用过。

## 4. 当前development物理库存与最小baseline核验

固定四条：`validation_000/031/063/095`；每条k=4…19，共64窗，全部normal/half/eighth、两gen、三个seed保留，包括无事件/失败窗。数据由已锁BAF2 `D.prepare`选出；97–99行调用native `identity`有序list比较，native85951第15–16行包含ID及K/T/keys/UV字节，因此不是仅比较ID集合。support203525又核全21帧物理/时间及选定端点mask。

最小baseline先验：两gen seed0全部光照的固定首8窗，用原native readout复原冻结RAW C1的center UV、depth内部UV、log_size/log_shape/z（沿原2e-6字段parity）；原完整pose RAW baseline还须和已存原native summary按既有`rtol=1e-6, atol=1e-8`一致，且所有64窗仍在。通过后才完整六组。候选完整pose必须重跑原`shared.frozen_rotation`（`pose_light_robustness.py:155`）的RAW图+候选中心+原预测size裁剪及rayframe，再由原`shared.score:168`评分。

sigma和读出规则固定后，即使某一rid明显退步，也不得删去；按原规则记录所有rid/seed负变化，供解释稳定性。

## 5. 仅在36通过后可准备的新合成确认：可复用路径，不立即生成

最小物理生成入口为已锁`pose_fresh_confirm_20261106_v3.py:generate(args,count,physical_seed)`（48–90行），同一campaign的冻结`pose_physical_worker.py`、sensor profiles及两event replay脚本不变。函数显式接受`count`和`physical_seed`，原worker每rollout使用base seed+1009*i。不能直接再跑其旧main（其正式main硬编码旧seed/96且还含条件性real-on-pass入口）。

如果机制36全过，根会话可在新独立root登记少量同分布轨迹的**新base physical seed**，先按实际已用bank/H5 `scene_seed`清单核无重叠，再调用既有generate。只换名字/目录、复用旧96或旧smoke不算新轨迹。没有核全已用seed清单之前，此审查不替根会话擅自宣布某个整数绝对未用过。

同物理光照覆盖可复用`prepare_pose_radiance_pairs.py:relight(path,out,rollout,gain,tone)`（40行起），repo `reviews/pose_light_execution_20261107/prepare_pose_radiance_pairs.py`，SHA `0526adbb228a1787cbd1fb35ded40173775e4034a1516f7561a26bbea901068e`。gain固定0.5/0.125，原canonical tone mapper不变，只改radiance/luma，几何/标签外链回原新H5；每level双方event replay共享物理轨迹。该旧CLI main固定指向已见96及旧SELECTED，不能原样用来声称新confirmation；复用其函数即可，无需现在另写generator。

这项少量全新轨迹确认是本轮机制复验，不自动等同原N=96的confirmation8。不得编辑原db98 development criterion以容纳新ID；新确认若发生，应另登记其固定库存与同一读出/不扫描规则。36失败则停止，根会话不生成、不占卡、不长训。

## 6. 当前canonical C1的机器绑定

来源：`outputs/pose_support_attention_scale_20261108/research_evidence_lite/full_v2/completion.json`，SHA `16be2e8e12c23b4791bdf312be1268621d56e99ff4893f0bff6e6eb2876e9e6c`。下表由其model_records与同一receipt的source_records逐路径join生成，不手抄或选择checkpoint。所有路径公共前缀：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-pose-slow-features-v1/study/`。

| 方法 | seed | branch | 文件 | file SHA256 | model-state SHA256 |
|---|---:|---|---|---|---|
| eventgenesis | 0 | center | eventgenesis_seed0_physical_supervised_localizer.pt | 89740206546637abac868305e7faf04fd45aa133cff549827474fc2d7a39bc2c | 60f3173f17fe2da9b6aff2154477e636f6a61babd4e35c44c44f0d397cf44737 |
| eventgenesis | 0 | depth | eventgenesis_seed0_physical_supervised.pt | 61af93c4a37832198e4b9e7eaf2f5ae2124c57f9608a84ff0a56b3ce81d3dfa4 | 0a71cd36fa840d7c2e3d6abf64b53ab438b90882eae2542aa741efb556233498 |
| eventgenesis | 1 | center | eventgenesis_seed1_physical_supervised_localizer.pt | 82e475e31c1d7e404ff2e8f5062906b5eb0fa5bbcb82cf43cccc1cad9f857eed | f218075289151eb816d935de29a3805b06a8d561531614d31c944b5bb5f55a25 |
| eventgenesis | 1 | depth | eventgenesis_seed1_physical_supervised.pt | 22284a16c7d315cb85083d3156fc41914922a79ed07718f31ddd2fb7843af3cb | 747e2326612f37fef00bba2ba72d6ca4b3ab8de189a26543f9d0ec7f2e903d71 |
| eventgenesis | 2 | center | eventgenesis_seed2_physical_supervised_localizer.pt | 143cad7f2d3d598b01f3b5bfd0bd28d86452e32672672db04abe0986b83eede4 | c0b493f28b08245bcea25d3e61e128f9d8627a7adcb86969fbc3ae3b670f49d6 |
| eventgenesis | 2 | depth | eventgenesis_seed2_physical_supervised.pt | c5afc900654bc18267f886c5bdedf744614a14088e5a8dfd60abf7f7cbf23ab5 | a7fe2bf8b6e4e720817979fd3724f62ab154b5d392d88477f9cb34b32be986f7 |
| v2e | 0 | center | v2e_seed0_physical_supervised_localizer.pt | 9cd27239f240b4a10215c997e1421f456383aec3382e0ff29ce8f757fb996bdd | 8b34a71d000145c21fae5f941beb27ede617fe32b9007fda0760dc5a4642e1df |
| v2e | 0 | depth | v2e_seed0_physical_supervised.pt | 78af29fb4fc1ddb9f0697e950a8620d77e9335a2c7cd0a6d64b611bf9087177f | 5d314021303a57cb288f20a4200be7d3ca23634c4add232f5529cd3508d4e3a0 |
| v2e | 1 | center | v2e_seed1_physical_supervised_localizer.pt | 00fc4f0f02af71b79289871064b01b5b6ea5c2722e0975d378ec5a18dc0ad9e9 | a4249a02c1b2d7146e380184a73387742ea4abc08e52c5f1d83aaaf8b42162de |
| v2e | 1 | depth | v2e_seed1_physical_supervised.pt | 22941c5fd8955d845dcdf36f03824c0657e610bcb7b0b29ecf13cce754fcf192 | f9b1c10bf8e73b26a39099e969ed14aef5fdde7bae97a12ec982ba9a6b8bc1e5 |
| v2e | 2 | center | v2e_seed2_physical_supervised_localizer.pt | 7827e0a57c6d7695895d996a2e32d0ec630e4143f5dacfe0aa40b6e6f76ba961 | efa60d51df17c25fc558c8c3d5e6639f38908386399eb2b4dd34d98713eb68df |
| v2e | 2 | depth | v2e_seed2_physical_supervised.pt | c992d0fdb6781ee3a6205e1c71e032dba37be3e53125d1ea294fe9c5bfb08c11 | 4c38ab3e5ec3535d8d6667e02ac6048d3a6abdc8f7f89da96bd376a617dfca11 |

完整pose还须复用下列原rotation checkpoint（不是attention诊断输出）。来源为已归档BAF完整pose执行收据的source_records，原文件SHA为 be98929cbc2359557d9693e89e3517b788ef3348088c2a4a4c5f1d511fbdaeff；此处只读取已发布metadata，没有重新读大权重或推断新的state hash。

| rotation文件路径 | file SHA256 |
|---|---|
| /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/recovery_20261105_mainline/jobs/117_pose_rotation_paired_20261106/results/eventgenesis_seed0_rotation.pt | 30c80365636d2f8270c3e84ab1f46c1bdd7ed28993b4293a0956200589a7cd14 |
| /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/recovery_20261105_mainline/jobs/117_pose_rotation_paired_20261106/results/eventgenesis_seed1_rotation.pt | 51b6fc8e3d9ae6f19a910bb43f00000e1d5bcb291e633b550dda9749fc36ea8e |
| /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/recovery_20261105_mainline/jobs/117_pose_rotation_paired_20261106/results/eventgenesis_seed2_rotation.pt | da0019729484424e1edfc58d737b70cd1b888f1cbc0670cbc3eaa3edc5e032a6 |
| /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/recovery_20261105_mainline/jobs/117_pose_rotation_paired_20261106/results/v2e_seed0_rotation.pt | 7623bf5d97bfeb910eea90954f168eaca078b7daadb6a9a89c24906c2304791c |
| /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/recovery_20261105_mainline/jobs/117_pose_rotation_paired_20261106/results/v2e_seed1_rotation.pt | eef0e74846ce85ed2c910dde2a81aec82ed4a67555e62dc7eb5b70cfcfa55a3c |
| /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/recovery_20261105_mainline/jobs/117_pose_rotation_paired_20261106/results/v2e_seed2_rotation.pt | 471e4e2f8cb3ac6c8a0b0862cedf162c0374fcbcb93702e1d44e9954ab76bbd7 |
