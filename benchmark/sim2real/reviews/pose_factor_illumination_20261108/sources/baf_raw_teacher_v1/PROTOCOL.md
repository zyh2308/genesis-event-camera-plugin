# BAF 学生 + RAW 正常光教师：等预算中心短训

旧的固定 BAF 中心短训已完整结束但仅20/36阶段检查通过，原结果不覆盖、不采用。随后零更新诊断发现 source-init teacher 的正常光 TRAIN 中心误差明显增加：EventGenesis RAW2.72→BAF8.36px，V2E4.26→17.11px。这仅支持“过滤输入使旧 teacher 保留目标偏移”的机制假设，不证明唯一根因。本轮只检验改回 RAW teacher 输入的效果，禁止进一步调参、换教师或扩训练预算。

## 唯一干预：教师正常光输入

学生输入继续使用旧 smoke 的固定 BAF128 缓存。normal/half/eighth 的过滤规则、源事件、窗口、增强、监督、模型与原初始化均不改。两个生成器各自用自己的原 teacher，不跨生成器蒸馏。全部 3 seeds，六个新 center 分支，各288步，共1,728更新；不在旧 C1 或旧 BAF 权重上继续训练。

新 `raw_teacher_policy.py` 从原 `e1883b…` 文件原字节生成，仅有两处语义改动：

1. `x,y=shared.affine(BAF_normal[idx],target[idx],rng)` 改为 `x,raw_teacher_x,y=native.matched_affine(BAF_normal[idx],RAW_normal[idx],target[idx],rng)`。
2. normal retention 的 `teacher(x)` 改为 `teacher(raw_teacher_x)`。

`native.matched_affine` 给 RAW 对照使用正式 NumPy RNG 状态的独立副本，只让第一条 BAF 增强消费正式 RNG。因此标签和后续 draw 不变，不允许再独立抽一次教师增强。原 normal teacher 权重2、half概率.75、auxiliary teacher权重0、lr1.5e-4、AdamW/clip5、batch32、6epochs、center的size-head冻结与所有其他源字节都不改。`POLICY_DERIVATION.json` 与标准库测试验证两次替换唯一、反向替换可逐字还原原文件；新 policy SHA `9e713cc63aefe10c53b27505101f1a4271483a20a1657b5c64ca2de838d9aa04`。

## 缓存、准入和来源

复用旧 db07 source / 60db contract / dbfd smoke receipt 绑定的两份 BAF128 缓存，不重生成、不修改。所有旧 input_lineage 的实际 NPZ、原始事件、HDF5、manifest SHA 重核。

额外 RAW normal128 只来自原 pinned NPZ loader 的1,529窗，ID顺序必须和旧 BAF cache.normal_ids完全一致；原标签必须和旧 cache.y **逐元素完全相同**，不引入 fresh GT 的舍入变化。仍用原 `shared.tensors` 在 CUDA 做 area128再回CPU，每生成器约300MB；smoke保存新RAW缓存、source/contract SHA，study只读回这两份缓存。RAW只作为训练中的 teacher输入，学生的 BAF缓存和推理输入不变。GT不输入teacher或filter。

任何实际更新前，两个生成器×三seed全部六组都先完成完整288步原 C1 的 source-init、indices、aux indices、half mask、增强标签 hash 核验。新 paired-normal 路由再独立重放并与原 C1 exact 比较。正式训练重新初始化原 NumPy/torch RNG，admission 不推进实际训练 RNG。训练后日志和标签 hash 再核。失败即保留并停止，不自动续训或追加预算。

## 四臂与门槛

新 driver 复用已锁 db07 旧 worker 的原评价函数：RAW C1；冻结 C1 看BAF；新 `raw_teacher_baf_center`。再按相同 method/seed 从已完成旧study复用 `filtered_teacher_BAF`，只有 RAW C1 和 frozen_C1_BAF 在 clean384 / matched_normal64 / half64 / eighth64 的范围、中心误差与完整pose summary 都重新核对一致后才发布四臂。

depth仍固定C1且看RAW；rotation仍固定原head且看RAW，用每臂预测中心做原crop/rayframe；禁止GTcrop。新36项中心阶段门槛仍相对 RAW C1：正常/半亮中心EPE不升，暗光中心EPE严格下降，正常pose下降≤2pp，半亮/暗光pose不降，逐seed检查。额外报告新旧teacher路线差值，不能用“比旧 BAF 好”替代原门槛。

原全局30项和原8项仍为 **not_evaluated**，不放松，不因中心阶段分数启用真实 E-POSE；无真实数据拟合、评分或选权重。checkpoint仅独立 model/Adam 保存读回与重建，不宣称恢复RNG/cursor后的下一步或整段轨迹等价，`resume_supported=False`。

smoke只对两生成器seed0各1实际更新（共2），独立读回后study六组各288。两个阶段各3,600秒只是安全上限，不增加更新数。

## 验证与执行

旧目录11文件/包不改。`test_raw_teacher` 在严格核对旧db07源后，原样载入旧15项CPU测试，再加2项生成diff测试和6项RAW路由/成对RNG/288重放/实际双微更新/无突变/缓存/门禁测试，合计23项。Windows标准库只先运行2项源生成测试与最终AST；Torch完整23项由根会话在服务器CPU实际执行，不虚称本地完成模型测试。

```text
python3 -B -m unittest -v test_raw_teacher
python3 -B run_raw_teacher.py --contract contract.json --phase smoke --out <new-smoke>
python3 -B run_raw_teacher.py --contract contract.json --phase study --out <new-study> --smoke-receipt <new-smoke/completion.json> --smoke-receipt-sha256 <actual-sha>
```

合同 SHA `d682813400a1709c90b47b4dcc73a83c8f7ea3fe830e9d837316555dea11ca41`，更新前登记 commit `36441ca`。根会话审核部署；本子会话不SSH、不占卡、不启动实验、不Git。
