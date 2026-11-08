# 固定soft-focus读出：完成，但不采用

这次零更新配对对照已完成，机制门槛仅 **27/36通过、9项失败**。保留原RAW C1读出，不采用候选，不生成后续独立confirmation，不启动长训或真实E-POSE评分，也不继续扫描sigma。

候选只把独立center分支的原attention乘以围绕预测峰、sigma固定1/16的Gaussian空间权重后重新读UV；两生成器、三seed、三个光照和原四条轨迹全部应用相同规则。没有使用GT区域决定读出。depth/size与权重固定，完整pose实际按两臂各自预测中心重跑原rotation crop/rayframe。

以下是四轨迹等权、再对三seed等权的完整64窗合成验证均值，不能当作真实E-POSE结果，也不能代替逐seed门槛。

| 生成器 | 光照 | 中心误差：RAW C1→候选，px↓ | 完整pose成功率：RAW C1→候选 |
|---|---|---:|---:|
| EventGenesis | 正常 | 8.2147→7.9664 | 35.9375%→39.5833% |
| EventGenesis | 半亮 | 9.9930→8.4566 | 37.5000%→46.3542% |
| EventGenesis | 最暗 | 29.2777→30.9513 | 20.3125%→22.9167% |
| V2E | 正常 | 14.6062→13.4195 | 29.6875%→30.2083% |
| V2E | 半亮 | 15.7818→14.2475 | 33.3333%→34.3750% |
| V2E | 最暗 | 15.7452→16.0714 | 31.2500%→31.2500% |

均值部分上升，仍没有达到预先登记的稳定性要求。例如，EG最暗三个seed的中心误差全部退步；EG seed2正常光中心8.71475→9.19154px；V2E seed2半亮成功率48.4375%→45.3125%。这些失败不能被其他seed的收益抵消，也不能按光照选择不同读出。完整九项失败已按实际工件逐项重算，列于同目录`AUDIT_RESULT.md`。

此前固定k12图的数学分解证明：部分窗口的区域外attention尾部确实对softargmax偏移有正贡献。这不意味着围绕预测峰进行任意软聚焦都有益。本候选根据预测峰抑制远处权重，并没有精确识别背景；峰在物体上也不等于峰位于目标CAD anchor。前景内部定位、合法空间支持以及后续rotation裁剪仍可能产生相互影响。这些是解释限制，当前实验没有识别出唯一新根因。

执行完整性与质量结论分开：本轮完成1152个配对窗口、36组原生评分，0次optimizer更新，原18个method/seed/light控制cell的预测/完整pose、模型状态与源文件哈希均通过。执行侧独立CPU复算记录1152份概率、2304份完整pose以及36cell共1260个原指标比较，七项指标最大绝对差为0，success完全一致。进程正常结束、复算一致，不等于候选通过质量门槛。

原formal30继续以C0为参考，原confirmation8定义不变；本轮36项是相对RAW C1的机制advance，未宣称formal30+8通过。本轮不修改仿真器、噪声、遮挡规则或已有正式权重，不产生新的Sim-to-Real泛化结论。

本地已核对lite包58项工件大小/SHA，从实际原生scores独立重算六行均值及36项门槛，并逐值比较g1独立CPU复算后的1260个标量，差值均为0。原始概率map、逐窗预测及CAD大文件仍在g1；原1152概率/2304pose的几何重放由已归档g1独立执行收据证明，不宣称本地重放了这些大raw。详见`AUDIT_RESULT.md`。

## 可复核入口

- [三seed均值表](evidence/three_seed_results.csv)、[18组配对表](evidence/paired_seed_results.csv)、[72组逐轨迹表](evidence/paired_rollout_results.csv)。
- [预登记36项及实际结果](evidence/mechanism_advance_result.json)、[独立审计](AUDIT_RESULT.md)、[原30+8门槛边界](AUDIT_PROTOCOL.md)。
- [实际退出收据](actual_execution_receipt.json)、[冻结前向完成收据](evidence/full/completion.json)、[独立CPU数值重算](evidence/independent_verification/AUDIT.json)。
- [新driver](evidence/driver_sources/run_readout.py)、[纯概率读出](evidence/math_sources/softfocus_probability.py)、[独立核验器](evidence/verifier_sources/verify_readout.py)。
- [原始大工件路径与SHA](evidence/raw_server_artifact_index.json)、[紧凑原字节包](research_evidence_lite.zip)。本轮g1根目录为 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261109-pose-predicted-peak-readout-v1/`。全部源和结果始末固定，原大maps/cases/几何仍在g1。

## 另外两条线

重建g3在2026-11-09 01:11 JST的独立快照为4/15组完整训练完成：EventGenesis三个seed及V2E seed0完成，Vid2E seed0与V2E seed1/2继续，其余8组等待；HQF开发三序列评价器仍等待全15冻结。见 [原始状态](reconstruction_status.json)。光流正式训练15/15不变，01:17 JST固定MVSEC评分8/15完成、2组运行、5组等待，未读取新真实分数调参。见 [状态证据](evidence/other_line_status.json)。
