# 接触参数接入只读审查

2026-11-08；审查共享工作树 `gec-robot-catch-hdr-utility-20261108`，主会话部署基线 `2f063b7`、`source_contact_1a013dfe46f1`。本审查不修改运行代码。

结论：当前调用方式符合共同物理场景校准目的，未发现按生成器区别调参或修改成功标准的行为。参数校准仍须用完成后的真实逐回合结果判断，不能由单个成功直接选最终配置。

| 审查项 | 证据 | 判断 |
|---|---|---|
| profile来源与内部步长 | `genesis_adapter.py:34–36,52`：环境变量选择共同profile；外层dt仍使用原模型timestep，仅substeps随profile | 不改变policy或传感器外层节拍 |
| 手部选择 | `:93–106`：按实际Genesis link对应原生body，找本地位置最近的实际collision geom，再判断原任务手部语义范围；位置误差>1e-5或候选分类矛盾即拒绝 | 避免重复metadata名称造成非手部误选；当前实际17个GS collider33–49，与掌部和16指节一致 |
| 基准与首次应用 | `:107–110`：scene.build后初始化ContactCalibration，再apply | 基准从未缩放的live参数取得，不会把1.5倍当作下一次基准 |
| 重置后应用 | `:176–178`：scene.reset后apply，再设置实际qpos/速度/重力、初始化传感器 | 每集一致；不把profile应用到已经开始的飞行途中 |
| 参数记录 | `:136–142`：profile、完整contact audit、手部选择、外层dt、内部dt都存档 | 可核对球、轮子、地面未改变 |
| 物理推进 | `:203–223`：仍原PID/IK力控制，scene.step一次后只增加原外层dt，按原event_endpoint_stride采集 | 子步2不增加事件帧预算，不额外改变控制频率 |
| 核心/策略/成功判据 | 新调用只涉及scene选项、摩擦setter和metadata；无新增控制器GT输入、附着或计分修改 | 与用户授权范围一致 |

两个非阻塞记录点：

1. 首次apply后在内层audit加入`selection`，reset后新audit不再有该内层键；完整清单仍保留在顶层`contact_hand_selection`。最终审计应读取顶层，不误报白名单丢失。
2. 旧`geom_map`仍可能因导入metadata重名而复用逐指原ID；新摩擦白名单没有依赖它。复用ID都位于同一原手部大类，因此不改变当前手部接触分类或strict成功规则。此次无需为它改动运行代码；若后续分析逐指接触，应使用实际body/localpose匹配。

主会话已取得的首批runtime参数：球摩擦.95保持不变；所选手部按1.5倍；substeps1/2对应实际最小时间常数.004/.002秒。后者包含接触约束参数变化，所以报告应称“共同接触profile”，不要称为仅增加计算精度。

绘图工具：`plot_contact_calibration.py`。它只读统一JSON中的已审计`verified_catch`，不从raw success自行推断；例如`success=true, terminated=true`仍须由上游审计传入false。缺失回合标灰，不当作成功或失败；`--expect-baseline 3/6`要求实际baseline六条逐回合数据确实为三次严格成功。绘图不读取或选择事件方法结果。
