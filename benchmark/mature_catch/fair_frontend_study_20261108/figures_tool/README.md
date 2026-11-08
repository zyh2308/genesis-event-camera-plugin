# 公平前端与盲测结果制图

`make_fair_figures.py` 只读已完成的 collector 与固定动作成本记录，输出独立 PNG、可编辑矢量 PDF、逐格统计 CSV、逐帧率配对检验，以及输入/输出哈希清单。脚本不连接服务器，不运行物理、事件生成或训练，不更改仿真器。

## 三张正式图

1. `frontend_crossover`：三个公共候选，Ours/V2E 全交叉，25/12.5 Hz 各一面板；每格同四个开发 seed。用于选择前端，不能作为盲测排名。
2. `blind_success_vs_rate`：两来源各锁一套全局前端，在相同 32 个新 seed 上比较 100/50/25/12.5 Hz。两条来源曲线带 Wilson 95% 区间；初始化后停止更新、含噪状态参考各只运行一次/seed，画独立水平参考线。
3. `blind_success_vs_cost`：盲测成功率对实测成本。成本来自相同固定动作轨迹的 0.32–1.12 秒完整窗口，与任务成功/提前失败时长无关。横轴为运行墙钟秒/模拟秒；每来源×帧率至少三次，画均值和极差。只接纳相同主机及同一张 GPU 的原生非审计、非详细仪表测量。没有成本数据就不生成该图。

配对表 `blind_paired_by_rate.csv` 给出双方都成功、仅 Ours 成功、仅 V2E 成功、双方都失败，并报告逐帧率双侧 exact McNemar 与四次检验的 Holm 调整。每档分母是 32 个配对 seed，不能把四档合并声称 128 个独立样本。可选 bootstrap 每次整体抽取 seed，保留该 seed 在所有帧率、来源的相关结果。

## 输入与停止条件

每个正式 stage 目录须有以下三项，并已由 collector 完成所有实际计划回合：

- `FAIR_COLLECTION_COMPLETE.json`：PASS、所有任务正常退出、任务失败全部保留、结果 JSON 哈希。
- `FAIR_STAGE_RESULTS.json`：metadata.stage 为 validation 或 blind，实际计划回合数及逐回合记录。
- `FAIR_EPISODES.csv`：与 JSON 的规范化逐回合数据精确一致。

开发完整网格是 4 seed × 2 档 × 2 来源 × 3 候选，共 48 回合。盲测是 32 seed × 4 档 × 2 来源，加两种各一次/seed 的参考，共 320 回合。初始化失败保留为任务失败；基础设施错误不能伪装成任务失败。参考回合 endpoint_hz 必须为空，避免复制参考扩大分母。盲测还须有 `FRONTEND_SELECTION.json`，其 frontend_by_method 与全部实际回合配置一致。

成本 CSV 由 `../cost_design/build_cost_table.py` 从 `COST_RESULT.json` 生成。必须具备完整 .8 秒窗口、正确 scope、锁定前端、完整原生事件验证、相同 trace/source/policy 哈希。若输入含多张卡，须显式用 `--cost-gpu-uuid` 选出一张同卡且完整的数据。episode 的 wall_s 只参与 CSV/JSON 一致性检查，从不计算成本。

历史显存不足不等于最终回合失败：如果 `all_actual_jobs_normal_exit=false`，只接受 `all_final_actor_attempts_normal_exit=true` 且历史故障保留的显式恢复分支。此时还必须校验 raw 清单与最终审计、DATA_COMPLETION_RECEIPT、原协议和不改算法/场景/seed 的 OOM 恢复清单互相绑定；全部注册 shard 有完整 claim，受中断 shard 由注册 replacement 完整重跑。缺失恢复凭据、未闭合调度或 raw 审计失败仍拒绝。

成本 v3 还要求同组具有相同 `physics_reference_sha256` 与 `physics_reference_kind`，避免把跨主机漂移的物理轨迹混成一种工作量。`reference_trace_sha256` 是旧动作来源，`physics_reference_sha256` 是当前共同真实物理参考，二者可以不同；参考采集本身不算正式成本。

成本 v4 区分 `raw_exact` 和 `task_functional` 两种等价合同，整条曲线必须只用一种。两种都要求 `functional_query_exact=true`、每次完整28次实际actor查询的payload指纹与有效收据一致、同一来源/帧率/前端的重复具有同一指纹；不同来源/帧率可以有不同指纹。`raw_exact` 还须原始事件精确一致；`task_functional` 允许原始事件不同，但必须如实保留该字段，图脚明确“actor输入/输出精确一致，raw events可能不同”，不能称为原始事件精确复现。成本窗口、native、同硬件、共同物理参考、无审计/详细仪表等条件不变。

已交付开发图所用的旧 `8cc5cd88716d...` 源码原样保存在 `code_versions/make_fair_figures_8cc5cd88716d.py`；后续成本schema扩展不追改已交付图及其历史manifest。

输入缺失、不完整、重复、前端在盲测途中改变或哈希不匹配时，退出码为 2、说明 NOT_READY，不生成占位结果。全部请求的数据先验完后再写图；不允许自动覆盖已有正式图。

## 命令

在具备 NumPy 和 Matplotlib 的 Python 环境运行；本项目可用 Windows 的 genesis 环境。

```powershell
$py = 'C:/Users/zhaoyuhan/miniconda3/envs/genesis/python.exe'
$tool = 'C:/Users/zhaoyuhan/Documents/ChatGPT/New project/robot_closed_loop_reuse_20261107/fair_frontend_blind_20261108/figures_tool/make_fair_figures.py'

# 先验输入，不创建图或输出目录
& $py -X utf8 $tool --validation '开发collector目录' --check-only

# 开发全交叉完成即可先出第一张，不必等待盲测
& $py -X utf8 $tool --validation '开发collector目录' --output '正式图目录'

# 盲测完成后，整体 seed bootstrap 10000 次
& $py -X utf8 $tool --blind '盲测collector目录' --selection 'FRONTEND_SELECTION.json路径' --bootstrap-repetitions 10000 --output '新正式图目录'

# 成本测完后独立生成三图中的第二、第三张
& $py -X utf8 $tool --blind '盲测collector目录' --selection 'FRONTEND_SELECTION.json路径' --cost 'FAIR_COST_RUNS.csv路径' --output '另一个正式图目录'

# 输出所需 CSV 表头；不造任何观测值
& $py -X utf8 $tool --print-schema episodes
& $py -X utf8 $tool --print-schema cost
```

`test_fair_figures.py` 的全部 fixture 仅为临时契约检查，包含空表、缺回合、错误前端、重复参考、成本缺样、混合硬件、CSV/JSON 不一致等情况；不作为正式实验数据，也不保留模拟结果图。

```powershell
& $py -X utf8 -m unittest -v test_fair_figures.py
```

## 解释范围

这些图验证现有原策略在两种事件来源及各自等预算选择的前端下的闭环效用，不是感知网络训练数据优劣实验。Wilson 是各档点估计的不确定性，跨帧率统计不独立；配对检验才使用对应 seed 的成对结果。成本曲线反映本次硬件和固定动作窗口的实测原生流水线成本，不外推实时性能或生产吞吐。
