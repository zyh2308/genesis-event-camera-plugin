# 公共 Sim-to-Real：职责、仓库交接及第一阶段任务

更新：2026-10-16。遵循用户本轮纠正：研究会话负责完整实验逻辑、任务拆解、研究决策与独立验收；执行会话负责实现、环境、运行及工程证据。本文件只制定任务与验收，不代表已经向目标会话派发。

## 1. 职责边界

研究会话：审查文献证据和可比性；明确核心变量、训练/测试边界、指标与统计；给阶段任务卡；检查指定commit的实现、原始结果和日志；记录通过、条件通过、不通过或无法核验。遇到问题交回执行会话修复，不在本会话接管训练代码开发。

执行会话：维护工作仓库与分支；安装和锁依赖；实现export/loader/training/evaluator；生成数据和hash；运行smoke/pilot/formal；保存机器可读原始证据、运行状态与资源记录。

前一轮研究协议的具体数字属于候选研究决策，尚未经工程核验。执行会话必须报告官方实现与文档之间的冲突，不能为满足稿件中的数字而伪造“官方默认”，也不能默默切换实现。共同修订记录在protocol changelog中。

## 2. 已核实的仓库入口

- 远端：<https://github.com/zyh2308/genesis-event-camera-plugin>，本研究会话已用当前GitHub连接成功读取。
- 本次读取的main：`568ebb5480b487340172230fa9490b017141d339`。
- 指定frozen simulator core：`e79e656ac00f7847b98db9e12bf43b093f3afff1`，远端可读取。
- GitHub compare显示main相对core ahead 16、behind 0；本次列出的差异仅新增benchmark及文档，没有`genesis_event_plugin/`实现变化。
- 当前研究工作目录`C:\Users\zhaoyuhan\Documents\ChatGPT\New project`不是Git仓库。
- `repository-source/zyh2308-genesis-event-camera-plugin-568ebb5/`是已解压源码快照，没有`.git`，不能作为自动获取远端新结果的入口。
- 远端已有`benchmark/`属于质量验证资产。公共sim-to-real必须在`benchmark/sim2real/`独立命名，不能把其quality smoke状态当作本分支成果。

资料入口：[main](https://github.com/zyh2308/genesis-event-camera-plugin/tree/568ebb5480b487340172230fa9490b017141d339)、[frozen core](https://github.com/zyh2308/genesis-event-camera-plugin/commit/e79e656ac00f7847b98db9e12bf43b093f3afff1)、[比较](https://github.com/zyh2308/genesis-event-camera-plugin/compare/e79e656ac00f7847b98db9e12bf43b093f3afff1...568ebb5480b487340172230fa9490b017141d339)。分支会变化，以上仅为读取时事实。

## 3. 仓库作为唯一成果入口

默认在插件仓库的专用Sim-to-Real分支开发；第一阶段由执行会话确定并公布分支名。研究会话按**明确commit SHA**审查，不依赖可能变化的HEAD，不修改或自动合并main。若执行环境确需独立实验仓库，则显式固定插件dependency commit，仍沿用同一证据结构。

建议目录（待工程会话实际创建，本地快照未增加这些目录）：

```text
benchmark/sim2real/
  README.md
  sim2real_status.md
  protocol/
    research_candidate_v1.yaml
    protocol_changelog.md
    formal_protocol.yaml
  tasks/
    S0_repository_and_feasibility.md
    S1_shared_trajectory_smoke.md
    S2_synthetic_training_pilot.md
    S3_freeze.md
    S4_formal_flow.md
    S5_reconstruction.md
    S6_pose.md
  manifests/
  configs/
  export/
  tasks_impl/
  evidence/<stage>/<run_id>/
  reports/
  reviews/
```

代码、配置、任务、报告和小型原始证据进入Git。大数据、checkpoint和长日志可在可访问artifact storage或共享磁盘；仓库保存URI/相对路径、大小、sha256、生成命令、所属run_id。不能只保存执行机器的私有绝对路径或容易过期的链接。

每次交付统一给出：repo URL、branch、交付commit、stage、status文件路径、报告路径、证据索引路径。研究会话直接读取，不要求用户重新上传。未推送且不在共享磁盘的内容，本会话不能视为已交付。

## 4. 下一张任务卡：S0，仅做仓库接入与可行性核对

目标：建立可直接检查的真实Git工作副本，并将候选协议中的事实、项目决策、工程障碍区分清楚。此阶段不启动完整语料生成或正式训练，不输出真实模型排名。

### 任务

1. 记录remote URL、工作目录、branch、HEAD、dirty状态、插件core依赖的精确SHA；先读目标仓库AGENTS.md等本地约定。不通过更改frozen core解决benchmark接口问题。
2. 将本目录的`RESEARCH_PROTOCOL.md`、`protocol_candidate_v1.yaml`、`SOURCE_AUDIT.md`及两份用户基线导入上述仓库交接位置，并标记candidate/not-formal。用户本轮的职责纠正以本文件为准。
3. 对STE-FlowNet、E2VID训练/评测、YOLO-6D/E-POSE分别检查准确repo+commit、训练入口、原生输入、训练监督、license、可获得的数据版本和环境限制。引用代码路径/行或配置字段，不以README标题代替事实。
4. 重点回答少数已知疑点：STE编码与计数精度/边界、E2VID配置5/10-channel差异、E-POSE三通道状态语义及CAD/pose原点。对每项给出无需真实评分即可验证的办法和最小适配范围；这一阶段先给核查结论，不要求完成所有修复。
5. 报告执行硬件、GPU/驱动/CUDA/Python/Torch、可用磁盘，实测一个小规模无真实评分的基础算例或forward所需资源；无法实测则如实标记估计/未运行。
6. 给出最小下一阶段任务与时间/存储估计，说明哪些候选数字可直接沿用，哪些必须在synthetic pilot中修订。默认优先L2，L1次之，L3不阻塞主线。

### 必交付

- `sim2real_status.md`：S0状态、阻塞、下一步，不以“完成”替代证据。
- `reports/S0_feasibility.md`：逐项结论与出处；配置建议偏离列表。
- `evidence/S0/<run_id>/environment.json`：设备与依赖版本，区分检测值/估计值。
- `evidence/S0/<run_id>/repository_state.json`：repo/branch/HEAD/dirty/core SHA。
- `evidence/S0/<run_id>/source_audit.json`：外部repo commit、准确配置/入口路径、未确认项。
- `evidence/S0/<run_id>/commands.txt`和原始stdout/stderr（若运行了任何算例），附exit code及时间。
- `evidence/S0/<run_id>/artifact_index.json`：上述文件和外部artifact的位置、大小与sha256。
- 一个可直接访问的交付commit。无需为了任务状态对外发消息、提交表单或发布正式结果。

### 本研究会话的验收

1. 从交付SHA直接读取文件，确认报告对应的实现和配置确实存在。
2. 核对core依赖、外部repo版本、入口/监督/表示与源码是否相符；检查是否误用了质量验证的旧报告。
3. 对日志和环境信息检查run_id、命令、exit code、时间、版本、文件大小/hash是否互相一致。
4. 能访问执行环境或小型artifact时，独立复算关键hash和少量确定性检查。不能访问环境时，不宣称已经独立复跑训练。
5. 将结果写入`reviews/S0_<commit>.md`：通过/条件通过/不通过/无法核验；每个结论列证据与未覆盖范围。执行会话自报PASS不是研究验收PASS。

## 5. 后续阶段只在前一阶段验收后下发

| 阶段 | 目标 | 验收重点 |
|---|---|---|
| S1 | 小型shared bank→两generator→共同labels/loader | physical IDs、GT和split hash一致；空窗、单位、坐标、时间因果性 |
| S2 | synthetic-only训练到synthetic val最小闭环 | 真实数据未用于选参；学习曲线、步数、初始化与增强配对；真实资源估计 |
| S3 | 正式protocol冻结 | 完整可执行config、manifests、metrics、排除规则、版本、hash，消除关键TBD |
| S4 | formal multi-seed光流 | 原始逐帧误差、checkpoint、运行完整性与汇总复算，3 seeds逐个核对 |
| S5 | 重建补充 | 同样的隔离与配对；指标/归一化/时序对齐可核验 |
| S6 | 可选物体位姿 | 无GT mask/crop泄漏；CAD坐标、pose计分和漏检分母正确 |

下一阶段若改变候选模型/数据/预算，要说明工程或研究证据；不能用真实test排名决定。smoke/pilot只能证明对应阶段可运行，不能升级为论文formal结论。

## 6. 当前派发状态

用户已明确：执行会话尚未创建，将来在本机WSL运行，科研资料主要位于WSL。本文件未发送到其他会话，也未创建新会话。

本机已检查到`Ubuntu-22.04`，WSL2，状态Running；`wslpath`确认当前交接文件可从下列Linux路径访问：

```text
/mnt/c/Users/zhaoyuhan/Documents/ChatGPT/New project/docs/sim2real-research-2026-10-16/COORDINATION_AND_STAGE0.md
```

默认由工程会话在WSL Linux文件系统中维护真实checkout、数据及运行环境；研究会话通过`wsl.exe -d Ubuntu-22.04 --exec ...`读取明确仓库路径的文件、Git状态和小型证据，或读取已推送commit。大型科研数据无需复制到Windows。Windows现有研究文档可由工程会话直接从`/mnt/c/...`导入仓库，不要求用户手动搬运。

尚未定位用户WSL内插件checkout、数据目录和训练环境，因此不宣称已读取这些本地资源。未来工程会话创建后，只需一次登记`WSL distro + repo绝对路径 + branch/remote + 执行会话入口`。随后所有任务和验收都引用同一仓库及交付SHA，不再逐次要求上传报告。
