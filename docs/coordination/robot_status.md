# Robot Closed-loop Utility 状态登记

**状态更新时间：** 2026-10-08  
**工作流：** `gec-robot/`  
**研究方向：** Planar Sliding-object Intercept-and-Grasp

## 1. Branch 与基线

- Branch：`exp/robot-closed-loop-v1`
- Worktree：`/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot`
- Base：`origin/main`
- Worktree HEAD：`568ebb5480b487340172230fa9490b017141d339`
- 本次创建前 `origin/main` 已通过 `git fetch origin` 更新到上述提交。

原始机器人工作区保持不变：

- Repository/worktree：`/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin`
- Branch：`main`
- 原工作区 local main HEAD：`15479d97154274796043d5e5cc42a336f57b2dd9`
- 原工作区未执行 reset、强制同步或修改；原有工作区状态保持 clean。

## 2. Frozen simulator

- Frozen simulator commit：`e79e656ac00f7847b98db9e12bf43b093f3afff1`
- 检查：在新 worktree 中执行

  ```text
  git diff --exit-code e79e656ac00f7847b98db9e12bf43b093f3afff1 -- genesis_event_plugin/
  ```

- 结果：PASS；相对于 frozen commit 未发现 `genesis_event_plugin/` 差异。
- 本 robot branch 不拥有特制 simulator，也没有为机器人任务修改 simulator core。

## 3. 历史机器人工作与 provenance

历史 Task1/Task2 及其相关脚本、数据和结果属于旧工作流，统一标记为 `LEGACY`，继续保留在原路径，不在本轮大量复制进 Git repository。

主要外部代码路径：

- `/home/zhaoyuhan/Eventbased_WAM/code/task0_dp`
- `/home/zhaoyuhan/Eventbased_WAM/downstream`
- `/home/zhaoyuhan/Eventbased_WAM/validation/downstream`

历史内容包括：

- Task1 Dynamic Target Reaching
- Task2 Fast Intercept and Hold
- paired rollout 与事件表示转换工具
- voxel grid、time-surface 和相关 metrics
- RGB/Event 训练与评估脚本
- 历史 checkpoint、rollout、HDF5/raw AER、视频和数据集

本轮仅建立新的 robot worktree 与本状态登记文件；未移动、删除、重命名、覆盖或修改上述旧代码和资产。

## 4. External Immutable Legacy Assets

以下资产按 `EXTERNAL_IMMUTABLE_LEGACY_ASSETS` 处理：

- 主要目录：`/home/zhaoyuhan/Eventbased_WAM/data`
- 主要目录：`/home/zhaoyuhan/Eventbased_WAM/runs`
- 主要目录：`/home/zhaoyuhan/Eventbased_WAM/logs`
- 主要目录：`/home/zhaoyuhan/Eventbased_WAM/research`
- 主要目录：`/home/zhaoyuhan/Eventbased_WAM/downstream`
- 主要目录：`/home/zhaoyuhan/Eventbased_WAM/validation/downstream`
- 资产类型：Task1/Task2 数据集、paired rollout、raw AER、HDF5、checkpoint、视频、评估结果和历史日志。

本轮不复制完整大型数据备份。既有清单和小文件备份保留在：

```text
/home/zhaoyuhan/Eventbased_WAM/reorg_backups/robot20261007_164200/robot_assets_manifest.txt
/home/zhaoyuhan/Eventbased_WAM/reorg_backups/robot20261007_164200/external_data_paths.txt
/home/zhaoyuhan/Eventbased_WAM/reorg_backups/robot20261007_164200/large_assets_manifest.txt
/home/zhaoyuhan/Eventbased_WAM/reorg_backups/robot20261007_164200/files/
```

其中 `large_assets_manifest.txt` 记录大型资产的绝对路径和大小；已有小型源码、配置、Markdown 副本及 SHA256 清单继续作为 provenance 记录。

本次 Git 重组不会触碰这些资产。后续新实验必须写入新的、独立命名的目录，禁止覆盖上述历史路径、Task1/Task2 数据、rollout 或 checkpoint。

## 5. paired rollout 与 voxel grid 状态

### paired rollout

- 历史 Task1/Task2 exact-online、replay、time-surface 和 formal 目录继续保留，身份为 `LEGACY`。
- 已知 Task2 canonical paired debug smoke 位于：

  ```text
  /home/zhaoyuhan/Eventbased_WAM/downstream/smoke/task2_paired_20260929_v122b
  ```

- 该 smoke 包含 common state/action/timestamp archive、Ours 事件输出、official V2E 输出及元数据。
- `formal_metrics_executed` 为 false；该产物只证明输入血缘、接口和 smoke 链路，不是 policy success-rate 结果。
- 不把历史 event count 或旧 smoke 输出写成机器人性能结论。

### voxel grid

- `downstream/voxel_representation.py` 提供 raw-AER 到 polarity-separated temporal voxel 的确定性转换。
- 已知 voxel self-test 通过：时间顺序、无未来事件泄漏和重复运行一致性均已检查。
- 5-bin/8-bin 仍属于 representation candidate/development protocol；正式候选尚未在本新 workflow 中冻结。
- voxel smoke 不是正式训练结果，也不代表事件策略已完成 Task1/Task2。

## 6. 已完成测试与未完成事项

已完成的历史基础设施或 debug/smoke 检查：

- Git 与 frozen simulator provenance 检查
- Task2 canonical paired rollout smoke
- independent reference event-domain smoke
- voxel representation self-test
- Task2 metrics self-test
- 外部机器人资产与路径审计

上述检查均不等同于正式 policy 性能结果。

尚未在本新 workflow 中完成：

- 新任务正式数据生成
- formal policy training
- multi-seed training/evaluation
- held-out test、full stress sweep
- wrist-camera formal experiment
- Franka + planar sliding target + oracle 环境实现

## 7. 迁移标记

- `MIGRATION_PENDING`：旧 Task1/Task2 代码是否在新任务冻结后以何种最小形式复用；新任务环境、oracle 和独立输出目录尚待设计。
- `CROSS_BRANCH_CORE_CHANGE`：未发现。frozen core 检查 PASS；本轮未修改 simulator core。
- `LEGACY`：历史 Task1/Task2、paired rollout、旧训练脚本、数据集、checkpoint 和视频。

## 8. 下一阶段

下一阶段只进行设计与环境准备：

1. 设计 Franka + planar sliding target 场景；
2. 设计可审计的 oracle/interception 参考行为；
3. 固定新的输出目录和数据 provenance；
4. 在获得明确协议后再考虑小规模验证。

本状态文件对应的迁移阶段不直接训练复杂 policy，不修改 frozen simulator，也不改写任何历史大型资产。
