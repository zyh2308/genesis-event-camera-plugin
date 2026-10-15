# Genesis Event Camera：Robot R0 状态

**更新时间：** 2026-10-16（Asia/Tokyo）
**阶段：** 场景实现与 GT oracle feasibility；候选配置尚未冻结。
**研究边界：** 机器人任务位于 benchmark 层；`genesis_event_plugin/` frozen core 不改；不使用未来 GT、逐帧重设目标、焊接或吸附。

## 当前进度

### 已完成并有证据

- 正式研究 baseline、robot handoff、R0 任务说明与候选场景配置已阅读；按其中的物理/碰撞规则实现 Panda + Plane + 平面方块。没有额外桌面碰撞盒。
- 已将下游场景、固定减速度预测器、共享控制器入口、episode CSV/轨迹/GIF、配置与运行时 setup 提交到指定分支。
- 已在本地完成工程级静态抓取调试：使用官方 Panda IK/位置驱动和手指力控，出现双侧接触并抬升保持。该调试不是服务器实验、不纳入成功率，不作为 R0 结论。
- 本地 frozen-core 差异检查通过：`git diff --exit-code e79e656ac00f7847b98db9e12bf43b093f3afff1 -- genesis_event_plugin/`。
- 服务器 `g1` 的隔离 worktree 已同步到 `bd6824b`；原 `main` 工作区未切换、未修改。服务器基础 Genesis 为 1.2.2，基础 Torch 为 2.5.1+cu121。
- 资源检查时 GPU 0 有其他用户计算进程、GPU 3 有 CARLA；GPU 1、2、4、5 当时空闲。后续 EGL 运行显式限制在经再次检查的空闲设备，不触碰其他用户进程。

### 服务器实测结果

**尚无完整成功的服务器 episode。** 单 episode 静态 smoke 在 `bd6824b209f63cb72c7b7c4f6df2ef38351d7f9b`、配置 SHA256 `7a7f73803856f1bcf11f67d10c97d72cf5e46b84888bbd2f6c60247c83480ba6`、seed `2026101600` 下运行，但 Genesis 接触查询抛出 `gather(): Expected dtype int64 for index`，exit 1；Torch 为 2.5.1+cu121，Genesis 在启动时也明确警告 Torch <2.8 不受支持。该目录只有未完成 `run_manifest.json`，不含完整 episode CSV，不计成功或失败样本。静态抓取、滑动标定/holdout、oracle 成功率、相机 preflight 与 Direct-HDR 成本均尚无可报告结论。此前本地静态调试不纳入统计。

## 分支、冻结基线与候选配置

- GitHub：`zyh2308/genesis-event-camera-plugin`
- 分支：`exp/robot-closed-loop-v1`
- 本地/服务器运行时修复前提交：`bd6824b209f63cb72c7b7c4f6df2ef38351d7f9b`；包含本次静态 smoke 的代码。后续 Torch 2.8 隔离支持正在修改。
- 服务器隔离 worktree：`~/Eventbased_WAM/repositories/gec-robot`
- 服务器原始仓库工作区 `~/Eventbased_WAM/repositories/genesis-event-camera-plugin` 保持在 `main`，HEAD `15479d97154274796043d5e5cc42a336f57b2dd9`。
- Frozen simulator core：`e79e656ac00f7847b98db9e12bf43b093f3afff1`；任务新增代码不在 `genesis_event_plugin/`。
- 场景配置：[`scene_pilot_r0_v1.json`](../../benchmark/robot_closed_loop/configs/scene_pilot_r0_v1.json)，SHA256 `7a7f73803856f1bcf11f67d10c97d72cf5e46b84888bbd2f6c60247c83480ba6`。这是待测候选，不是冻结值。
- 当前候选：40 mm 立方体、80 g、低摩擦 Plane、1 ms 物理步、0.30/0.40/0.50 m/s 发射、固定外置 320×240 相机、100 Hz 控制、10 ms 观测延迟、手爪闭合起始提前量 100 ms。减速度 0.0981 m/s² 仅是分析初值；正式 holdout/oracle 必须使用 no-hand calibration 共享拟合值。

## 运行时与运行命令

在服务器只使用上述隔离 worktree；先检查 GPU 进程归属，再将 `CUDA_VISIBLE_DEVICES` 设为一个当时空闲的物理卡。仓库 setup 将已有 Genesis 1.2.2 包复制到 project venv，在私有 venv 安装 Torch 2.8 CPU 并应用 HDR overlay，不修改共享 conda 环境。修正兼容问题后必须重跑单 episode smoke。

```bash
source ~/miniconda3/bin/activate Eventbased_WAM
cd ~/Eventbased_WAM/repositories/gec-robot
ROBOT_TORCH_WHEEL="$HOME/Eventbased_WAM/torch-2.8.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl" bash scripts/setup_robot_runtime.sh
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python \
  -m benchmark.robot_closed_loop.run --mode static --episodes 1 --seed 2026101600 \
  --out benchmark/robot_closed_loop/results/server_static_smoke
```

后续每个输出路径必须是新目录；跑批使用 `tmux`，写 project log 和 PID，并低频查看。建议顺序：

1. `--mode static --episodes 20`：真实双侧接触、抬升和 0.5 s 保持。
2. `--mode slide --split calibration --episodes 10 --speeds 0.3,0.4,0.5`：无手方块物理滑动和共同减速度拟合。
3. `--mode slide --split holdout ... --a-eff <calibration 输出>`：独立种子检查漂移、到达时间和预测误差。
4. `--mode oracle --episodes 20 --speeds 0.3,0.4,0.5 --a-eff <calibration 输出>`：GT 当前状态 → 固定预测器 → 固定控制器；不读取未来状态。
5. 分别运行小幅 `--bias-y`、`--ttc-bias-ms`、`--extra-latency-ms` 注入；检查成功率和错误类别变化。
6. `--mode preflight` 与 `--mode reference --reference-duration 0.20`：固定相机可见性、目标像素尺度、遮挡，以及直接 HDR 1/2 kHz reference 成本。

## 证据入口与 provenance

- 代码、配置和复跑命令：[`benchmark/robot_closed_loop/README.md`](../../benchmark/robot_closed_loop/README.md)
- 每次运行必须记录代码 commit、分支、Genesis/Torch/资产/overlay、配置 SHA256；产物 SHA256 清单在运行目录 `SHA256SUMS`。
- 结果目标位置：服务器 `~/Eventbased_WAM/repositories/gec-robot/benchmark/robot_closed_loop/results/<run-id>/`；轻量原始 CSV、JSONL、摘要、短录像纳入分支。大型产物另存可访问位置并提供 URL 与校验清单。
- 当前阶段原始结果哈希：未产生（服务器实验尚未启动）。

## 尚未完成 / 风险

- 服务器端 GitHub fetch 曾等待超过 90 秒；之后改用 Git bundle 完成同步，隔离 worktree 当前为 `bd6824b`。首次 runtime setup 因共享 Genesis 安装中的无权限 `utils/__pycache__` 文件失败；修复后 setup exit 0，overlay 接口检查通过，日志 `logs/robot_r0/setup_attempt2.log`。
- 静态 smoke 证实 Torch 2.5.1+cu121 与接触查询不兼容；已在 project-local setup 中加入 Torch 2.8 CPU 安装路径，等待提交和重跑验证。
- 方块摩擦/减速、自由滑动漂移与拦截时间尚未实测；`a_eff` 未拟合，不能据此做 holdout 或正式 oracle。
- 固定相机的视场、目标像素尺度、机器人遮挡和 1/2 kHz Direct-HDR 运行成本尚无测量。
- 当前代码尚未给出 >95% 成功率证据；需先小批试运行和调整，再明确验证数量与 Wilson 95% 区间。正式 Ours/V2E 比较、大规模数据生成和训练均未开始。

## 冻结建议与研究会话需决策事项

暂不冻结任务配置。先依据服务器证据调整摩擦、发射区、截距、闭合提前量与控制限幅；随后在独立 holdout 上确认单一减速度模型可用，再报告 oracle。若 Torch 兼容、HDR reference 成本或在合理调参后 oracle 可达性仍有实质障碍，将在此处附日志/哈希和建议方案后提交研究会话决定。主分支合并留待验收。
