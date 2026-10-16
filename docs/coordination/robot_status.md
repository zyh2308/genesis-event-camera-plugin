# Genesis Event Camera：Robot R0 / R0.5 状态

**更新时间：** 2026-10-16（Asia/Tokyo）
**阶段：** R0 已由研究会话验收；R0.5 strict scoring 与统一时间步已实现，服务器静态与无手校准通过；动态 oracle 与 Direct-HDR/event-state 验证进行中；协议尚未冻结。
**研究边界：** 机器人任务位于 benchmark 层；`genesis_event_plugin/` frozen core 不改；不使用未来 GT、逐帧重设目标、焊接或吸附。

## 当前进度

R0.5 strict static smoke 与 0.5 ms no-hand 校准已在服务器运行。R0 的 300/300 旧评分结果及历史 0.2 s Direct-HDR pilot 均不作为 R0.5 验收证据。R0.5 的逐集样本、代码提交、配置哈希和结果清单在下方单独登记。

### 已完成并有证据

- 正式研究 baseline、robot handoff、R0 任务说明与候选场景配置已阅读；按其中的物理/碰撞规则实现 Panda + Plane + 平面方块。没有额外桌面碰撞盒。
- 已将下游场景、固定减速度预测器、共享控制器入口、episode CSV/轨迹/GIF、配置与运行时 setup 提交到指定分支。
- 已在本地完成工程级静态抓取调试：使用官方 Panda IK/位置驱动和手指力控，出现双侧接触并抬升保持。该调试不是服务器实验、不纳入成功率，不作为 R0 结论。
- 本地 frozen-core 差异检查通过：`git diff --exit-code e79e656ac00f7847b98db9e12bf43b093f3afff1 -- genesis_event_plugin/`。
- 服务器 R0 worktree `~/Eventbased_WAM/repositories/gec-robot` 在主要批次时为 `d3bc9e3732b8e898c2412d34f7ea00bd07f4acd9`；相机诊断 worktree `~/Eventbased_WAM/repositories/gec-robot-camera-sync` 为 `062c3811684a854110e01cfb500222083f0f6deb`；最终机位录像与 HDR pilot 在干净 worktree `~/Eventbased_WAM/repositories/gec-robot-finalcam-e63fb94`、代码 `e63fb94872ddca204f6740981fe99b982e0b6777` 运行。原服务器 `main` 工作区未切换、未修改。R0 project venv 使用 Genesis 1.2.2 + Torch 2.8.0+cpu，Torch wheel SHA256 `cb06175284673a581dd91fb1965662ae4ecaba6e5c357aa0ea7bb8b84b6b7eeb`，文件位于服务器 `~/Eventbased_WAM/torch-2.8.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl`。基础共享环境仍是 Torch 2.5.1+cu121。
- 最终运行前 GPU 2 空闲、GPU 3 有 CARLA 进程；Direct-HDR 和动态 smoke 均设置 `EGL_DEVICE_ID=2`，短 context probe 确认 `GL_RENDERER=Tesla V100S-PCIE-32GB/PCIe/SSE2`。未干预 CARLA 或共享 Quadrants cache。

### 服务器实测结果

**服务器静态抓取复测（开发级）：** 在代码提交 `418718af6c0052c6a368cedd9d5928a1bccec458`、配置 SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a` 下运行 20 集，20/20 完整成功：均有双侧接触、净空 ≥0.1015 m、连续保持 0.5 s；首次接触 0.062–0.066 s，中位数 0.065 s；抬升触发 0.131–0.136 s。静态 20/20 的 Wilson 95% 区间为 83.9%–100%，不能据此证明 95% 下界达标，且不代表动态拦截成功率。代表性抓取 GIF 与完整关键状态 trace 已归档。

**服务器单集 smoke（早期开发）：** 代码提交 `af0548c3ab6ccc663f25fea47e8dd59989b5d1a3`、旧配置 SHA256 `7a7f73803856f1bcf11f67d10c97d72cf5e46b84888bbd2f6c60247c83480ba6` 下，静态 1/1 双侧接触并抬升：首次接触 0.064 s、抬升触发 0.134 s、底面净空 0.1016 m、保持 0.5 s，结果 `server_static_smoke_torch28/r0_static.csv`。仅作为早期 runtime/控制 smoke，不计入新配置 20 集统计。

**服务器 no-hand free-slide 校准：** 代码提交 `865b79e777ff054856ea49cbf36bf0aa3f1ccf1a`、配置 SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a` 下运行 30 集（0.30/0.40/0.50 m/s 各 10 集），Panda 不在场，30/30 到达 x=0.55 m 拦截平面。由三档轨迹共同拟合的固定减速度 `a_eff=0.09804843 m/s²`；逐 episode 拟合中位数 `0.09804806 m/s²`、IQR `[0.09799079, 0.09808144] m/s²`。各档中位到达时间为 0.9704/0.6908/0.5402 s，到达速度为 0.2048/0.3323/0.4470 m/s。中位横向漂移（相对起始 y）为 -1.0/4.9/1.4 mm，平均绝对漂移为 7.0/7.5/9.0 mm。拟合仅来自 calibration seeds；holdout 尚待运行。轨迹中方块 z 保持约 0.01998–0.020004 m；离散 support-contact 查询为间歇 true，需结合稳定高度解释，不把该布尔采样当作连续接触测量。

**服务器 no-hand holdout：** 使用新种子（每档 10 集）与校准阶段固定 `a_eff=0.09804843035177806 m/s²`，代码提交 `93e739eec055b440516c02335a295fe43167d948`、配置 SHA256 不变。Panda 仍不在场，30/30 到达拦截平面。各速度档中位到达时间 0.9798/0.6873/0.5346 s，中位逐 episode 减速度 0.098027/0.098064/0.098043 m/s²；校准预测初始 TTC 误差中位数 -1.45/-1.14/-1.08 ms，横向位置误差中位数 -0.83/-0.27/0.43 mm，全部绝对横向误差最大 2.36 mm。该结果支持单一固定减速度作为本候选场景的运动预测器；它不是机器人抓取成功率证据。

历史低机位 camera preflight 在代码 `9c83cbd040388e50e49489e6d83b1d8431b863cd` 下输出过有效 HDR 与 43 px bbox，但其 0% occlusion 对照使用非法全零 Panda qpos，不能作为遮挡结论。修正后采用 target-only reference 的新测量已选出正 x 侧相机候选；详情与哈希见“通过的正 x 侧固定相机候选”。Torch 2.5.1 下的一次失败接触 smoke 仍作为历史兼容性证据保留，不计任务样本。

**服务器 GT-current oracle 早期开发批次（已由后续 speed-conditioned controller 复验取代）：** 代码提交 `4d70b8f0f796ca5163d47ab3b5aac98f71aa45a4`、配置 SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a`，校准固定 `a_eff=0.09804843035177806 m/s²`、GT 状态延迟 10 ms，每档 20 集。0.30 m/s 为 20/20 成功（Wilson 95% CI 83.9%–100%）；0.40 与 0.50 m/s 均为 0/20（各 CI 0%–16.1%）。目前配置未达到各主速度档可行性目标。

失败轨迹细查发现：0.40 m/s 的 20 集全部出现机器人接触，其中 19 集有短暂双指接触；0.50 m/s 有 19 集出现接触、18 集有短暂双指接触；高档均没有成功抬升。代表失败 trace 显示目标接触后速度反向并从手指逃逸。摘要将所有失败都分成 `oracle_unreachable`，是因为控制器在物体离开拦截区后最后一次预测不可达，掩盖了之前发生的接触；失败分类优先级需修正。0.40 m/s 批次还有手指停滞触发空夹爪抬升的 episode，说明仅凭关节停滞不能区分物体夹持与夹爪自身硬停；后续控制器会加入同一估计状态下的几何接近/速度门限，仍不使用 evaluator 的 GT 接触标签。

**闭合提前 20 ms sensitivity（开发批次）：** 使用代码 `eff4f592fe4e503b8f4264415c2b3b0b0b326165`、配置 SHA256 `c85cb2b117d90704f2d824ccf7f0e102019b7d8e5d814a2edcdf3a6004932edf`、每档 5 集，`--ttc-bias-ms -20`（相对 100 ms lead，触发提前到预测 crossing 前约 120 ms）。结果 0.30 档 5/5；0.40、0.50 档均 0/5。0.40 档 3/5、0.50 档 5/5 有双侧接触，但均未发生合格抬升；高档失败为 8 个 `unstable_grasp`、2 个无接触 `oracle_unreachable`。代表 trace 显示在高速度下手指到目标前已接近/到达闭合硬限位，方块停在夹爪前缘，没进入有效夹持开度；此前闭合 latch 与状态门控阻止了错误开爪和空夹爪触发。该单因子试验不支持更早闭合；较晚闭合 sensitivity 找到三档成功点。

**统一 speed-conditioned GT-current oracle 开发复验：** 代码提交 `d3bc9e3732b8e898c2412d34f7ea00bd07f4acd9`、配置 SHA256 `9b313bfc634e4eb39fa7591593bc808e6543f4a8a78f9dd15a332fb6be106439`、固定减速度 `0.09804843035177806 m/s²`、延迟 10 ms、每档 20 集、新 seed。首次延迟补偿当前速度估计中位数为 0.2951/0.3951/0.4951 m/s；实际 closure lead 中位数为 80.49/70.49/60.49 ms，与三个单因子成功点吻合。三档均 20/20 完整成功（Wilson 95% CI 各为 83.9%–100%）；每档 20/20 有双侧接触并启动抬升，保持均 0.5 s；夹持开度中位数约 39.8 mm，目标底面最终净空范围 0.10045–0.10263 m。该开发批次未证明真实成功率下界超过 95%，仍需独立样本估计不确定性。

## 分支、冻结基线与候选配置

- GitHub：`zyh2308/genesis-event-camera-plugin`
- 分支：`exp/robot-closed-loop-v1`
- 早期静态 smoke 使用提交 `af0548c3ab6ccc663f25fea47e8dd59989b5d1a3`；相机/HDR preflight 使用 `9c83cbd040388e50e49489e6d83b1d8431b863cd`；静态 20 集复测使用 `418718af6c0052c6a368cedd9d5928a1bccec458`；no-hand slide calibration 使用 `865b79e777ff054856ea49cbf36bf0aa3f1ccf1a`；slide holdout 使用 `93e739eec055b440516c02335a295fe43167d948`；首批 oracle 使用 `4d70b8f0f796ca5163d47ab3b5aac98f71aa45a4`；GT oracle 控制门控修正使用 `eff4f592fe4e503b8f4264415c2b3b0b0b326165`；较晚闭合敏感性运行使用 `ae34e8c94fb6c0b8646cc0b901e42bd218617151`，结果归档在 `47f5427cb1aedb8cd783f8d890351dd507cd9f62`；speed-conditioned lead 使用 `d3bc9e3732b8e898c2412d34f7ea00bd07f4acd9`；preflight 同步修复为 `b2e06934eb6ae190e5063210455538bf86f05f55`，camera launcher 路径修正为 `062c3811684a854110e01cfb500222083f0f6deb`；正 x 静态视野结果归档于 `e63fb94872ddca204f6740981fe99b982e0b6777`；动态机位和 Direct-HDR 显式 EGL 复测及结果归档为 `c3492fd52cd7c3420a60171e9e15616ac7fca1cb`。均在指定分支，run manifest 保存每次代码与配置来源。
- 服务器隔离 worktree：`~/Eventbased_WAM/repositories/gec-robot`
- 服务器原始仓库工作区 `~/Eventbased_WAM/repositories/genesis-event-camera-plugin` 保持在 `main`，HEAD `15479d97154274796043d5e5cc42a336f57b2dd9`。
- Frozen simulator core：`e79e656ac00f7847b98db9e12bf43b093f3afff1`；任务新增代码不在 `genesis_event_plugin/`。
- 场景配置：[`scene_pilot_r0_v1.json`](../../benchmark/robot_closed_loop/configs/scene_pilot_r0_v1.json)，当前配置 SHA256 `124d87188e9a383a054fb7e6e6eaf106c6d331ad0abfaf2a8a959364eafdc549`（状态估计门控、校准来源、speed-conditioned closure lead 与通过静态视野复测的正 x 相机）；oracle holdout 使用的旧相机配置 SHA256 为 `9b313bfc634e4eb39fa7591593bc808e6543f4a8a78f9dd15a332fb6be106439`。较晚闭合 sensitivity 使用 `c85cb2b117d90704f2d824ccf7f0e102019b7d8e5d814a2edcdf3a6004932edf`，首批 oracle 使用 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a`。当前 schedule 已通过统一控制器每档 20 集开发复验和每档 100 集独立 oracle holdout；Direct-HDR 初步成本和代表性动态视角复核已完成，场景/机位候选建议提交研究会话冻结；采样率收敛尚未确认。
- 当前候选：40 mm 立方体、80 g、低摩擦 Plane、1 ms 物理步、0.30/0.40/0.50 m/s 发射、固定外置 320×240 相机（候选机位 `[1.15,0.0,0.22]` 看向 `[0.65,0.0,0.02]`，垂直 FOV 30°；合法 target-only 分母静态复测通过）、100 Hz 控制、10 ms 观测延迟。闭合 lead 使用首次因果速度估计 `v0` 下的 `clip(0.08 - 0.10·(v0-0.30), 0.05, 0.10)` 秒；开发复验三档各 20/20，独立 holdout 三档各 100/100，实际 lead 中位数约 80.49/70.49/60.49 ms。独立 holdout 的逐档 Wilson 95% CI 下界为 96.3%。目标短边在新机位静态路径采样为 30–55 px；拦截中心短边 30 px。共享减速度候选为 0.09804843 m/s²，no-hand holdout 已支持该模型。误差注入已完成。

## 运行时与运行命令

在服务器只使用隔离 worktree；启动前核对 GPU 进程归属，EGL 运行必须显式设置经空闲检查和 renderer probe 确认的 `EGL_DEVICE_ID=2`（`CUDA_VISIBLE_DEVICES` 本身不限制 EGL）。仓库 setup 使用 project venv 中的 Genesis 1.2.2、Torch 2.8 CPU 和 HDR overlay，不修改共享 conda 环境。静态抓取 20/20、相机/HDR preflight、no-hand 标定/holdout、GT-current oracle 三档各 100/100 均已完成；正 x 相机静态和动态视野复核已完成；Direct-HDR 1/2 kHz 小样本 reference 已运行。

```bash
source ~/miniconda3/bin/activate Eventbased_WAM
cd ~/Eventbased_WAM/repositories/gec-robot
ROBOT_TORCH_WHEEL="$HOME/Eventbased_WAM/torch-2.8.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl" bash scripts/setup_robot_runtime.sh
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python \
  -m benchmark.robot_closed_loop.run --mode static --episodes 1 --seed 2026101600 \
  --out benchmark/robot_closed_loop/results/server_static_smoke
```

最终相机候选 preflight 复跑命令（当前默认配置 SHA256 `124d8718...`）：

```bash
CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python \
  -m benchmark.robot_closed_loop.run --mode preflight --episodes 1 \
  --out benchmark/robot_closed_loop/results/camera_preflight_final_20261016
```

Direct-HDR 1/2 kHz reference pilot（canonical EGL 复测已在服务器运行，代码 `e63fb948...`、配置 SHA256 `124d8718...`；设置 `EGL_DEVICE_ID=2`）：

```bash
CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python \
  -m benchmark.robot_closed_loop.run --mode reference --speed 0.4 \
  --reference-duration 0.20 --seed 2026102900 \
  --out benchmark/robot_closed_loop/results/r0_direct_hdr_finalcam_dev_20261016
```

静态 20 集复跑命令（实际 run 使用 tmux；PID/log/exit 位于 `results/runtime_logs/r0_static_dev20_20261016.*`）：

```bash
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode static --episodes 20 --seed 2026101610 --out benchmark/robot_closed_loop/results/r0_static_dev20_20261016
```

服务器 no-hand calibration 命令（30 集，target-only scene）：

```bash
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode slide --split calibration --episodes 10 --speeds 0.3,0.4,0.5 --seed 2026101700 --out benchmark/robot_closed_loop/results/r0_slide_calibration_20261016
```

服务器 no-hand holdout 命令（CPU-only；运行时隐藏 CUDA 设备）：

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode slide --split holdout --episodes 10 --speeds 0.3,0.4,0.5 --a-eff 0.09804843035177806 --seed 2026101800 --out benchmark/robot_closed_loop/results/r0_slide_holdout_20261016
```

GT-current oracle 开发批次（60 集；输出和代表性成功/失败 trace、GIF 已归档）：

```bash
CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode oracle --episodes 20 --speeds 0.3,0.4,0.5 --a-eff 0.09804843035177806 --seed 2026101900 --out benchmark/robot_closed_loop/results/r0_oracle_dev60_20261016
```

闭合时序单因子扫描（server tmux 启动器见对应 `results/runtime_logs/*.launch.sh`）：

```bash
CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode sensitivity --episodes 5 --speeds 0.3,0.4,0.5 --a-eff 0.09804843035177806 --ttc-bias-ms -20 --seed 2026102100 --out benchmark/robot_closed_loop/results/r0_sens_early20ms_dev15_20261016
```

较晚闭合扫描（各 run 均新 seed、新输出目录）：

```bash
CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode sensitivity --episodes 5 --speeds 0.3,0.4,0.5 --a-eff 0.09804843035177806 --ttc-bias-ms 20 --seed 2026102200 --out benchmark/robot_closed_loop/results/r0_sens_late20ms_dev15_20261016
CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode sensitivity --episodes 5 --speeds 0.4 --a-eff 0.09804843035177806 --ttc-bias-ms 30 --seed 2026102400 --out benchmark/robot_closed_loop/results/r0_sens_late30ms_mid04_20261016
CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode sensitivity --episodes 5 --speeds 0.5 --a-eff 0.09804843035177806 --ttc-bias-ms 40 --seed 2026102300 --out benchmark/robot_closed_loop/results/r0_sens_late40ms_dev05_20261016
```

统一 speed-conditioned lead oracle 开发复验（配置 `9b313bfc...`；每档 20 集）：

```bash
CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode oracle --episodes 20 --speeds 0.3,0.4,0.5 --a-eff 0.09804843035177806 --seed 2026102500 --out benchmark/robot_closed_loop/results/r0_oracle_adaptive_dev60_20261016
```

后续每个输出路径必须是新目录；跑批使用 `tmux`，写 project log 和 PID，并低频查看。建议顺序：

1. ✅ `--mode static --episodes 20`：已验证 20/20 真实双侧接触、抬升和 0.5 s 保持；样本量仍仅为开发级。
2. ✅ `--mode slide --split calibration --episodes 10 --speeds 0.3,0.4,0.5`：target-only scene 完成 30 集 no-hand 轨迹；共享拟合 `a_eff=0.09804843 m/s²`。
3. ✅ `--mode slide --split holdout --episodes 10 --speeds 0.3,0.4,0.5 --a-eff 0.09804843035177806`：新种子 30/30 到达，固定模型 TTC 与横向误差均较小。
4. ✅ 统一 speed-conditioned lead 开发集每档 20/20；新 seed 独立 holdout 每档 100/100，Wilson 95% CI `[96.3%,100%]`。GT-current oracle 可行性达到每档成功率下界 >95% 的统计门槛。
5. ✅ `--ttc-bias-ms -20` 早闭合 sensitivity：每档 5 集，0.30 档 5/5，0.40/0.50 档 0/5；更早时序使手指在目标到达前过度闭合，已不作为候选。
6. ✅ 较晚时序 sensitivity：+20 ms 下 0.30/0.40 各 5/5；+30 ms 下 0.40 为 5/5；+40 ms 下 0.50 为 5/5。
7. ✅ 统一 speed-conditioned lead 开发复验每档 20/20；横向偏差、额外延迟注入完成；新 seed 独立 oracle 每档 100/100，Wilson 95% 下界 96.3%。
8. ✅ 新相机候选 target-only preflight 9 点静态视野采样通过；显式选择 V100S 的真实 Panda 动态 oracle 代表录像已检查；0.20 s Direct-HDR 1/2 kHz 路径和成本已复测。该 pilot 未验证长时采样率收敛或任何视觉估计器。

## 证据入口与 provenance

- 代码、配置和复跑命令：[`benchmark/robot_closed_loop/README.md`](../../benchmark/robot_closed_loop/README.md)
- 每次运行必须记录代码 commit、分支、Genesis/Torch/资产/overlay、配置 SHA256；产物 SHA256 清单在运行目录 `SHA256SUMS`。
- 结果目标位置：服务器 `~/Eventbased_WAM/repositories/gec-robot-finalcam-e63fb94/benchmark/robot_closed_loop/results/<run-id>/`（较早 runs 位于 `gec-robot/`）；轻量原始 CSV、JSONL、摘要、短录像及 SHA256 清单纳入分支。当前新增产物均为小型文件，无需外部大型产物托管。
- 服务器静态成功原始数据 SHA256：CSV `cca2129f62c502bc2367128284065f6e006b21b8466c445471bf33b247138f66`；轨迹 `fa8f8c0d540f6106490a583979e054015213d78bb532a25e082a9a8479efb68b`；GIF `f363f4de5cd15c3a4d25e2c6d3062f2e7d16e97d60d19171f35a29c4e1c2e76d`；`SHA256SUMS` `c586944bf6ef0cea126dfeafffa2c6ded327d33b317b34a8c661df8b2bf78179`。静态结果在服务器 `server_static_smoke_torch28/`，实际运行代码 commit `af0548c...`、配置 hash `7a7f738...`。
- 历史低机位 preflight 位于 [`camera_preflight_low_20261016/`](../../benchmark/robot_closed_loop/results/camera_preflight_low_20261016/)；它的遮挡比例对照无效，已由新 target-only 测量取代。
- 静态 20 集原始结果位于 [`r0_static_dev20_20261016/`](../../benchmark/robot_closed_loop/results/r0_static_dev20_20261016/)；运行代码 commit `418718a...`、配置 hash `39c7aff...`。CSV SHA256 `6bf0bf909ffd358a7edc51df738ab55713c98eeeff78a2df3cea6fb584b6c8ae`；代表性 trace `2154ab5d81a59bc146cc865c5f5d489594d4fb18491132729f366374ac371ab6`；成功 GIF `8dcd8ac511a432763d7073b9182470f0d2677869022f7a3bf4d626d5c84b787e`；run manifest `ca0d87b4f609f0520590fb77b7a1ee853bfde792533b5276fa28110ba76c89d0`；`SHA256SUMS` 文件本身 `603ff03fef8558ad79e49851d0bebc72a7da355c38eccec8de5fc9b52edca869`。日志 SHA256 `bbbf64d424ff6e4bf4dcdc4990424d238d37374044ef2e9680e0dd6dd5a053c7`，exit `0`，PID `1537609`。
- no-hand slide calibration 原始结果位于 [`r0_slide_calibration_20261016/`](../../benchmark/robot_closed_loop/results/r0_slide_calibration_20261016/)；运行代码 commit `865b79e...`、配置 hash `39c7aff...`、`panda_asset_in_scene=false`。CSV SHA256 `fef5cf0c22be2a2c5aa62ea4a7b53d4dfe878738f7f833ae6fc8b2a695be0336`；拟合 JSON `4d1886c0e18c5f9d7b2a381ddbaee64da7815af8a8c3430d3e092db690587178`；manifest `c92c006e80bc7531da2c275c40748a8f712c26631755a032139ecf6cc14eb3e9`；`SHA256SUMS` 文件本身 `16a2374d16292d71c0d6b4ce49e302e127fd253bb23f6f5ca27e20ff0b6e7c03`。全部 30 条逐集 trace 包含在校验清单。运行日志 SHA256 `217a93ba8cc1fa7117a1a515f41739dfd6c7d1c9bae63e1a53bd032df122bee9`，exit `0`，PID `1543261`。
- no-hand holdout 原始结果位于 [`r0_slide_holdout_20261016/`](../../benchmark/robot_closed_loop/results/r0_slide_holdout_20261016/)；运行代码 commit `93e739e...`、配置 hash `39c7aff...`、`panda_asset_in_scene=false`。CSV SHA256 `2588b58dfe2d13b7397d5956cdbf545efb8aec7a9491d7174351b38eaeb33bae`；manifest `7ba5d4097a4079a00c534e71d2e14697c864a803c91cd4cb77a6dbb764ce0915`；`SHA256SUMS` 文件本身 `51288e81254627504e6c393a6800dbb215207571d1e5415a04d8ab60d8abd053`；逐集 traces 均含于清单。日志 SHA256 `bf7ce564d21c03f5e08ab59613b0f4e321bc98ec47011a7311696163bc1a851e`，exit `0`，PID `1548544`。
- 首批 GT-current oracle 原始结果位于 [`r0_oracle_dev60_20261016/`](../../benchmark/robot_closed_loop/results/r0_oracle_dev60_20261016/)；运行代码 commit `4d70b8f0f796ca5163d47ab3b5aac98f71aa45a4`、配置 SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a`、`a_eff=0.09804843035177806`、GT observation latency 10 ms。CSV SHA256 `2b3fc7ea3294a371d2b9d8cfc51cb20583df9da661a7eea6a2ebefa21fdfa7d`；summary SHA256 `bb87bcda337e3edd4f71b9968b20515221bb6813e22429907721b712dbdf6bab`；manifest SHA256 `e31d39b053de368e57313fb52a5a75164ef15779a5591193ecdb76243cc9efc5`；`SHA256SUMS` 文件 SHA256 `3d339625fcc2f5897d11a30c447ac9fb7ebf494793204dc07b36dc70bedeb6b6`。逐集结果 CSV、代表性成功/失败 trace 与 GIF 均在清单。日志 SHA256 `3b1c69c2750de50eb4988563f7e537ece0ff3c61d264bdd71b7fc64b11f7e2fa`，exit `0`，PID `1551511`。
- 20 ms earlier closure sensitivity 结果位于 [`r0_sens_early20ms_dev15_20261016/`](../../benchmark/robot_closed_loop/results/r0_sens_early20ms_dev15_20261016/)；代码 commit `eff4f592fe4e503b8f4264415c2b3b0b0b326165`，配置 SHA256 `c85cb2b117d90704f2d824ccf7f0e102019b7d8e5d814a2edcdf3a6004932edf`，每档 5 集、`--ttc-bias-ms -20`。CSV SHA256 `dcc9b9cacceb810655a1b86a6b084de6582399f529a2621f5a96e2f1aa814993`；summary SHA256 `af44f9b96169ea20268833db48cf617f02460e1aa15124347f8d8da6c2a6a683`；manifest SHA256 `9c2a57cee87c646e1a7cfd8dcd5f62b97bcbceabc3a740aee87ed6cb2ce05899`；`SHA256SUMS` 文件 SHA256 `fe6a3b83635a2146cffc39f9d55f5788e017bdd8b17210698b541511fd42c744`。成功/失败 traces、GIF 与清单含于目录；launcher SHA256 `a5745c079001edc780f61d00b547f9feaba11ed6e5a14f33f8f0c7c82d1b3b15`；运行日志 SHA256 `c44e5351df71f87a05081b3bb4254359a15763e117d6f6fbc36d699b75c508dd`，exit `0`，PID `1565416`。
- 较晚闭合 `+20 ms` 结果位于 [`r0_sens_late20ms_dev15_20261016/`](../../benchmark/robot_closed_loop/results/r0_sens_late20ms_dev15_20261016/)；代码 commit `ae34e8c94fb6c0b8646cc0b901e42bd218617151`、配置 SHA256 `c85cb2b117d90704f2d824ccf7f0e102019b7d8e5d814a2edcdf3a6004932edf`，0.30/0.40 m/s 各 5/5 成功，0.50 m/s 0/5。manifest `ddaac966ffbbf7c84dbf3b77c7c2b2b2682abf49ff782a7a721b8949e5be446c`；`SHA256SUMS` `0d48368bf7be48108051512a70a6c5c67b35fcce4ad53a730914c398589c81db`；launcher `50b718ec347ebfbb7b437aa643dfad9f5a29712b4f16a87f4525a90becca4a3b`；log `3f94104713fb67f0e782f3c87dfad11aba166cb948aa852bbb6d9b331cd249e1`，exit `0`，PID `1570682`。
- 较晚闭合 `+30 ms` 的 0.40 m/s 结果位于 [`r0_sens_late30ms_mid04_20261016/`](../../benchmark/robot_closed_loop/results/r0_sens_late30ms_mid04_20261016/)；代码 commit `ae34e8c94fb6c0b8646cc0b901e42bd218617151`、配置 SHA256 `c85cb2b117d90704f2d824ccf7f0e102019b7d8e5d814a2edcdf3a6004932edf`，5/5 成功；manifest `15130a5b6c08360e2759e209ad5661893a97d6c9759581b189966fd52b202261`；`SHA256SUMS` `a1c955731fe0eb5dc6adf8e7eaef2d0b1b347877ff24039b3aa06cb0c624944c`；launcher `97bed4d5c72151ea60bb8f89c2c1dce359b55c23c2efbe76805a5eed8ea32d77`；log `2e9cff43544194d22d445b1ca1782d8bdf39955f17d447f9bcf63aeaa3787c91`，exit `0`，PID `1577375`。
- 较晚闭合 `+40 ms` 的 0.50 m/s 结果位于 [`r0_sens_late40ms_dev05_20261016/`](../../benchmark/robot_closed_loop/results/r0_sens_late40ms_dev05_20261016/)；代码 commit `ae34e8c94fb6c0b8646cc0b901e42bd218617151`、配置 SHA256 `c85cb2b117d90704f2d824ccf7f0e102019b7d8e5d814a2edcdf3a6004932edf`，5/5 成功，底面净空 0.10095–0.10237 m；manifest `81ff27e3df895d3afdcb1e3a6e6423e2ccb053ba5a693621ac55eacab76ad06b`；`SHA256SUMS` `0fe24e890ded18e3368d5b7f6ce4f836f994f5a9d7f318e4bbaf1c4704063924`；launcher `4a3cbe287104c72ad107b69bd0e9cb8a887ac4d9dd2721d5a1e4943612c63082`；log `420d95bc9d9bf5b30c1cac830bb30ed7197b8a38ffa2dbdc97ac91d9941ae5b2`，exit `0`，PID `1574449`。
- 无效旧机位 preflight 仅作 bug 证据保留：报告 SHA256 `eb586dd8e7dffb69c480ac4e1d3e7c16c42bdd2008c900642a5fb7f477f08ea0`，输出清单 SHA256 `3d92ac2fcb06440c4cf16b9f69d7a145bd6cef2b867228cb1470160b86277ae6`；不得用于相机可见性结论。

## 尚未完成 / 风险

- GitHub fetch 从服务器等待超时后，改用已推送 commit bundle 串行同步。runtime setup 第一次因共享包缓存权限失败，第二次成功但 Torch 2.5.1+cu121 的接触查询失败；project venv 安装 Torch 2.8 CPU 后静态抓取 1/1 成功。
- Camera render 返回值与 preflight 遮挡测量均已修正。旧高位负 y 机位在合法 home 姿态遮挡 98.31%，拒绝；修正同步后的 top-down 候选在拦截中心遮挡 100%，拒绝；正 x 侧候选 `[1.15,0,0.22]` 静态 home 视角通过，0–6.7% 遮挡、30–55 px 短边。单次动态录像检查的 9 个均匀抽样帧中，方块始终有可见区域，接触时被夹爪部分遮挡；此观察是定性 smoke，不是跨随机轨迹的遮挡率统计。
- no-hand calibration 与独立 holdout 均已完成；单因子 timing 扫描成功时序点已记录。
- 新 seed oracle 独立 holdout 三档均 100/100，Wilson 95% CI 下界 96.3%；仅支持当前物理场景、固定预测器与固定控制器的 oracle 可行性，不代表视觉估计器成功率。10 mm 横向位置偏差敏感性显著；总观测延迟增至 30 ms 的小样本仍 15/15。
- 正 x 侧机位已更新为当前候选；9 点 preflight 静态采样保持 Panda 在 pre-grasp home。代表性动态录像显示接触附近有部分遮挡但目标未完全消失；多 seed 动态遮挡率仍未量化。Direct-HDR 在显式 V100S EGL device 下完成短 pilot：1/2 kHz 事件数量相差 17.18%，较早未显式钉住 EGL device 的探索运行差 18.16%；两者均提示不得宣称时间采样已收敛，正式使用前应做较长窗口采样率收敛核验。
- Top-down 相机的首次 launcher 误指向旧 preflight worktree；由本会话启动的 PID `1609155` 已经 `tmux` 定向中断，未生成完整相机结果且不计数。launcher 根路径已修正后重跑。
- 新种子 oracle 独立 holdout 已完成：服务器 tmux `r0_oracle_holdout100x3_20261016`，PID `1594166`、exit `0`；实际代码 commit `d3bc9e3732b8e898c2412d34f7ea00bd07f4acd9`、配置 SHA256 `9b313bfc634e4eb39fa7591593bc808e6543f4a8a78f9dd15a332fb6be106439`、seed `2026102800`，每档 100/100，Wilson 95% CI 下界 96.3%。
- 正式 Ours/V2E 比较、大规模数据生成和训练均未开始。

## 冻结建议与研究会话需决策事项

建议研究会话冻结 R0 场景候选：40 mm、80 g 方块，Plane 低摩擦参数，1 ms 物理步；发射速度 0.30/0.40/0.50 m/s；固定相机 `[1.15,0,0.22]` 看向 `[0.65,0,0.02]`、320×240、垂直 FOV 30°；100 Hz 控制、10 ms GT observation latency；固定 `a_eff=0.09804843035177806 m/s²`；闭合 lead=`clip(0.08 - 0.10·(v0-0.30), 0.05, 0.10)` 秒，`v0` 为首次延迟补偿的因果速度估计。当前配置文件 SHA256 `124d87188e9a383a054fb7e6e6eaf106c6d331ad0abfaf2a8a959364eafdc549`。物理与控制候选的独立新 seed oracle holdout 使用前一版相机配置 `9b313bfc...`，每档 100/100，Wilson 95% 下界约 96.3%；新机位配置下另有 0.40 m/s 1/1 dynamic smoke 成功。相机只改变观察渲染，不进入 GT oracle 决策，因此 holdout 支持任务物理/控制可行性而不构成新机位视觉成功率。Direct-HDR 短 pilot 已在显式 V100S EGL 设备下运行，按 render-only 540×0.2 s 外推约 0.233 h@1 kHz、0.466 h@2 kHz；事件计数差异 17.18%，早期未固定 EGL 的探索运行差 18.16%，采样收敛未确定。建议冻结任务和相机候选，后续 reference 正式比较前先决定是否要求较长的 1/2 kHz 收敛试验；不把共享像素模型或理想阈值响应称为真实 EVK4 标定。主分支合并留待验收。

### 最终机位动态 smoke 与 Direct-HDR reference（显式 EGL 设备）

- canonical 复测由服务器隔离 worktree `gec-robot-finalcam-e63fb94` 执行，代码 commit `e63fb94872ddca204f6740981fe99b982e0b6777`，配置 SHA256 `124d87188e9a383a054fb7e6e6eaf106c6d331ad0abfaf2a8a959364eafdc549`，Genesis 1.2.2 + Torch 2.8.0 CPU。两次运行均显式设 `EGL_DEVICE_ID=2`；短 EGL context probe 得到 `GL_RENDERER=Tesla V100S-PCIE-32GB/PCIe/SSE2`、NVIDIA OpenGL 4.1，确认选中 V100S。任务期间 GPU 3 仍由 CARLA PID `2319055/2320180` 使用，未受干预。更早未设置 `EGL_DEVICE_ID` 的 pilot 原样保留，但下列结果作为 canonical。
- 动态相机 smoke 命令：`CUDA_VISIBLE_DEVICES=2 EGL_DEVICE_ID=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode oracle --episodes 1 --speeds 0.4 --a-eff 0.09804843035177806 --seed 2026102920 --out benchmark/robot_closed_loop/results/r0_oracle_finalcam_egl2_video_smoke_20261016`。结果 1/1 成功，双侧接触、抬升和 0.5 s 保持均达标。抽查 9 个均匀分布视频帧，接触时方块被夹爪部分遮挡，未完全消失；这是定性检查，不是动态遮挡率统计。GIF SHA256 `03cc1334b1af567270bfe1516da5538759b59510eaec3a7328ae0517fc2111eb`；trace `e7017b20e9437823135654fd7288ac5037c83ac9eb8f0ba10a43c578031c9d61`；逐集 CSV `c2629bde47da9b605c86bbaf2b5926af617e4e3038ad9fd3ecb39a6b3178477f`；manifest `f70e1ffed4d5c56d585580e5c3b91709c4f4bcd46c4d8e0ac752a9f23474759b`；SHA256SUMS `520a67b1e932ab3c42fd74fd53697b8b84b724be53904b820a453ef4b4a31d2c`。服务器 log SHA256 `c2ea8a978e222d5d2b4fc5a97feef352c2a96cbcac930005b60e21f012e085f6`，exit `0`，launcher PID `1626504`。
- Direct-HDR 命令：`CUDA_VISIBLE_DEVICES=2 EGL_DEVICE_ID=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode reference --speed 0.4 --reference-duration 0.20 --seed 2026102930 --out benchmark/robot_closed_loop/results/r0_direct_hdr_finalcam_egl2_dev_20261016`。0.20 s 输入由 401 个 0.5 ms HDR render 构成；1 kHz 产生 45,751 events，2 kHz 产生 55,244 events，计数相差 17.18%；同像素同极性、2 ms 容差匹配率为 1 kHz 97.39%、2 kHz 80.65%。总采样计时 9.62 s（render 与 DVS sample 计时和，不含启动和物理步）；首帧 render 313.90 ms，warm render mean/p50/p95 为 7.76/7.91/8.52 ms，DVS mean/p95 每采样 15.47/26.03 ms。线性外推 540 条、每条 0.2 s 的 render-only 成本为 0.233 h@1 kHz、0.466 h@2 kHz，不含物理、DVS、启动、数据写盘与模型成本。输入是线性 HDR luminance 的 log；响应复用 `genesis_event_plugin.DvsEmulator`，ideal threshold 0.2、无噪声/无限带宽；source 绕过 Ours 时间重建和 V2E 插帧，不代表 EVK4 标定。`reference_convergence.json` SHA256 `d564e9f446318bd942594e4d020a1964fc38a9818e93b89447e682c31dab1866`；1 kHz NPZ `e79ba48c2bc44b3df4f67d36caed54be46d9339f53db75bde1641db587924ac1`；2 kHz NPZ `c793b0df0ca31e2bbe56106c55a1829f8f20a622bd10495652b67a09826df1ab`；HDR/RGB snapshots `086f65d04de59867bbd815d0057b8777c67441b0da0d97b55b5dfa3934261187`；manifest `ab00c7624cfbd457679371ed23eb368febc13cd95a9ad277871552847992011f`；SHA256SUMS `13733abd2fab97ca4f151102140e91262501c82f912cb0c35bb08b628840265c`。服务器 log SHA256 `c5c5ca2c70d6d8e97a2c66366a7a1115d33d7421eade25599fe95d9b000dcfca`，exit `0`，launcher PID `1628136`。
- 完整复跑 launcher 与日志在 `results/runtime_logs/r0_oracle_finalcam_egl2_video_smoke_20261016.*` 和 `r0_direct_hdr_finalcam_egl2_dev_20261016.*`；结果目录内 `sha256sum -c SHA256SUMS` 全部通过。此前未显式设 EGL 的运行结果仍在 `r0_oracle_finalcam_video_smoke_20261016/` 与 `r0_direct_hdr_finalcam_dev_20261016/` 供审计，不作为本节成本与动态视野主证据。

### speed-conditioned oracle 开发批次校验

- 归档目录：[`r0_oracle_adaptive_dev60_20261016/`](../../benchmark/robot_closed_loop/results/r0_oracle_adaptive_dev60_20261016/)。服务器运行 commit `d3bc9e3732b8e898c2412d34f7ea00bd07f4acd9`，配置 SHA256 `9b313bfc634e4eb39fa7591593bc808e6543f4a8a78f9dd15a332fb6be106439`，运行命令见“统一 speed-conditioned lead oracle 开发复验”。每档 20/20；Wilson 95% CI 为 `[0.8389, 1.0]`。
- 结果 SHA256：逐集 CSV `3b44f6cd783562594da04fc246cdc1cc708c3ab240aa3227cc5cc56e1c7ebb78`；摘要 `0d48fb7adb6003e1d2bdd7aeec128e65ee36b4d9363531e4021bf6ed2961d1ac`；代表 trace `d21fcfc1858ed84d7888041b84793b867145244fee87f24225cdae3365e4d214`；GIF `57cd46752368283ba27d46ea7962c52bf368a218d1bc53482b49c4016f0a5651`；manifest `5dedfb11e9c68bc4d354c6672c599982f4a1335813ea00000fc8ae09a452db25`；SHA256SUMS `128ede7a3a32588b1778481368b7825cb07688e0963aa7a52569dc2823418800`。
- 运行日志 `results/runtime_logs/r0_oracle_adaptive_dev60_20261016.log` SHA256 `f487e018a1934edc1198393acd3303cb4fe7ce713e39d9e6c85b8c49714b1951`；launcher SHA256 `a47f4d9fb01143aff29cae51a4061716fa8ee1433243c500aa34b965c8a1cdaf`；exit `0`，PID `1582044`。


### 位置偏差与附加延迟敏感性

- 两批运行均用代码 commit `d3bc9e3732b8e898c2412d34f7ea00bd07f4acd9`、配置 SHA256 `9b313bfc634e4eb39fa7591593bc808e6543f4a8a78f9dd15a332fb6be106439`；每档 5 集，敏感性结果不计入独立成功率估计。
- `r0_sens_biasy10mm_dev15_20261016/`：命令参数 `--bias-y 0.010 --seed 2026102600`。0.30/0.40/0.50 m/s 分别 0/5、1/5、0/5，合计 1/15；14 个失败均为 `unstable_grasp`。注入的横向偏差为目标宽度的 25%，显示抓取对横向状态误差敏感。CSV SHA256 `81321b67711f04938851de4d167dd443156f63a0d813bc4e42b362a22573ce2e`；manifest `ba13329b623253ecea1458ac780fb01864f99f7c198ce6d8f878ed8ea74b4a9b`；目录 SHA256SUMS 文件 `a3ce6e24e40dc93f17329b09d8df8d2550532b1161c53d3a934e87594b5013f3`；运行日志 `99ffa4d9dee1cf5b8fac9fdad3e2a707a6ec41efd2c605ced15056d404a33a73`，launcher `641aca95f3be085446898aacd512cd966cf657608f3dc6fead8ec676be331ab3`，exit 0，PID 1588265。
- `r0_sens_latency20ms_dev15_20261016/`：命令参数 `--extra-latency-ms 20 --seed 2026102700`。总观测延迟从配置的 10 ms 增至 30 ms；0.30/0.40/0.50 m/s 均 5/5，合计 15/15，故该扰动幅度下未观察到性能退化。CSV SHA256 `79782d3aaa00d8e6570ed986eeb4cc402a08b6a09caa6a51b8435b684ad91518`；manifest `2d7e820763a065a234f718e0438075f7124fb294026745b74a1ea5cb2a8f27fe`；目录 SHA256SUMS 文件 `5cf2d81249b303dfa9354a666be680b3ce567400314fb9d18718ccda653fc787`；运行日志 `eac734660b1b8dc7657db8cb15dbc1783b0778fe6687aa584cf4913c379d1335`，launcher `1a30b5a20caca9045b4802118964536a2da618397743e311d560f45fd83d1490`，exit 0，PID 1591125。两个目录中的代表性 trace/GIF 均由 SHA256SUMS 覆盖；两份清单均已在本地逐项验证。


### 独立 GT-current oracle holdout（100 集/速度）

- 服务器命令：`CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode oracle --episodes 100 --speeds 0.3,0.4,0.5 --a-eff 0.09804843035177806 --seed 2026102800 --out benchmark/robot_closed_loop/results/r0_oracle_holdout100x3_20261016`。
- 实际服务器代码 commit `d3bc9e3732b8e898c2412d34f7ea00bd07f4acd9`，分支 `exp/robot-closed-loop-v1`；配置 SHA256 `9b313bfc634e4eb39fa7591593bc808e6543f4a8a78f9dd15a332fb6be106439`；Frozen core `e79e656ac00f7847b98db9e12bf43b093f3afff1`。Genesis 1.2.2 + Torch 2.8.0 CPU，Panda XML SHA256 `e502e07fcef6219024be02dfb9cd65d5c194b4767adf381ca51f10172ff8a32d`。
- 0.30/0.40/0.50 m/s 各 100/100 完整成功；各档 Wilson 95% CI `[0.9630,1.0000]`。固定 10 ms GT observation latency；成功定义含真实接触、合格抬升、净空与 0.5 s 保持。每档使用连续且不重叠 seed block，总成功数 300/300。
- 归档 [`r0_oracle_holdout100x3_20261016/`](../../benchmark/robot_closed_loop/results/r0_oracle_holdout100x3_20261016/)；逐集 CSV SHA256 `17b4f6ff5c0abc7615b31b80f4c48aa336ea3cfbf237c4d5b7d660d4e6e7894c`；summary `420e9f994ff688d3ce64e33d1d779e1ab70c44ce87939b38a980dbd9826b30e3`；代表 trace `1a733d85e111332d5ddc3835e0cc9a82cb70e78b792af884422b77741f822247`；GIF `3a1806a61f92c37a6aa2cf7ea7dcae40ae5406faf3c4e1d30fa29a45f2b75574`；manifest `7bc3c5b9b69f52834c2dc78243d1fd77b8edf5ea134706f51a6aba53e5d46588`；SHA256SUMS 文件 `de9d6c2f5fb07e61dcc51fc33c689d6621362bf791c0471d82a5f679f76aae35`。本地 `sha256sum -c SHA256SUMS` 全项通过。
- 服务器日志 SHA256 `e0f82252cd7f3cf42fc0d917b12709e432f5c583b630f6b787a86b62ef7eb578`，launcher `933e76254030e99fc939c3c925f967cf4f8c59e0f165519211ac4d294b4d3f2a`，exit 文件 `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`（内容 0），PID `94bd2aef56d16331563e9025cab15eb6a69f031f32344f58130c3763da89a307`（内容 1594166）。Genesis 有共享 Quadrants cache lock 警告，但运行完成并 exit 0；未运行提示中的共享缓存清理命令。


### 首轮高位负 y 相机候选（拒绝）

- 运行代码 commit `a0db01337cbf9266ede5518cdfe034b2e0a995f1`，配置 `high_oblique_neg_y.json` SHA256 `eb20e525fa63ddc89887be244086c406aad9ae17e98dafb59d301dfeb333d05f`；相机 `[0.65,-0.25,0.75]` 看向 `[0.65,0,0.02]`，320×240/FOV30°。
- 合法 pre-grasp home 姿态、拦截点的 target-only 像素数为 771，机器人场景仅见 13 px，遮挡 98.31%、短边 2 px，候选拒绝。RGB 显示目标几乎被 Panda 手掌遮住。
- 首轮输出 [`camera_preflight_high_oblique_neg_y_20261016/`](../../benchmark/robot_closed_loop/results/camera_preflight_high_oblique_neg_y_20261016/) 中 `intercept_render` 有效；9 个路径采样的 robot-scene mask 因没有在 `set_qpos()` 后推进物理/渲染同步而重复，已在 `NOTE.md` 标为无效。后续复测使用同步修复。JSON SHA256 `cff9ef8a8719bb7ab726312f937577df0816c06dc8eda96a282226f3b66d322b`；RGB `0285aa68a016f70faba00ed87f9458d4b3119ea7f11b5b8a46e888e3a76d4773`；HDR/depth/seg `372cb9987986b040c359f7bbd49fe5fc7774e6fd3a6fb4af3203e5e356bbdaa7`；manifest `5e8df2512103b25eb666bf64747b1e99640227a10eab32bfb1682de2d039bdd8`；SHA256SUMS `c61e86218f36a6be59a58937a98cd0c51f19586520914b4b04010004781d5718`。服务器 log `fd1efae6872e8638fea90f731955dafbcfc2c41f679c29589bfc48cc03199972`，exit `0`，PID `1605800`。


### 通过的正 x 侧固定相机候选

- 当前候选配置 [`scene_pilot_r0_v1.json`](../../benchmark/robot_closed_loop/configs/scene_pilot_r0_v1.json) 的 SHA256 为 `124d87188e9a383a054fb7e6e6eaf106c6d331ad0abfaf2a8a959364eafdc549`；相机位姿 `[1.15,0.0,0.22]`，看向 `[0.65,0.0,0.02]`，320×240、垂直 FOV 30°。通过测量运行代码 `062c3811684a854110e01cfb500222083f0f6deb`；实验配置 SHA256 同为 `124d87188e9a383a054fb7e6e6eaf106c6d331ad0abfaf2a8a959364eafdc549`。
- 合法 Panda home 姿态下拦截中心目标 1060 px 全可见，0% 遮挡，短边 30 px。发射与中段路径样本 0% 遮挡、短边 36–55 px；拦截 y 边缘遮挡 5.29–6.65%，短边 32 px。HDR 是有限 float32 `[240,320,3]`。RGB 已人工查看，方块在夹爪下方可见。九点是静态姿态采样，未覆盖 Panda 动态横移过程。
- 结果目录 [`camera_preflight_side_x_positive_20261016/`](../../benchmark/robot_closed_loop/results/camera_preflight_side_x_positive_20261016/)：JSON SHA256 `38448fcc48c4a7671cf467012eef1db86fd144d89e681aa79be6526fe20f393d`；HDR/depth/seg `3124df3e100360306ede3bd9900e167aef3863f24f0ce94dd6f36529de3ce591`；RGB `a76203b0d8f3fbc8813d6a3492454387f594898a9e7b7ceb7a4e17765e3fbe2b`；manifest `9655d8f6afcaa045969dce8c7baefbab8a9aa6c0f5422ba0ae708dcce584ec30`；SHA256SUMS `983d7b9608c178447cc6ce8c7baefbab8a9aa6c0f5422ba0ae708dcce584ec30`。日志 `1e5e84ae0d1ea87c8f831cdb90ac8eaeac4c16afb6fa06b9a1f9e99b7e412dc8`，exit `0`，PID `1612441`。结果清单本地全项验证通过。
- Top-down 失败候选已归档在 [`camera_preflight_top_down_20261016/`](../../benchmark/robot_closed_loop/results/camera_preflight_top_down_20261016/)；配置 SHA256 `b987145661715118ae77ef5d879a980762e6ece316445d84392d42f93f301096`，代码 commit `062c381...`。发射/中段目标无遮挡，但拦截中心 100% 被挡，exit `1`，拒绝该机位；目录 SHA256SUMS `b70826cf369e3bfdb6741f41a53e5e8547a735ca60d347da740f51f8be5efcff` 已为归档文件重新生成并校验。


## R0.5 strict closeout（进行中）

### 已落地并有初步服务器证据

- 新增下游严格评分：每个物理步计算旋转后立方体 8 个角点相对 Plane 的真实最低净空；连续 0.5 s 同时要求双侧指接触、无 Plane 接触、真实净空 ≥80 mm，且从首次双侧接触起 object-to-TCP 相对位置漂移 ≤10 mm。逐集结果和 1 kHz 状态 trace 保存严格评分统计、接触/支撑样本数、滑移峰值及最小净空。
- 保留 R0 旧规则 `r0_v1_legacy_center_clearance_historical_contact`，在旧规则首次满足停止条件时单独复算并记录；新正式 `success` 字段采用 `r0_5_strict_continuous_contact_clearance_slip`。历史归档文件不改写。
- 新 R0.5 候选配置将 GT oracle、Direct-HDR 与自由滑动校准统一到 0.5 ms 物理步；控制仍为 100 Hz，预测器/闭合调度/机器人任务候选不变。因物理时间步改变，需要先重做无手减速度拟合，再运行 oracle。
- 同一 oracle 控制与评分循环可接 GT 当前状态或因果事件状态。事件源只接收相机、固定针孔标定与直接 HDR 生成的 `[t_s,x_px,y_px,p]`；不读目标 GT、GT segmentation、depth 或未来状态。GT 误差评估和 segmentation 可见性审计在控制循环的独立输出侧进行。
- 新 reference 每物理步 direct-render 线性 HDR，在同一条闭环物理轨迹中并行产生 1/2 kHz 事件流；日志计划覆盖 event-time 对齐误差、10 ms voxel 一致性、事件状态误差、动态目标遮挡率及完整 episode wall time。此时的 centroid/regression 仅用于状态接线 smoke，不构成可用 estimator 的精度结论。
- 跨会话诊断输入（研究会话提供，R0 代码 `e70ba12`）：三个单种子重跑在旧停止窗口的真实底面最低净空为 `79.406/79.090/78.935 mm`；原命令各延续 50 ms 后都通过严格检查。此证据指向停止窗口/底面测量缺陷，而非需要立即改控制参数；R0.5 本轮据此修复评分与计时，未重调机器人机械控制。

### 尚未运行 / 不可声称

- strict oracle 小样本试跑已通过；尚未完成更新评分下的独立大样本统计和误差注入 strict 复测。旧 R0 的 300/300 不能引用为新成功率。
- 0.5 ms no-hand 拟合数据已有 60 集，但固定 `a_eff` 的新种子预测 holdout（30 集）尚未运行；将用于校验自由滑动到达参考。
- 完整拦截/接触/抬升 Direct-HDR 的 paired 1/2 kHz、横向边缘轨迹动态可见性、事件状态接管控制和实际成本均待服务器验证。
- 本地只做了 Python 编译、JSON 解析、diff whitespace 与 frozen-core 差异检查；物理实验均在服务器执行。

### strict static smoke 与 0.5 ms no-hand 校准

- 运行代码 commit：`b757a83a7b8a391bb7553e4ee05d5fbce2cbbb94`。实验在隔离 worktree `~/Eventbased_WAM/repositories/gec-robot-r0_5-b757a83`（detached 于机器人分支提交）；服务器原 `main` 与旧 R0 worktree 未修改。
- strict static smoke：1/1 success；最长连续严格保持窗 0.500 s，窗内最小真实底面净空 `0.08008784 m`，双侧接触 1000/1000 样本、支撑面接触 0/1000，object-to-TCP 最大漂移 `0.001288 m`。旧评分在 0.807 s 独立复算成功；严格评分于 0.808 s 完成。source config SHA256 `4f58c58e3b0db367066ae126db7f3f4c603bd4c273aede33777e8381db240517`；逐集 CSV `09a4cf4c795a4ddb19c8f692a5d8678fabdd151cce0addb356c2ebbaeea8b954`；trace `fde20ca377a445f0117a17b84a66ff851fda3721ab14e59dd407bd47be31ec2b`；GIF `682315c67458610cb6f443dd49af70da3793b5998463f74614ecd697fcbdbb9e`；目录 SHA256SUMS `b3e4290c2b5b897929b1c9840f4ddd879fbcf24c5c4362e8b9c2e4f42b6097d1`。仅验证 strict 测量链，不代表动态成功率。
- no-hand 校准：同一代码 commit、0.5 ms physics，60 集（0.30/0.40/0.50 m/s 各20），60/60 均通过拦截平面。联合拟合 `a_eff=0.09803606718731761 m/s²`；单集拟合中位数 `0.0980347534`，IQR `[0.0979928904, 0.0980804825]`。各速度档中位到达时间 `0.99112/0.66988/0.52403 s`，中位到达速度 `0.20285/0.33438/0.44863 m/s`。校准运行用的 source config hash `4f58c58e3b0db367066ae126db7f3f4c603bd4c273aede33777e8381db240517`；拟合后更新的候选配置 SHA256 `dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd`。结果目录 `r0_5_slide_calibration_20261016/` 的 SHA256SUMS `f7ac352ffb2fd75475dd3c85c53f185bb188c886873b1133f6c4bb9f20939059`；log `f2834f979f4f0ebca968e893c6c11010692a08a2bd1470f2e4b4be72e2ace0a6`；launcher `b53243fb50f47f1fadf26c1e103d5781274913465a102915c919ec7ac7cccc0a`；exit 文件内容 `0`、SHA256 `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。结果清单校验通过。

### 严格评分 oracle 先导批次

- 代码 commit `6cd7c70ea64ece2c274c00ae00ebabf30ebbc65b`，配置 SHA256 `dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd`，0.5 ms physics、100 Hz control、10 ms GT observation latency、`a_eff=0.09803606718731761 m/s²`。每档 3 个新种子，三档各 3/3 strict success（9/9）；Wilson 95% CI 每档 `[0.4385,1.0000]`，仅用于诊断，远不足以验证 95% 成功率门槛。旧 R0.5 分数复算也是各 3/3。
- 每集要求严格连续保持 0.500 s；长保持窗真实底面最低净空 `80.0000–80.1152 mm`，其间支撑面接触样本 0，首次双侧接触后最大 object-to-TCP 漂移 `1.856–3.582 mm`（限制 10 mm）。从闭合开始到校准 no-hand 模型的自由滑动到达时刻为 `−77.8 至 −50.6 ms`；首次接触相对该同一到达参考为 `−61.8 至 −37.1 ms`；接触位置在拦截面前 `12.0–19.8 mm`，平面实际过线均未发生；首次接触的目标/TCP 平面相对速度为 `0.189–0.416 m/s`。因此后续时序统计统一相对**no-hand free-slide 预测到达**，并单列 `actual_crossing_t_s`（被截停时为空），不将未发生过线记作缺失的接触时间。
- 逐集 CSV `d3ed150caf8db357c9b5b961cca993ee68078fba952e1b92c07fe332891b55e0`；摘要 `bb1f84cc1ed168b85c2fd93cc0916b3b155c13ea7e30172d0cce7d73610cc27e`；代表成功 trace `cc9ffd3f4b3bfb2d247a43ee037ddb1ffc8f248f298e401ed1297589a38ef033`、GIF `6975d31e0dee2cc248b114cf1107fbe928471d18d8328511e4c1ef74bbf42b6d`；目录 SHA256SUMS 文件 `b88fe6c39e4ae83d91222d0ff6d2a065836676560e13af6b0ca0574aee9ca474`，已在本地和服务器 `sha256sum -c` 全项通过。服务器日志 SHA256 `628ce15d0e47429c57c5ed273409e78a0f815c49de1a80d067acbc96edb55f64`，launcher `d4feb20f93d7c0e946fe828ec0d2f0954157f634261817e743716275a9eb0a2f`，exit 文件内容 `0`；PID `1932999` (launcher) / `1933002` (python)。Genesis 共享 Quadrants cache lock 仅告警；没有清理共享缓存。
- 本地核验：`python -m compileall -q benchmark/robot_closed_loop`、`git diff --check`、结果目录 `sha256sum -c SHA256SUMS` 均通过。旧评分保留在每集 `legacy_score` 字段，原 R0 归档不变。

### 当前复跑入口

配置起点：[`scene_pilot_r0_5_candidate.json`](../../benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json)。命令、评分边界和事件接入说明见 [`benchmark/robot_closed_loop/README.md`](../../benchmark/robot_closed_loop/README.md) 的 R0.5 小节。服务器只在新的 branch worktree 执行，保留服务器原 `main` 工作区；启动前检查 GPU PID/归属，EGL 仅选空闲的 GPU 2。

机器人分支代码、配置、报告与运行器已推送至 `9f15d01`。其中 `1b2cb24` 加入旧评分窗口最低真实净空，`96b6450`/`d7da857` 补充 lift-gate 因果状态和逐时刻事件置信度日志，`0f2fafb`/`3a6a366`/`5fbe64a` 分别准备 strict 位置偏差、速度×横向位置 reference、0.5 ms no-hand holdout 运行器。严格 oracle 新种子 100×3 批次正在服务器隔离 worktree `~/Eventbased_WAM/repositories/gec-robot-r0_5-02b1afc` 上运行，实际 benchmark code commit `8f22be28c7f334062de3e9a34b6c10352a2524f9`、配置 SHA256 `dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd`、seed `2026103300`、Python PID `1939038`、tmux `r0_5_oracle_strict100x3_20261016`。该命令 `CUDA_VISIBLE_DEVICES=`，无 GPU 占用；launcher 使用 `/usr/bin/time` 记录完整 Python 进程 wall/CPU 时间。待主批次完成后，先运行固定 `a_eff` 的 30 集 no-hand holdout，再运行 strict 位置误差注入与完整 Direct-HDR/event-state smoke，并在本节补录逐集结果、轨迹/录像、数据与日志哈希及协议建议。
