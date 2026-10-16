# Genesis Event Camera：Robot R0 / R0.5 / R0.6 状态

**更新时间：** 2026-10-17（Asia/Tokyo）
**阶段：** R0.5 strict oracle 已核验三档各 100/100；GT-shadow 9/9 是延迟当前 GT 驱动的 sensor-reference，不是事件控制。R0.5 event-controller smoke 为 strict success 0/9、接触 3/9、抬升 0/9。R0.6 已修复速度回归和状态新鲜度缺陷、完成三集 raw-event 离线回放，并在 g1 完成新一组三集 2 kHz event-controller rollout：strict success 0/3、接触 1/3、抬升 0/3。任务、相机、物理时钟与 strict scorer 保持不变。事件空间归因和 lift 状态仍未通过；尚未启动大规模配对数据、正式多 seed 训练或 Ours/V2E 排名。
**研究边界：** 机器人任务位于 benchmark 层；`genesis_event_plugin/` frozen core 不改；不使用未来 GT、逐帧重设目标、焊接或吸附。

## 当前进度

R0.5 场景可行性与 R0.6 状态接口诊断分开报告。R0.5 oracle 是三档各 100/100；旧版 event-controller smoke 是 0/9 strict success、3/9 接触、0/9 抬升。R0.5 GT-shadow 是独立的完整成功 reference 轨迹，控制器使用 10 ms 延迟当前 GT，事件只用于评估。R0.6 已完成 raw-event 离线回放和三集 2 kHz 真实事件闭环；随后会用已保存 GT-shadow 事件做 tiny overfit 管线诊断，不新增 Direct-HDR 渲染轨迹。逐批次的实际代码 commit、配置/数据哈希、日志和复跑命令如下。

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
- event-controller smoke 中边缘轨迹的低可见比例与失控后的机器人姿态同时出现；独立 GT-shadow 的九条完整成功轨迹在相同左右边缘范围内仍有 98.86% 最低动态可见比例、100% 中心可见比例。按该批 bbox 半开区间口径，目标短边为 30–54 px。当前证据支持正 x 侧机位作为任务相机冻结候选；结论限于当前成功轨迹、配置和 evaluator-only target mask，不代表估计器精度或真实传感器标定。
- Top-down 相机的首次 launcher 误指向旧 preflight worktree；由本会话启动的 PID `1609155` 已经 `tmux` 定向中断，未生成完整相机结果且不计数。launcher 根路径已修正后重跑。
- 新种子 oracle 独立 holdout 已完成：服务器 tmux `r0_oracle_holdout100x3_20261016`，PID `1594166`、exit `0`；实际代码 commit `d3bc9e3732b8e898c2412d34f7ea00bd07f4acd9`、配置 SHA256 `9b313bfc634e4eb39fa7591593bc808e6543f4a8a78f9dd15a332fb6be106439`、seed `2026102800`，每档 100/100，Wilson 95% CI 下界 96.3%。
- 正式 Ours/V2E 比较、大规模数据生成和训练均未开始。

## R0.5 冻结建议与研究会话需决策事项

建议把以下内容作为正式协议冻结候选：40 mm、80 g 方块，Plane 低摩擦参数；0.5 ms 物理步；0.30/0.40/0.50 m/s；正 x 侧固定相机 `[1.15,0,0.22]` 看向 `[0.65,0,0.02]`、320×240、垂直 FOV 30°；100 Hz 控制、10 ms 因果观测延迟；固定 `a_eff=0.09803606718731761 m/s²` 和现有 speed-conditioned closure schedule；成功定义为真实双侧接触、无支撑、真实底面净空 ≥80 mm、相对滑移 ≤10 mm、连续保持 0.5 s。配置 `scene_pilot_r0_5_candidate.json` SHA256 `dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd`。strict oracle 独立 holdout 300/300，各档 Wilson 95% 下界 96.3%；GT-shadow sensor-reference 九集也全部通过严格评分，但其控制使用 10 ms 延迟当前 GT，不能计作视觉成功率。相机的九条成功轨迹动态审计通过。事件控制 smoke 仍为 0/9 抬升；1/2 kHz 采样下 event-derived post-contact 状态位置 P95 约 7.8/8.0 cm、速度 P95 约 0.68/0.67 m/s，尚未证明事件状态可用于稳定 lift gate。建议冻结任务和相机候选，暂不冻结 event-state adapter 或 reference 采样充分性；由研究会话决定下游是否先改进无 GT 因果状态估计，以及是否需要关键接触窗口高于 2 kHz 的 reference。pixel response 复用插件 `DvsEmulator` 的 ideal threshold 0.2、无噪声、无限带宽；绕开 Ours 时间重建与 V2E 插帧，不是 EVK4 标定。主分支合并留待验收。

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

### 已实现的测量与因果边界

- strict scorer 逐物理步用旋转后方块的 8 个角点计算真实底面净空。合格保持要求连续 0.5 s 同时满足净空至少 80 mm、双侧指接触、无支撑面接触，且从双侧接触锚点起物体相对 TCP 平移不超过 10 mm。逐集 CSV、trace 保存真实净空、接触/支撑计数、连续窗口与相对滑移。
- 旧评分 r0_v1_legacy_center_clearance_historical_contact 保留并并行复算；新 success 使用 r0_5_strict_continuous_contact_clearance_slip。R0 历史结果未改写。
- GT oracle 与事件状态连接相同的固定预测器、控制器和 strict scorer。事件状态入口只接收带时间戳的事件、固定针孔标定和机器人本体状态；事件路径不读取目标 GT、GT segmentation/depth 或未来状态。GT 状态误差、接触真值和可见性 segmentation 只在独立评估侧记录。
- Direct-HDR reference 绕开 Ours 时间重建和 V2E 插帧。HDR 帧经共享 plugin DVS 像素响应产生事件；当前 smoke 用理想对比阈值配置、无噪声/无限带宽，不是 EVK4 物理校准。reference 与两个 estimator 共享的是像素响应模型，不共享时间重建/插帧算法。
- 本轮候选场景/机械控制未因 R0 旧停止窗缺陷而重调。研究会话提供的旧窗真实净空 79.406/79.090/78.935 mm，延长原命令 50 ms 均通过；这是触发修正测量和计时的证据，不计入 R0.5 结果。

### 服务器静态与无手校准

- strict static smoke 1/1 成功；最长连续合格窗 0.500 s，最低真实底面净空 80.08784 mm，双侧接触 1000/1000、支撑接触 0/1000、最大相对滑移 1.288 mm。此项只验证测量链，不代表动态成功率。
- 0.5 ms no-hand calibration 60 集（0.30/0.40/0.50 m/s 各 20），60/60 通过拦截平面。固定共同减速度为 0.09803606718731761 m/s²；episode 拟合中位数 0.0980347534，IQR [0.0979928904, 0.0980804825]。各档中位到达时间 0.99112/0.66988/0.52403 s，到达速度 0.20285/0.33438/0.44863 m/s。
- 校准先导运行代码 commit b757a83a7b8a391bb7553e4ee05d5fbce2cbbb94、初始配置 SHA256 4f58c58e3b0db367066ae126db7f3f4c603bd4c273aede33777e8381db240517；拟合后候选配置 SHA256 dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd。校准数据清单 SHA256 f7ac352ffb2fd75475dd3c85c53f185bb188c886873b1133f6c4bb9f20939059。

### strict oracle 先导批次（诊断样本）

- commit 6cd7c70ea64ece2c274c00ae00ebabf30ebbc65b，配置 SHA256 dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd，0.5 ms 物理步、100 Hz 控制、10 ms GT observation latency，固定 a_eff=0.09803606718731761 m/s²；每档 3 集，9/9 strict success。每档 Wilson 95% CI 下界仅 43.85%，不用于验证成功率门槛。
- 合格保持窗最低真实净空 80.0000–80.1152 mm，窗内支撑接触 0，最大相对滑移 1.856–3.582 mm。闭合和首次接触时序均相对固定 no-hand 预测到达时间报告；接触发生在拦截平面前，实际过线不存在，因此不能用“过线到接触时差”作为主指标。

### 独立 strict oracle 批次（正式 R0.5 可行性证据）

- g1 隔离 worktree /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-02b1afc，运行代码 commit 8f22be28c7f334062de3e9a34b6c10352a2524f9，候选配置 SHA256 dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd；CPU-only，0.5 ms physics、100 Hz 控制、10 ms 状态延迟，固定 a_eff=0.09803606718731761 m/s²，seed block 起始 2026103300。
- 0.30/0.40/0.50 m/s 分别 100/100 strict success（总计 300/300）；各档 Wilson 95% CI [0.9630, 1.0000]。旧版 R0 scorer 并行复算也各档 100/100；这里明确报告为 R0.5 新数据，不引用旧 R0 的 300/300。
- 每集合格连续保持 0.500 s；300 集窗内最低真实净空范围 80.000246–80.117938 mm；双侧接触与净空合格样本均为每集 1000 个，合格窗内支撑接触为 0；最大物体/TCP 相对滑移跨集 1.829–4.416 mm（10 mm 限值）。
- 可解释到达参考为校准的 no-hand free-slide 预测到达时间。各档闭合开始相对预测到达的中位数为 -76.16/-65.25/-55.49 ms；首次接触中位数为 -60.67/-50.34/-42.82 ms。所有目标都在被机械臂截停前未越过拦截平面（实际 crossing 0/300）；首次接触位置在平面前：各档范围 11.18–13.76、14.95–19.63、16.38–22.34 mm。首次接触目标相对 TCP 的平面速度中位数 0.1947/0.3089/0.4150 m/s。
- 逐集结果和代表录像： [r0_5_oracle_strict100x3_20261016](../../benchmark/robot_closed_loop/results/r0_5_oracle_strict100x3_20261016/)。CSV SHA256 0678027d67c8068aa8b4764651c8df98c67e892fe4ea5792d23ecf0824462df3；summary 48f5cdd47f4ded4677a11ef2baadf7a3f8868b2be61ccdfff23eea72f417f550；manifest 51decf3f9780f60a0b2c12fb7b4d5ef25971da67769b3bd79b15f0a95c3c81fd；代表 trace b8bd06981d0ceec6a97685779fb2724411efbf30a99782db9197283d896aa35f；GIF b527e85a396aa5c3503faa58a58e2e23c2209839480b2e7609ca96f197668577；目录 SHA256SUMS 文件 27ca298d8715747830c3f6b434e4064c7c9c136c915295390055fe5dcdd8775f。服务器和本地 sha256sum -c SHA256SUMS 全部通过。
- 运行命令：CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode oracle --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json --episodes 100 --speeds 0.3,0.4,0.5 --a-eff 0.09803606718731761 --seed 2026103300 --out benchmark/robot_closed_loop/results/r0_5_oracle_strict100x3_20261016。完整 Python 进程墙钟 2315.97 s、用户 CPU 2274.06 s、系统 CPU 35.33 s、最大 RSS 3,404,592 KB，exit 0；launcher PID 1939036，Python PID 1939038。log SHA256 8acf9ccbd518549c46f05d82a09c9b3d51cda4ae2058d540956d0f1dee4df0c8，runtime 0b1ac74c9c786368219808171f7808a69f943e631c158716b492ddfe282cadd5，launcher bedf777f7388d4948bb1dd57866e41df7ee9373e37f0bb55e528324c7f245972，exit 文件内容为 0。
- 分支结果 commit 8573279936d40102efeabf00286e34b6c99e7681。运行 manifest 保留实际运行的旧代码 commit 8f22be28c7f334062de3e9a34b6c10352a2524f9。Genesis 提示 Quadrants cache lock 警告但批次正常完成；未执行共享 cache 清理。

### 已完成运行与待完成决策

- 固定减速度 no-hand holdout 已完成：30 集、0.5 ms、0.30/0.40/0.50 m/s 各 10 集，使用 calibration 固定 a_eff=0.09803606718731761 m/s²，未在 holdout 重拟合；30/30 到达拦截平面。各档 TTC 误差绝对值 P95 为 1.004/0.669/0.569 ms，横向预测误差绝对值 P95 为 2.150/1.120/0.571 mm。holdout 预测只使用固定模型与每集初始测量状态，未来 crossing 只供 evaluator 计分。代码 commit 8573279936d40102efeabf00286e34b6c99e7681、配置 SHA256 dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd、seed 2026103600；CSV 3093e28c4fab340904f9592540114d6418cd0ad24dd9ddabc08ae76b00edfc39，summary 0eb3bec1c54e5eee975a9e6fd1251bad8b60c5a760b72b39bef668248ed7a013，manifest 2e0c22aaf0c792cfe23f6073f5215b183a7a030c5662a0e3dfd311d91bca839c，SHA256SUMS dc637f2aad7acc0af9bf0b3e28b6ebcc4fc3cc2e7930ca362458582585eea004；服务器和本地逐项校验通过。完整进程 wall 88.06 s、CPU user 60.52 s、max RSS 848,140 KB，exit 0，launcher PID 1990576 / Python PID 1990580；log SHA256 94c02ee2188976210a8eac6c9f65dc0b7141550482dfbb2914fd8cceeab2caf3，runtime f808314c50a732e9f77e6250657b40bf43aa92686a2f634c59c02b8f53b0d8ae。
- strict 横向位置偏差 +10 mm sensitivity 首次运行未形成统计结果：commit 8573279936d40102efeabf00286e34b6c99e7681、配置 SHA256 dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd、seed 2026103500。首次物理仿真完成后，历史评分停止条件未触发，结果组装因 legacy_true_bottom_min_at_stop 未初始化而 exit 1；没有 CSV/summary，该次不计 episode 指标。目录 r0_5_sens_biasy10mm_strict15_20261016/ 留存 manifest 与 FAILURE_NOTE.md；log SHA256 9af60700ff7b7cade5b26183ee3856ac6d968d5ba982a68a123ed2c742739bd2，exit=1，Python wall 87.82 s。benchmark 层 fix 将未达到旧停止条件时的诊断记为 null，不改控制和评分；修复 commit fcf567f7485e590b075b378c4ff726079e0a287a。
- 修复后独立 retry 使用新 seed 2026103515、同一配置和严格评分，15 集（每档 5）。0.30/0.40/0.50 m/s strict success 分别 0/5、4/5、0/5，失败均为 unstable_grasp（11/15）。这是证明任务对 10 mm 横向感知偏差敏感的单因子小样本，不用于估计总体成功率或冻结控制参数。运行 commit fcf567f7485e590b075b378c4ff726079e0a287a，完整 Python wall 276.26 s、CPU user 302.23 s、max RSS 1,683,192 KB、exit 0，launcher PID 1995457。逐集 CSV SHA256 97182cd66067bbe6f08b48769958e26f709312c3544befa9262392b02be10dc0；summary da6362fd6613743d681771d13df5ad8e5858b5950d6683be29d5936e75dcfb69；manifest 37acea4a37be9bb5999148e8ed12fb0f5a5875bf3294bcdd61d1804ec39fb794；SHA256SUMS beb3e517e78d82a491ae7d4d4e249d00c55e812281e5994852f5999dfdf27012，服务器和本地逐项通过。成功/失败代表 trace 与 GIF 在 retry 结果目录中；log SHA256 06320ef6d176486654351415f43e794445857097d89366141132100392be2b68，runtime 14facb9cc8038a1e01bab87b498ae89f1efaa4e59d576acec3db0f29b40abc68。
- Direct-HDR 首次启动仅在 argparse 失败，仿真未创建、没有占用 GPU：服务器代码 commit fcf567f7485e590b075b378c4ff726079e0a287a，配置 SHA256 dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd；负号开头的 y 列表作为独立参数触发 expected one argument，exit 2，wall 0.17 s，PID 2002116。日志 SHA256 706ec89d59905711a7d7f78105495f4e8fc1e505d50b52e49cf9866dd52fdcd5；runtime 85cdc13684c35f1f7d0a96efbdfe0ea00b8a460d44c5e99a33f9b4cfc41ff553。失败日志保存在 runtime_logs 的 launch_argparse_failed 文件中。
- 已把 y 列表改为 --reference-y-values=...，并同步 README；本地 parser 解析检查和 Python compileall 通过，修复提交 4a23458266508c902df545e66f82a2ea396012d6。
- Direct-HDR/event-controller smoke 已完成：代码 commit `4a23458266508c902df545e66f82a2ea396012d6`、配置 SHA256 `dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd`、seed 2026103400；9 集覆盖三档速度 × 左/中/右位置，统一 0.5 ms 物理步、50 Hz 动态可见性审计，同一物理/HDR 轨迹配对生成 1/2 kHz event streams；2 kHz 因果事件态接入同一固定预测器、横向控制器与 lift gate。strict success 0/9、首次接触 3/9、抬升 0/9，接触只在中心轨迹。事件数相对差均值 16.12%；同像素同极性 0.5 ms 匹配比例均值 90.12%/75.60%；10 ms voxel cosine 均值 0.980、voxel event-count 差均值 11.35%。post-contact controller state 有 7,110 个样本，但位置 P95 误差 1.15–3.15 m、速度 P95 误差 0.62–1.61 m/s；lift gate eligible 350/7,110，状态有陈旧/错误复用，不能声称事件闭环可用或采样率足够。
- 完整 Direct-HDR/event-controller 进程 wall 1,791.35 s、CPU user 54,971.47 s、system 263.84 s、max RSS 1,926,592 KB、exit 0；launcher PID 2004113 / Python PID 2004117。每个 3 s rollout wall 平均 184.11 s（RTF 0.0163）。Genesis physics backend 为 CPU，HDR EGL renderer 绑定 V100S GPU2；未用 GPU-time profiler，不能报告专属 GPU 累计时间。结果归档 [`r0_5_direct_hdr_event_smoke_20261016/`](../../benchmark/robot_closed_loop/results/r0_5_direct_hdr_event_smoke_20261016/)（131 MiB；本地清单逐项校验通过），服务器原始地址 `g1:/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-02b1afc/benchmark/robot_closed_loop/results/r0_5_direct_hdr_event_smoke_20261016/`。逐集 CSV SHA256 `36a7ed3d93c3a93e8052f1128b37baf72d74551396679e2a884a75d29f84da2a`；summary `5ebcfb0bb83828a93c37cb76c2a30dce83f4c1e20e2af8aac8420bab08e8d02c`；visibility `7a6489894ae4bb83cf019cb4c0674590c3cbfc717e6a5966c403a44d887a1fa2`；manifest `ec4258689526a6e74ecaf09366371ca3c0848c8b260e5db92d87604993239913`；SHA256SUMS 文件 `fed1383d62c14d243753f43ccd6ef9ca0c5e8bf526fc0d87960fb86076f654c9`。log `947ddf9311eacc6448cf67e447122fc3de3e93f9d4d98acab82277d958623f6b`；runtime `3984be6812e8a255a4c7964f79181ed9d25e66c14362768355bf96ee1af34d7a`；launcher `fc1447215944cbb2617fe113ab494740db14fbd120c953abb25e1f5f2b8163b5`；exit file 内容 0。pixel response 是共享 `DvsEmulator` ideal threshold 0.2、无噪声、无限带宽；Direct-HDR 绕开 Ours 时间重建和 V2E 插帧，不是 EVK4 标定。
- `EventKinematicsTracker` 仍是因果状态接入 smoke，不是已验证 estimator。GT-shadow 中，1/2 kHz 的 post-contact event state episode-level P95 位置误差均值为 7.83/8.03 cm、速度误差 0.682/0.671 m/s；在同一 tracker/controller 上，event-controller 九集 strict success 0/9、contact 3/9、lift 0/9。这个状态接口只消费带秒时间戳的事件与固定针孔标定；shadow 的延迟 GT 只驱动参考物理轨迹，evaluator 的 GT segmentation/depth 不进入事件输入或控制。
- Direct-HDR 采样质量和事件控制能力已分开记录。全程 0.5 ms 物理/HDR 轨迹配对生成 1/2 kHz event streams；10 ms voxel cosine 均值 0.988、voxel event-count 差 7.89%，但采样率变化没有改善上述下游状态误差，现有证据不足以冻结采样充分性。运行成本为完整 Python wall 1911.25 s、CPU user 44907.19 s、system 141.00 s、max RSS 1,799,356 KB；九条 rollout 平均 195.37 s，事件源 steady HDR render 均值 7.48 ms/sample，DVS mean 11.96–14.09 ms/sample。物理后端为 CPU，EGL renderer 是 V100S GPU2；没有 GPU 累计时间 profiler。规模化 Direct-HDR 成本需要优化或由研究会话接受，不对更大数据量外推。
- 本阶段 GT-shadow reference 已完成，事件状态 adapter 和 1/2 kHz 采样结论未闭环。保持 frozen simulator core；本阶段不启动规模数据生成、正式训练或 Ours/V2E 排名。

### GT-shadow Direct-HDR 完整轨迹、成本与数据 provenance

- 服务器原始结果：`g1:/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/`，约 140 MiB；服务器原目录中 9 个 `direct_hdr_events_1khz_2khz.npz` 保留完整 1/2 kHz 原始 event arrays，共 93,553,247 bytes。原始 `SHA256SUMS` 文件 SHA256 为 `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`，g1 上已对全目录逐项 `sha256sum -c SHA256SUMS` 通过。因 raw arrays 体积较大，本地仓库保存其余 105 个逐集指标、状态/控制 JSONL、key HDR snapshots、成功 GIF 和副本说明；raw arrays 由上面的 g1 地址提供，校验值在原始清单中。仓库副本 [`r0_5_direct_hdr_gt_shadow_20261016/`](../../benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/) 的 `SHA256SUMS.local-artifacts` SHA256 `379d60a1f6059bad628fd1764eca1a7860a0c684b821a939a47beee3c9f8302a`，本地 105 项均通过；它覆盖当前仓库副本，原始 `SHA256SUMS` 保留完整服务器数据清单。逐集 CSV SHA256 `26042f8d1b9d162a64400e94a002778e6ac464e7f85d8e1dd8f610abc7e210f1`；summary `cbff52f36467d92d2b23683af362c5164a76707d51186cf25cdc53dee7d45909`；dynamic visibility `4f281a2eb23651477149a5465e638d5632bddc87bd4efed083603f261e6b251e`；manifest `bb11951ece4c6dc0bf35c713fb6ed3b151c935c11db9613601a358fd228d0b8b`。
- 实际运行代码 commit `2100645b0473c294760789bc4475b644f261d9cf`，配置 SHA256 `dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd`，seed `2026103420`–`2026103428`；0.30/0.40/0.50 m/s 各 3 集，launch y 为 -55/0/+55 mm，Physics/HDR 共用 0.5 ms 时间轴，控制 100 Hz、观测延迟 10 ms，visibility evaluator 50 Hz。每集 strict success 9/9；汇总 9/9 的 Wilson 95% CI 为 70.1–100%，每速度 3/3 的 Wilson 下界 43.9%，样本只支持路径 smoke 与遮挡复核，不替代各档 oracle 100/100。首次接触至抬升分别 67.5–77.5 ms。
- 旧评分仍保留在逐集记录中，`legacy_success_at_legacy_stop` 为 true；新评分版本为 `r0_5_strict_continuous_contact_clearance_slip`。逐集严格合格窗口真实底面净空范围 80.002–80.110 mm、最大相对滑移 1.964–6.235 mm、连续保持各 0.500 s；每集窗口含 1000 个 qualified samples，9 集在合格净空期间累计支撑接触样本为 0。闭合启动相对校准 no-hand free-slide 预测到达时刻为 -76.2 至 -51.0 ms，首次接触为 -61.7 至 -40.5 ms；目标首次接触发生在拦截平面前 11.5–23.6 mm，目标/TCP 相对速度 0.189–0.425 m/s。目标多数被 Panda 截停，实际过线时差不作为参考；此处以“无机械臂 free-slide 预测到达时间”为清晰的时序基准。
- 动态机位审计在 645 个样本中覆盖 launch、接触、抬升和保持。`visible_bbox_xyxy` 是半开区间 `[xmin,ymin,xmax+1,ymax+1]`；短边按 `min(xmax-xmin, ymax-ymin)` 计算，因此全样本短边最小/中位/P95/最大为 30/35/48/54 px。可见比例最小/中位为 98.86%/100%。target-only segmentation 只进入 evaluator，不进入事件源或控制器。证据支持当前正 x 侧相机候选在所测工作区有动态视野。
- 相同物理/HDR 轨迹配对采样比较：0.5 ms 轨迹分别生成 1 kHz（每两步）和 2 kHz（每步）event streams。9 集事件数均值 539,185/572,429；2 kHz 相对 1 kHz 的增加量为 6.17%（分母为 1 kHz 总数），若以 2 kHz 总数为分母则相对差为 5.81%。按同像素/极性、±0.5 ms 匹配，1/2 kHz 平均匹配比例 93.40%/87.81%。旧 sanity check 使用 10 ms、2×64×64 voxel：同 bin cosine 均值 0.988、p10 均值 0.968；事件计数差按每个共同非空 10 ms 窗口的 `|n1-n2|/max(n1,n2)` 计算，9 集 episode-mean 的非加权均值为 7.89%，按 1,286 个共同窗口加权为 8.09%。它不是 R0.6 因果五时间 bin 表示的采样结论。post-contact 区间在 9 集合计有 667 个 event-state 样本/采样率，先逐集计算 P95 后，1/2 kHz 的位置误差均值为 7.83/8.03 cm、速度误差为 0.682/0.671 m/s；两种采样率结果接近，均不足以支持抓取控制。估计状态使用秒时间戳的事件和固定针孔标定；没有目标 GT、GT segmentation/depth 或未来状态。shadow 路径中的延迟当前 GT 只供控制器驱动 sensor-reference 轨迹，事件仅用于测量和 evaluator。
- 同一因果 event-state 接口确实接入固定预测器、控制器、lift gate 和 strict scorer 的 Direct-HDR/event-controller smoke，但结果是 strict success 0/9、接触 3/9、抬升 0/9；因此接入链路通过 smoke，估计精度与任务性能没有通过。此结果和 GT-shadow 的 9/9 必须分开引用。
- 完整端到端 Python 进程 wall 为 1911.25 s，CPU user 44907.19 s、system 141.00 s、最大 RSS 1,799,356 KB；episode loop 与产物写入为 1831.56 s，平均单 rollout wall 195.37 s，平均模拟时长 1.426 s，实测 rollout RTF 均值 0.00739。steady HDR render mean 为 7.48 ms/sample，DVS mean 为 14.09 ms/sample（1 kHz）和 11.96 ms/sample（2 kHz）。物理后端为 CPU、HDR EGL 使用 Tesla V100S GPU2；没有 GPU time profiler，因此不报告专属 GPU 累计时间。该实测成本用于本次九集完整轨迹，不外推规模数据。
- provenance：PID launcher `2039840`、Python `2039844`、exit `0`；run log SHA256 `6914249cf4f7be183fdce7090dcb65bc05d3057dfa1151ee3c40c96c0269e620`，runtime `f981402e63ad754a97f155e85cdf7053d4883bb96bd5c1d97f0084a2dfeaa97c`，launcher `5cc7e643d5456b5d81f210149cc296bbe72a84490e26df84b4614ea361f56653`，exit file SHA256 `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。运行日志和 runtime/PID/exit 文件在 `benchmark/robot_closed_loop/results/runtime_logs/`；run manifest 保存精确命令、代码 commit、配置和输出地址。

### R0.6 确定性缺陷修复与 g1 原始事件离线诊断

- 修复代码 commit `b3dc69d6ec447e0272573b395c82ac56b5a02bff`；g1 回放在分支 HEAD `30177915e0e260d6161feb48dddf087809268f73` 执行。配置 `scene_pilot_r0_5_candidate.json` SHA256 `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`。源 raw-event 清单 SHA256 `86cf6c0801365c0c1e5949e4e04d3178fa03f334997db6d09547f69a2b5af37c`，g1 复跑逐项校验成功。
- 回归缺陷：原来时间向量用 `times-times[-1]`，位置却减去位置均值，导致六个 5 ms bin 的理想 0.4 m/s 运动被估成 0.1272727 m/s；统一中心化后为 0.4000000 m/s。状态位置时间戳改为最后有效 bin 的真实事件时间中位数；因果资格以回归生成时刻不晚于观测 cutoff 判定，避免把生成时已包含 cutoff 之后事件的状态拿来控制。
- 状态策略：`fresh`（年龄 ≤30 ms 且有支持轨迹）才提供状态；`stale`、`no_event`、`tracking_lost`、`unavailable` 均不输出可用 MotionState。缺状态时保持横向命令、不启动闭合；已开始闭合则继续闭合；状态无效时 lift gate 不得触发。30 ms 是 fail-closed 工程阈值，尚未冻结。合成接口探针覆盖 no-event、tracking-lost、fresh 和静默后 stale。
- 投影自检：以候选配置重建 Genesis look-at 相机基向量和 W/2,H/2 主点，25 点 workspace round-trip 最大误差为 `2.2e-13 mm`；旧半像素主点假设造成中位 `1.53 mm`、最大 `2.39 mm` 平面偏差。此前运行没有序列化 camera transform，因此该结果仅为配置自洽核验，不能称为 renderer 外参实测或 EVK4 标定。
- 离线回放三个保存轨迹：episode 0（0.30 m/s，y=-55 mm）、episode 4（0.40 m/s，y=0）、episode 8（0.50 m/s，y=+55 mm），每档一个。输入为时间戳 raw events 与固定配置针孔参数；GT 只在估计形成后供离线对齐/误差评估。源轨迹由 GT-shadow 控制产生，严格成功 3/3 不属于事件控制成绩。逐集结果和 poll trace 在 [`r0_6_event_state_offline_20261016/`](../../benchmark/robot_closed_loop/results/r0_6_event_state_offline_20261016/)；完整输入 raw arrays 仍位于 g1 `~/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/`，哈希见源 `SHA256SUMS`。
- 分阶段误差以每集 P95 再对三集取均值，并附三集范围。1/2 kHz 接触前位置误差 P95 为 `18.4 cm (13.6–23.2)` / `20.0 cm (17.5–22.7)`，速度为 `3.36 m/s (2.75–4.51)` / `3.57 m/s (2.63–4.23)`。接触至 lift start 位置误差 P95 为 `7.93 cm (5.11–9.86)` / `8.46 cm (5.02–11.43)`，速度为 `1.77 m/s (1.57–1.91)` / `2.18 m/s (2.03–2.33)`。post-lift 位置/速度 P95 均值仍约 `21.7 cm / 4.70 m/s`（1 kHz）和 `21.4 cm / 4.48 m/s`（2 kHz）；固定滑面高度假设在升起后也不成立。
- 到达误差参考采用独立 no-hand 标定的自由滑动到达时间，仅为 evaluator 对齐。接触前 194 个 poll/rate 中可预测到达比例为 1/2 kHz `51/194=26.3%` / `65/194=33.5%`；仅在可达预测子集，绝对到达时间误差 MAE 为 `0.296/0.288 s`、P95 为 `0.799/0.657 s`。它们离闭环时序需要的精度很远。
- 新鲜度与状态可用性：接触前分别 194/194 polls fresh；contact-to-lift 分别 21/21 fresh。控制器观测 age 的 episode-P95 均值：接触前 `12.90/12.84 ms`，contact-to-lift `13.10/13.19 ms`；post-lift fresh 为 `194/202=96.0%` 和 `195/202=96.5%`，其余主要为 stale/tracking-lost。接触前所有 polls 都 fresh 但误差仍很大，因此 age/有效标记不能代替跟踪准确性；宽 ROI 里没有 no-event poll，也没有证明 tracker 定位到的是方块。
- evaluator-only 事件空间归因将射线投到方块中心高度平面后，落在 GT 方块中心 80 mm 内的事件仅约 `13.8–17.6%`，且假设保留目标面视差。这支持当前宽 ROI centroid 受机械臂/背景事件和固定高度回投影影响；不把 GT 投影用于 estimator/controller。简单 centroid 仅为接线 smoke，未验证为感知器。
- 采样率 sanity check 已改为拟用的因果输入：控制器每 10 ms 更新，观测 cutoff 落后 10 ms；输入为 5×5 ms 时间 bin、25 ms 历史、2 polarity、64×64。三个轨迹共 417 个窗口，1/2 kHz 窗口均非空；以每窗 `|n1-n2|/max(n1,n2)` 计算，窗口加权事件计数差均值 `8.43%`，加权 voxel cosine `0.98597`，各集 count-diff P95 为 `21.00%/26.96%/14.55%`。这比旧 10 ms 单 bin cosine sanity check 更贴近控制输入，但与状态误差一起看仍不能说明任一采样率可控。
- g1 replay 命令：

~~~bash
cd /home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_6-b3dc69d
PYTHONPATH=. /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python \
  benchmark/robot_closed_loop/diagnostics/replay_event_state.py \
  --source-results /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016 \
  --out benchmark/robot_closed_loop/results/r0_6_event_state_offline_20261016 \
  --episodes 0,4,8
~~~

- 输出哈希（已从 g1 拉回并复核）：`per_episode.json` `0cbd5967da0ced9a8f83568fa5c5f3bf0f948650415ea5a16f3186d366152e75`；`causal_observation_polls.csv` `8b440b056c974d4d7a9a059ad683d37068df04aafd0e44cc4375bb2e73dac0bc`；`five_bin_sampling_comparison.json` `cf1d59f19c9b778e254373bcee06d7e689f19c5dd7fb14aa83275ec4e83330f0`；`run_manifest.json` `b7f3894498d9c5812655fc0c88da8bd0101f07b9cafd45e6f673543fb47f05ef`；结果 `SHA256SUMS` `c05742db048d4109397fdadac401c4ea0c4231d0f47e570eca609cc089260e3d`。

### R0.6 小规模真实事件控制 rollout

在 g1 上的 branch snapshot worktree `/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_6-b3dc69d`、detached HEAD commit `30177915e0e260d6161feb48dddf087809268f73` 下，使用同一因果 centroid 状态接口、固定预测器/控制器、strict scorer 做 0.30/0.40/0.50 m/s 各一集，seed `2026103700`–`2026103702`，事件状态 2 kHz、物理步 0.5 ms、GPU3 Direct-HDR。配置 SHA256 `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`；结果目录 [`r0_6_event_controller_3sample_20261016/`](../../benchmark/robot_closed_loop/results/r0_6_event_controller_3sample_20261016/)，g1 原始路径为 `~/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_6-b3dc69d/benchmark/robot_closed_loop/results/r0_6_event_controller_3sample_20261016/`。完整 Python 进程 exit 0，wall/user CPU/system CPU/max RSS 为 `662.89/18618.97/79.01 s / 1,809,900 KB`；episode rollout wall `196.44/182.08/193.98 s`（每集模拟 3 s），平均 `190.83 s`。物理 CPU、Direct-HDR EGL GPU3；steady HDR render mean `7.47–7.51 ms/sample`，DVS mean `5.64–6.52 ms/sample`（1 kHz）、`5.14–5.99 ms/sample`（2 kHz）；无 GPU-time profiler。总进程 wall 才是本次三集端到端成本，不将 render/DVS 均值线性外推到大数据。

| Episode | 速度 / 发射 y | 首次接触 | lift start | Strict success | 动态最小可见比例 | 全轨迹 2 kHz 状态位置 P95 |
|---|---|---:|---:|---:|---:|---:|
| 0 | 0.30 m/s / -55 mm | 无 | 无 | 否 | 0.0 | 161.3 mm |
| 1 | 0.40 m/s / 0 mm | 0.6285 s | 无 | 否 | 1.0 | 216.6 mm |
| 2 | 0.50 m/s / +55 mm | 无 | 无 | 否 | 0.0 | 500.5 mm |

结果为 0/3 strict success、1/3 接触、0/3 lift；不估总体成功率。唯一接触样本（中心轨迹）接触后 2 kHz 状态可用 274/2,374 帧（11.54%），可用状态位置 P95 `275.5 mm`、速度 P95 `7.29 m/s`，状态 age P95 `9.5 ms`，lift gate eligible `0/2,374`。两条边缘轨迹没有接触且动态最小可见比例为 0；录像显示它们后段方块不在可见区域。需要保留因果限制：这是感知误差/视野受限的证据，但三集不足以分离 tracker 偏差与机器人事件/遮挡的贡献。逐集事件、状态控制 trace 和失败录像都已纳入仓库；本地与 g1 `sha256sum -c SHA256SUMS` 通过。结果清单 SHA256 `90c4823bf1dfb6dec0b42ec7ee97d5241b718d6fed55c06a3f9746ba68290a1f`，summary `40981bbf3541859550c62bef70a884a6227b35371a56998f13ffbfc2ba4255ba`，逐集 CSV `fe715a0773d9fed75bd08b6dc121b42bb400edf5f2a335c9b9c159fc0fe0f5ff`，manifest `df3fe104e044c39eab59898ce9b3c87b314167eb10d68d936b44b52cc5328c8a`。

运行日志 provenance：tmux session `r0_6_evt_20261016`，launcher PID `2445286` / Python PID `2445290`，exit 0；log SHA256 `12f9585d909d27184ef87083204ba1c2179ab01dd9033849bb42de1f56455b8c`，runtime `109228a1fd87c4bcee0e10221e05ae59893c3c71d3acc141f92b3f4b3de954b8`，launcher `49973e333315859564f7f747918147ef2ef2b2a063e4f5ba1b1af88ff66b0c73`，exit 文件 SHA256 `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。

命令：

~~~bash
benchmark/robot_closed_loop/results/runtime_logs/r0_6_event_controller_3sample_20261016.launch.sh
~~~

该小样本用于检查真实无 GT 接入、接触和 lift，不估计总体成功率。像素响应共用插件 `DvsEmulator` 的 ideal threshold 0.2、无噪声、无限带宽；绕开 Ours 时间重建和 V2E 插帧，不是 EVK4 标定。可视化 segmentation 只用于 evaluator。

### Tiny event-to-state overfit 诊断（小规模、尚未启动）

质心诊断显示接触前位置 P95 为 18–50 cm、速度 P95 为 3.2–18.1 m/s，中心轨迹接触后 lift gate 无一次 eligible。下一步已增加简单 2D CNN 诊断脚本：输入固定五时间 bin 的 `10×64×64` log-count tensor，输出 observation cutoff 时刻的 world xyz 位置和速度；只用同三条已存 GT-shadow Direct-HDR raw-event 轨迹训练并做 in-sample overfit 检查。GT pose/velocity 只生成训练标签和离线评分，不进入模型输入或推理控制；没有 episode holdout，因此不报告泛化性能。此处不会重渲染或改 lift trigger。

脚本 [`train_event_state_overfit.py`](../../benchmark/robot_closed_loop/diagnostics/train_event_state_overfit.py) 与 launcher [`r0_6_event_state_overfit_20261017.launch.sh`](../../benchmark/robot_closed_loop/results/runtime_logs/r0_6_event_state_overfit_20261017.launch.sh) 已在分支。实际运行 code commit `1e2ddc98907138b21225306cdea3c4b5882111a2`，配置 SHA256 `aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`，raw 输入复用上节 episode 0/4/8 和源清单 SHA。使用服务器 venv CPU PyTorch 2.8.0+cpu、4 CPU threads、120 epochs；tmux session `r0_6_overfit_20261017`、launcher PID `2460434`，当前运行中。输出目录 `results/r0_6_event_state_overfit_20261017/`；训练摘要、model-only latency 和数据哈希待完成后补。

### 建议的下一步配对训练 pilot（协议候选，未执行/未冻结）

1. 保持场景、机器人、0.5 ms physics、固定相机、五-bin 输入、状态接口、predictor/controller/lift gate 和 strict scorer 不变。先复用已完成的 tiny overfit 轨迹确认标签、事件窗口和状态时间戳接线；过拟合只能说明 pipeline/capacity 可行，不作为性能证据。
2. 若诊断可拟合，再生成 18 个固定场景 episode ID：`3 speeds × 3 lateral positions (-55/0/+55 mm) × 2 seeds`。每个 ID 由完全相同的初始条件分别驱动 Ours 和 V2E 输入路径；按 episode 划分 9 train / 9 heldout（每个 speed-position cell 一集一边），不能按相邻时间窗随机拆分。Direct-HDR 只做 sensor reference/标签时序校验，不馈入学习器或替代 Ours reconstruction/V2E interpolation。
3. 训练相同的小型 temporal/event-to-state 网络、相同预算和标签，只让 Ours 与 V2E 各自看到自己的事件输入。训练监督使用 simulator object pose/velocity；验证/闭环推理只允许事件 tensor、时间戳和固定相机标定。模型输出必须携带 causal timestamp、置信度/有效性，并实现 `fresh/stale/no_event/tracking_lost/unavailable`；GT segmentation/depth 只可离线评分，不能进入 sensor、tracker 或 controller。
4. 先看 heldout 每阶段 position/velocity/arrival error、状态可用率与年龄，达标后各做少量 live rollout；不从 18 集 pilot 排名。可讨论的工程入口门槛是：接触前和 contact-to-lift 状态有效率 ≥95%、age ≤30 ms、平面位置 P95 ≤5 mm、arrival P95 ≤10 ms；这些是根据 +10 mm sensitivity 和 100 Hz 控制周期提出的讨论值，不是已冻结阈值。最终成功必须由同一 strict scorer 的真实接触、无支撑抬升和 0.5 s 连续保持定义。
5. 参考采用共享 `DvsEmulator` ideal threshold 0.2/no-noise/infinite-bandwidth 时，Ours/V2E/Direct-HDR 三者均明确记录该共享像素模型；两种 estimator 的输入差异只来自 Ours 时间重建与 V2E 插帧。端到端成本分别记 Direct-HDR render、Ours reconstruction、V2E interpolation、event generation/binning、network/controller、physics 与 I/O，不用早期单次 render 数值外推。

### 复跑命令与运行约定

在服务器 branch worktree 中执行，运行前记录 commit/config hash、核查项目路径和资源。长任务用 tmux，stdout/stderr、PID、exit 与 /usr/bin/time runtime 均保存在 benchmark runtime_logs。下列命令目录需使用未存在的新输出路径。

~~~bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode oracle --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json --episodes 100 --speeds 0.3,0.4,0.5 --a-eff 0.09803606718731761 --seed 2026103300 --out benchmark/robot_closed_loop/results/r0_5_oracle_strict100x3_20261016
~~~

Direct-HDR/reference 复跑入口（服务器 tmux launcher 会记录 PID、日志、exit 和完整 Python wall/CPU time；启动前核验 GPU2）：

~~~bash
EGL_DEVICE_ID=2 CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python -m benchmark.robot_closed_loop.run --mode reference --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json --speeds 0.3,0.4,0.5 --reference-episodes 3 --reference-y-values=-0.055,0,0.055,-0.055,0,0.055,-0.055,0,0.055 --event-controller-hz 2000 --visibility-hz 50 --seed 2026103400 --out benchmark/robot_closed_loop/results/r0_5_direct_hdr_event_smoke_20261016
~~~


GT shadow sensor-reference 复跑命令（事件流只用于采样/状态评估，delayed current-GT 只驱动完整参考轨迹）：

~~~bash
EGL_DEVICE_ID=2 CUDA_VISIBLE_DEVICES=2 PYOPENGL_PLATFORM=egl PYTHONPATH=. \
  /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python \
  -m benchmark.robot_closed_loop.run --mode reference \
  --config benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json \
  --speeds 0.3,0.4,0.5 --reference-episodes 3 \
  --reference-y-values=-0.055,0,0.055,-0.055,0,0.055,-0.055,0,0.055 \
  --event-controller-hz 2000 --reference-controller-source gt_current \
  --visibility-hz 50 --seed 2026103420 \
  --out benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016
~~~

Frozen core 仍与基线 e79e656ac00f7847b98db9e12bf43b093f3afff1 一致；主分支未合并。
