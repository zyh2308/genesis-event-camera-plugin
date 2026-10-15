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
- 服务器 `g1` 的隔离 worktree 最近同步到 `d3bc9e3732b8e898c2412d34f7ea00bd07f4acd9`；原 `main` 工作区未切换、未修改。R0 project venv 使用 Genesis 1.2.2 + Torch 2.8.0+cpu，Torch wheel SHA256 `cb06175284673a581dd91fb1965662ae4ecaba6e5c357aa0ea7bb8b84b6b7eeb`，文件位于服务器 `~/Eventbased_WAM/torch-2.8.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl`。基础共享环境仍是 Torch 2.5.1+cu121。
- 资源检查时 GPU 0 有其他用户计算进程、GPU 3 有 CARLA；GPU 1、2、4、5 当时空闲。后续 EGL 运行显式限制在经再次检查的空闲设备，不触碰其他用户进程。

### 服务器实测结果

**服务器静态抓取复测（开发级）：** 在代码提交 `418718af6c0052c6a368cedd9d5928a1bccec458`、配置 SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a` 下运行 20 集，20/20 完整成功：均有双侧接触、净空 ≥0.1015 m、连续保持 0.5 s；首次接触 0.062–0.066 s，中位数 0.065 s；抬升触发 0.131–0.136 s。静态 20/20 的 Wilson 95% 区间为 83.9%–100%，不能据此证明 95% 下界达标，且不代表动态拦截成功率。代表性抓取 GIF 与完整关键状态 trace 已归档。

**服务器单集 smoke（早期开发）：** 代码提交 `af0548c3ab6ccc663f25fea47e8dd59989b5d1a3`、旧配置 SHA256 `7a7f73803856f1bcf11f67d10c97d72cf5e46b84888bbd2f6c60247c83480ba6` 下，静态 1/1 双侧接触并抬升：首次接触 0.064 s、抬升触发 0.134 s、底面净空 0.1016 m、保持 0.5 s，结果 `server_static_smoke_torch28/r0_static.csv`。仅作为早期 runtime/控制 smoke，不计入新配置 20 集统计。

**服务器 no-hand free-slide 校准：** 代码提交 `865b79e777ff054856ea49cbf36bf0aa3f1ccf1a`、配置 SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a` 下运行 30 集（0.30/0.40/0.50 m/s 各 10 集），Panda 不在场，30/30 到达 x=0.55 m 拦截平面。由三档轨迹共同拟合的固定减速度 `a_eff=0.09804843 m/s²`；逐 episode 拟合中位数 `0.09804806 m/s²`、IQR `[0.09799079, 0.09808144] m/s²`。各档中位到达时间为 0.9704/0.6908/0.5402 s，到达速度为 0.2048/0.3323/0.4470 m/s。中位横向漂移（相对起始 y）为 -1.0/4.9/1.4 mm，平均绝对漂移为 7.0/7.5/9.0 mm。拟合仅来自 calibration seeds；holdout 尚待运行。轨迹中方块 z 保持约 0.01998–0.020004 m；离散 support-contact 查询为间歇 true，需结合稳定高度解释，不把该布尔采样当作连续接触测量。

**服务器 no-hand holdout：** 使用新种子（每档 10 集）与校准阶段固定 `a_eff=0.09804843035177806 m/s²`，代码提交 `93e739eec055b440516c02335a295fe43167d948`、配置 SHA256 不变。Panda 仍不在场，30/30 到达拦截平面。各速度档中位到达时间 0.9798/0.6873/0.5346 s，中位逐 episode 减速度 0.098027/0.098064/0.098043 m/s²；校准预测初始 TTC 误差中位数 -1.45/-1.14/-1.08 ms，横向位置误差中位数 -0.83/-0.27/0.43 mm，全部绝对横向误差最大 2.36 mm。该结果支持单一固定减速度作为本候选场景的运动预测器；它不是机器人抓取成功率证据。

修正后的 camera preflight 在代码 `9c83cbd040388e50e49489e6d83b1d8431b863cd`、配置 SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a` 下通过。320×240、垂直 FOV 30° 的真实 Panda 抓取预备姿态画面中，目标 bbox 为 `[50,97,93,144]`、短边 43 px、可见 859 px；HDR 为有限 float32 `[240,320,3]`，范围 `[0.00441,1.12109]`。相机候选 `[0.65,-0.45,0.25]` 下目标能见且尺度可用。报告中的 0% occlusion 是与全零 Panda qpos 的对照；Genesis 日志提示该 qpos 越界，因此这个比例不能作为定量遮挡结论。应以后续用合法停放姿态或独立无机器人参考重新测量；当前只确认实际任务姿态下目标未被完全遮挡。旧高机位/旧 HDR preflight 仍保留为无效 bug 证据。Torch 2.5.1 下的一次失败接触 smoke 仍作为兼容性证据保留，不计任务样本。

**服务器 GT-current oracle 开发批次：** 代码提交 `4d70b8f0f796ca5163d47ab3b5aac98f71aa45a4`、配置 SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a`，校准固定 `a_eff=0.09804843035177806 m/s²`、GT 状态延迟 10 ms，每档 20 集。0.30 m/s 为 20/20 成功（Wilson 95% CI 83.9%–100%）；0.40 与 0.50 m/s 均为 0/20（各 CI 0%–16.1%）。目前配置未达到各主速度档可行性目标。

失败轨迹细查发现：0.40 m/s 的 20 集全部出现机器人接触，其中 19 集有短暂双指接触；0.50 m/s 有 19 集出现接触、18 集有短暂双指接触；高档均没有成功抬升。代表失败 trace 显示目标接触后速度反向并从手指逃逸。摘要将所有失败都分成 `oracle_unreachable`，是因为控制器在物体离开拦截区后最后一次预测不可达，掩盖了之前发生的接触；失败分类优先级需修正。0.40 m/s 批次还有手指停滞触发空夹爪抬升的 episode，说明仅凭关节停滞不能区分物体夹持与夹爪自身硬停；后续控制器会加入同一估计状态下的几何接近/速度门限，仍不使用 evaluator 的 GT 接触标签。

**闭合提前 20 ms sensitivity（开发批次）：** 使用代码 `eff4f592fe4e503b8f4264415c2b3b0b0b326165`、配置 SHA256 `c85cb2b117d90704f2d824ccf7f0e102019b7d8e5d814a2edcdf3a6004932edf`、每档 5 集，`--ttc-bias-ms -20`（相对 100 ms lead，触发提前到预测 crossing 前约 120 ms）。结果 0.30 档 5/5；0.40、0.50 档均 0/5。0.40 档 3/5、0.50 档 5/5 有双侧接触，但均未发生合格抬升；高档失败为 8 个 `unstable_grasp`、2 个无接触 `oracle_unreachable`。代表 trace 显示在高速度下手指到目标前已接近/到达闭合硬限位，方块停在夹爪前缘，没进入有效夹持开度；此前闭合 latch 与状态门控阻止了错误开爪和空夹爪触发。该单因子试验不支持更早闭合；较晚闭合 sensitivity 找到三档成功点。

**统一 speed-conditioned GT-current oracle 开发复验：** 代码提交 `d3bc9e3732b8e898c2412d34f7ea00bd07f4acd9`、配置 SHA256 `9b313bfc634e4eb39fa7591593bc808e6543f4a8a78f9dd15a332fb6be106439`、固定减速度 `0.09804843035177806 m/s²`、延迟 10 ms、每档 20 集、新 seed。首次延迟补偿当前速度估计中位数为 0.2951/0.3951/0.4951 m/s；实际 closure lead 中位数为 80.49/70.49/60.49 ms，与三个单因子成功点吻合。三档均 20/20 完整成功（Wilson 95% CI 各为 83.9%–100%）；每档 20/20 有双侧接触并启动抬升，保持均 0.5 s；夹持开度中位数约 39.8 mm，目标底面最终净空范围 0.10045–0.10263 m。该开发批次未证明真实成功率下界超过 95%，仍需独立样本估计不确定性。

## 分支、冻结基线与候选配置

- GitHub：`zyh2308/genesis-event-camera-plugin`
- 分支：`exp/robot-closed-loop-v1`
- 早期静态 smoke 使用提交 `af0548c3ab6ccc663f25fea47e8dd59989b5d1a3`；相机/HDR preflight 使用 `9c83cbd040388e50e49489e6d83b1d8431b863cd`；静态 20 集复测使用 `418718af6c0052c6a368cedd9d5928a1bccec458`；no-hand slide calibration 使用 `865b79e777ff054856ea49cbf36bf0aa3f1ccf1a`；slide holdout 使用 `93e739eec055b440516c02335a295fe43167d948`；首批 oracle 使用 `4d70b8f0f796ca5163d47ab3b5aac98f71aa45a4`；GT oracle 控制门控修正使用 `eff4f592fe4e503b8f4264415c2b3b0b0b326165`；较晚闭合敏感性运行使用 `ae34e8c94fb6c0b8646cc0b901e42bd218617151`，结果归档在 `47f5427cb1aedb8cd783f8d890351dd507cd9f62`。均在指定分支，run manifest 保存每次代码与配置来源。
- 服务器隔离 worktree：`~/Eventbased_WAM/repositories/gec-robot`
- 服务器原始仓库工作区 `~/Eventbased_WAM/repositories/genesis-event-camera-plugin` 保持在 `main`，HEAD `15479d97154274796043d5e5cc42a336f57b2dd9`。
- Frozen simulator core：`e79e656ac00f7847b98db9e12bf43b093f3afff1`；任务新增代码不在 `genesis_event_plugin/`。
- 场景配置：[`scene_pilot_r0_v1.json`](../../benchmark/robot_closed_loop/configs/scene_pilot_r0_v1.json)，当前配置 SHA256 `9b313bfc634e4eb39fa7591593bc808e6543f4a8a78f9dd15a332fb6be106439`（状态估计门控、校准来源与 speed-conditioned closure lead）；较晚闭合 sensitivity 使用 `c85cb2b117d90704f2d824ccf7f0e102019b7d8e5d814a2edcdf3a6004932edf`，首批 oracle 使用 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a`。当前 schedule 已通过统一控制器每档 20 集开发复验，仍未冻结。
- 当前候选：40 mm 立方体、80 g、低摩擦 Plane、1 ms 物理步、0.30/0.40/0.50 m/s 发射、固定外置 320×240 相机（机位 `[0.65,-0.45,0.25]`；实际姿态 preflight 已通过，遮挡比例仍待合法参考姿态复核）、100 Hz 控制、10 ms 观测延迟。闭合 lead 使用首次因果速度估计 `v0` 下的 `clip(0.08 - 0.10·(v0-0.30), 0.05, 0.10)` 秒；开发复验三档各 20/20，独立 holdout 三档各 100/100，实际 lead 中位数约 80.49/70.49/60.49 ms。独立 holdout 的逐档 Wilson 95% CI 下界为 96.3%。目标短边 43 px（旧机位参数，定量遮挡复核正在更新）。共享减速度候选为 0.09804843 m/s²，no-hand holdout 已支持该模型。误差注入已完成。

## 运行时与运行命令

在服务器只使用上述隔离 worktree；先检查 GPU 进程归属，再将 `CUDA_VISIBLE_DEVICES` 设为一个当时空闲的物理卡。仓库 setup 将已有 Genesis 1.2.2 包复制到 project venv，在私有 venv 安装 Torch 2.8 CPU 并应用 HDR overlay，不修改共享 conda 环境。静态抓取 20/20、相机/HDR preflight、no-hand 标定/holdout 已通过；speed-conditioned GT-current oracle 开发集三档均 20/20。Direct-HDR reference 成本尚未测量。

```bash
source ~/miniconda3/bin/activate Eventbased_WAM
cd ~/Eventbased_WAM/repositories/gec-robot
ROBOT_TORCH_WHEEL="$HOME/Eventbased_WAM/torch-2.8.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl" bash scripts/setup_robot_runtime.sh
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python \
  -m benchmark.robot_closed_loop.run --mode static --episodes 1 --seed 2026101600 \
  --out benchmark/robot_closed_loop/results/server_static_smoke
```

校正后的 camera preflight 复跑命令：

```bash
CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONPATH=. .venv-robot/bin/python \
  -m benchmark.robot_closed_loop.run --mode preflight --episodes 1 \
  --out benchmark/robot_closed_loop/results/camera_preflight_low_20261016
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
5. ⚠️ `--ttc-bias-ms -20`：每档 5 集，0.30 档 5/5，0.40/0.50 档 0/5；更早时序使手指在目标到达前过度闭合。继续试较晚闭合，不运行正式独立成功率批次。
6. ✅ 较晚时序 sensitivity：+20 ms 下 0.30/0.40 各 5/5；+30 ms 下 0.40 为 5/5；+40 ms 下 0.50 为 5/5。
7. ✅ 统一 speed-conditioned lead 开发复验每档 20/20；横向偏差与额外延迟注入已完成，下一步运行新种子每档 100 集独立成功率样本。
8. `--mode preflight` 与 `--mode reference --reference-duration 0.20`：固定相机可见性、目标像素尺度、遮挡，以及直接 HDR 1/2 kHz reference 成本。

## 证据入口与 provenance

- 代码、配置和复跑命令：[`benchmark/robot_closed_loop/README.md`](../../benchmark/robot_closed_loop/README.md)
- 每次运行必须记录代码 commit、分支、Genesis/Torch/资产/overlay、配置 SHA256；产物 SHA256 清单在运行目录 `SHA256SUMS`。
- 结果目标位置：服务器 `~/Eventbased_WAM/repositories/gec-robot/benchmark/robot_closed_loop/results/<run-id>/`；轻量原始 CSV、JSONL、摘要、短录像纳入分支。大型产物另存可访问位置并提供 URL 与校验清单。
- 服务器静态成功原始数据 SHA256：CSV `cca2129f62c502bc2367128284065f6e006b21b8466c445471bf33b247138f66`；轨迹 `fa8f8c0d540f6106490a583979e054015213d78bb532a25e082a9a8479efb68b`；GIF `f363f4de5cd15c3a4d25e2c6d3062f2e7d16e97d60d19171f35a29c4e1c2e76d`；`SHA256SUMS` `c586944bf6ef0cea126dfeafffa2c6ded327d33b317b34a8c661df8b2bf78179`。静态结果在服务器 `server_static_smoke_torch28/`，实际运行代码 commit `af0548c...`、配置 hash `7a7f738...`。
- 通过的 camera preflight 位于 [`camera_preflight_low_20261016/`](../../benchmark/robot_closed_loop/results/camera_preflight_low_20261016/)；运行代码 commit `9c83cbd...`、配置 hash `39c7aff...`。JSON SHA256 `41733c813f37ccab58e211db7f646b29d8a98c292285b6220748ff1d5262018c`；RGB `c3fb74d89182d28b7a92579604ead018550ecc93e6f14721e075b4fff6050ef6`；HDR/depth/seg NPZ `fea84331929a5bfb29831d504c1893224d019505ae0bfb1a6c925b573c372179`；manifest `426e67b15058095f0906dfaf3be44ad2d91869c6d4c62535aeaf5563ecf970e1`；`SHA256SUMS` 文件本身 `fedc2831f0d34502b58b55f0a0ad7a129d94eeb9652aba957a0bd540fe5c8b63`。运行日志 SHA256 `93b36b3b9c2e1bb33c1534a9f4a83c1c93c80ff646f3a70398e71b48ae1b264c`。
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
- Camera render 返回值处理存在 benchmark bug，已修正并在服务器复测；旧高机位 preflight 记录到 0 可见像素，仅保留作 bug 证据。低机位图像与 HDR 输出有效，但保存的 RGB 可见夹爪对目标有遮挡；旧 0% 遮挡来自非法 Panda qpos，不能采用。已把 preflight 改为同相机 target-only 参考，并新增发射到拦截区的 9 个静态视野采样；新实现待服务器实测，完整动态机械臂遮挡仍未测量。
- no-hand calibration 与独立 holdout 均已完成；单因子 timing 扫描成功时序点已记录。
- 新 seed oracle 独立 holdout 三档均 100/100，Wilson 95% CI 下界 96.3%；仅支持当前物理场景、固定预测器与固定控制器的 oracle 可行性，不代表视觉估计器成功率。10 mm 横向位置偏差敏感性显著；总观测延迟增至 30 ms 的小样本仍 15/15。
- 已测单个抓取姿态的相机视场与目标像素尺度；完整运动轨迹视野、基于合法参考姿态的定量遮挡，以及 1/2 kHz Direct-HDR reference 路径成本仍未测量。
- 新种子 oracle 独立 holdout 已完成：服务器 tmux `r0_oracle_holdout100x3_20261016`，PID `1594166`、exit `0`；实际代码 commit `d3bc9e3732b8e898c2412d34f7ea00bd07f4acd9`、配置 SHA256 `9b313bfc634e4eb39fa7591593bc808e6543f4a8a78f9dd15a332fb6be106439`、seed `2026102800`，每档 100/100，Wilson 95% CI 下界 96.3%。
- 正式 Ours/V2E 比较、大规模数据生成和训练均未开始。

## 冻结建议与研究会话需决策事项

建议暂定冻结候选为当前配置：40 mm、80 g 方块，Plane 低摩擦参数，1 ms 物理步；发射速度 0.30/0.40/0.50 m/s；相机 `[0.65,-0.45,0.25]`、320×240、垂直 FOV 30°；100 Hz 控制、10 ms GT observation latency；固定 `a_eff=0.09804843035177806 m/s²`；闭合 lead=`clip(0.08 - 0.10·(v0-0.30), 0.05, 0.10)` 秒，`v0` 是首次延迟补偿后的因果速度估计。该物理候选在独立新 seed 的 oracle holdout 每档 100/100 成功，Wilson 95% 下界约为 96.3%，达到 >95% 统计门槛。该结论只适用于 GT-current oracle。相机机位正经有效 target-only 参考重测，后续还需完成视野/遮挡确认与 Direct-HDR reference 成本，再建议研究会话冻结任务配置。主分支合并留待验收。


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
