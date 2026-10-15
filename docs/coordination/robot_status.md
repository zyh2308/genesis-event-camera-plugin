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
- 服务器 `g1` 的隔离 worktree 最近同步到 `9c83cbd040388e50e49489e6d83b1d8431b863cd`；原 `main` 工作区未切换、未修改。R0 project venv 使用 Genesis 1.2.2 + Torch 2.8.0+cpu，Torch wheel SHA256 `cb06175284673a581dd91fb1965662ae4ecaba6e5c357aa0ea7bb8b84b6b7eeb`，文件位于服务器 `~/Eventbased_WAM/torch-2.8.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl`。基础共享环境仍是 Torch 2.5.1+cu121。
- 资源检查时 GPU 0 有其他用户计算进程、GPU 3 有 CARLA；GPU 1、2、4、5 当时空闲。后续 EGL 运行显式限制在经再次检查的空闲设备，不触碰其他用户进程。

### 服务器实测结果

**服务器静态抓取复测（开发级）：** 在代码提交 `418718af6c0052c6a368cedd9d5928a1bccec458`、配置 SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a` 下运行 20 集，20/20 完整成功：均有双侧接触、净空 ≥0.1015 m、连续保持 0.5 s；首次接触 0.062–0.066 s，中位数 0.065 s；抬升触发 0.131–0.136 s。静态 20/20 的 Wilson 95% 区间为 83.9%–100%，不能据此证明 95% 下界达标，且不代表动态拦截成功率。代表性抓取 GIF 与完整关键状态 trace 已归档。

**服务器单集 smoke（早期开发）：** 代码提交 `af0548c3ab6ccc663f25fea47e8dd59989b5d1a3`、旧配置 SHA256 `7a7f73803856f1bcf11f67d10c97d72cf5e46b84888bbd2f6c60247c83480ba6` 下，静态 1/1 双侧接触并抬升：首次接触 0.064 s、抬升触发 0.134 s、底面净空 0.1016 m、保持 0.5 s，结果 `server_static_smoke_torch28/r0_static.csv`。仅作为早期 runtime/控制 smoke，不计入新配置 20 集统计。

**服务器 no-hand free-slide 校准：** 代码提交 `865b79e777ff054856ea49cbf36bf0aa3f1ccf1a`、配置 SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a` 下运行 30 集（0.30/0.40/0.50 m/s 各 10 集），Panda 不在场，30/30 到达 x=0.55 m 拦截平面。由三档轨迹共同拟合的固定减速度 `a_eff=0.09804843 m/s²`；逐 episode 拟合中位数 `0.09804806 m/s²`、IQR `[0.09799079, 0.09808144] m/s²`。各档中位到达时间为 0.9704/0.6908/0.5402 s，到达速度为 0.2048/0.3323/0.4470 m/s。中位横向漂移（相对起始 y）为 -1.0/4.9/1.4 mm，平均绝对漂移为 7.0/7.5/9.0 mm。拟合仅来自 calibration seeds；holdout 尚待运行。轨迹中方块 z 保持约 0.01998–0.020004 m；离散 support-contact 查询为间歇 true，需结合稳定高度解释，不把该布尔采样当作连续接触测量。

**服务器 no-hand holdout：** 使用新种子（每档 10 集）与校准阶段固定 `a_eff=0.09804843035177806 m/s²`，代码提交 `93e739eec055b440516c02335a295fe43167d948`、配置 SHA256 不变。Panda 仍不在场，30/30 到达拦截平面。各速度档中位到达时间 0.9798/0.6873/0.5346 s，中位逐 episode 减速度 0.098027/0.098064/0.098043 m/s²；校准预测初始 TTC 误差中位数 -1.45/-1.14/-1.08 ms，横向位置误差中位数 -0.83/-0.27/0.43 mm，全部绝对横向误差最大 2.36 mm。该结果支持单一固定减速度作为本候选场景的运动预测器；它不是机器人抓取成功率证据。

修正后的 camera preflight 在代码 `9c83cbd040388e50e49489e6d83b1d8431b863cd`、配置 SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a` 下通过。320×240、垂直 FOV 30° 的真实 Panda 抓取预备姿态画面中，目标 bbox 为 `[50,97,93,144]`、短边 43 px、可见 859 px；HDR 为有限 float32 `[240,320,3]`，范围 `[0.00441,1.12109]`。相机候选 `[0.65,-0.45,0.25]` 下目标能见且尺度可用。报告中的 0% occlusion 是与全零 Panda qpos 的对照；Genesis 日志提示该 qpos 越界，因此这个比例不能作为定量遮挡结论。应以后续用合法停放姿态或独立无机器人参考重新测量；当前只确认实际任务姿态下目标未被完全遮挡。旧高机位/旧 HDR preflight 仍保留为无效 bug 证据。Torch 2.5.1 下的一次失败接触 smoke 仍作为兼容性证据保留，不计任务样本。

## 分支、冻结基线与候选配置

- GitHub：`zyh2308/genesis-event-camera-plugin`
- 分支：`exp/robot-closed-loop-v1`
- 早期静态 smoke 使用提交 `af0548c3ab6ccc663f25fea47e8dd59989b5d1a3`；相机/HDR preflight 使用 `9c83cbd040388e50e49489e6d83b1d8431b863cd`；静态 20 集复测使用 `418718af6c0052c6a368cedd9d5928a1bccec458`；no-hand slide calibration 使用 `865b79e777ff054856ea49cbf36bf0aa3f1ccf1a`；slide holdout 使用 `93e739eec055b440516c02335a295fe43167d948`。均在指定分支，run manifest 保存每次代码与配置来源。
- 服务器隔离 worktree：`~/Eventbased_WAM/repositories/gec-robot`
- 服务器原始仓库工作区 `~/Eventbased_WAM/repositories/genesis-event-camera-plugin` 保持在 `main`，HEAD `15479d97154274796043d5e5cc42a336f57b2dd9`。
- Frozen simulator core：`e79e656ac00f7847b98db9e12bf43b093f3afff1`；任务新增代码不在 `genesis_event_plugin/`。
- 场景配置：[`scene_pilot_r0_v1.json`](../../benchmark/robot_closed_loop/configs/scene_pilot_r0_v1.json)，当前相机候选配置 SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a`；旧相机配置 hash `7a7f73803856f1bcf11f67d10c97d72cf5e46b84888bbd2f6c60247c83480ba6`。配置仍未冻结。
- 当前候选：40 mm 立方体、80 g、低摩擦 Plane、1 ms 物理步、0.30/0.40/0.50 m/s 发射、固定外置 320×240 相机（机位 `[0.65,-0.45,0.25]`；实际姿态 preflight 已通过，遮挡比例仍待合法参考姿态复核）、100 Hz 控制、10 ms 观测延迟、手爪闭合起始提前量 100 ms。实际目标短边 43 px，高于原候选下限 16 px。共享减速度候选为 0.09804843 m/s²，独立 no-hand holdout 已支持该模型；机器人 oracle 尚未运行。

## 运行时与运行命令

在服务器只使用上述隔离 worktree；先检查 GPU 进程归属，再将 `CUDA_VISIBLE_DEVICES` 设为一个当时空闲的物理卡。仓库 setup 将已有 Genesis 1.2.2 包复制到 project venv，在私有 venv 安装 Torch 2.8 CPU 并应用 HDR overlay，不修改共享 conda 环境。新 runtime 下静态抓取 20/20 与校正后的相机/HDR preflight 均已通过；Direct-HDR reference 成本尚未测量。

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

后续每个输出路径必须是新目录；跑批使用 `tmux`，写 project log 和 PID，并低频查看。建议顺序：

1. ✅ `--mode static --episodes 20`：已验证 20/20 真实双侧接触、抬升和 0.5 s 保持；样本量仍仅为开发级。
2. ✅ `--mode slide --split calibration --episodes 10 --speeds 0.3,0.4,0.5`：target-only scene 完成 30 集 no-hand 轨迹；共享拟合 `a_eff=0.09804843 m/s²`。
3. ✅ `--mode slide --split holdout --episodes 10 --speeds 0.3,0.4,0.5 --a-eff 0.09804843035177806`：新种子 30/30 到达，固定模型 TTC 与横向误差均较小。
4. `--mode oracle --episodes 20 --speeds 0.3,0.4,0.5 --a-eff <calibration 输出>`：GT 当前状态 → 固定预测器 → 固定控制器；不读取未来状态。
5. 分别运行小幅 `--bias-y`、`--ttc-bias-ms`、`--extra-latency-ms` 注入；检查成功率和错误类别变化。
6. `--mode preflight` 与 `--mode reference --reference-duration 0.20`：固定相机可见性、目标像素尺度、遮挡，以及直接 HDR 1/2 kHz reference 成本。

## 证据入口与 provenance

- 代码、配置和复跑命令：[`benchmark/robot_closed_loop/README.md`](../../benchmark/robot_closed_loop/README.md)
- 每次运行必须记录代码 commit、分支、Genesis/Torch/资产/overlay、配置 SHA256；产物 SHA256 清单在运行目录 `SHA256SUMS`。
- 结果目标位置：服务器 `~/Eventbased_WAM/repositories/gec-robot/benchmark/robot_closed_loop/results/<run-id>/`；轻量原始 CSV、JSONL、摘要、短录像纳入分支。大型产物另存可访问位置并提供 URL 与校验清单。
- 服务器静态成功原始数据 SHA256：CSV `cca2129f62c502bc2367128284065f6e006b21b8466c445471bf33b247138f66`；轨迹 `fa8f8c0d540f6106490a583979e054015213d78bb532a25e082a9a8479efb68b`；GIF `f363f4de5cd15c3a4d25e2c6d3062f2e7d16e97d60d19171f35a29c4e1c2e76d`；`SHA256SUMS` `c586944bf6ef0cea126dfeafffa2c6ded327d33b317b34a8c661df8b2bf78179`。静态结果在服务器 `server_static_smoke_torch28/`，实际运行代码 commit `af0548c...`、配置 hash `7a7f738...`。
- 通过的 camera preflight 位于 [`camera_preflight_low_20261016/`](../../benchmark/robot_closed_loop/results/camera_preflight_low_20261016/)；运行代码 commit `9c83cbd...`、配置 hash `39c7aff...`。JSON SHA256 `41733c813f37ccab58e211db7f646b29d8a98c292285b6220748ff1d5262018c`；RGB `c3fb74d89182d28b7a92579604ead018550ecc93e6f14721e075b4fff6050ef6`；HDR/depth/seg NPZ `fea84331929a5bfb29831d504c1893224d019505ae0bfb1a6c925b573c372179`；manifest `426e67b15058095f0906dfaf3be44ad2d91869c6d4c62535aeaf5563ecf970e1`；`SHA256SUMS` 文件本身 `fedc2831f0d34502b58b55f0a0ad7a129d94eeb9652aba957a0bd540fe5c8b63`。运行日志 SHA256 `93b36b3b9c2e1bb33c1534a9f4a83c1c93c80ff646f3a70398e71b48ae1b264c`。
- 静态 20 集原始结果位于 [`r0_static_dev20_20261016/`](../../benchmark/robot_closed_loop/results/r0_static_dev20_20261016/)；运行代码 commit `418718a...`、配置 hash `39c7aff...`。CSV SHA256 `6bf0bf909ffd358a7edc51df738ab55713c98eeeff78a2df3cea6fb584b6c8ae`；代表性 trace `2154ab5d81a59bc146cc865c5f5d489594d4fb18491132729f366374ac371ab6`；成功 GIF `8dcd8ac511a432763d7073b9182470f0d2677869022f7a3bf4d626d5c84b787e`；run manifest `ca0d87b4f609f0520590fb77b7a1ee853bfde792533b5276fa28110ba76c89d0`；`SHA256SUMS` 文件本身 `603ff03fef8558ad79e49851d0bebc72a7da355c38eccec8de5fc9b52edca869`。日志 SHA256 `bbbf64d424ff6e4bf4dcdc4990424d238d37374044ef2e9680e0dd6dd5a053c7`，exit `0`，PID `1537609`。
- no-hand slide calibration 原始结果位于 [`r0_slide_calibration_20261016/`](../../benchmark/robot_closed_loop/results/r0_slide_calibration_20261016/)；运行代码 commit `865b79e...`、配置 hash `39c7aff...`、`panda_asset_in_scene=false`。CSV SHA256 `fef5cf0c22be2a2c5aa62ea4a7b53d4dfe878738f7f833ae6fc8b2a695be0336`；拟合 JSON `4d1886c0e18c5f9d7b2a381ddbaee64da7815af8a8c3430d3e092db690587178`；manifest `c92c006e80bc7531da2c275c40748a8f712c26631755a032139ecf6cc14eb3e9`；`SHA256SUMS` 文件本身 `16a2374d16292d71c0d6b4ce49e302e127fd253bb23f6f5ca27e20ff0b6e7c03`。全部 30 条逐集 trace 包含在校验清单。运行日志 SHA256 `217a93ba8cc1fa7117a1a515f41739dfd6c7d1c9bae63e1a53bd032df122bee9`，exit `0`，PID `1543261`。
- no-hand holdout 原始结果位于 [`r0_slide_holdout_20261016/`](../../benchmark/robot_closed_loop/results/r0_slide_holdout_20261016/)；运行代码 commit `93e739e...`、配置 hash `39c7aff...`、`panda_asset_in_scene=false`。CSV SHA256 `2588b58dfe2d13b7397d5956cdbf545efb8aec7a9491d7174351b38eaeb33bae`；manifest `7ba5d4097a4079a00c534e71d2e14697c864a803c91cd4cb77a6dbb764ce0915`；`SHA256SUMS` 文件本身 `51288e81254627504e6c393a6800dbb215207571d1e5415a04d8ab60d8abd053`；逐集 traces 均含于清单。日志 SHA256 `bf7ce564d21c03f5e08ab59613b0f4e321bc98ec47011a7311696163bc1a851e`，exit `0`，PID `1548544`。
- 无效旧机位 preflight 仅作 bug 证据保留：报告 SHA256 `eb586dd8e7dffb69c480ac4e1d3e7c16c42bdd2008c900642a5fb7f477f08ea0`，输出清单 SHA256 `3d92ac2fcb06440c4cf16b9f69d7a145bd6cef2b867228cb1470160b86277ae6`；不得用于相机可见性结论。

## 尚未完成 / 风险

- GitHub fetch 从服务器等待超时后，改用已推送 commit bundle 串行同步。runtime setup 第一次因共享包缓存权限失败，第二次成功但 Torch 2.5.1+cu121 的接触查询失败；project venv 安装 Torch 2.8 CPU 后静态抓取 1/1 成功。
- Camera render 返回值处理存在 benchmark bug，已修正并在服务器复测；旧高机位 preflight 记录到 0 可见像素，仅保留作 bug 证据。新低机位已确认实际抓取姿态下目标清晰可见，HDR 输出有效；遮挡百分比对照姿态不合法，仍待实现有效定量对照。
- no-hand calibration 与独立 holdout 均已完成；GT-current oracle 动态抓取与误差注入仍待运行。静态 20/20 的置信下界只有 83.9%，也没有给出动态 oracle >95% 证据。
- 已测单个抓取姿态的相机视场与目标像素尺度；完整运动轨迹视野、基于合法参考姿态的定量遮挡，以及 1/2 kHz Direct-HDR reference 路径成本仍未测量。
- 正式 Ours/V2E 比较、大规模数据生成和训练均未开始。

## 冻结建议与研究会话需决策事项

暂不冻结任务配置。先依据服务器证据调整摩擦、发射区、截距、闭合提前量与控制限幅；随后在独立 holdout 上确认单一减速度模型可用，再报告 oracle。若 Torch 兼容、HDR reference 成本或在合理调参后 oracle 可达性仍有实质障碍，将在此处附日志/哈希和建议方案后提交研究会话决定。主分支合并留待验收。
