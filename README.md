# Genesis Event Camera Plugin — Physics/Radiance v2 Prototype

> This directory is an independent experimental copy. The original plugin is
> left unchanged. The v2 path adds linear HDR/radiance input, full SE(3)
> pose-driven depth-aware warping, visibility guards, and an explicit
> adaptive temporal sampler. It is not yet a calibrated EVK4 renderer: the
> Genesis public camera API currently exposes RGB/depth/segmentation/normal,
> so a real HDR provider and calibrated radiance scale must be supplied.

New entry points:

```python
from genesis_event_plugin import (
    AdaptiveSamplingConfig,
    GenesisPhysicsEventPlugin,
    PhysicsInterpolator,
)
```

The production-facing wrapper requires explicit `radiance_provider` and
`motion_state_provider` callbacks. This prevents display RGB or incomplete
entity velocities from being silently reported as physical input.

For fast camera motion, create the Genesis camera with an expanded render
resolution and shifted principal point (`pad_intrinsics`), then pass the same
`CameraMargin` to `GenesisPhysicsEventPlugin`. The wrapper performs the final
crop before DVS generation; the expanded depth/seg/radiance images remain
available to the physical warp so disoccluded pixels are not fabricated by a
display-space resize.

This directory is the independent research implementation intended for Git
distribution. It contains the physics/radiance v2 plugin, Torch/CUDA reference
paths, Genesis adapters, tests, and an optional Genesis 1.2.2 HDR renderer
overlay. The original direct plugin, training jobs, and generated datasets are
deliberately excluded.

从 [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 物理仿真生成 DVS（Dynamic Vision Sensor）事件数据的研究型插件。

本项目是 **Event-based World Action Model（事件世界模型）** 研究的一部分，用于把标准 RGB 相机替换为事件相机，并为仿真数据生成、模型训练和 Sim2Real 评估提供统一的数据管线。

---

> **研究状态（2026-09-28）**：physics/radiance v2 原型可运行；独立 Genesis renderer overlay 已通过 HDR smoke；Torch/CUDA flow+warp 参考路径可用。目标真机的曝光、辐射、像素电路和读出参数尚未完成 EVK4 标定。`analytic`/`fixed` 仅作 legacy/消融，不代表 ground truth 或真实相机。完整设计和证据边界见工作区中的《Genesis 事件相机插件学术综述与研究设计总纲（2026-09-28）》。

**重要状态边界**：本仓库不宣称已经完整复现 EVK4，也不把普通 display RGB 转换称为真实 radiance。原生 HDR 需要使用 `integrations/genesis_renderer_hdr_v2_overlay/` 对 Genesis 1.2.2 源码应用补丁后，显式调用 `camera.render(radiance=True)`。

## 特性

- **Genesis 场景直出事件**：`scene.step() -> render -> 插帧 -> DVS 像素模型 -> AER/事件帧`
- **自适应插帧**：
  - `direct`：逐物理/渲染步使用真实 endpoint；当前可信主基线
  - `analytic`：实验性简化流，仅含部分实体平移，缺相机 ego-motion/轴向运动/角速度/遮挡
  - `fixed`：帧交叉淡化的历史基线，可能制造非物理事件
- **DVS 像素模型**：移植 V2E 的 `lin-log -> IIR 低通 -> 差分量化 -> 泄漏电流 -> 散粒噪声` 管线，并补充 ESIM 风格的带宽/阈值建模
- **传感器基线/历史 stress profile**：`clean / moderate / noisy / evk4_nominal_1klux / low_light / high_speed / overexposure`；EVK4 预设只采用公开名义值，仍需目标真机重新标定
- **双格式输出**：
  - `aer_events.h5`：原始事件流 `[t_us, x, y, p]`
  - `event_frames.h5`：ON/OFF 双通道事件帧 `(T, H, W, 2)`
  - `rgb_frames.h5`：RGB 帧 + 多曝光运动模糊（用于 RGB vs 事件公平对比）
- **质量评估**：
  - 非循环 RGB-edge、覆盖、极性、显式静止 mask 等诊断指标
  - 修正后的 PECS CD/GD；论文 baseline 仅在完全匹配 PECS rig/protocol 时启用

---

## 安装

```bash
# 克隆后进入仓库目录
pip install -e .

# 或者使用 requirements.txt
pip install -r requirements.txt
```

依赖：

- Python >= 3.8
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) >= 1.2
- PyTorch >= 2.0
- numpy, h5py, scipy
- 运行 demo / 质量评估时另需：`pillow`, `imageio`, `opencv-python`

---

## 快速开始

```python
import genesis as gs
from genesis_event_plugin import GenesisEventPlugin

gs.init(backend=gs.cpu, logging_level="warning")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, 0)),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(
    gs.morphs.Box(pos=(0, 0, 0.5), size=(0.2, 0.2, 0.2)),
)
cam = scene.add_camera(res=(320, 240), pos=(1, 1, 1), lookat=(0, 0, 0.5), fov=45)
scene.build()

plugin = GenesisEventPlugin(
    output_dir="output/",
    preset="real_v2",
    interpolation_mode="direct",  # 本例 dt=1 ms，每个 physics step 直接采样
    output_event_frames=True,
    output_rgb_frames=True,
)
plugin.attach(scene, cam, [cube])

cube.set_dofs_velocity([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

plugin.start_episode()
for step in range(100):
    scene.step()
    events = plugin.capture()
plugin.end_episode()
plugin.close()
```

`capture()` 返回 `(N, 4)` 的事件数组，列顺序为 `[t(s), x, y, p]`。

### Prophesee EVK4-HD 名义光学配置

若使用 EVK4-HD 原配 Soyo SFA 0820-5M 镜头，可先用厂商名义值创建相机：

```python
from genesis_event_plugin import GenesisEventPlugin, get_camera_profile

evk4 = get_camera_profile("evk4")
cam = scene.add_camera(
    pos=(...),
    lookat=(...),
    **evk4.genesis_camera_kwargs(),  # 1280x720, vertical FOV 23.6°, pinhole
)

plugin = GenesisEventPlugin(
    output_dir="output/",
    preset="evk4_nominal_1klux",
    camera_profile=evk4,
    strict_camera_profile=True,
)
```

EVK4 档案记录 1280×720、4.86 µm、8 mm、F/2.0、H/V/D-FOV
41.4°/23.6°/47.0°。Genesis 的 `fov` 是**垂直视场**，故配置为
23.6°。F/2.0 不会被错误映射为 Genesis 的 `aperture=2.0`；当前以
pinhole 为可复现主基线。真实相机的内参、畸变、焦点、bias 和逐照度噪声
仍须实物标定，详见 [EVK4 配置记录](docs/research/Prophesee_EVK4_光学与传感器配置_2026-09-01.md)。

---

## 目录结构

```
genesis_event_plugin/
├── genesis_event_plugin/
│   ├── plugin.py             # 顶层 API（GenesisEventPlugin）
│   ├── interpolator.py       # direct / 实验 analytic / legacy fixed
│   ├── dvs_emulator.py       # DVS 像素模型主循环
│   ├── recorder.py           # HDF5 双格式记录
│   ├── noise_model.py        # 噪声预设系统
│   ├── utils/
│   │   ├── dvs_core.py       # V2E 核心函数（lin-log / low-pass / event_map / shot_noise）
│   │   ├── flow_utils.py     # 解析光流 + 自适应插帧率
│   │   └── motion_blur.py    # RGB 多曝光运动模糊
│   └── evaluation/
│       ├── evaluator.py      # 统一评估框架（内部指标 + CD/GD）
│       ├── pecs_metrics.py   # PECS Chamfer / Gaussian Distance
│       ├── event_reader.py   # HDF5 / RAW / numpy 事件读取
│       └── plan_a_runner.py  # Plan A scaffold（未完成物理匹配前 fail-closed）
├── scripts/
│   └── demo_runner.py        # Franka 官方 demo 事件动图生成
├── tests/
│   ├── test_core_correctness.py # 核心物理/格式回归测试
│   └── bench_quality.py      # 历史 benchmark（科研结论禁用）
├── docs/
│   └── debugging.md          # 调试指南
├── setup.py
└── requirements.txt
```

---

## 噪声预设

| 预设 | 用途 | 关键参数 |
|------|------|---------|
| `clean` | 完美传感器 / 调试 / 对比 | 无噪声、无限带宽 |
| `moderate` | 工程开发 baseline | 经验参数，未绑定真机 |
| `noisy` | v2e-compatible baseline | V2E 默认风格，不等于任意真实 DVS |
| `evk4_nominal_1klux` | EVK4/IMX636 初始基线 | C±=0.25、CTNU 上界近似、1 klux 背景率上限；其余待标定 |
| `low_light` | legacy stress profile | 会改变 sensor bias；不能与其他场景作单变量比较 |
| `high_speed` | legacy stress profile | 同上 |
| `overexposure` | legacy stress profile | 同上；不能靠降阈值恢复 clipping 丢失的信息 |

也可用 `custom_config(...)` 自定义任意参数。

---

## 输出格式

| 文件 | 内容 | 说明 |
|------|------|------|
| `aer_events.h5` | `events: (N, 4) uint64 [t_us, x, y, p_01]` | 避免 71.6 分钟溢出；含 schema/unit/polarity/config/resolution/episode ranges |
| `event_frames.h5` | `ep_N/frames`, `actions`, `timestamps_s` | 默认 decision/capture interval + `log1p`；静默 interval 也保存零帧 |
| `rgb_frames.h5` | `ep_N/clean`, `timestamps_s`, `blur_Xms` | RGB + 仅供对照的多曝光版本 |

---

## 质量评估

### 核心回归测试

```bash
PYTHONPATH=. python -m unittest -v tests.test_core_correctness
```

`plan_a_runner.py` 目前会主动拒绝运行：旋转/平移执行器、棋盘纹理、镜头/光照和 real-sim 同步尚未实现。完成 matched physical rig 后才可启用 PECS CD/GD 外部评估。历史 `bench_quality.py` 也默认禁用，避免把不完整流和循环指标用于科研结论。

---

## 参考与致谢

- V2E: Delbruck, Hu, He. *V2E: From Video to Events.* CVPRW 2021. [GitHub](https://github.com/SensorsINI/v2e)
- ESIM: Rebecq et al. *ESIM: an Open Event Camera Simulator.* CoRL 2018.
- Graca & Delbruck. *Unraveling the Paradox of Intensity-Dependent DVS Pixel Noise.* arXiv 2021.
- PECS: Han et al. *Physical-Based Event Camera Simulator.* ECCV 2024.

`dvs_core.py` 中核心 DVS 数学函数移植自 V2E（MIT License），详见 [NOTICE](NOTICE.md)。

---

## License

本项目代码以 [MIT License](LICENSE) 发布。
