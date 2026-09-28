# Genesis 事件相机插件 — 功能调试指南

## 目录结构

```
genesis_event_plugin/
├── genesis_event_plugin/
│   ├── __init__.py           # 包入口
│   ├── plugin.py             # 顶层 API (GenesisEventPlugin)
│   ├── interpolator.py       # Phase 1-6 插帧
│   ├── dvs_emulator.py       # DVS 像素模型
│   ├── recorder.py           # HDF5 双格式记录
│   ├── noise_model.py        # 噪声预设
│   └── utils/
│       ├── dvs_core.py       # V2E 核心函数 (lin_log, low_pass, event_map...)
│       ├── flow_utils.py     # 解析光流
│       └── motion_blur.py    # RGB 运动模糊
├── setup.py
└── README.md
```

## 一、环境检查

```bash
python -c "import genesis; print(genesis.__version__)"        # ≥ 1.2.2
python -c "import torch; print(torch.cuda.is_available())"    # GPU: True/False
python -c "import h5py; print(h5py.__version__)"              # ≥ 3.0
```

## 二、模块级功能验证

### 2.1 DVS 核心函数 (dvs_core.py)

```bash
cd genesis_event_plugin
python -c "
from genesis_event_plugin.utils.dvs_core import *
import torch, math
x = torch.arange(0, 256, dtype=torch.float32)
y = lin_log(x)
print('lin_log(0)=', y[0].item(), '(should be near 0)')
print('lin_log(255)=', y[255].item(), f'(should be near ln(255)={math.log(255):.3f})')
print('lin_log OK')
"
```

**预期输出**: `lin_log(0) ≈ 0`, `lin_log(255) ≈ 5.54`

### 2.2 噪声模型 (noise_model.py)

```bash
python -c "
from genesis_event_plugin.noise_model import get_preset, PRESETS
for name in PRESETS:
    p = get_preset(name)
    print(f'{name:15s} thr={p.pos_thres} σ={p.sigma_thres} cutoff={p.cutoff_hz}Hz shot={p.shot_noise_rate_hz}Hz refrac={p.refractory_period_s}s')
print('Noise presets OK')
"
```

### 2.3 DVS 仿真器 (dvs_emulator.py)

```bash
python -c "
from genesis_event_plugin import DvsEmulator
import numpy as np

emu = DvsEmulator(res=(60, 80), preset='clean')
# 静态场景: 无事件
frame0 = np.random.randint(0, 256, (60, 80), dtype=np.uint8)
emu.initialize(frame0, t=0.0)
events = emu.generate_events(frame0, t=0.033)
assert len(events) == 0, f'Static scene should have 0 events, got {len(events)}'
print(f'Static scene: {len(events)} events (expected 0)')

# 运动场景: 应有事件
frame1 = np.roll(frame0, shift=2, axis=1)  # 平移 2 像素
emu.reset()
emu.initialize(frame0, t=0.0)
events = emu.generate_events(frame1, t=0.033)
assert len(events) > 0, f'Moving scene should have events, got {len(events)}'
print(f'Moving scene: {len(events)} events (expected >0)')
print('DVS emulator OK')
"
```

### 2.4 解析光流 (flow_utils.py) — 需要 Genesis

```bash
# 在服务器上运行:
cd ~/Eventbased_WAM/code/genesis_event_plugin
PYTHONPATH=. python -c "
from genesis_event_plugin.utils.flow_utils import compute_analytic_flow
import genesis as gs, numpy as np

gs.init(backend=gs.cpu, logging_level='warning')
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.001, gravity=(0,0,0)), show_viewer=False)
plane = scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(gs.morphs.Box(pos=(0,0,0.5), size=(0.2,0.2,0.2)))
cam = scene.add_camera(res=(320,240), pos=(1,1,1), lookat=(0,0,0.5), fov=45)
scene.build()

cube.set_dofs_velocity([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
scene.step()
rgb, depth, seg, _ = cam.render(rgb=True, depth=True, segmentation=True)

F = compute_analytic_flow(depth, seg, cam.intrinsics, [cube], dt=0.03)
mask = seg == cube.idx
assert mask.sum() > 0, 'Cube not visible'
vx_mean = F[0][mask].mean()
print(f'Flow vx at cube: {vx_mean:.2f} px/frame (should be <0 for rightward motion)')
print('Analytic flow OK')
"
```

### 2.5 插帧器 (interpolator.py) — 需要 Genesis

```bash
# 在服务器上运行:
cd ~/Eventbased_WAM/code/genesis_event_plugin
PYTHONPATH=. python -c "
from genesis_event_plugin import GenesisInterpolator
import genesis as gs, numpy as np

gs.init(backend=gs.cpu, logging_level='warning')
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.001, gravity=(0,0,0)), show_viewer=False)
plane = scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(gs.morphs.Box(pos=(0,0,0.5), size=(0.2,0.2,0.2)))
cam = scene.add_camera(res=(320,240), pos=(1,1,1), lookat=(0,0,0.5), fov=45)
scene.build()

cube.set_dofs_velocity([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
scene.step()
rgb0, depth0, seg0, _ = cam.render(rgb=True, depth=True, segmentation=True)
for _ in range(30): scene.step()
rgb1, depth1, seg1, _ = cam.render(rgb=True, depth=True, segmentation=True)

interp = GenesisInterpolator(K=cam.intrinsics, mode='analytic')
gray_frames, ts = interp.interpolate(rgb0, depth0, seg0, rgb1, depth1, seg1, [cube], 0.0, 0.03)

print(f'Interpolation: {len(gray_frames)} frames generated (U≈{interp.get_stats()[\"avg_upsample_factor\"]})')
assert len(gray_frames) >= 2, f'Expected at least 2 frames, got {len(gray_frames)}'
for i, (gf, t) in enumerate(zip(gray_frames, ts)):
    print(f'  Frame {i}: t={t*1000:.2f}ms, shape={gf.shape}, min={gf.min()}, max={gf.max()}')
print('Interpolator OK')
"
```

### 2.6 RGB 运动模糊 (motion_blur.py)

```bash
python -c "
from genesis_event_plugin.utils.motion_blur import apply_motion_blur
import numpy as np

# 100fps 序列, 10 帧
frames = np.random.randint(0, 256, (10, 60, 80, 3), dtype=np.uint8)
ts = np.arange(10) * 0.01  # 每 10ms 一帧
blurred = apply_motion_blur(frames, ts, exposure_time_ms=30.0)

print(f'Blurred shape: {blurred.shape}')
assert blurred.shape == frames.shape
# 模糊后的帧应该更平滑 (方差更小)
orig_var = frames.var(axis=(1,2,3)).mean()
blur_var = blurred.var(axis=(1,2,3)).mean()
print(f'Original variance: {orig_var:.1f}, Blurred variance: {blur_var:.1f} (should be lower)')
print('Motion blur OK')
"
```

## 三、端到端测试

```bash
# 在服务器上运行:
cd ~/Eventbased_WAM/code/genesis_event_plugin
PYTHONPATH=. python -c "
from genesis_event_plugin import GenesisEventPlugin
import genesis as gs, numpy as np, os, shutil

OUT = '/tmp/genesis_event_test_e2e'
shutil.rmtree(OUT, ignore_errors=True)

gs.init(backend=gs.cpu, logging_level='warning')
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.001, gravity=(0,0,0)), show_viewer=False)
plane = scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(gs.morphs.Box(pos=(0,0,0.5), size=(0.2,0.2,0.2)))
cam = scene.add_camera(res=(320,240), pos=(1,1,1), lookat=(0,0,0.5), fov=45)
scene.build()

cube.set_dofs_velocity([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

plugin = GenesisEventPlugin(
    output_dir=OUT,
    preset='noisy',
    interpolation_mode='analytic',
    output_event_frames=True,
    output_rgb_frames=True,
)
plugin.attach(scene, cam, [cube])

plugin.start_episode()
total_events = 0
for step in range(100):
    scene.step()
    events = plugin.capture()
    total_events += len(events)
plugin.end_episode()
plugin.close()

print(f'E2E test: 100 steps, {total_events} events')
print(f'Output files:')
for f in sorted(os.listdir(OUT)):
    size_kb = os.path.getsize(os.path.join(OUT, f)) / 1024
    print(f'  {f}: {size_kb:.1f} KB')

import h5py
with h5py.File(os.path.join(OUT, 'aer_events.h5'), 'r') as f:
    events = f['events'][:]
    print(f'AER: {len(events)} events, shape={events.shape}')
    if len(events) > 0:
        print(f'  Sample: t_us={events[0,0]}, x={events[0,1]}, y={events[0,2]}, p={events[0,3]}')

if os.path.exists(os.path.join(OUT, 'event_frames.h5')):
    with h5py.File(os.path.join(OUT, 'event_frames.h5'), 'r') as f:
        print(f'Event frames: groups={list(f.keys())}')

print('E2E test PASSED')
"
```

## 四、噪声模型调试

### 4.1 IIR 低通滤波 — τ ≫ dt 条件

一阶 IIR 正常工作的前提是 `τ = 1/(2π·cutoff_hz) ≫ dt`（即 `eps = inten01 × dt/τ ≪ 1`）。

**各预设达标情况**：

| 预设 | cutoff_hz | τ | 所需 dt (eps≤0.3) | U因子 | 可行？ |
|------|:--:|:--:|:--:|:--:|:--:|
| clean | 0 | ∞ | — | — | 无滤波 ✅ |
| noisy | 30 | 5.3ms | ≤1.6ms | ≥20 | ✅ |
| low_light | 10 | 15.9ms | ≤4.8ms | ≥7 | ✅ |
| high_speed | 300 | 0.53ms | ≤0.16ms | ≥200 | ❌ 不可能 |
| overexposure | 0 | ∞ | — | — | 无滤波 ✅ |

**high_speed 特殊说明**：cutoff_hz=300 时 τ=0.53ms，IIR 天然退化为恒等映射。这是物理正确的——带宽 300Hz 的传感器本就近乎瞬时响应，不需要平滑。`eps` clamp 到 1 不是 bug。

**调试**：
- `eps > 0.3` 警告 → dt 太大，增加插帧 U 或降低 cutoff_hz
- 如果 high_speed 下 IIR 退化是预期行为，不需要"修复"

### 4.2 散粒噪声 (Shot Noise) — 物理直觉

散粒噪声来自光子本身的量子性质：即使光照完全均匀，每个像素在 dt 内收到的光子数服从 Poisson 分布。涨落超过阈值就产生噪声事件。

**核心规律**：`σ/N = 1/√N`——越暗，相对涨落越大。

| 光照 | 平均光子数 N | 相对噪声 σ/N | 超过 15% 阈值的概率 |
|------|:--:|:--:|:--:|
| 明亮 | 10000 | 1% | ≈ 0 |
| 一般 | 400 | 5% | ≈ 0.3% |
| 偏暗 | 100 | 10% | ≈ 13% |
| 很暗 | 25 | 20% | ≈ 45% |

**关键结论**：低光场景下噪声多是对的。如果 low_light 数据很干净，反而不真实。

DVS 参考值在两个阈值边界之间来回震荡进一步放大了噪声——Poisson 涨落推参考值过界 → 触发 → 参考值追上 → 反向涨落再推 → 持续震荡。

**调试**：
- 暗区噪声率高是**物理正确**的，不要当作 bug 修复
- 未来可优化：`shot_noise_rate_hz ∝ 1/inten01`（直接反比于亮度），让 Poisson 的 1/√N 关系直接落地

### 4.3 两条噪声路径 — 直接 Poisson vs Graca 光电校准

| | 路径 A：直接泊松 | 路径 B：Graca 2021 校准 |
|------|------|------|
| **触发条件** | `shot_noise_rate_hz > 0` 且 `photoreceptor_noise=False` | `shot_noise_rate_hz > 0` 且 `photoreceptor_noise=True` |
| **实现** | Bernoulli 随机直接"贴"噪声事件 | 高斯噪声注入光感受器 → IIR → 阈值截断 |
| **噪声时序** | 白噪声（帧间独立） | 有色噪声（IIR 平滑后缓慢起伏） |
| **物理保真度** | 中等 | 高（Graca 2021 精确映射） |
| **计算开销** | 极低 | 初始化一次，后续每帧仅高斯采样 |
| **前提条件** | 无 | `cutoff_hz > 0`，`sample_rate` 不能太低 |
| **适用场景** | 默认、日常、快速原型 | 精确校准、发论文、匹配特定芯片 |

**两者互斥**——同时开会产生双重噪声，结果不可控。

**调试**：
- 日常开发用路径 A，发论文对噪声有严格要求的实验用路径 B
- 路径 B 初始化时 `eps > 0.1` 会报警，需要增加插帧或降 cutoff
- 路径 A 的 `shot_noise_rate_hz × dt > 1` 警告表示单帧噪声概率超 1，需减小 dt


## 四、常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `cube not visible` | 相机位置没对准物体 | 改 camera_pos/lookat 使物体在视野内 |
| `Non-monotonic time` | 时间戳回退 | 检查 Genesis dt 设置 |
| `0 events` | 阈值太高/无运动/纯白墙 | `preset='clean'` 降阈值, 或 `sigma_thres=0` |
| `Too many events` | 插帧太密 | 增大 `C_nom` 或设 `lambda_b=0.3` |
| IIR 警告 `eps > 0.3` | dt 太大，τ≫dt 不成立 | 增加插帧 U 或降低 cutoff_hz；high_speed 下退化是预期行为 |
| 低光场景噪声太多 | Poisson 散粒噪声物理正确 | 暗区 σ/N 更大，噪声多是对的；未来可做 ∝1/inten01 优化 |
| `shot_noise × dt > 1` | 单帧噪声概率超 1 | 减小 dt（增加插帧）或降低 shot_noise_rate_hz |
| 两道噪声同时开 | 路径 A+B 互斥 | 选一种：日常用直接 Poisson，论文用 Graca 校准 |
| 程序卡住 | GPU 后端不兼容 | 切 `backend=gs.cpu` |
| 输出文件为空 | `close()` 没调用 | 确保 `plugin.close()` 在脚本末尾 |

## 五、性能基准

| 组件 | CPU 耗时 | GPU 耗时 (预期) |
|------|:--:|:--:|
| 解析光流 (320×240, 2物体) | 0.03ms | <0.01ms |
| 插帧 warp+融合 (320×240, U=5) | ~2ms | <0.5ms |
| DVS 事件生成 (320×240) | ~3ms | <1ms |
| **capture() 总延迟** | **~10ms** | **~2ms** |

> 基准: Genesis CPU 后端, 320×240, Ubuntu 20.04, RTX 3090 (实际 GPU 测试待驱动修复)
