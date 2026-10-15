# Genesis 事件相机项目：通用研究基线

**版本：2026-10-07**  
**三个研究 Chat 均应上传本文件；不再上传 2026-09-30 的旧总结。**

---

## 1. 核心研究问题

本项目要证明的不是“能生成事件”或“事件更多”，而是：

> **Genesis 中可获得的 scene-linear HDR/radiance、depth、segmentation、camera/object/link pose 等物理状态，能否让事件生成在光度、时空几何和可见性上更正确；这种正确性是否能进一步带来真实事件数据泛化与机器人闭环效用。**

当前论文证据链：

```text
Genesis physics
→ physics-aware HDR/radiance event simulation
→ stronger Real-vs-Sim event fidelity
→ stronger public real-data Sim-to-Real
→ stronger robot closed-loop utility
```

三条验证线：

1. 仿真器质量：Real EVK4 vs Ours vs baseline。
2. 公共真实数据 Sim-to-Real：不同 simulator 的合成训练数据 → 同一模型 → 真实公开事件测试。
3. 机器人闭环：Ours/V2E 合成训练数据 → 同一感知/控制结构 → 动态机器人任务。

---

## 2. 当前 simulator 核心架构

正式数据流：

```text
Genesis physical scene
    ↓
scene-linear HDR / radiance endpoints
+ depth
+ segmentation / entity-link identity
+ camera pose
+ object/link pose
+ timestamps
    ↓
SE(3) / geometry-aware correspondence
    ↓
target-domain occupancy / z-buffer visibility
/ unknown / disocclusion semantics
    ↓
physically constrained spatiotemporal radiance reconstruction
    ↓
adaptive temporal sampling / real sub-render requests where needed
    ↓
log-intensity DVS response
    ↓
AER events (x, y, t, p)
```

核心思想：事件生成发生在 renderer/physics state 层，而不是把 display RGB 当作唯一真值后处理。

---

## 3. HDR→Event 为什么是主线

display RGB 通常经历 tone mapping、gamma、exposure、clipping、dynamic-range compression。事件相机响应的是光强对数变化，若先压缩到显示域再做 RGB→Event，会丢失事件生成关心的光度动态范围。

因此当前主线是：

> **scene-linear radiance → log intensity → event response**

而 V2E 等 video-to-event 方法作为重要 baseline 保留。

---

## 4. 几何/可见性方法

source pixel 通过 depth 反投影到三维场景点，再依据 camera/object/link SE(3) 运动投影到 target/intermediate domain。

必须明确区分：

- source-valid
- projected occupancy
- z-buffer visibility
- target visibility/support
- intermediate visibility
- unknown / unsupported region
- disocclusion（新暴露区域）

核心不是单纯“算光流”，而是回答：

> 源时刻的哪一个物理表面，在目标时刻投影到哪里；它是否仍然可见；哪些 target pixels 根本没有可靠 source correspondence。

未知/新暴露区域不能简单被图像插值伪装成可靠 radiance。

---

## 5. DVS 响应

对 radiance/luminance：

`L(x,y,t) = log(I(x,y,t))`

达到 ON/OFF contrast threshold 时输出：

`e = (x, y, t, p)`

传感器非理想性可包括 threshold mismatch、temporal response、leak、shot/background noise 等，但必须区分参数证据：

- measured（实测）
- fitted（拟合）
- nominal（名义）
- assumed（假设）

不能把 nominal/assumed 参数描述为“精确 EVK4 模型”。

---

## 6. v3.1 正确性修复与冻结状态

当前 frozen simulator core：

`e79e656ac00f7847b98db9e12bf43b093f3afff1`

Genesis：1.2.2 + HDR renderer overlay。  
Genesis-facing adapter：当前 single-environment。

v3.1 已修正：

- valid01 / valid10 correspondence 坐标域语义；
- source-valid 与 target-domain support 分离；
- projected occupancy；
- z-buffer target visibility；
- intermediate visibility；
- unknown/disocclusion mask；
- disocclusion fraction 使用 target-domain unsupported mask；
- radiance residual 只在 projected target support 上计算，其余为 NaN；
- depth change 采用 source-material-point correspondence：`|z_target-z_source|/z_source`；
- Torch 路径增加 occupancy/visibility/z-buffer diagnostics；
- NumPy/Torch 对照验证；
- single-env 边界明确。

这些现在属于**方法正确性基线**，不是可以为 benchmark 自由调整的参数。

---

## 7. Genesis 优势与贡献边界

不要声称这些物理量“Genesis 独有”。更准确的表述：

> Genesis 把渲染、物理、机器人状态放在统一仿真环境中，使 radiance、depth、segmentation、camera/object/link motion 等信息可以同步、可控、可重复地服务于事件生成和机器人实验。

方法贡献更适合表述为：

> 将 scene-linear HDR 与显式几何、target-domain visibility/occlusion/disocclusion 建模结合到 Genesis 机器人仿真中，形成 physics-aware event simulation。

---

## 8. 三条验证线的职责

### 质量验证
回答：“事件本身在 `(x,y,p,t)` 上是否更接近真实 EVK4？”  
主指标目标：PECS 风格 CD/GD；弱 aggregate statistics 仅 sanity check。

### 公共 Sim-to-Real
回答：“同样的 synthetic physical trajectories，只换 event generator 后，训练出的模型在真实公开事件数据上谁泛化更好？”

### 机器人闭环
回答：“训练事件质量的差异是否能经由感知误差、拦截误差传播到机器人任务成功率？”

---

## 9. 统一公平性原则

同一个 latent physical sample/rollout 必须共享：

- scene
- object/camera/robot trajectories
- timestamps
- labels
- train/val/test split

下游共享：

- model architecture
- optimizer
- training budget
- seeds
- event representation
- normalization
- augmentation
- batch schedule

**核心实验变量只允许是 event generator。**

---

## 10. Privileged information 规则

Genesis depth、segmentation、object pose、link pose 可以用于：

- simulator 内部物理重建；
- synthetic training labels；
- offline evaluation GT。

除非真实部署同样可获得，否则不得作为 Ours 下游模型的额外 inference input。

---

## 11. Event representation

下游默认统一使用 causal 5-bin event voxel grid：

- 无未来事件泄漏；
- Ours/V2E 使用相同 binning；
- 不因事件密度差异改变 representation。

历史 event-frame/time-surface 资产可保留，但不应成为新的主比较变量。

---

## 12. 已降级/废弃的旧思路

以下不再作为论文主证据：

- event count 更接近真实；
- event rate / polarity ratio / active pixels 作为主要质量指标；
- readiness/smoke 当正式结果；
- 静态普通 pick-and-place 作为机器人主任务；
- 用 Ours 自己作为 Ours-trained 模型的独家测试环境；
- 为某一个 benchmark 后验修改 simulator；
- 向下游网络泄漏 Genesis GT。

---

## 13. Formal 实验共同规则

必须预注册/冻结：

- core commit
- generator config
- baseline version
- data manifest/hash
- split/seeds
- metrics
- model/training budget
- alignment policy（如适用）
- exclusion/STOP rule

正式结果产生后不得无记录修改。

---

## 14. 计算现实

1280×720 HDR replay 曾表现出较高 CPU 成本。因此：

- smoke/readiness 可低分辨率；
- 正式实验保持预注册分辨率；
- 优先先打通最小完整 pipeline；
- 不用大规模 brute-force sweep 代替研究设计。
