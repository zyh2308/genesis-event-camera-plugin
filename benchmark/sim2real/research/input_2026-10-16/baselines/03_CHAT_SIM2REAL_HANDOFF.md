# Chat 分支交接：公共真实数据 Sim-to-Real

**本会话只处理 synthetic-only training → public real event dataset testing。**  
开始时同时上传 `01_COMMON_RESEARCH_BASELINE.md`。

---

## 1. 核心研究问题

> 在相同 Genesis latent physical data、相同 labels、相同网络和训练预算下，仅改变 event generator，Ours 生成的数据是否能训练出在真实公开事件数据上泛化更好的模型？

这个分支是论文从“信号真实性”走向“真实任务泛化”的中间桥梁。

---

## 2. 逻辑链：层层递进

推荐三层：

### Layer 1：事件光度/结构
synthetic events → event-to-image reconstruction → real HS-ERGB。

### Layer 2：运动几何（主实验）
synthetic events → event optical flow → real MVSEC。

### Layer 3：机器人相关物体状态（加强）
synthetic events → 6DoF object pose → real E-POSE。

论文叙事：

`event structure → motion geometry → manipulation-relevant object state`

资源有限时：先 Layer 2，后 Layer 1；Layer 3 不得拖住主实验。

---

## 3. Shared synthetic trajectory bank

必须先生成不含 simulator-specific event 的 latent corpus：

```text
trajectory_bank_v1/
- scene_id
- seed
- HDR/radiance endpoints
- depth
- segmentation
- camera pose(t)
- object/link pose(t)
- timestamps
- flow GT / reconstruction GT / pose GT
```

然后从同一份 bank 分叉：

```text
trajectory_bank_v1
├── Ours → events_ours
└── V2E → events_v2e
```

所有 label 完全相同。

---

## 4. 主实验：事件光流

### 为什么是主实验
它直接敏感于：

- event timing
- motion correspondence
- depth/parallax
- independently moving objects
- occlusion/disocclusion
- camera ego-motion

与 Ours 的 geometry/visibility novelty 最对齐。

### 推荐模型
默认优先：**STE-FlowNet**，原因是 event-only、官方训练/评估代码、MVSEC 支持明确，混淆变量少。

ADMFlow 是合理 fallback/secondary reference，因为其论文本身研究 rendered synthetic events → real MVSEC。若正式前切换模型，必须基于工程可复现性/资源，而不是看哪一个更利于 Ours；切换记录在 protocol changelog。

### Real test
MVSEC，具体 sequence / dt=1/dt=4 必须 formal 前冻结。

### 主指标
- AEE/EPE ↓
- outlier rate ↓

禁止 real-test-specific fine-tuning。

---

## 5. Layer 1：E2VID → HS-ERGB

目的：验证 synthetic event 的 photometric/structural usefulness，尤其对应 HDR/radiance 路线。

训练：

- Ours events + same GT images
- V2E events + same GT images
- same E2VID architecture/training budget

Real test：HS-ERGB。

指标：

- SSIM ↑
- LPIPS ↓
- PSNR ↑

图像归一化/对齐规则必须在看到 test ranking 前冻结。

---

## 6. Layer 3：E-POSE 加强实验

目标：把 synthetic event quality 延伸到 manipulation-relevant 6DoF object state。

Genesis 中使用公开 YCB meshes/对应物体，Ours/V2E 共享相同 pose labels。

Real test：E-POSE。

优先复现数据集/相关公开工作的标准事件表示与 6D pose metric；若代码/训练成本不可控，则把该层明确标为 optional strengthening experiment，而不是为了“三级链条”硬做不可靠结果。

---

## 7. 公平性

Ours/V2E 必须共享：

- exact trajectory IDs
- label files
- train/val split
- dataset size
- event window duration
- voxelization
- network architecture
- optimizer/schedule
- seeds
- augmentation

真实测试集不得用于：

- tuning simulator parameters
- tuning synthetic randomization based on final ranking
- selecting favorable scene subsets

---

## 8. Synthetic validation 与 real test 隔离

所有 hyperparameter selection 只能看：

- synthetic train
- synthetic validation

真实 public test 在 config frozen + hash 后才运行。

若 real test 发现 code bug：

- 记录 bug；
- 修复必须方法无关；
- changelog；
- 所有 generator/model 统一重跑受影响部分。

---

## 9. 推荐仓库结构

```text
benchmark/sim2real/
├── protocol_v1.yaml
├── protocol_v1.sha256
├── trajectory_bank_v1.yaml
├── manifests/
│   ├── reconstruction.yaml
│   ├── flow.yaml
│   └── pose.yaml
├── export/
│   ├── export_ours.py
│   └── export_v2e.py
├── tasks/
│   ├── e2vid_hsergb/
│   ├── steflo_mvsec/
│   └── epose_pose/
└── results/
```

第三方模型建议放 `external/`，记录 repo URL + commit SHA + local patches。

---

## 10. 阶段 Gate

### S0：model/dataset audit
确认官方代码、license、input/output、metrics、环境与 GPU 需求。

### S1：trajectory bank smoke
验证 Ours/V2E 同 trajectory hash、same GT、loader 可读。

### S2：光流最小闭环
小规模 synthetic train → MVSEC evaluator 跑通，但标记 PILOT。

### S3：protocol freeze
固定 corpus size、seeds、model config、real sequences、metrics。

### S4：formal multi-seed flow
至少 3 seeds，报告 mean±std/CI。

### S5：E2VID complement
主光流结果冻结后再加。

### S6：E-POSE optional
资源/复现性达标再进入。

---

## 11. 本 Chat 与 Sim-to-Real Codex 的分工

本 Chat：

- 研究任务选择；
- 模型选择 Gate；
- 数据泄漏审查；
- protocol；
- 结果解释。

Codex：

- 外部 repo 环境；
- dataset converter；
- training scripts；
- manifests/hashes；
- smoke/pilot/formal runs；
- `sim2real_status.md`。

---

## 12. 论文 Claim

如果 Ours-trained model 在真实 public data 上稳定更好，可支持：

> Ours 的 synthetic events 缩小了下游 real-event perception 的 domain gap。

不要把某个网络的性能提升直接说成“所有真实任务都更好”。
