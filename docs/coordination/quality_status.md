# Genesis Event Camera Quality Validation

## 迁移状态

| 项目 | 状态 |
|---|---|
| Branch | `exp/quality-cdgd-v4` |
| HEAD at migration start | `568ebb5480b487340172230fa9490b017141d339` |
| Original worktree | `/home/科研/Eventbased_WAM/repositories/genesis-event-camera-plugin` |
| Original branch / HEAD | `main` / `568ebb5480b487340172230fa9490b017141d339` |
| Original worktree dirty | `NO` |
| New worktree | `/home/科研/Eventbased_WAM/repositories/gec-quality` |
| Backup | `/home/科研/Eventbased_WAM/reorg_backups/quality20261008_003158` |
| Frozen simulator core | `e79e656ac00f7847b98db9e12bf43b093f3afff1` |
| Frozen core diff in this branch | `EMPTY / PASS` |

本次是无损工作区整理和质量验证分支迁移，不是实验执行。原工作区未提交修改为空，因此没有需要从工作区复制的未提交质量代码；`origin/main` 上已有的 benchmark、loader、baseline wrapper、readiness script、历史 protocol 和结果记录由新 worktree 完整继承。

## 旧质量验证的真实状态

- Phase 4.7 runtime readiness 已记录为 PASS。
- Real EVK4 LR/RL 完整 loader 曾使用官方 ECF filter 通过；记录了完整 `CD/events`、事件数量、首末时间戳和前后静默时间。
- Genesis/Ours 0.30 s active smoke、native HDR、独立 display RGB、segmentation coverage 和事件输出曾通过 readiness gate。
- 官方 V2E SuperSloMo smoke 曾通过，使用 `100 Hz` source、`0.0001 s` timestamp resolution、slowdown factor `100` 和固定 seed `42`。
- Phase 5 formal benchmark 已 STOP；没有有效 formal metric、aggregate、ranking 或 significance result。
- 现有历史 smoke 和 exploratory/distributional 比较不得重新标记为 formal benchmark。

## Frozen protocol 和 pair manifest

历史文件保持原样：

| 文件 | SHA-256 |
|---|---|
| `benchmark/protocol_v1.yaml` | `b1b4519446bbc821220b3502a0eb3811bf8c0baf8bd457eca022a0d082ebdd97` |
| `benchmark/protocol_v2.yaml` | `c6382e3cd94ce02c2f92d8d12a8d5604affb8a4fd9a17b0694c7609120ab28df` |
| `benchmark/protocol_v3.yaml` | `c6d0926147e3fb08b15892c1f9ee368c5d7380ad3326cca0b8d962ae1f78d763` |
| `benchmark/evaluation_pairs_v2.yaml` | `e62e7cc0fb18730c53213277ceea20f7598b7f22cca102b0695201d857b2c9b9` |
| `benchmark/evaluation_pairs_v3.yaml` | `86c9596bb3c6e79788e1ae5aa627319ade886a64c88571f4ccc5e7ede6fbc175` |

`protocol_v3` 是 formal execution 前的 readiness protocol，明确：

- CD/GD 被禁用；
- 5 ms 局部窗口只能用于 deterministic qualitative visualization；
- temporal offset search 和 first-event alignment 被禁用；
- 旧的弱统计指标不能直接作为新的论文主质量指标；
- 不得通过修改 `protocol_v3` 偷开 CD/GD；
- 新的 CD/GD 方案必须另建 `protocol_v4`；
- 当前迁移没有产生新的 formal result。

本分支暂不创建看似已冻结的 `protocol_v4` 结果。若后续已有 v4 草案，应明确标记为 `DRAFT`，并不得覆盖 v1/v2/v3。

## 已有 Real EVK4 数据引用

正式 readiness/历史 pair manifest 引用的真实动态片段为：

```text
/home/科研/Eventbased_WAM/data/real/evk4_checkerboard_translation_20260914/derived/one_round_h5/segment_01_LR.hdf5
/home/科研/Eventbased_WAM/data/real/evk4_checkerboard_translation_20260914/derived/one_round_h5/segment_02_RL.hdf5
```

`real_v2` 校准引用为：

```text
/home/科研/Eventbased_WAM/validation/simulator-quality/real_v2/calibration/evk4_real_noise_fit_v2.json
/home/科研/Eventbased_WAM/validation/simulator-quality/real_v2/calibration/evk4_real_noise_map_v2.npz
```

这些外部数据没有复制到 Git worktree；绝对路径、大小清单和引用扫描保存在备份目录中。

## Baseline 和 loader

- Ours：frozen core `e79e656`，`real_v2`，配置 `benchmark/configs/ours.yaml`。
- V2E：本地 `/home/科研/Eventbased_WAM/code/v2e_source`，配置 `benchmark/configs/v2e.yaml`；历史 core smoke 和 official SuperSloMo readiness smoke 均有记录，但不是本次正式 benchmark。
- ESIM：配置和 wrapper 保留；当前官方 build 不可用时不得用其他脚本冒充 ESIM。
- ICNS/IEBCS：无可用本地 checkout，保持 unavailable 状态。
- PECS：作为 real-to-sim、CD/GD 和下游验证设计参考，不因本次整理自动加入可执行 baseline。
- Real loader：`benchmark/io/real_stream.py`，要求官方 vendor ECF filter，并拒绝用第一事件做后验对齐。

## 本轮迁移内容

新 worktree 保留了 origin/main 中已有的：

- `benchmark/` 协议、pair manifest、replay、metrics、runner、baseline wrapper、loader 和 readiness scripts；
- `calibration/` schema 与质量验证引用；
- `genesis_event_plugin/evaluation/` 质量读取和评估接口；
- `docs/quality_benchmark_asset_audit.md` 及已有 Phase 4/5 状态记录；
- 所有历史 protocol 和 hash 文件。

本轮没有修改 `genesis_event_plugin/`，没有修改任何历史 protocol，没有复制大型 EVK4 HDF5/RAW、生成事件流、checkpoint 或临时结果。

```text
MIGRATION_PENDING: false
CROSS_BRANCH_CORE_CHANGE: false
FORMAL_BENCHMARK_EXECUTED_THIS_MIGRATION: false
SIMULATOR_MODIFIED_THIS_MIGRATION: false
```

## Stage 5 Q2 — QualityAdapter interface audit

Q2 establishes only the engineering boundary for a future v4 evaluation. It
does not create or freeze `protocol_v4`, run alignment, fit normalization,
execute CD/GD, rank methods, add a baseline, or modify the frozen simulator.

```text
Q2_INTERFACE_IMPLEMENTED: true
FORMAL_BENCHMARK_EXECUTED_Q2: false
FORMAL_METRICS_EXECUTED_Q2: false
SIMULATOR_MODIFIED_Q2: false
```

Added under `benchmark/quality_v4/`:

- `event_schema.yaml`: canonical `(N,4)` float64 `[x,y,p,t_s]` schema;
- `artifact_schema.yaml`: required provenance sidecar contract;
- `adapters/base.py`, `real_ev_k4.py`, `genesis.py`, `v2e.py`: raw artifact
  adapters and strict metadata validation;
- `tests/pecs_metric_selftest.py`: non-formal legacy metric sanity tests;
- `README.md` and `stage5_q2_quality_adapter_report.md`.

The adapter boundary is:

```text
raw artifact + sidecar -> QualityAdapter -> canonical events + metadata -> future metric
```

Real EVK4, EventGenesis, and official V2E adapters reuse existing loaders and
perform only explicit layout/unit/polarity conversion. First-event alignment,
per-method temporal offset search, normalization fitting, and parameter fitting
are forbidden at this layer. Sidecar event count and observed first/last
timestamps are checked against the loaded artifact rather than silently
overwritten.

The self-test passed identity, empty-input, timestamp-unit, polarity, and
determinism checks. This is not a formal result. Historical `protocol_v1`,
`protocol_v2`, and `protocol_v3` remain unchanged; any formal CD/GD design
must be a separately reviewed `protocol_v4`.

## 下一步

等待人工审阅本分支和 Q2 报告后，才可以进入“设计 `protocol_v4`”阶段。
`protocol_v4` 应独立定义配对事件流、CD/GD、轨迹/时间同步、数据来源和
V2E 公平执行条件；不得改写旧 v3，也不得把本次 Q2 接口审计、迁移或历史
readiness smoke 当成 formal benchmark。
