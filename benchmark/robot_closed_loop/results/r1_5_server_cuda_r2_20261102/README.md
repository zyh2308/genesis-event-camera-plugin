# g1 稀疏重建 CUDA 验收与 R2 抓取复核（2026-11-02）

**服务器 CUDA 已通过；R2 抓取尚未通过。本报告不宣布全流程冻结。**

代码固定为 `6ac417c7a03e125f40ebf967796bf28191d55859`。CPU 定位修复之后，本轮未再修改核心文件，没有用高频 HDR 替代稀疏重建。核心源码逐文件哈希与已验收 CPU/CUDA 版本一致。

## 与原定位验收的关系

此前 9 条固定开发轨迹的 581 次查询，定位中位数/P95 从旧 Ours 的 8.18/106.72 px 降到 2.33/5.77 px；另外 3 条归档轨迹 193 次查询为 2.20/5.54 px。候选生成和训练仅用原有 100 Hz HDR 端点、深度、分割和物体/相机位姿。Direct-HDR 只作共同评估源。本轮检查 CUDA 是否保留该已验收路径的输出，不把 16 次一致性查询冒充重做 774 次精度实验。

## 服务器 CUDA 实测

| 项目 | RTX 3090（GPU0） | V100S（GPU2） |
|---|---:|---:|
| 受控检查 | 33 项通过 | 33 项通过 |
| 原 Genesis 100 Hz 缓存片段 | 4 段、16 个区间通过 | 同 4 段通过 |
| 每种后端事件总数 | 19,725 | 19,725 |
| 事件位置、极性、数量 | 一致 | 一致 |
| 支持掩码分歧 | 0 | 0 |
| 事件时间最大差 | 0.260 µs | 0.260 µs |
| 固定模型，CUDA 生成导致的定位变化 | 0 px（16 次） | 0 px（16 次） |
| CPU/CUDA 模型推理最大变化 | 0.000212 px | 0.000205 px |

两次完整三阶段进程均 exit 0。环境为 g1 既有 Eventbased_WAM，torch 2.5.1+cu121；没有修改共享环境。与用户先前授权的本地 RTX 2050 检查结论一致。计数按 CUDA 后端独立汇总，不把 Torch CPU 与 CUDA 两份重复相加。

## R2：独立事件输入驱动真实物理抓取

固定已验收稀疏事件训练出的 checkpoint，SHA256 `dda51f2d5a9665295eeeeccd98682183946f8b080ceb610f82c2baeec3f8d896`。在线使用独立 Direct-HDR 事件；没有将目标真值交给事件控制器，也没有把 Direct-HDR 在线评估说成候选的稀疏生成路径。

| seed | 速度 | 横向位置 | 接触 | 抬升 | 严格保持 |
|---|---:|---:|---|---|---|
| 2026110200 | 0.30 m/s | 0 mm | 是 | 否 | 0 s |
| 2026110201 | 0.40 m/s | −55 mm | 是 | 否 | 0 s |
| 2026110202 | 0.50 m/s | +55 mm | 是 | 否 | 0 s |

严格成功为 **0/3**。这是三个开发验证回合，每档一个，不能作为可靠成功率估计。原评分仍要求连续保持 0.5 s、真实底面净空至少 80 mm、滑移不超过 10 mm、双侧接触且无桌面支撑；没有降低标准。

逐条 trace 复核：观测时刻均不晚于控制时刻减 10 ms，因果时间违规为 0。接触后，夹爪主要因开口落在物体夹持范围之外而未满足抬升条件，合格样本为 0。仅“接触到了”不等于夹住。

## 唯一补充对照：速度窗口 50 ms → 100 ms

同模型、同 seed 2026110200、同场景/评分，仅将因果位置历史窗口从 50 ms 延长至 100 ms。未改核心、训练、传感器或抬升门槛。

闭合时刻由 0.69 s 推迟到 0.81 s；仍在约 0.860 s 接触，未抬升，严格成功 **0/1**。这证实速度处理会改变动作，但不足以恢复抓取。接触时夹爪已合得过窄，不能把结果解释成“速度窗口已修好”，也不能据此断言稀疏生成器仍存在新漏洞。此回合是已看过 seed 的针对性对照，不与原三个回合合并为独立成功率。

## 冻结建议与可直接复用的配置

已准备 `benchmark/robot_closed_loop/configs/event_sparse_reconstruction_20261102.json`：显式选 `coverage_time_log_linear`、Torch/CUDA、0.995 支持门槛和近似稀疏模式。构造器历史默认值不变；严格采样模式保留，不冒充本次稀疏修复。

建议把已验收的事件生成核心与未成功的抓取控制分别管理。用户已被明确询问：先冻结事件生成核心，还是等至少一次严格抓取成功再冻结。在得到该方向性答复前，状态是 CUDA 已验收、冻结决定待定；旧冻结标签和主分支不变。

```python
import json
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.noise_model import DvsNoiseConfig

profile = json.load(open(PROFILE_PATH, encoding="utf-8"))
plugin = GenesisPhysicsEventPlugin(
    noise_config=DvsNoiseConfig(**profile["noise_config_kwargs"]),
    frame_provider=frame_provider,
    motion_state_provider=motion_state_provider,
    **profile["plugin_kwargs"],
)
plugin.attach(scene, camera)
```

该配置验证范围是 100 Hz 端点的刚体机器人/方块场景；新柔性/流体场景当前仍是场景探索，并非该定位验收的扩展结论。传感器仍为理想、未标定 profile。

## 归档、成本及资源收尾

完整数据在 g1 的三个唯一运行目录：

- `/home/zhaoyuhan/Eventbased_WAM/runs/robot_cuda_r2_server_20261102a`：RTX 3090 CUDA 与三集原配置 R2。
- `/home/zhaoyuhan/Eventbased_WAM/runs/robot_sparse_cuda_v100_20261102b`：V100S CUDA。
- `/home/zhaoyuhan/Eventbased_WAM/runs/robot_R2_velocity100_20261102c`：单集 100 ms 速度窗口对照。

各目录 `SHA256SUMS.server` 索引全部保留产物，含 raw events、轨迹、录像、运行代码与日志；仓库纳入轻量结果、逐集数据和索引。下载归档的原始文件哈希全部核对；大型源缓存继续只读共用。

三集原 R2 记录墙钟 8553.25 s，其中 7604.60 s 是用户要求暂停期间，扣除后活跃墙钟 948.65 s；不能把暂停时间当计算成本。第三集原始耗时字段保留，并另列扣除值。100 ms 单集对照墙钟 300.02 s，exit 0。

GPU0 的本任务已结束并释放预约；GPU2 按用户要求在任务完成后恢复低显存待命，计算占用为零。没有访问 g3，没有停止他人任务。
