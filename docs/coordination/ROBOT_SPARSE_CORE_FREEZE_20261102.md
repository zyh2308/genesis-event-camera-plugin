# 原稀疏重建与 CUDA 再冻结（2026-11-02）

**已达到用户指定的最低冻结条件：完成一集由事件感知驱动的严格物理抓取。冻结基线为 `robot-sparse-cuda-r2-20261102`。**

冻结分支：`freeze/robot-sparse-cuda-r2-20261102`。完整描述在 `FROZEN_BASELINE.json`，精确提交由该分支固定。主分支与旧冻结标签不变。

## 验收结果

| 项目 | 结果 |
|---|---|
| 原稀疏生成路径 | 100 Hz 原生线性 HDR 端点 + 深度/分割/位姿，经原几何重建 → DVS；没有高频补图参与生成或训练 |
| 已归档像素定位验收 | 主开发集 581 次查询，中位数/P95 由 8.18/106.72 px 降至 2.33/5.77 px；另外 193 次归档查询为 2.20/5.54 px |
| g1 CUDA | RTX3090、V100S 分别通过 33 项检查、4 段原缓存和 16 次固定模型查询；支持掩码分歧 0，事件位置/极性/数量一致 |
| CUDA 生成对定位的影响 | 0 px；CPU/CUDA 推理变化分别不超过 0.000212/0.000205 px |
| 事件驱动严格抓取 | seed 2026110202，0.50 m/s，y=+55 mm，10 ms 观测延迟 |
| 连续合格保持 | **0.500 s，即连续 1000 个 0.5 ms 物理步** |
| 合格窗口最低真实底面净空 | **80.082 mm**（门槛 80 mm） |
| 合格窗口最大相对滑移 | **3.202 mm**（限制 10 mm） |
| 接触与支撑 | 合格窗口持续双侧接触，桌面支撑样本为 0 |
| 因果与独立复算 | 未来观测违规 0；500 条 1 ms 轨迹记录的合格标记与几何复算一致 |

该回合的模型始终为稀疏候选事件训练的固定 checkpoint，SHA256 `dda51f2d5a9665295eeeeccd98682183946f8b080ceb610f82c2baeec3f8d896`。在线用共同 Direct-HDR **评估事件**驱动模型，随后由模型状态和机器人自身反馈控制；目标真值仅供评分。这不是高频 HDR 绕过候选的生成或训练，也不计入真值控制成功。

## 已固定的修改

核心版本仍为 `6ac417c7a03e125f40ebf967796bf28191d55859` 的已验收内容：像素覆盖面积处理、精确端点、随时间连续的 log 亮度融合、保留未知标记，以及 Torch/CUDA 一致实现。支持门槛仍为 0.995。CUDA 复核之后没有再修改核心源码。

R2 中另行修正了下游衔接：用过去观测和公开滑动标定联合估计纵向位置/速度，闭合前更新速度参考，补机器人自身位置反馈。最后修正了抬升保护：运动中检查夹口完整性；机器人稳定后才建立负载下的微小漂移基线。初始探测抬升、60 µm 漂移门槛、10 ms 违规持续要求，以及独立物理成功评分均保留。

保护器的五个**夹爪状态受控检查**通过：正常运动负载、稳定夹持、稳态松脱、空夹和不对称。机器人自身跟随另在 g1 Genesis 做了正负 65 mm 的四次检查，关键时刻误差由约 3.5 mm 降至 0.5–0.7 mm，峰值速度低于 0.4 m/s。这些检查不计入事件抓取回合。

原保护版本 K 在仍然双侧接触、净空约 93.8 mm、滑移约 2.7 mm 时，因微小关节变化主动降下方块，只有 0.1595 s 合格保持。最终版本 O 完成 0.5 s。两次渲染 GPU 不同，原报警前的实测目标位置最大差约 0.146 mm；报告保留该差异，不宣称位级完全相同的反事实。

## 使用方式

事件生成须使用 `benchmark/robot_closed_loop/configs/event_sparse_reconstruction_20261102.json` 中的构造参数，显式启用 `coverage_time_log_linear`。只更新代码而继续使用旧合成参数，不等于采用这次冻结基线。构造器历史默认值和独立 strict-sampled 路径没有被静默替换。

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

R2 复跑使用新控制配置 `shared_control_r2_verified_20261102.json`，其余 scene、模型与严格 scorer 保持原设置。原样命令和运行配置见 `evidence/R2_plan.json`；在新独立输出目录运行，不覆盖原结果。

## 证据与解释范围

- `evidence/strict_event_grasp_audit.json`：独立时间、来源、几何、滑移、源码及模型核对。
- `evidence/qualified_hold_trace.csv`：合格保持窗口的 500 条记录；原评分在 0.5 ms 的全部物理步执行。
- `evidence/development_rollout_ledger.csv`：13 次开发运行，保留失败、重复 seed 和配置差异；**不能把它们合并成独立成功率**。最终配置当前有一集严格成功。
- `evidence/loaded_guard_controls.json`：保护器正负例；`evidence/all_run_manifests/`：全部开发运行的大产物哈希索引。
- 原始 raw events、完整状态/控制 trace、录像、运行快照在 g1：`/home/zhaoyuhan/Eventbased_WAM/runs/robot_R2_loaded_guard_20261102o`。该回合 2 kHz raw events 为 657,370 条。

最终回合 exit 0，完整墙钟约 317.24 s。导出的录像来自 33 张真实 Genesis RGB 记录画面，0.5 倍速播放；不是实时吞吐量展示。源 GIF 与全部原始数组仍保留。

本次完成的是原刚体机器人/方块任务的核心再冻结及功能 smoke。正式生成器排名、多 seed 成功率，以及新柔性/流体场景的感知效用仍需各自实验。传感器仍是理想、未标定响应。
