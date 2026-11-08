# 原任务 GT 阶段标记是否控制机器人：独立只读审查

## 结论

**在当前事件运行器上，GT `ee_distance → stage` 没有进入 actor 输入、动作缩放、IK、抓手动作或底层 PID。它用于原任务奖励与提前终止。** 原任务内部将阶段写为 `grasping`，不等于它替事件控制器发出了闭手命令。当前证据不支持为此另开控制泄漏修复。

但它不只是一个画在报告中的标签：GT 判断会决定何时停止这一回合。因此准确表述是“事件估计驱动动作；评分与提前终止沿用原任务 GT”，不能说全流程完全不访问 GT。各方法共享同一原任务终止规则即可；保留该规则，不必因查到 GT 名称就更改实验。

本轮未连接 SSH、未运行物理、未使用 checkpoint、未修改工作树源码。审查对象是主会话从实际 upstream 取得的两个源码副本，以及本地当前运行器/adapter。

## 1. 已确认的数据流

| 路径 | 源码证据 | 对动作的作用 |
|---|---|---|
| 真实球位置 → `ee_distance` | `DcmmVecEnv.py:455–472` | 返回 `info`，不直接返回 actor observation |
| `ee_distance` → `stage` / `terminated` | 同文件 `849–853` | 在当前一次物理动作执行之后切阶段，或令回合终止 |
| `stage` → 奖励分支 | 同文件 `705–744` | 改奖励；当前 runner 只累加/保存奖励，不在线更新权重 |
| event state → `obs['object']` | `run_original.py:171–184` | 完整替换球的位置和速度字段 |
| 缺 event state → 零动作 | 同文件 `188–191` | 不把原 GT 球 observation 作为回退输入 |
| observation → actor | 原 PPO 文件 `378–398、418–433` | 拼接本体状态、球位置速度、手部状态；没有 stage/info/reward/contact |
| actor action → arm/base/hand | 原 PPO 文件 `400–416` | 仅按 `task`（Catching / Tracking）选择固定缩放，不按 stage 选择 |
| action → IK/手指目标/PID | `DcmmVecEnv.py:780–798、474–490` | 使用动作缓冲、关节状态和时间；没有球真值或 stage |
| 终止输出 → 结束 rollout | `run_original.py:220、232` | 影响可观测轨迹长度；不会成为下一次 actor 的 GT 提示 |

`NativeEnv.step()` 的执行顺序尤其重要：先 `_step_mujoco_simulation(action)`，再取 observation/info，再检查 GT stage 与终止。因此这一检查不会重写刚执行的 action；若未终止，下一次动作仍由覆盖后的 event observation 产生。

原 PPO 的 `model_act` 会把观测拆给 `obs/obs_t/obs_c`，但这些全部派生自同一 observation tensor。它没有读取 `env.stage`。类名 `PPO_Catch_TwoStage` 不能被当成“在线用 GT 切换策略”的证据。

## 2. stage 与其他 flag 的消费者

对完整 `DcmmVecEnv` 类做 AST 枚举，`self.stage` 仅见于：

- 构造器、`update_stage()`、`reset()`：赋值/合法值校验；
- `compute_reward()`：奖励分支；
- `step()`：距离检查与提前终止。

`step_touch` 来自真实接触，在 Catching 模式中用于接触记录、奖励及成功分析；它没有进入 `_get_ctrl` 或当前事件 tracker。Tracking 模式下它还可触发原任务结束，但当前任务固定为 Catching。

`reward_touch/reward_stability/catch_time` 的写入和读取都留在奖励逻辑；`arm_limit` 由 IK 可行性决定并供奖励使用。没有查到这些值绕回 actor 或用于替代事件估计闭合抓手。

当前 `genesis_adapter.py:246–250` 的 `event_state()` 只向 sensor 传时间、手部刚体变换及手关节状态，没有传 `stage`、`ee_distance` 或物理接触 flag。`_genesis_step:202–217` 中的 release 是模拟器维持/释放球的环境动力学，不是事件估计器的运动开始提示。

## 3. GT observation、噪声与 reset

上游 `_get_obs():407–436` 确实仍计算带噪 GT 球 observation，并更新 `prev_obj_pos3d`。当前 runner 随后用事件位置/速度**完整替换** `obs['object']`；两条字段均被替换，没有残留第三条球真值字段。`obs2tensor` 只取这些明确字段。

因此原 `_get_obs` 的球噪声抽样仍消耗随机数，但该球数值没有进入 actor；`prev_obj_pos3d` 也只影响下一次将被覆盖的原球速度。不能把“内部曾算出 GT”与“GT 被用于控制”混为一谈。

机器人 base/arm/hand observation 仍由原仿真本体状态加原噪声获得，这对应本实验允许保留的本体反馈。`_get_ctrl():481–486` 读取关节状态并加入原控制噪声，是原 PID 所需的机器人状态，不是球位置泄漏。

reset 保留原随机来球、重力、PID、动作延迟和噪声机制（`581–656`），设置 `stage='tracking'` 并生成初始 observation。来球参数用于建立环境；当前事件基线使用 `gate_until=0`、`gate_after_release=None`，`run_original.py:203` 的表达式不会把 release_time 用作行动延迟。保留该配置收据即可。

## 4. 本地离线核验

新增只读脚本 `audit_gt_stage_dataflow.py`，直接用 AST 编译原作者的 `_get_obs/_get_ctrl/obs2tensor/model_act/action2dict`。不导入 Genesis，不加载权重。执行了两组互相矛盾的 GT 球位置和 tracking/grasping 阶段；保留同一事件 observation、本体输入及随机数状态。

核验通过：

- 完整覆盖 object 后，actor 的30维 observation tensor 逐元素完全一致；
- spy actor 记录的接口只查询 `task`，从不查询 stage/info/contact；
- 对给定动作与同一噪声，原 `_get_ctrl` 输出逐元素完全一致。

spy actor 仅用于证明数据接口，不是原 checkpoint 的新性能实验。结论来自完整源代码消费者审查与接口探针共同支持。结果保存为 `gt_stage_dataflow_audit.json`。

源代码 SHA256：

- `DcmmVecEnv.py`：`085b45272f111e87b18406ca7e17b32e969f1f328dc62758950d4e39a75e759d`
- `ppo_dcmm_catch_two_stage.py`：`1777eadfd71dc589113ab782c027b39225bf70ee80512c178d8e8a2002a6c130`

当前最小行动：保持原 task/scorer；在交付说明写清 GT 只用于环境、评分和提前终止。后续窗口适配及事件对照继续依据已锁定配置进行，本项不需要修改 actor/阶段逻辑。
