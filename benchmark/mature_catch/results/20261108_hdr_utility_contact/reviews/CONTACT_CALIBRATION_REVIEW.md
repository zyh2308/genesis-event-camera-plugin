# 共用 Genesis 抓持接触场景：独立审查与最小校准方案

本审查只读源码和已完成结果，不连接服务器、不修改代码或物理参数。用户已授权校准所有方法共用的接触场景；原策略权重、冻结事件核心和最终评分保持不变。参数选择只看原项目含噪 GT 参考，不能查看 Ours/V2E 输赢后挑参数。

## 1. 当前证据支持校准，但还没有证明摩擦就是唯一原因

`remote_queue_g3/018_live_status.response.json` 中六条新 GT 参考全部正常退出：

| Case / seed | 接触阳性的控制步数 | 最后一步接触 | 最终成功 |
|---|---:|---:|---:|
| h01 / 2026110821 | 2 | 无 | 否 |
| h02 / 2026110822 | 2 | 无 | 否 |
| h03 / 2026110823 | 28 | 有 | 是 |
| h04 / 2026110824 | 27 | 有 | 是 |
| h05 / 2026110825 | 28 | 有 | 是 |
| h06 / 2026110826 | 15 | 无 | 否 |

因此当前基线为 **3/6**。六条均触球，三个失败表现为接触后未在终点保留球。接触阳性控制步并不等于整步持续接触，不能据 15 步就声称连续保持了 0.6 秒。该结果足以支持审查接触动力学，而不能独自区分摩擦、碰撞柔顺性、抓手几何、控制动作与姿态等原因。

此六 seed 已被查看，并将用于选接触参数，应转为**场景开发/校准 bank**；后续不能再称它是参数冻结后的独立测试。

## 2. 源码和真实应用参数必须分开审计

### 2.1 当前 adapter

`benchmark/mature_catch/genesis_adapter.py` 当前读到：

- **49–50 行**：外步长来自原 MuJoCo 模型，目前 0.002 s，`substeps=1`；启用 self collision、neutral collision、MuJoCo compatibility。
- **53 行**：MJCF 导入保留 `default_armature=0.0`，使用 convex hull。不要在本轮接触校准中改变几何或控制器。
- **77–92 行附近**：碰撞几何映射包含 metadata 名字优先的匹配。部分 mesh metadata 名字重复，不适合据此恢复原材料参数。本轮仅登记，不扩大成映射重构。
- **110–116 行附近**：质量与原模型相符的检查是有用的，但不验证摩擦和有效接触 solver 参数。
- **188–192 行附近**：原控制力每 0.002 s 设置一次，再调用一次 `scene.step()`。使用 substeps2 时仍保持该外层控制更新，不能顺便把策略或 PID 频率翻倍。
- **224–233 行附近**：评分接触按原几何 ID 的手部区间分类。它用于同类接触评分，不应被解释成每个重复 mesh 的材料 ID 已准确对应。

### 2.2 Fresh h01 原始 MJCF

来源为 `019_contact_api.response.json` 中实际运行的 `fresh_state_reference/cases/h01/state_reference/migration_audit/genesis_task.xml`，不是猜测的原项目默认值：

- **XML 319 行**：球质量 0.035 kg，惯性明确给出。
- **XML 320 行**：球 free-joint damping `0.014969252352111547`；本轮保持不变，并记录 runtime 值。
- **XML 322 行**：球半径 0.04 m，`friction="0.95 0.3 0.1"`，`condim="4"`，`solref="0.001 1"`，`solimp="0.998 0.998 0.001"`。
- **XML 213–306 行**：手指关节/碰撞 mesh；手指 geom 未单独声明 friction，不能把视觉 RGBA/material 当机械摩擦。
- **XML 208–209 行**：palm 的 `hand_link` visual/collision mesh 直接挂在 `link6`，并没有独立 palm body。
- **XML 201–207 行**：同一 `link6` 还挂腕部和 camera mesh，不能简单把 link6 的所有几何都加摩擦。

Genesis 的 MJCF parser（实际包 `genesis/utils/mjcf.py` **620–621 行**）把 friction 导成 `mj_geom.friction[0]`，并把 solref/solimp 拼成 7 维参数。因此 **.3/.1 两个旋转摩擦项不能直接宣称已由该 Genesis 导入路径实现**。原始 `condim=4` 也不能代替实际 contact solver 的 readback。

### 2.3 有效 solref 和 substeps

实际 Genesis `engine/solvers/rigid/rigid_solver.py`：

- **191–215 行**：`TIME_CONSTANT_SAFETY_FACTOR=2.0`，导入 time constant 小于 `2*substep_dt` 会被钳制。
- **1043–1084 行**：构建几何数据时 sanitize 后写入 live `geoms_info`。
- **2864 行起**：`get_sol_params(...)` 读取 live 参数。
- **3178–3179 行**：`get_geoms_friction(...)` 读取 live 摩擦。

所以 XML 中的球 timeconst 0.001 s 在旧 2 ms/substeps1 设置下可能有效为 **0.004 s**；substeps2 时可能成为 **0.002 s**，必须真实读回。手部默认 timeconst 与球–手的参数混合也会影响接触，不能只看球单侧数值。

因此 substeps2 不应被宣传成“只改善数值精度、接触刚度完全不变”：若 clamp 改变，它同时改变了有效接触柔顺性。保持该作用、如实登记即可，不需要再加入一个新的 solver 参数搜索。

## 3. 必须记录的 runtime 审计

每个 profile 构建后、每次 reset 应读回并保存：

1. 所有实际 collision geom 的 global ID、body/link ancestry、type、局部 pose、selected-hand 状态；白名单不含 object、floor、wheel 或 camera。16 个手指碰撞体；若协议含 palm，另外明确选择 link6 上的 hand_link collision mesh，通常共 17 个。
2. 各 geom 的 Python friction、solver friction、friction ratio、两者乘积，以及球–手接触实际使用的 friction 混合规则/可读的接触值。**不要假设两侧取 max、min、均值或乘积。** 若混合规则使手侧提升无效，应先报告其无效性，再决定是否需要方向性更改。
3. 所有 selected 与 unselected geom 的 build/reset 前后 7 维 live sol_params；外 dt、`substeps`、真实 solver `_substep_dt`、最小/默认 time constant 和 solver iteration/tolerance。
4. 质量、惯性、dof damping、几何、PID/actuator 配置和原权重哈希保持原值的检查。只读 getter 不额外抽取随机噪声。
5. 评分输出完全保留原 success/truncated 与 terminated 标记，再使用同一个既定完成判据：到原任务时限、无失败终止、有真实手部接触且末步接触。

已有 `contact_calibration.py` 的方案方向正确：从 scene.build 后 live baseline 取得 μ，单次因子作用于不可变 baseline，reset 不重复相乘；Python/live μ 一致、ratio/sol_params 不变、非选中 geom 不变均有检查。调用者还应断言 object ID 不在白名单，并存储 body/geom 白名单证据；helper 自身不能从一份任意 ID 列表证明“手部”语义。

## 4. 固定三个候选，不依据事件方法结果选择

本轮最小矩阵如下。球、地面、轮子、其他机械臂链接、质量、damping、原权重、事件核心和 scorer 全部不动。

| Profile | 内部 substeps | 手侧实际 baseline μ 的倍数 | 目的 |
|---|---:|---:|---|
| baseline（已完成） | 1 | 1.0 | 保留原 3/6；不可删除 h01/h02/h06 |
| hand_friction1p5 | 1 | 1.5 | 单独检查抓持表面摩擦敏感性 |
| substeps2 | 2 | 1.0 | 检查较细积分及对应的有效接触参数 |
| combined | 2 | 1.5 | 检查两者是否需要共同作用 |

新增运行 **3 profiles × 同6个 seed = 18 个实际 GT-only 回合**。每条都保存，不因失败更换 seed，不用感知误差或 Ours/V2E 完成数作选择。μ×1.5 是有限的非负 Coulomb 参数敏感性测试，不违反该模型的参数域，但未经实测不能称为实物手材/球材标定。

选择规则应在看候选结果前固定：主要看最终完成数，完全同分时选最少改变物理设置的配置。建议 tie 顺序 baseline → substeps2 → hand_friction1p5 → combined；次要接触/滑移指标只解释现象，不临时改变排序。保留原动力学参数的 baseline 若并列，应优先保留。

同时检查球被“粘住”的伪成功、明显穿透、异常接触冲量、数值爆炸、关节/执行器异常。μ 变大后完成数上升，本身不保证更贴近真实机器人。

## 5. 最少新增独立验证

参数选择后锁定 profile，再注册**至少6个未被本研究查看过的 seed**，左右各3个，来球分布与六条校准 bank 相同。最好同这6个新 seed 跑 **chosen＋baseline 共12条 GT-only**，这样可以检验校准改善是否离开开发 bank 后仍存在，而不依赖不同 seed 分布的旧结果。

建议把 5/6 作为这一步可行性目标，6/6更好但不要求十全十美；六个样本不构成精确成功率估计。若独立 bank 仍不稳定，保留为压力测试，不继续根据 Ours/V2E 赢输增加摩擦扫描。

通过后，所有事件方法共享这一个冻结后的 contact profile，再使用另一批未参与场景选择的来球做配对。用于校准/验证物理的 bank 不能又被称作未见过的正式测试 bank。

## 6. 不阻塞本轮的资产审计附记

原 XML 中 35 g、半径40 mm的球保留了 `7.33516e-5 kg·m²` 的各向同性惯性。均质实心球对应约 `2.24e-5`，原值甚至高于质量完全位于该球体范围内时的单轴上界 `m*r²=5.6e-5`。这说明当前资产参数不能被宣称为真实球的物理标定。按既定三个候选保持原惯性、不扩大本轮搜索是合理的；将质量/几何/惯性原值和这个边界登记即可。

## 来源

本地只读证据副本位于本目录：`018_live_status.json`、`019_contact_api.json`、`020_contact_solver.json`、`021_contact_details.json`。它们来自主会话已有结果与源码读取；本审查没有自行连接服务器。

## 补充：最新接入代码的独立复核

在主会话接入后，再次只读了最新 `genesis_adapter.py`：**92–110 行**改用 actual link 的 MuJoCo body＋局部 pose 找 native collision 候选，只有最近候选全部属于同一 hand/non-hand 类别时才通过。这解决了本次材料白名单不能依赖重复 mesh metadata 名的要求，没有扩大原评分映射。构建后应用 profile（107–110 行），reset 后再次恢复 profile（175–179 行）；**52 行**仅改变内部 substeps，**140–142 行**记录外 dt 和请求的内部 dt。

`contact_calibration.py` 的 **72–77、103–123 行**使用不可变 live baseline μ，不会每次 reset 再乘1.5；非白名单摩擦、ratio 和 live sol_params 均检查未变。该模块的 μ 范围检查只是安全限界，不能替代真实材料标定。需在实际运行审计中确认16个手指加是否含palm的白名单，并补球–手实际接触摩擦合成规则。`effective_substep_dt_s` 当前记录由外 dt/请求 substeps 计算，若需要宣称 solver 确认值，还应读回 solver 真正使用的值。

主会话已完成的部分候选结果出现 `success=True`、`terminated=True`（substeps2/h01）。按当前明确判据，它仍为 `verified_catch=False`，不能因原环境 success 标记就改计为成功。当前报告给出的完全同分选择规则属于本审查建议；若尚未在候选运行之前登记，应如实称为分析规则，不能补称预注册。
