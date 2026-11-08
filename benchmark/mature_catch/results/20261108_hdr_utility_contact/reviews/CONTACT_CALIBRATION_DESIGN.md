# Genesis 接取接触参数校准方案

日期：2026-11-08。状态：源码审查完成；benchmark 独立参数模块和纯测试已编写，尚无新增物理运行结论。

## 目的与固定项

当前开发结果中，多数失败已经发生在接触之后。该实验应先回答“接触模型移植后，原策略能否稳定留住球”，再比较事件来源。校准方案由所有事件方法共用；原策略权重、冻结事件核心、估计器、初始 RGB-D、传感器帧率、控制频率和成功判据均不修改。不允许物体附着、GT 控制替代或按事件方法分别调物理参数。

本方案只涉及 benchmark 的 Genesis 场景构造参数。当前共享工作树基线由主会话指定为 `5e695ce`；这里没有改动该工作树。

## 已从本地源码确认的接口

本地包根目录：`C:/Users/zhaoyuhan/miniconda3/envs/genesis/Lib/site-packages/genesis`；安装元数据为 `genesis-world 1.2.2`。版本号与服务器一致不代表源码完全一致；实施前须由主会话核对服务器实际文件及接口。

| 事项 | 本地源码证据 | 对实施的含义 |
|---|---|---|
| 单个碰撞几何体设置摩擦 | `engine/entities/rigid_entity/rigid_geom.py:343–352`：`RigidGeom.set_friction(value)` 同时更新 `_friction`，若 solver 已构建则调用 `solver.set_geom_friction(value, idx)` | 可以精确选择夹爪接触几何体，不必覆盖整台机器人 |
| 活跃 solver 读回 | `engine/solvers/rigid/rigid_solver.py:3178–3179`：`get_geoms_friction(geoms_idx)`；`:3215–3222` 提供单个/批量 setter | 验收须读回实际 solver 数值，不能只检查 Python 属性 |
| 整实体 setter 范围过大 | `engine/entities/rigid_entity/rigid_entity.py:4441–4464`：遍历实体所有 link；文档声明稳定范围 `[0.01,5]` | 本任务禁止 `robot.set_friction(...)`，避免影响底盘轮子、地面及其他环境几何体 |
| 接触合成规则 | `engine/solvers/rigid/collider/contact.py:329–338`：每侧 scalar friction×ratio，然后 `max(mu_a,mu_b,0.01)` | 1.5 倍应按真实接触对的有效摩擦来解释；不能把两侧相乘或平均 |
| MJCF 摩擦导入 | `utils/mjcf.py:620`：只读取 `mj_geom.friction[0]`；`:621` 单独读取 `solref+solimp` | 本地导入的是滑动摩擦标量，不代表完整保留 MuJoCo 的扭转、滚动摩擦或 `condim` |
| material 覆盖关系 | `engine/entities/rigid_entity/rigid_entity.py:2391–2395`，另见 `2645–2649`：material.friction 非空时覆盖文件，否则使用导入值；`engine/materials/rigid.py:82` 默认 `None` | 需要同时记录导入值、material 覆盖值、运行时有效值 |
| 子步长 | `engine/simulator.py:106–108`：`substep_dt=dt/substeps`；`:281–288` 每次 scene.step 执行指定次数的内部子步 | 保留外层 `dt=.002s`；`substeps=2` 仅将内部积分改为 `.001s`，外层控制与事件时间不改 |
| 约束时间常数下限 | `engine/solvers/rigid/rigid_solver.py:191–214,273`：下限 `2*substep_dt`，低于下限的正时间常数会被提升 | 子步2还可能改变实际接触柔顺性，须存档修改前后真实 `sol_params`；不能将结果全部归因为更精细积分 |

实际 adapter 的证据：`benchmark/mature_catch/genesis_adapter.py:49–54` 当前 `substeps=1`，完整 MJCF 作为一个 entity 导入；`:75–88` 已有 Genesis geom 到原 MuJoCo geom 的对应；`:228–233` 用实际 Genesis 接触映射原任务手部/物体接触分类。因此整个 robot entity 的摩擦修改会波及非抓取部件，不能使用。

### 原生 MuJoCo 与导入后的区别

MuJoCo 的 geom 摩擦有滑动、扭转、滚动三项；相同 priority 的普通碰撞对按分量取最大值，较高 priority 可主导参数，显式 contact pair 可另行定义。`condim` 决定是否使用扭转/滚动约束。参见 [MuJoCo 官方接触参数说明](https://mujoco.readthedocs.io/en/3.3.3/modeling.html#contact-parameters)。上表 Genesis 标量导入不能证明这些机制全部等价。必须先取得实际原 MJCF 的三项 friction、condim、priority、显式 pair，再判定原任务的保球能力有没有依赖导入时丢失的机制。

如果原任务使用 `condim=4/6` 和非零扭转/滚动摩擦，而 Genesis 实际仅保留滑动摩擦，应在记录中明确这是移植后的共同接触参数校准，不宣称逐项等价复刻 MuJoCo。此次不为补齐这一差异而修改 Genesis 引擎。

## 最小四个配置

| 配置 | 夹爪/物体目标几何摩擦倍率 | 内部子步 | 外层 dt | 用途 |
|---|---:|---:|---:|---|
| `baseline` | 1.0 | 1 | 2ms | 复现原有场景 |
| `hand_friction1p5` | 1.5（仅手） | 1 | 2ms | 区分保球摩擦不足 |
| `substeps2` | 1.0 | 2 | 2ms | 区分碰撞积分/约束时间分辨率 |
| `combined` | 1.5（仅手） | 2 | 2ms | 检查两者交互 |

倍率是针对每个选中碰撞几何体的**导入基准值**；每集重置后设置 `baseline_mu[geom] * ratio`，不能在当前值上重复乘，避免多集累积。拒绝超出稳定范围的值，不静默截断。保持 `friction_ratio` 为原值或显式记录，防止绝对值与 ratio 双重放大。

### 碰撞体白名单与地面保护

1. 先用实际原模型的名字、body 祖先及现有 `geom_map` 列出球和 LEAP 手的碰撞几何体。将清单存档。只允许这些经过核对的 collider，不依赖宽泛字符串或整数区间静默匹配。
2. 手掌/手指是否都纳入，按实际接触记录与 LEAP 子树决定；相机外壳、前臂、轮子、底盘、地面全部排除。
3. 球的摩擦并非仅用于手—球接触。若将球也乘1.5，在 `max(mu_ball,mu_floor)` 规则下可能改变球—地面有效摩擦，即便地面参数没动。因此须计算修改前后球—地面、球—非手碰撞对的有效摩擦。
4. 若球系数变化会影响非抓取接触，优先采用**手部白名单1.5倍、球保持基准**，并在配置名/文档中写清 `hand_only`。不能为达到“手与球都1.5倍”的形式要求而暗改地面交互。若主会话确认应同时改球，必须显式记录这一物理变化，并仍保持地面/轮子参数本身逐项不变。
5. 所有未选中 collider 的有效 friction、所有质量/惯量、碰撞网格、contype/conaffinity 和 DOF 摩擦损失保持逐项一致。`set_dofs_frictionloss` 是关节摩擦，不在本方案内。

## 实施前服务器必须提供的字段

主会话获取并存档以下 JSON，不需要额外物理运行：

- Python、Genesis、MuJoCo、torch 版本；实际 Genesis 包路径；下述源码文件 SHA256。
- 原生 MuJoCo：`opt.timestep`、integrator、cone、solver、iterations/noslip_iterations；每个 geom 的 id/name/body/parent ancestry/type、contype/conaffinity、friction三项、condim、priority、solmix、solref、solimp；显式 contact pair 与 exclusion。
- Genesis：每个实际碰撞 geom 的全局/实体局部 id、name、link、对应原 geom、scalar friction、material.friction、运行时 friction_ratio 与 solver friction 读回；geom sol_params；模拟 dt/substeps/substep_dt；`enable_mujoco_compatibility` 及约束 solver 有效配置。
- 实际服务器 `RigidGeom.set_friction`、`RigidSolver.get_geoms_friction`、MJCF摩擦解析、接触摩擦合成、`_sanitize_sol_params` 的源码摘录。若缺 getter，主会话确认受支持的只读后端字段转换方式，不能猜 Taichi/Quadrants 存储布局。
- 当前夹爪—球、球—地面碰撞对实际有效摩擦，基准与拟议profile逐对比较；未选中 collider 差异列表必须为空。

本地证据 SHA256：

```json
{
  "engine/entities/rigid_entity/rigid_geom.py": "0a57b34719c7ea091db6a3e3d675d2b044062eb7c630b1a1f3bd3171b2edab11",
  "engine/solvers/rigid/rigid_solver.py": "cf664fdc9bc7b7fda5560f12ef4bb56cd4837848d1449327891d2058b5117c7e",
  "engine/solvers/rigid/collider/contact.py": "24a66e7c31d5e0b21f0e46ea2875a4982b708b64e067f2f3ece678694c11f2a3",
  "utils/mjcf.py": "ef7320cb1f050360c106a1b741ee832094e94826a05f054d674a5541940f0500",
  "engine/simulator.py": "b385b363dfb56512ae189d3ad86fd9bd75acc7209e2cf4b9c934ec76e98073da",
  "options/solvers.py": "ccea060d5769b02776954cd5f6b6c7a07922c234cecf19a3473e013139263090"
}
```

## 小规模验证顺序

1. 四profile先检查无渲染 metadata：白名单、摩擦读回、外层clock、质量/惯量、policy/corehash不变；连续reset不累积。`substeps=2` 的有效solref单独报告。
2. 用同一组已知开发seed与原策略的共同真值/原任务参考输入，检查是否接触、接触后保留、最终成功。该步骤隔离物理可行性，不作为事件结果；保留全部成功和失败。
3. 优先选择能恢复共同参考可行性、未改变非抓取碰撞、额外参数最少的profile。不得按 Ours 胜过 V2E 的某一个seed来选择。
4. 冻结profile后，先在已知开发seed上为 Ours/V2E各跑相同条件，确认输入/时钟正常，再在新的位置、速度、seed上配对检验；全量保留。报告分开列共同物理参考、Ours、V2E，接触率与最终留球成功率分开。
5. 如果所有profile都不能改善共同参考，暂不扩大摩擦扫参。检查接触时物体速度、手指闭合姿态、法向力、滑移及原策略抓取阶段触发时刻；这些问题不能靠无限增加摩擦掩盖。

建议每集保存：首次接触时刻、首次接触相对位置/速度、手指闭合状态、接触对、接触持续时间、接触后最大滑移、最后时刻接触、原成功/终止标志、现有严格成功判定、实际profile/readbacks及演示录像。球坐标和接触力只用于离线诊断/计分，不能额外提供给事件控制器。

## 实施边界

当前glare八集仍属于原物理profile；后续校准结果必须另起目录并保留原profile对照。

## 主会话取得 g3 实际源码后的更新

主会话已取得并保存 `../G3_CONTACT_API.json`、`../G3_CONTACT_SOLVER.json`。后者的实际服务器 `rigid_solver.py` SHA256 为 `cf664fdc9bc7b7fda5560f12ef4bb56cd4837848d1449327891d2058b5117c7e`，与本地逐字相同。已确认实际运行包的单 geom setter、摩擦/ratio getter 与 live sol_params getter 均存在；服务器 `utils/mjcf.py:620` 同样仅导入第一项。

实际归档 `genesis_task.xml` 中：

- 球：`friction="0.95 0.3 0.1"`、`condim="4"`、`solref="0.001 1"`、`solimp="0.998 0.998 0.001"`、sphere `size="0.04"`；body质量 `.035kg`。
- 手部和地面没有显式 friction，最终值应以实际编译模型和 Genesis solver 读回为准；MuJoCo默认滑动摩擦1.0。
- floor-wheel 有显式接触 pair，friction五项 `1.1 1.1 .005 .0001 .0001`。本校准不调整这些 pair 或任何轮子/地面 geom。
- 部分导入 metadata 重用 MCP/pip/fingertip 名称，造成 `geom_map` 的原ID重复；所以手部白名单应按实际 Genesis link、原生body祖先与精确碰撞体清单核对，不能只相信metadata名称。原任务宽分类仍可能一致，但逐指ID不是天然可靠的。

据此确定**仅手部摩擦×1.5，球保持.95**。如果球与地面原有效摩擦1.0，将球也×1.5会提升至1.425；只改手可以避免这一额外改动。

主会话随后授权只编写独立模块，已交付共享工作树中的：

- `benchmark/mature_catch/contact_calibration.py`：`profile_config(name)` 给出 `substeps` 和 `hand_friction_scale`；`ContactCalibration(robot, profile_name, hand_geom_ids)` 在build后抓取不可变基准，`apply()`在每次reset后设置绝对目标，并返回全体碰撞体的基准、设置前、设置后live参数。
- `benchmark/mature_catch/test_contact_calibration.py`：7项纯测试通过；不运行Genesis。覆盖手部选择性修改、地面/物体/轮保持不变、连续应用不累积、子步profile不触碰摩擦、未知/重复ID拒绝、Python/live值不一致拒绝、ratio/solparams意外变化拒绝。

live `solver.get_sol_params(geoms_idx=...)` 已在服务器源码`:2864–2885`确认。它直接读取后端参数，因此审计报告记录的是子步数构建后实际生效的7项参数，不是原XML副本。

没有连接服务器、没有运行GPU/Genesis，没有修改core/权重/估计器/计分。主会话负责adapter接入、白名单核对、部署和物理验证。
