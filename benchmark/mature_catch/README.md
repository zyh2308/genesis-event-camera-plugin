# Catch It! 任务复用与事件输入适配

复用 [Catch It!](https://github.com/hang0610/Catch_It) 的 RangerMiniV2 移动底盘、XArm6、LEAP 手和原两阶段 PPO 权重。上游固定提交：`b6af28c67586fc029f3ab43faf5d787e7d3a1609`；接取 checkpoint SHA256：`81585772adf9b368e299875a3dce29247e23c641d8294295bec07dfb6450aaf6`。

本目录是任务适配与验证代码，不修改冻结事件核心。实际物理后端为 Genesis 1.2.2；原 MuJoCo 模型仅供原控制器的运动学与 IK 使用。Genesis 后端运行时禁止 `mj_step`，碰撞由 Genesis 计分。

## 当前协议

- 用户批准首版使用原项目球体，固定半径 40 mm。
- 原相机安装、640×480、垂直视场 58°，原控制器和动作缩放、延迟、25 Hz 策略频率不变。
- 每集 reset 时只读一帧实际 RGB。由圆形轮廓和已知尺寸求初始三维位置，不向检测器传对象 GT、深度或分割。
- 后续状态只由事件和机器人自身状态更新；运动起始由事件观测判断，不读仿真释放标记。
- 事件来自真实 100 Hz 稀疏 Genesis scene-linear HDR 端点和冻结稀疏重建路径，响应为该冻结版本的 ideal、阈值 0.2；不是高频 Direct-HDR 替代，也不是 EVK4 标定模式。
- 任务结束与失败沿用原任务规则：超过 2.5 s 为原成功条件，掉落/非手碰撞等提前失败。与历史 Panda 方块的 strict scorer 是不同任务。
- 控制器一旦进入已估计的持握状态，本来就只按手部自身状态传播目标。开启 `--stop-events-after-grasp` 时停止此后不参与控制的传感器采样，物理和原评分继续运行。此切换不读取真实接触标记。输出 `capture_schedule.json` 记录真实采样终点。

采用“图像建立参考、事件跟踪”的输入协议，参考 [EKLT](https://rpg.ifi.uzh.ch/docs/IJCV19_Gehrig.pdf) 与 [CVPR 2023 Data-Driven Feature Tracking](https://openaccess.thecvf.com/content/CVPR2023/papers/Messikommer_Data-Driven_Feature_Tracking_for_Event_Cameras_CVPR_2023_paper.pdf)。这里没有声称实现了这两篇论文的算法。

## 三条件和对照

开发条件：水平速度 1.7 m/s 正常、2.2 m/s 高速、2.2 m/s 高速逆光。同一模型构造 seed 1702、同一来球 seed 1702，原权重固定。逆光组在逆光下获取初始化图像，未替换成正常光照图像。

每种生成器将使用同一初始化信息、控制器及评分。当前先验证完整任务，不做生成器排名。另设 `--initialization-only`，保留初始图像、禁止事件更新，作为动态更新的对照。

## 运行与证据

服务器根目录：`/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/catchit_20261107a`。g1/g3 共享该目录；每次记录实际运行节点和 GPU UUID。`hybrid_v8_job.json`、`hybrid_v8_commands.json`、对应日志和结果为完整命令与运行证据。各 `source_versions` 为不可变部署快照。

`SOURCE_RECEIPT.json` 保存实际部署文件 SHA256；`FROZEN_EVENT_PROFILE.json` 是冻结参数的内容副本，原始配置路径和文件哈希见 receipt。实际事件 worker 每次核验 17 个冻结核心文件。

环境：物理和渲染使用现有 `.venv-robot`（Python 3.11、Genesis 1.2.2、torch 2.8 CPU）与任务依赖 overlay；CUDA 重建使用现有 `Eventbased_WAM_torch28`。两进程以明确的数组字节格式通信。没有修改共享 Python 环境。

CUDA 的额外适配仅分组执行相同时间采样，保留全部时间点、阈值与几何规则；3090 和 4090 的实际缓存对照已验证事件数量、位置、极性与时刻一致。分组脚本与测试在此目录。

## 代码入口

- `run_original.py`：原权重、输入替换、控制 trace 与原评分。
- `genesis_adapter.py`：物理后端迁移、原 PID/IK、Genesis 接触映射与实际渲染。
- `sparse_event_sensor.py` / `event_worker.py`：单帧初始化和冻结事件采集。
- `rgb_sphere_initialization.py`：只接受图像与相机参数的球体初始化。
- `event_sphere_tracker.py` / `event_sphere_tracker_kalman.py`：静止与运动状态估计。

当前状态：三条件真实闭环在运行，尚未将新方案宣称为接取成功。旧纯事件失败、单帧测距误差、所有成功与失败均保留。最新结果将单独写入 `RESULTS.md`。
