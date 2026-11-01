# E-POSE 三小时配对短训：训练前登记

状态：进行中。此页登记的是方案和已通过的预检，尚无本轮真实 E-POSE 分数。最终报告会在本目录补齐。

## 本轮目的

先解决完整事件图输入下的物体定位，再交付服务器短训后的真实 E-POSE 初测和实际预测视频。不会把定位改善说成完整 6DoF 已解决，也不以单对象、单序列、单 seed 宣称生成器普遍优势。

此前的开发 bank 只有 4 条 40 ms 训练轨迹、16 个训练窗，木块无纹理且相机静止。多次短程诊断后，EventGenesis 在旧 8 窗开发验证集上通过定位门槛，V2E 仍未通过；R2–R7 均没有开启新真实片段的评分。用户提供三小时时间窗口后，本轮改为扩充共享物理数据，再做配对短训。

## 训练前固定的设计

- 对应公开 YCB `036_wood_block` 的真实木纹；随附几何文件 SHA256 与 E-POSE CAD 完全一致。使用原 CAD 坐标，不换成通用立方体。
- 24 条合成训练轨迹、8 条独立合成验证轨迹。物体固定，相机运动，位置、方位、距离、高度、视角和速度变化。每条 120 ms，前 40 ms 预热，剩余窗口为 10 ms；共 192/64 个训练/验证窗。
- 两组共用物理帧、轨迹、标签和划分，不按各自事件数或成绩筛选。物理端点 1,000 Hz，EventGenesis 使用 frozen simulator core `e79e656ac00f7847b98db9e12bf43b093f3afff1`。
- V2E 采用锁定官方 EventEmulator。由同一物理端点映射的亮度图做线性 DN 插值，按 25 µs 更新传感器，不使用 SuperSlomo。预检拒绝直接使用 1 ms 传感器步长，其更新系数约 0.95。此设置是高帧率端点变体，应与原 100 Hz+SuperSlomo 协议区分。
- 沿用完整 SingleShotPose 架构和三通道完整事件图。每组从各自 R6 合成专用权重继续，追加 2,400 次全网络 Adam 更新，batch4、lr1e-4、30 步预热与余弦衰减、weight decay1e-5、梯度裁剪100。两组采样和增强相同。不能描述为从随机初始化仅训练了2,400次。
- 共同训练适配为原关键点坐标 MSE 加中心网格交叉熵。BN statistics 只在合成训练时更新，验证和测试使用普通 `eval()`。它不是原版 SingleShotPose 完整训练复现。
- 每100步在同一独立合成留出集上按相同定位规则选权重，冻结后才读取新真实样本。定位门槛为中心误差<25个网络像素且网格准确率≥75%；角点和6DoF质量另行报告。
- 首选两组均通过定位后做真实比较。预先登记的截止备用方案为仅测试通过定位的组，未通过的一组仍为失败，不作单组与失败组的生成器排名。
- 新真实片段固定为 `block/r_05_gl`，MAT `0003:3:0600` 共200个原生事件窗。训练前只检查文件存在，没有读取事件内容。旧v1权重在同片段作为参照。真实分数不参与选权重或调参。

## 已通过的预检

- CPU全网络单步反传、有限非零梯度、Adam更新和保存恢复通过，原坐标损失复算差为0。该单步权重单独废弃，不进入正式训练。
- 私有CPU OSMesa浮点FBO能精确读回亮度2.5，避免8位截断。使用进程级 framebuffer 兼容开关，没有修改共享环境或frozen core。
- SingleShotPose的56个文件和tree SHA256核对通过。
- V2E CPU API及25µs传感器更新检查通过。
- 场景预检曾发现CAD原点与几何中心不同导致一条轨迹裁切，已在训练前修正取景中心和物理可见性检查。部分失败bank和日志留在服务器，未进入训练。

## 执行位置

服务器：g1，完整run：
`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-epose-diverse-bank-v8`

tmux：`research_epose_r8_20261101`。CPU预算16线程，无GPU预留。生成按8个独立CPU进程并行，训练按两组各8线程并行。没有使用其他用户已占用的GPU。

`pretraining/` 为训练前协议、代码与预检证据快照；`protocol.json`、协议哈希和 `source_snapshot_before_training.json` 是核对入口。实际批量生成代码为 `generate_diverse_bank_workers.py`，原 `generate_diverse_bank.py` 是保留的初期单轨迹探针。大bank和checkpoint留在服务器，不纳入Git。

公开资产：[YCB](https://ycb-benchmarks.s3.amazonaws.com/index.html)。数据说明：[E-POSE论文](https://www.nature.com/articles/s41597-025-04536-5)。下游：[SingleShotPose](https://github.com/microsoft/singleshotpose)。

## 12:07 UTC 更新：窗内运动可观察性纠正

R8 的实际增强样本在 update0 被过严的绝对损失容差拦截。独立复算表明差异只有 FP32 求和顺序，相对差最大约2.14e-7，梯度完全相同；R9 仅修正 oracle 容差。早期 update400/500 checkpoint 的模型、Adam计数及 RNG 恢复检查通过。

完整256个合成窗口的检查发现，R8/R9相机始终看向物体，中心位移中位数仅0.03756px/10ms（最大0.28091）。跨轨迹视角和位置变化不能替代窗内运动检查。诊断框内事件数上界中位数为EventGenesis83、V2E14；框内可能有背景，不能全部解释成物体事件。R9固定2400步训练继续保留，真实自动评分已主动关闭，r_05_gl仍未读取。

R10改用固定相机朝向的对角平移，生成前统一检查可见性及中心位移>1px/10ms。两组共享公开纹理、精确CAD、物理端点和标签；frozen core、模型、R6初始化、损失、2400步预算及定位门槛保持原登记。剩余时间采用12条训练、4条验证轨迹，每条80ms，前40ms预热，得到48/16个窗口。这与R8不是等规模控制消融。先生成、检查完整事件图，再启动训练；R10不使用R9中途权重。

当前g1 run：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-epose-fixed-view-motion-v10`。8个CPU进程生成，后续每组8线程训练，无GPU预留。训练前协议及24个源文件在`motion_training/`；归档及内部源文件SHA256已在WSL独立核验。
