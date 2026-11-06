# 位姿光照鲁棒性 · 合成配对试验

先用原合成训练集试验事件增强和几何一致性。沿用两生成器原尺度分支的冻结权重作为各自起点，比较等预算干净续训、事件增强续训、增强加几何一致性三组。三组每步各看两个视图，使用同一几何仿射、相同训练样本顺序，监督尺寸、中心、形状因子和深度。一致性约束仅作用于同一合成姿态的干净与扰动视图。姿态旋转保持冻结。三个种子配对，固定最后权重。

盲于训练的 96 条原独立合成轨迹各取四个等间隔窗口。用干净、中等扰动、较强扰动三个固定视图评分。事件扰动在事件表示上改变计数强度、局部缺失和背景活动，用于合成压力测试。它没有物理照度含义。主效应为几何一致性分支相对等预算干净组在中等和较强压力下的深度 MAE、中心误差和完整 ADD-S。干净条件成功率容许回退 2 个百分点。三个 seed 的主深度效应均须为正，配对轨迹 95% 区间下界为正。

另外，从四条已冻结的独立合成轨迹生成共享三档辐照度缓存。沿用正常档，把线性辐照度分别乘 0.5 和 0.125，保持几何、时间、遮挡和标签。两生成器各走原官方输入和回放路径，V2E 的亮度帧用原固定映射重算。这项受控预试验只隔离全局辐照度，不能代表空间照明变化或真实传感器标定。当前 E-POSE 木块全部留作诊断，不在本试验中读入。

## 当前验证

本地 WSL 的 CPU 与 CUDA 均完成三组单步更新，确认同种子事件扰动可复现。独立几何样例验证尺度偏差、深度偏差、旋转正交性和完整 ADD-S；辐照度样例验证衰减倍率、原亮度转换一致性、物理标签与源文件哈希。样例为程序生成，用于验证实现。

g1 和 g3 各有两次 SSH 连接超时。服务器配对训练与回放等待网络恢复。没有项目数据上的新性能结果。

## 运行入口

在可用节点上复用固定 Torch 2.8 环境与已有的资源预约机制。将三个运行脚本放在同一独立目录，使用新的输出目录。先用 `pose_light_robustness.py` 的 `--smoke` 运行两种生成器的工程预检，通过后去掉 `--smoke` 完成三个配对 seed。

```sh
python -u pose_light_robustness.py --campaign "$CAMPAIGN" --out "$RUN/smoke" --smoke
python -u pose_light_robustness.py --campaign "$CAMPAIGN" --out "$RUN/robustness"
python -u prepare_pose_radiance_pairs.py --campaign "$CAMPAIGN" --out "$RUN/radiance" --replay
python -u evaluate_pose_radiance_pairs.py --campaign "$CAMPAIGN" --radiance "$RUN/radiance" \
  --robustness "$RUN/robustness" --out "$RUN/radiance_score"
```

前两步与辐照度回放可以在资源允许时分配给独立作业。辐照度评分要求回放和三组配对训练均已完成。真实评分仍由独立的预登记协议控制。

事件增强借鉴随机事件丢弃与背景干扰，几何一致性用于同一合成姿态的两个事件视图。[EventDrop](https://ijcai.org/proceedings/2021/97) · [AEE](https://doi.org/10.1109/lra.2025.3644151)
