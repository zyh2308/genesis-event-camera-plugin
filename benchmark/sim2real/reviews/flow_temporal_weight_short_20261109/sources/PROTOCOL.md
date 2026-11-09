# 光流共享灰度时间差权重配对短训

现有固定训练批次中，共同事件活跃有效区只获得最细尺度约12%的原光度梯度能量。此前按窗口调整运动采样和替换 Census 均未通过质量门槛。本轮检验像素级重加权是否改善运动区，仍使用原光度残差和所有原采样窗口。灰度时间差也可能来自遮挡或亮度变化，因此把结果限定为下游开发实验。

两组均恢复各自8910步 `last.pt`，沿用原模型、Adam、scheduler、原控制样本及增强顺序。灰度原始数据严格有限且在 `[0,1]`，已知浮点插值只允许8个 dtype eps 的舍入余量，不裁剪输入。每尺度使用原 OpenCV 线性缩放的共享增强端点 `I0/I1`，每像素权重为 `r=1+min(abs(I1-I0)/0.05,1)`，再除以同一窗口同尺度的空间均值。常量时间差权重全为1。`0.05` 为预定的5%灰度满量程封顶，固定用于两生成器，不调参。

原四尺度 Charbonnier、兼容 warp、各尺度像素求和和尺度均值、二阶平滑×10保持原样。候选只把常量1改为上述无梯度权重。模型仍仅输入各自事件。训练中的物理 flow GT 和有效区只用于共享增强标签身份核对与验证评分，不进入优化损失或像素权重。每个更新保存四尺度权重的SHA，四臂逐步序列必须一致。

CPU源/64个合成bank分支准入 → 每方法两臂各1步实际更新及独立模型/Adam读回 → 每臂固定200步。GPU smoke额外验证本地重建的原光度值逐值相同。只有完整四臂通过，才启动800步的主短训。训练后仅按固定200步末点评估全部528个合成窗，八条轨迹、四种运动分层。原12项质量门槛逐项沿用，要求两生成器的全有效及共同活跃 EPE 均改善，且各物理分层全有效 EPE 恶化不超过5%。

预登记的场景主假设为 `camera_and_independent_rigid_motion` 的共同事件活跃有效区。此前在该合成开发条件观察过轻微优势，本轮全量保留两条固定轨迹、132个窗。单独输出 EventGenesis 候选对自身控制及 V2E 候选的同窗差和逐轨迹差，窗口共同支持为空时保留 null。只有两个逐轨迹方向一致才支持同一开发场景的方向性复验。该场景结果不替代12项全局门槛，也不能称独立新场景泛化。所有其他运动分层和负结果按原评估保留。

数据与冻结来源由已部署 `20261108-flow-census-paired-short-v1/sources/contract.json` 绑定，源 manifest SHA `45343592cc7eff1c86f64bafe0d91362f96371f84071daf19ae32cff9060ff04`。复用的是旧审计、批次、恢复与评测代码，不读取旧候选输出。原训练配方、trainer、model tree、compat、两个原 `last.pt` 及数据 manifest 在准入时核验，源与权重在结束时复查。未使用MVSEC数据或成绩，无自动真实评分、换权重或后续长训。

服务器运行根为 `RUNS/20261109-flow-temporal-weight-paired-v1`。源在其 `sources`，CPU 和 GPU 模式均需各自全新目录。

```
CUDA_VISIBLE_DEVICES='' PY -B -u ROOT/sources/run_temporal_weight.py cpu --out ROOT/cpu
PY -B -u ROOT/sources/run_temporal_weight.py smoke --out ROOT/smoke
PY -B -u ROOT/sources/run_temporal_weight.py study --out ROOT/study
```

GPU 模式沿旧资源入口的显式 GPU3 UUID 绑定，由父会话准入。CPU夹具可用 `STE_NATIVE_LOSS_SOURCE` 与 `STE_NATIVE_COMPAT_SOURCE` 指向已校验的原 loss/compat 文件。夹具验证常量权重、范围和空间归一化、真实原函数的四尺度前向及梯度恒等，以及逐窗/逐轨迹库存。服务器完整训练另需实际完成收据。
