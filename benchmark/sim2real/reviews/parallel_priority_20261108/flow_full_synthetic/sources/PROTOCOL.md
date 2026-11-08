# 冻结正式光流模型的完整合成分解评估

这是一项只读评估，不是新的训练。原拟在 RTX 3090 上增加 seed1/2 的采样短训，但现有原生 batch8 反传实测峰值25,566MiB，超过24GB卡容量。因此不降 batch、不做未经验证的梯度累积，不部署这条短训方案。

固定五种生成器、三个 seed 的原正式 synthetic_best（每次正式训练45epoch/8910更新），不更换任何权重或选点。模型、源文件、初始化、数据 manifest、每个 NPZ 和 checkpoint SHA 均冻结核验。每个模型完整8条独立合成验证 rollout，每条66个30ms窗口，共528窗；不抽样。两张同型号3090分别评 seed0/1和seed2。只做 batch1 原生352输入前向，使用训练原 reference correlation、原 warp 兼容和原输出单位/裁剪，FP32、TF32关闭。每个模型与其原 checkpoint 对应历史完整验证 EPE 核对，绝对误差不超过1e-5；否则记录失败并停止该 shard。

分解报告全物理有效区、固定 EventGenesis 与 V2E 事件活跃像素交集、其补集、四物理 stratum 和所有既有GT幅值 bins（0,.1,.5,1,2,5,inf px及其活跃子集）。五种方法使用完全相同的EG∩V2E支持区域，不使用各方法自己的事件支持区。零流参考同区计算。每个窗口保留区域像素数和支持比例，无支持为null。先各rollout内对有支持窗口等权，再8条rollout等权；四strata各2条等权。跨seed需报告三点、mean、sampleSD；不宣称显著性或真实泛化优势。

不构建 optimizer，不反传，不更新模型，不读真实事件、GT、结果，不改变真实测试checkpoint，也不自动启动长训。输出仅独立新root；历史结果保持原样。逐rollout最多同时缓存EG、V2E和当前方法，避免缓存整个验证bank；启动每worker至少8GiB可用RAM，父进程两任务合计预留16GiB。父进程原子lease重新核验GPU、注册表和活动app，若已被占用则拒绝，worker上限一小时。结束释放资源。
