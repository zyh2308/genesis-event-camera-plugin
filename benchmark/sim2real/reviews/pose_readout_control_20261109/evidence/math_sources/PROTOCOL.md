# 固定预测峰软聚焦：只读概率代数

唯一 API：`softfocus_predicted_peak(attention)`。输入恰为一个16×16、float32/float64、有限非负的原softmax概率图，总和误差不超过1e-5；此数值容差沿用冻结区域诊断的接受量级，不是结果门槛。拒绝错误shape、dtype、负值、NaN/Inf、全零或明显未归一化图。

grid固定为 `((column+.5)/16,(row+.5)/16)`，预测峰用原图C行优先首次argmax，sigma固定1/16。FP64私有副本计算 `q_i=p_i*exp(-||g_i-g_peak||²/(2*sigma²)) / sum(p*g)`，再求坐标期望。无clip、epsilon、硬裁剪、flat-map fallback、温度、调参接口或GT参数；原p字节不变。

返回 `map_sum`、预测 `peak`、FP64 `weights`、`uv`、`original_fp64_uv` 和 `numerical_diagnostics`。其中原全图FP64坐标特意不偷偷归一化p，用于与调用方保留的原Torch FP32控制比较；不宣称NumPy FP32约简等于原Torch算子。

平坦图仍按原argmax tie规则选择首cell，软聚焦后会偏向左上。这是待保留的失败风险，不添加有利回退。错误峰同样不会被GT纠正；硬argmax与3×3旧负结果不变，本模块不宣称已改善定位或完整位姿。

CPU测试9项，包含均匀图解析边界、角点/边缘onehot、精确Gaussian乘积、远尾仍为软权重、C顺序/Fortran及strided输入、不修改原图、原FP64控制算术、无效输入拒绝和API禁止GT/可调参。仅依赖NumPy和标准库，不运行模型/训练/真实评分。

本模块只服务已登记四轨迹开发机制对照；不等同原30+8检查。原30项见冻结native `pose_two_branch_coverage.py` SHA85951e…第214–224行；原8项由 `pose_coverage_real.original_confirmation()` 调用原fresh.gate，96轨迹全16候选窗。C1深度保持不变时不能因中心改善就宣布深度门槛修复，更不能自动开启真实E-POSE评分。
