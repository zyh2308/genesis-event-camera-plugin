# 原生光度约束是否误导光流：v2修复及严格支持补充

v1的151任务因把70个参与dt1梯度的张量误当模型named参数总数而失败，失败及26d7601446088f45d5194a9eef058caf90d2e8fd50f4e21e71fb9dd99a7f2dee锁源完整保留。本v2新root独立执行，不要求151成功，不改v1任何文件。固定源码模型实际82个named且82个requires_grad；对全部trainable参数allow_unused=True求梯度，完整记录names/shapes/requires_grad及canonical SHA。native dt1 photo有70个nonNone梯度，其余unused按零贡献处理，计算cos时保留所有存在分量的范数；不把total与nonNone混淆。

首次affine严格支持：使用原grid将raw valid用bilinear采样>=.9999，再与原NNvalid和in_crop相交，保持flow数值完全原样。训练样本各尺度的GT方向oracle使用此支持再次resize后的strict mask；验证窗原始GT只pad，也保持严格支持。逐训练样本报告NN支持被剔除比例、运动bins、原photo梯度能量/损失占比；原8batch参数oracle按NN/strict两种诊断支持并列。strict剔除不等同纯遮挡，不保证保留标签为绝对物理真值，不进入实际训练或optimizer。

范围：固定seed0、EventGenesis与V2E原完整8910更新的last.pt，不使用采样短训checkpoint。不读真实文件或真实分数；不选新checkpoint、不构建optimizer、不更新模型、不发布GT mask训练方案。运行新独立root，原模型/权重前后SHA均须一致。GPU3现有研究任务结束后，由root资源队列准入，整体900秒硬上限。失败或预算不足保留证据，不自动启动训练。

推荐的三个诊断构成同一登记试验：

1. 理想可控平移与亮度/遮挡干预。解析64×64纹理生成精确右2px、上1px平移；原warp必须实现`next(x+u,y+v)`对齐prev。无干预控制要求GT光度残差低于zero和minusGT，梯度及±epsilon必须向正确方向。固定亮度gain1.1/offset.05、16×16已知遮挡及二者组合只用于判断光度假设，干预后不要求算法改善。已知遮挡区不把被遮挡GT当可见对应真值。
2. 原登记八个训练样本：各stratum首条训练rollout的11/44窗，增强种子20261108+stratum×1000+index，与两生成器完全配对。原256联合变换后分析四个原生尺度（256/128/64/32）。另将同八个样本按原batch8、NN/strict诊断支持分别运行全参数梯度oracle：原四尺度photo、最细尺度有效区photo、物理无效区photo与最细尺度GT EPE对全部82 trainable参数求导，报告70nonNone及unused/cos/dot/norm；不step。strict复用helper的physical_invalid键包括被严格支持剔除的边界/插值区域，不能解释成纯物理遮挡。正cos仅表示局部一阶方向有利，不等价于训练一定有效。OOM或超时则保留新root失败，不自动重试/训练。
3. 原登记32个验证窗：八条验证rollout各0/22/43/65，四strata各八窗，不依照新结果改窗口。原352padding输入，原模型训练返回四尺度；无BN/dropout且最细返回与eval前向固定容差1e-6核对。验证光度分析是原公式在352分辨率的诊断，不冒充原256训练loss或完整528验证指标。

源函数全部沿用冻结trainer、reference correlation、warp兼容、multiscaleloss与模型14文件树。原native photo将cv2.INTER_LINEAR resize灰度到方形尺度，使用Charbonnier(alpha=.45, epsilon=.001)，每尺度全像素SUM后四尺度平均；event_images实参实际未mask原loss。逐例独立分量重建与原objective_dt1 photo必须精确相等，失败不放宽。

GT仅作诊断oracle：先按原联合仿射变换，再双线性resize并对u/v按w/W、h/H缩放。粗尺度oracle排除双线性GT混入无效零哨兵的像素（valid双线性值≥.9999）；同时保留nearest有效mask。运动bins在每尺度重新换回输入栅格像素单位，界限0/.1/.5/1/2/5/inf。GT无效处不计算正确方向，不把零哨兵当真值。

固定报告：物理有效区、EG∩V2E共同活跃区及其补集、灰度梯度阈值.01定义的强/弱纹理及活跃交集、四strata、运动bins、几何/可见性无效区、GT插值污染、padding/增强裁剪外区、预测出界及出界与物理无效的交集。几何无效不自动称为纯遮挡。每区均报告像素数、native photo(zero/pred/GT/minusGT)、损失/梯度能量占比、EPE和对GT方向cos；无oracle支持为null。

方向定义：d=(GT-pred)/|GT-pred|，仅有效GT及误差>1e-6处；`cos(-grad_photo,d)>0`、沿d光度方向导数<0表示局部方向有利。固定epsilon=.01/.05（各尺度像素），记录中心与双侧有限差分以及几何边界穿越数；不将双线性/二值mask的不光滑误认成原实现错误，不设事后阈值。

原warp先把出界采样置黑，但photo仍与prev全图比较，出界损失可包含与flow梯度无关的常数。另报告仅去除此常数的mask-only梯度差作为负对照；梯度相同不能称为有效学习改进。输出空间和共享参数空间分别标记，单seed/固定子集不作显著性或真实泛化结论。任何训练候选须在实际诊断后另行登记，不因本报告自动长训。
