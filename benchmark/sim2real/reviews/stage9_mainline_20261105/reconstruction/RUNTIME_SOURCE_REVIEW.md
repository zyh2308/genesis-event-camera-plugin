# HQF Stage9桥接运行时源码审阅

审阅日期：2026-11-05。依据仓库23761f2的原始infer、scorer、source verifier及metric helper只读副本。没有SSH、真实数据推理、指标评分或GPU操作。本文是静态审阅与CPU元数据合同结果，不表示实际Torch/LPIPS执行通过。

## 复用范围与两组硬编码

旧infer接受单个checkpoint和固定14个HQF序列，不限制generator枚举。其checkpoint_to_model严格加载`model`，读取checkpoint model_args或登记recipe，提供两份时必须一致，且设置eval和关闭参数梯度。模型只接收5-bin events；dataset返回的参考图像像素在调用模型前删除。输出经原cropper裁回native180×240并统一clip一次[0,1]。

旧scorer的两组schema/status、旧权重门禁、专用self-test包含明确的两组硬编码。桥接不调用这些旧CLI/schema验证或伪造两组身份，而是在自己的Stage9合同下绑定15个真实method_seed，独立补齐完整推理metadata守卫，再复用未改动的validate_data_manifest、score_pair及metric函数。score_pair的数据循环只依赖其私有模块中的EXPECTED_ARMS；指标算术没有单独依赖`eventgenesis`或`v2e`名字。旧self-test仍只能证明原两组fixture，不能冒称通过了15组测试。

## 审阅中提前修复的桥接问题

1. prepare后源码可能改变：verify、run启动和最终评分前都重新调用原源码树验证器；所有评价脚本/metric helper也核SHA。helper实际导入位置必须与绑定文件一致。
2. 旧infer provenance写死历史core e79：原始输出保留；新run manifest解释该字段，逐推理记录Stage9有效core62da。训练recipe、completion及checkpoint SHA是正式身份依据。没有把历史字段当作新版core证据。
3. 新schema跳过旧validate_run_manifest后会漏检查：新增15身份推理合同，检查状态、非fixture、权重/方法/结构、公共权重禁用、event-only、循环重置、native crop、源码commit、数据hash与完整时间端点。实际窗口时间、事件边界、浮点输出SHA和指标输入仍由原score_pair逐帧核查。
4. live run_manifest在评分后写完成状态会改变其SHA：评分改为绑定不可变inference_manifest_frozen.json；live状态单独更新。
5. metric环境缺件可能在210次推理后才暴露：run在推理前确认cv2、scikit-image及既存本地LPIPS/AlexNet文件，CPU初始化真实metric但不输入图像；不会安装或下载替代权重。评分前再次核资产SHA。
6. CUDA严格推理依赖CUBLAS_WORKSPACE_CONFIG：run要求启动前显式设置:4096:8或:16:8，README已给部署命令。

## 指标路径

原classical_metrics使用float32灰度[0,1]、float64累加MSE，PSNR=10log10(1/MSE)，SSIM使用scikit-image structural_similarity(data_range=1)。LPIPS采用已锁定PerceptualSimilarity v0.1 AlexNet，灰度重复RGB并线性映射至[-1,1]。真实图像只在评分阶段充当target，不进入模型或选点。零误差PSNR保留+inf，现有严格聚合器对此需单独处理，不能删除或改值。

## 验证及剩余边界

stdlib CPU测试26/26通过，覆盖完整矩阵、正式预算/配对初始化/权重哈希、缺组时CLI准备提前拒绝、完整推理metadata及错误身份/GT输入/公共权重/循环状态/数据hash拒绝、prepare后源码篡改与无授权启动拒绝。测试没有真实影像或Torch模型。

部署后仍需在已登记服务器环境运行原有合成infer与metric fixture，确认Torch、cv2、scikit-image、锁定源树及本地backbone缓存可用；通过后才准备正式HQF15协议。所有15个12000更新收据与各自synthetic_best必须齐全。该桥接不提供部分真实推理的自动续跑；失败保留目录及日志，禁止覆盖。
