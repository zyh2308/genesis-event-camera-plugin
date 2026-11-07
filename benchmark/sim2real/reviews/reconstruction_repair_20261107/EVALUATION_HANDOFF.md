# 修复后重建的评分交接边界

此说明仅准备后续评分，当前不启动真实推理或评分。

1. 新权重只取 `20261107-recon-sigmoid-formal-v2/training/seed{0,1,2}/{method}/`，五种方法全部完成12,000次更新、退出码0、通过输出稳定性检查之后，再冻结15个 `synthetic_best.pt`、配套选择记录和SHA。不得用旧线性输出的seed0/1填补新矩阵，也不得按真实分数选checkpoint。
2. 权重必须含 `model_args.final_activation="sigmoid"`，并与正式v2配方、trainer SHA、数据manifest一致。初始化SHA继续对应原登记seed；补跑学习参数和预算公平一致。
3. 现有 `infer_hqf_e2vid_candidate.py:checkpoint_to_model` 已用真实保存函数做CPU数值检查，能读取checkpoint明确保存的model_args；如传入旧的线性输出配置，预期行为是拒绝加载。不能为了兼容而删除这项严格检查。真实评价应明确绑定新sigmoid模型配置。
4. 旧 Stage9 HQF 桥接器的checkpoint根目录与配方指纹指向原campaign。它不能直接代表新v2权重。新评分manifest须明确映射v2完整15权重，保留原14条HQF序列、事件窗口、首图anchor排除、循环状态重置、裁剪、float输出及LPIPS-Alex/SSIM/PSNR/MSE协议，不能静默换权重或修改GT亮度/直方图处理。
5. HQF已经看过；重复结果应如实标记为既见数据上的开发评测。修复方案和采用门槛仅根据合成实验确定，没有使用HQF指标选择配置。若后续需要盲测确认，应另行预登记尚未查看的真实测试集，不能再称HQF未见。
6. 新结果必须与此前原始完整表并存。算法共同修复的收益不能作为EventGenesis相对V2E的优势证据；生成器结论必须来自统一新配方、相同完整预算、多seed的真实配对结果。
