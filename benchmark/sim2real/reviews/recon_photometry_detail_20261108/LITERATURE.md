# 本轮辅助损失的依据与适用边界

本轮不更换 E2VID+ 架构，不采用新网络或真实训练；只在原 LPIPS-VGG 1 + TC 2 上测试两个有作者实现依据的辅助项。

1. **V2V: Scaling Event-Based Vision through Efficient Video-to-Voxel Simulation (NeurIPS 2025)**。作者 E2VID 训练配置给出 LPIPS-VGG 1、L1 1，同时保留时序目标。这支持把像素 L1 作为最小的光度精度对照。这里只借用 L1 权重，不声称复现其 RAFT、小网络、TC 权重及完整数据协议。 [作者配置](https://github.com/HYLZ-2019/V2V/blob/main/config/train_v2v_e2vid_10k.yaml) · [论文](https://papers.nips.cc/paper_files/paper/2025/file/b14cf0a01f7a8b9cd3e365e40f910272-Paper-Conference.pdf)
2. **Multi-Stage Progressive Image Restoration (CVPR 2021)**。MPRNet 作者训练脚本采用 Charbonnier 主项和 0.05 的 Laplacian edge 辅助项。这里只借用边缘目标及其权重；主项保持本研究的 LPIPS/TC/L1，不改称 MPRNet 复现。边缘模块保留作者滤波、下采样、零插值及 epsilon=1e-3，单独验证灰度与 RGB 重复输入的前向/梯度等价。 [论文](https://openaccess.thecvf.com/content/CVPR2021/html/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.html) · [作者损失代码](https://github.com/swz30/MPRNet/blob/main/Deblurring/losses.py) · [作者训练权重](https://github.com/swz30/MPRNet/blob/main/Deblurring/train.py)
3. **E2VID+ 作者训练配置**。本项目沿用 LPIPS-VGG 1 + TC 2；新实验保留这两个分量和全部时序规则，用同期控制识别增加像素/细节目标的局部作用。 [原配置](https://github.com/TimoStoff/event_cnn_minimal/blob/master/config/reconstruction.json)

L1 更直接约束亮度与逐像素误差，LPIPS 保护特征层面的外观；二者可能有取舍，不能只看某一项提升。边缘损失也不是天然有效，须用逐像素梯度误差和感知指标共同判断。本轮候选只有两个生成器都通过预登记门槛，才值得进一步验证；不保证对所有 seed 或真实数据有效。

## CUDA 数值检查的边界

PyTorch 官方文档明确指出 CUDA grid_sample 的反传可能非确定，不能仅以相同初始化和 RNG 宣称整段轨迹逐位复现。本次 v1/v2 中原函数自身的重复反传也有微小梯度差；失败保留。v3 直接使用原 Trainer 函数，独立测试新增分支，未改写 TC，也未把原生重复检查重标为通过。 [PyTorch grid_sample 文档](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grid_sample.html)

AMiner 本地未配置凭据，因此改用论文及作者代码作为依据；没有索取或记录任何访问密钥。
