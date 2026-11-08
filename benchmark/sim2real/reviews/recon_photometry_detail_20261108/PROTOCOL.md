# 合成重建光度与细节：有限配对续训

本次仅验证优化效果，不启动正式长训或真实评分。使用现有独立 checkout；冻结模拟器 62da4f35ce3c5c96f04268bd1687a27657af9238、事件数据、噪声、物理标签、模型结构和 sigmoid 输出。

## 设计（观察本轮结果前登记）

- EventGenesis / V2E，各 seed 0；各自共同 synthetic_best update 4800 检查点，恢复模型、Adam 和 RNG，续训 300 次至 5100。每方法三臂共 1800 次更新。
- C0：原 LPIPS-VGG 1 + TC 2。
- C1：C0 + 像素 mean L1 1。
- C2：C1 + MPRNet Laplacian Charbonnier 边缘损失 0.05，epsilon=0.001。仅将原作者 RGB 核适配为灰度/输入设备。
- 学习率 1e-4 固定，batch 2 × 40 帧；原顺序采样及共享增强，实际标签、事件及样本序列哈希检查。无 GT 真实输入。
- 三臂均恢复相同源检查点及 RNG；合成评估恢复 RNG。每方法 GPU 一张，原子 UUID 租约，阶段超时两小时，结束释放。原三线作业不受改动。

## 指标与采用条件

源检查点与最终 5100 均评估相同 8 个独立 rollout，每个完整 200 帧；先做 rollout 等权平均。all200 为主，tail195 为次。LPIPS-Alex/VGG、PSNR、MSE、SSIM、对比度、相关性、边缘幅度误差、平坦帧和边界比例全部保留。

候选必须对 **两个生成器分别** 相对同期 C0 满足以下合成门槛，才建议扩大验证；单 seed 通过也不是论文最终结论：

1. 全部 loss / 30 个梯度张量有限，实际 TC 每更新 76 次；输出对比度比 > 0.1、平坦帧比例 < 5%。
2. MSE 至少下降 5%，PSNR 至少提高 0.2 dB，SSIM 不下降。
3. LPIPS-Alex 和 VGG 均不比 C0 恶化超过 1%（预先指定容差，原始差值完整报告）。
4. 相对于 GT 的逐像素横纵梯度 MAE 不恶化，相关性不下降；边缘幅度比另作诊断，边缘更强本身不算细节更准。

没有候选通过则报告未通过，不放宽门槛、不以真实集挑损失或权重。正式多 seed、HQF 仅在后续明确决定后实施。

## 实现与核验步骤

1. 独立 auxiliary_losses.py，与作者 RGB 公式进行 loss / 梯度等价、gradcheck、L1 梯度测试。
2. C0 直接绑定原 Trainer.forward_sequence（函数身份检查），保留其全部 LPIPS/TC。C1/C2 通过只读 forward hook 获取同一批原预测，再以相同 40 帧平均加入辅助项；原 Trainer 不改写。新增损失额外在 GPU 对照作者公式的 loss/梯度。原生 CUDA 反传重复波动单独记录，不宣称轨迹确定性。
3. 每方法三臂各一次真实 optimizer smoke；源 CP hash、Adam 4800→4801、TC 梯度、固定小样本评估核对后运行。
4. 300 次结束写不可变 checkpoint、恢复模型/Adam/RNG并检查，生成指标与配对门槛报告；保存失败日志。正常推送研究分支。

## 成熟做法来源

- V2V (NeurIPS 2025) 作者 E2VID 配置同时使用 LPIPS-VGG 1 和 L1 1： https://github.com/HYLZ-2019/V2V/blob/main/config/train_v2v_e2vid_10k.yaml 。仅借鉴像素项；并非复现其全部协议。
- MPRNet (CVPR 2021) 作者 edge loss 及 0.05 权重： https://github.com/swz30/MPRNet/blob/main/Deblurring/losses.py 与 https://github.com/swz30/MPRNet/blob/main/Deblurring/train.py 。这是复原任务的辅助项借鉴，不宣称事件重建标准强制使用。
- E2VID+ 作者原配置 LPIPS-VGG 1 + TC 2： https://github.com/TimoStoff/event_cnn_minimal/blob/master/config/reconstruction.json 。

限制：同一合成验证集参与研究开发；不是独立盲测。此次只识别短程局部效果，不能替代最终多 seed 和公共真实测试。

预检修订记录：v1 要求 CUDA 参数梯度逐位相等，在两组 smoke 中被拦截（尚未进行 optimizer 更新）。v2 在新路径保留原失败，以原函数重复反传对照定位可能的 CUDA atomic 累加差异。没有改变 loss、模型、训练预算或采用门槛；若数值一致性仍失败则继续拦截。

v2 中原函数自身重复反传也未通过 rtol=1e-5 / atol=1e-7（最大绝对差约 3–5e-7），扩展函数差异未超过原生重复波动。v3 不放宽该容差来宣称“通过”：取消 Trainer 改写，C0 直接使用原函数，新增辅助目标以独立且已核验的分支相加；CUDA 原生逐位/该容差复现失败仍保留为失败证据。正式研究采用门槛保持原值。
