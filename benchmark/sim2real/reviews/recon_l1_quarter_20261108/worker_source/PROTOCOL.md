# L1=0.25：单张 GPU2 上串行执行的有限配对试验

本轮仅改变资源安排。上一版本的两个独立 runner 在原子 GPU 门禁阶段被其他作业占用拦截，未完成 smoke 或 optimizer 更新；其日志保留，不重用输出目录，不终止或干扰其他作业。

本轮父输出目录为 `20261108-recon-l1-quarter-paired-gpu2-v1`，两个方法分别写入 `{eventgenesis,v2e}/{smoke,study}`。两方法全部使用 **g1 GPU2，V100S 32GB**，固定 UUID `GPU-a02affe0-7598-579e-ca39-f551f35ad6e9`；不混用不同型号 GPU。

## 研究假设与不变的训练协议

上一轮纯 L1=1 改善光度误差，但 V2E 的 LPIPS-VGG 与逐像素梯度 MAE 在全部 8 个验证 rollout 均退步。只改善光度不算成功。较弱的 **L1=0.25** 只是保护感知/细节的固定待检验假设，不是论文最佳参数或已证明有效的配置。

- EventGenesis/V2E，各 seed 0，各两个臂：`control`（原 LPIPS-VGG 1 + TC 2）和 `photometric`（C0 + mean 像素 L1 0.25）；没有边缘训练臂。
- 四臂各自从对应生成器相同已冻结的 synthetic_best update 4800 权重恢复完整模型、Adam、Python/NumPy/CPU/CUDA RNG 与 epoch 80/batch 0 游标，分别续训 300 步到 5100；不接用上一轮优化后的权重。
- 共同物理 bank、事件输入、标签、噪声与遮挡规则不变。frozen core：`62da4f35ce3c5c96f04268bd1687a27657af9238`。
- 原生 sigmoid、学习率 1e-4、有效 batch 2×40、TC、采样及增强完全保留。worker 逐字节复用已静态核验的 quarter 版本，仅管理器和资源协议改变。
- C0 直接使用原 Trainer 函数身份；C1 从同一原生 forward 的 40 帧预测上增加辅助项。原生 CUDA grid_sample 自重复微小梯度漂移不改记为通过，不宣称全程逐位确定性。

## 单租约执行与停止边界

唯一父管理器先验证 GPU index/UUID、同 UUID 原子锁、租约登记及实际 compute apps。GPU 被占用即保留失败证据并退出，不抢占、不等待占卡，不启用空闲占位进程。

顺序固定：`smoke_eventgenesis` → `smoke_v2e` → `study_eventgenesis` → `study_v2e`。两个 smoke 各两臂，共 **4 次**独立真实 optimizer 更新；两个 smoke 完成且实际标签/增强跨方法完全相同后才允许主短训。主训练总预算 **1200 次** optimizer 更新。没有双 runner 相互等待或双租约。

父任务整体上限 **2 小时**，不是每阶段两小时。超时或异常只终止本管理器启动的子进程组，在 finally 中释放自身 GPU 租约；清理等待至多 30 秒后必要时结束自身未退出的子进程。不触碰其他作业。完成后释放 GPU，不保留空进程。

## 验证及原 9 项采用门槛

相同 8 个独立合成验证 rollout，各完整 200 帧、rollout 等权；all200 为主，tail195 为次。对照只比较同一 5100 总步数，源权重 4800 单列参考。两个生成器各自必须同时满足：

1. 对比度比 `>0.1`。
2. 平坦帧比例 `<0.05`。
3. MSE `<=0.95×control`。
4. PSNR `>=control+0.2 dB`。
5. SSIM `>=control`。
6. LPIPS-Alex `<=1.01×control`。
7. LPIPS-VGG `<=1.01×control`。
8. 逐像素梯度 MAE `<=control`。
9. 相关性 `>=control`。

loss/30 个梯度张量全部有限、每更新 76 次有效 TC、确切更新数、配对哈希与 checkpoint 重建恢复同样是必需证据。方法内事件与标签/增强序列必须一致；跨方法只要求标签/增强一致，事件生成器本身是变量。

任何失败保留失败，不降门槛。不启动正式长训、不扫权重、不真实评分、不用真实集挑损失或后处理。本轮为单 seed 合成开发诊断，不是多 seed 论文最终结论。原 GPU0/1 的资源失败与历史数值检查失败均保留在原目录，与本轮运行收据分开。

思路来源：V2V 作者的 LPIPS+L1 配置 https://github.com/HYLZ-2019/V2V/blob/main/config/train_v2v_e2vid_10k.yaml ，以及 E2VID+ 原 LPIPS/TC 配置 https://github.com/TimoStoff/event_cnn_minimal/blob/master/config/reconstruction.json 。不声称本轮 .25 权重来自论文最佳参数。

