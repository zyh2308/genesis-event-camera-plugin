# Stage 4 三线交付研究核查

核查提交：61554fb0031609200bc5ddbf2340ec8d20722725。
服务器最新综合核查：2026-10-19 08:04 UTC / 17:04 JST。本文按 Asia/Tokyo 记录用户侧日期。

## 结论

重建与光流的 seed 0 配对训练目前均已完成。用户转述的重建 721/729 updates 是历史启动快照。重建的训练链路、更新、恢复和数据派生有证据支持，但发现其 temporal-consistency loss 的坐标实现错误，暂不通过研究正确性验收。MVSEC 评价器发现已复现的配置 schema 错误，协议和数据清单也未完成，真实评分继续关闭。位姿修复只在两个训练窗口上证明了机制可行，尚缺真实原始样本与匹配 CAD及独立合成验证。

## 重建训练事实

| 生成器 | 更新数 | 完成 epoch | 最佳合成 LPIPS-VGG | 最佳 epoch | 第 200 轮 LPIPS-VGG |
|---|---:|---:|---:|---:|---:|
| eventgenesis | 12000 | 200 | 0.0417374526 | 180 | 0.0431254068 |
| v2e | 12000 | 200 | 0.0859162696 | 195 | 0.0861119945 |

EventGenesis 最终状态时间为 10 月 19 日 10:08:10 JST，V2E 为 10:15:02 JST。每组 12000 条 optimizer 日志连续，每次记录 76 个有效 TC 调用和 30 个有限梯度张量；40 次合成验证（每 5 epoch）完整。12000 对 clip_indices/clip_ids 全部一致。日志没有记录独立的增强 seed，因此本次并未逐条验证所有增强结果，只核对了共同 seed 传递实现及既有增强 fixture。

已读取两组 last checkpoint，模型内容 SHA256 匹配，Adam 30 组状态均在 12000，cursor 为 next_epoch=200、next_batch=0。两组初始模型、配置与 data manifest 一致。update20→21 恢复的 oracle 有匹配的下一批内容摘要及损失记录，证明边界加载和下一步训练；它是恢复进程中先 no-grad probe 后实际 update 的对照，不是独立重写训练或完整参数更新等价证明。

10 ms 数据生成器代码确实对每个相邻物理端点重新投影，并与原 30 ms 标签做额外对照；没有除以三。两组共享标签摘要，源 bank 与原光流版本保留。本次核对实现、manifest 和运行证据，未独立重投影全部 6400 个物理区间或重新计算全部大文件摘要。

表中 LPIPS 是错误 TC 实现参与优化下产生的历史合成验证记录，暂不作为“正确 TC 协议下”的正式研究结果，更不能作为真实域优势。没有进行真实图像评分。

## P1 重建 TC 坐标约定错误

训练调用的锁定 `event_cnn_minimal/utils/loss.py`（SHA256 `3d94483e3bc05582560efd9fa2d50f751ecff52321ec812ae04429b36c88aeb4`）使用 `2*x/(W-1)-1`、`2*y/(H-1)-1` 建立采样网格，却在两处 `F.grid_sample` 未指定 `align_corners`。g1 的 PyTorch 2.5.1 默认 False，与该网格的像素中心约定不一致。训练器没有对应 compatibility patch，实际 pilot_stdout.log 和 resume_stdout.log 第 23 行均记录默认 False 的运行警告。

在 g1 实际训练环境中，CPU 独立调用该上游函数，输入 112×112、相同的正值棋盘重建图、精确零 flow；用于可见性估计的两幅参考图均为零，使可见性掩码恒为 1：

| 指标 | 当前调用 | 临时进程内指定 align_corners=True 的对照 |
|---|---:|---:|
| warp 最大绝对误差 | 0.5625 | 0.0000038147 |
| warp 平均绝对误差 | 0.18866089 | 接近浮点误差 |
| 未乘任务权重的 TC loss | 0.20792618 | 0.00000057191 |

此输入没有真实运动，正确恒等变换应保留图像。结果证明当前函数在零运动时也引入错误扭曲；非零梯度和 76 次调用不能证明数学正确。WSL 的同 SHA 上游文件在 PyTorch 2.13 上也复现同样结果。测试只在独立进程内改变函数调用作对照，未修改外部源码、训练代码或权重。

PyTorch 官方坐标说明：
https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grid_sample.html

影响：本轮 TC 训练目标包含系统性几何误差；目前无法仅凭记录确定对生成器差值的具体影响。需要修复、验证并对双方一致实施新版本训练。该问题不涉及冻结 simulator core，也没有证据要求重新生成物理 bank 或重跑已完成光流。

## P1 MVSEC 配置与评价入口不兼容

候选协议将权重索引放在 `protocol["model"]["checkpoint_index"]`，评价器第 235 行和输出构造却读取 `protocol["checkpoint_index"]`。原 `--self-test` 只验证局部算术和关闭状态，未覆盖有数据 artifact 的完整 manifest→authorization→load→score 路径。

本次在临时目录中只生成合成数组、假 checkpoint 标识、合成 manifest 和限该临时文件的授权记录，不接触真实数据、不改仓库关闭状态：

- 原候选 schema 的端到端入口报 `KeyError('checkpoint_index')`；
- 仅在临时协议中补 top-level alias 的定位对照完成，返回预期 AEE=0；
- 原算术 self-test 本身通过。

这是可复现的接线错误，修复评价器即可，不要求重新训练光流。应统一 schema 并增加完整合成 artifact 路径测试，覆盖通过、权重/hash不符、漏窗口、空支持等必要情况。

候选 manifest 的实际源文件路径/SHA256、APS index 数组、窗口列表和权重绑定仍是 null 或空，不能签发最终评测授权。候选裁剪采用锁定 legacy code 的 indoor 全幅/outdoor top190，与论文中心 256×256 表述不同，需要明确主协议与兼容性说明后固定，不能混称完全相同的官方口径。

## 光流与位姿的新增证据

光流最终 selected-best 诊断的 64 条样本再次独立加权汇总：共同 active union 的 EventGenesis EPE 0.3027907753、V2E 0.3151324278，零流 0.3520442492；改善 13.9907% / 10.4850%，像素数 27751、覆盖 0.48658%。与机器记录一致。checkpoint SHA 和选定 epoch 对应之前核查的 best 权重。本次未重新做 GPU forward，只核对实现、记录和重新汇总。该结果是小样本合成诊断，无真实域排名含义。

位姿的普通 eval、confidence argmax、仅用训练事件重估 BN buffers 与保存恢复机制在源代码和 fixture 日志中可核对。PnP 结果是两个同 rollout 训练窗口上的过拟合检查；不能升级为独立验证通过或正式 E-POSE ready。真实样本/CAD 的可核验对应仍缺失。本次未重复请求发布站点或重跑 CUDA pose fixture，数据访问状态按执行会话带时间的原始请求证据记录。

## 版本追溯

WSL 与 GitHub 分支指向 61554fb；g1 worktree 仍为 b4f671b 并有未提交文件，用户转述属实。重建 trainer、10 ms 数据生成器、final-best 诊断代码的 g1 内容 SHA256 与已提交 WSL 文件一致，trainer checkpoint 记录的 hash 也吻合。因此旧 HEAD 本身不能证明运行用了旧 trainer。

g1 没有 MVSEC 新评价脚本，位姿 adapter 与 WSL 内容也不同，不能将三线统一声称为“服务器已完全同步至 61554fb”。位姿 fixture 在 WSL 运行，这种差异未证明影响重建。建议用单独、干净且可记录依赖哈希的执行 checkout 同步新版本，避免重置共享工作区。

## 下一步交接

1. 重建：保留本轮完整权重/日志，标明 TC 坐标问题；在 benchmark compatibility 层修复 TC 的两处采样约定，不改冻结核心。先通过零流恒等、已知位移/子像素、非方形尺寸和真实模型 TC 反传检查，再从相同登记随机初始化进行两组一致的新版本训练。不要只重算验证，也不要把旧权重直接续训混为同一正式结果。门禁通过后按已有授权继续，首批 optimizer 更新和恢复证据及时交付；修复未验证前不扩展重建其他 seeds。
2. 光流：现有训练保留；修复 MVSEC schema 和完整合成 artifact 路径，补齐实际测试清单，明确裁剪和汇总，供研究协议核验。保持真实分数关闭。
3. 位姿：继续原始样本/CAD获取和独立合成 rollout 的普通 eval 检查；不以双窗口过拟合结论启动正式 E-POSE 训练。两项修复不阻塞位姿工作。
4. 三线分别更新状态，区分已完成训练、数学有效性验收与真实评分。对 bug 修复建立可复用的关键数学不变量检查，不用单纯“exit 0／梯度有限”代替。

本目录为新增研究审查材料，未提交 Git。只包含审查脚本、证据和说明，未修改训练实现或真实评分权限。
