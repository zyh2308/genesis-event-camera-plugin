# Stage 4 seed 0 首次训练交付独立核验

日期：2026-10-17（Asia/Tokyo）。被审查提交：`1895a154bb3c193cca0e64e8d7a9c5e8650cbe54`。实际训练代码提交：`1fab7e17e201943c77703aa8a1c0c6eaefbb8f5a`。

**结论：两组 seed 0 光流的 update-20 开训交付点通过，45 epoch 作业确实仍在继续。共享数据、配对初始化、采样顺序、真实 optimizer 更新和 checkpoint 文件均有独立证据。可继续当前作业；不需要因审查重训或中断。另有一个尚未被原恢复探针覆盖的 `--resume` RNG 设备问题，首次实际断点续训前应修复。**

本次只进行文件/进程/代码审查、哈希、CPU checkpoint 加载和 CPU 数据检查，没有分配 GPU、重跑恢复前向、停止或修改训练进程。

## 1. 当前进展：比提交快照更新

| 核查时间（日本时间） | EventGenesis | V2E |
|---|---:|---:|
| 2026-10-17 01:51:39 左右 | 228 / 8910 | 232 / 8910 |
| 2026-10-17 01:55:39 左右 | 298 / 8910 | 302 / 8910 |

两组均进入第 2 epoch。第一轮各有 198 个 optimizer updates、1,584 个不重复训练 sample IDs，没有 validation rollout 混入训练；随后各完成对 8 个留出 rollout 的 synthetic validation，指标有限，`synthetic_best.pt` 已存在。

01:51 检查时 tmux 目标 session 存在，两个 Python PID 为 2513255 / 2513256，启动参数与 run root/seed/generator 对应；GPU2/4 均为 V100S 32 GiB，利用率为 100%，显存约 25,678 MiB/卡。CPU 审查不分配 GPU。第二次读数仍有更新增长，支持作业在持续推进，而非只有一个滞留 PID。

GitHub / WSL / g1 HEAD 均为 1895a15。g1 的 tracked 修改只有实时状态文件。运行代码从 1fab7e1 到当前 HEAD 没有改变 trainer 或 objective，后续只变动状态监控与证据/报告；不能把 GitHub 58-update 快照当成实时停滞。

## 2. update-20 文件与优化器状态

两份不可变 checkpoint 均为 580,038,432 bytes，研究会话独立复算 SHA256：

- EventGenesis：`f7eb58f5dae4b914050a38c5c68596e23a64372f68584fa5c1ba00e3c14c63ec`。
- V2E：`ce2b5b076b9ba6df9af525a6ca478c8f3d0b20a37e7783de433986619d7cc763`。

先核对准确哈希，再用 CPU `weights_only=True` 和有限 NumPy RNG 类型 allowlist 加载。两组参数均有限，70 个 Adam states 的 step **全部为 20**；checkpoint 中 optimizer_update=20、sampler_position=160、batch=8、seed=0，config/data manifest hash 与实际文件一致。

checkpoint 初始模型 hash 均为 `acc6bbd11488a21868c5f147c7f397521480166c24914cc7988a6428d6ed71cf`。当前模型 state hash 与 checkpoint 自带 metadata 一致。Adam lr=4e-4、betas=(0.9,0.999)、weight decay 分组 0 / 4e-4，scheduler milestones=[5,10,20]、gamma=0.7、update-20 时 epoch 状态为 0，均符合登记。

两份入仓 first-20 JSONL 的 SHA256 与说明一致，每份准确包含连续更新 1–20，逐字段等于 g1 原日志前缀，loss/梯度有限、batch8、70 个梯度张量。更大范围的完整当前日志也没有非有限 loss/梯度或 update 缺口。

原训练进程的 GPU 恢复 probe 分别记录误差约 7.15e-7 / 6.71e-8，代码阈值为 1e-6，验证了 live/restored 模型预测相近。该数值来自执行进程日志与探针代码，本次未独立重跑 GPU 前向。**它验证模型/optimizer/scheduler 装载，不等于完整 `--resume` 路径已验收。**

## 3. 数据及公平性

- 完整 shared HDF5 为 1,868,960,874 bytes，独立 SHA256 与 `e9e57131…72ee337a` 完全一致；split hash 亦一致。
- 32 条 rollout、24 train / 8 validation，32 个不同 scene_seed；每条 201 端点、约 2 s。
- 64 个 adapter NPZ 全部复算哈希并匹配。
- 对 32 对 rollout 逐元素核对 gray_prev/next、flow_gt、flow_valid、起止时间、sample IDs、K：两组全部一致。
- 适配后的 flow/mask 与 shared bank 完全相同；事件窗与物理端点完全对应，66 个 30 ms 窗口覆盖 0–1.98 s；末 20 ms 为预定尾段，非按 generator 事件密度筛选。
- 独立比较截至第二次读取时共同的 **298 个 optimizer updates**，两组 sample ID 列表和顺序完全相同。augmentation 使用相同 seed/epoch/batch/micro 索引生成，因此执行次序一致。
- 第一轮训练每组覆盖所有 1,584 个登记窗口各一次，没有继续沿用旧 smoke 的“每条只取头尾两个样本”。

训练入口事件进入模型，公共合成灰度端点构成 photometric loss。代码虽然联合变换 flow/mask，但没有把 flow 用作模型输入或 optimizer 监督；validation 才用 flow EPE。没有发现新增真实数据路径或真实分数反馈。

722 条 repository artifact index 全部匹配。锁定 STE 源码 14 个文件全部匹配；trainer/objective 的 g1 与 WSL 哈希一致；frozen simulator core 相对 e79e656 无改动。

## 4. 运行后端与失败修复的解释

实际 generator 后端由 CPU NumPy 改为现有 Torch CUDA，这是配置变化，应明确保留 `STAGE4_TORCH_CUDA_RUNTIME_AMENDMENT.md`，不能说与原配置逐字节相同或事件 bitwise 相同。

本次核对确认，训练预算、模型、优化器、初始化、split、传感器数值及 V2E 25 μs 设置未改变。变化为 backend/device、bank 版本名和修订 metadata。修订记录发生在本轮数据生成/训练之前，没有真实分数选择路径。记录的三处数值对照覆盖投影、深度、warp 与 mask；它是有限 numerical diagnostic，不是完整事件输出等价证明。

研究侧接受其作为本轮 synthetic-only 运行配置的工程修订，继续使用已经完成的 bank；不要求为 NumPy 逐事件一致性重生成语料。正式报告应明确实际 Torch 后端及数值差异。

V2E shard 路径、finalizer/adapter 路径、灰度 CPU/shape 适配等失败均有保留记录。objective 修复只把灰度参考转换为官方 loss 要求的 CPU `[B,H,W]`，flow 前向/梯度仍在 GPU。研究会话独立执行该灰度 helper：数据值不变、形状正确、只禁用参考目标的梯度；官方 Charbonnier、warp、smoothness 配方未被替换。两组均从同一 seed 重新开始，不是一组沿用失败后的权重。

## 5. [P2] 首次断点重启前修复 RNG 恢复设备

位置：`scripts/train_steflow_stage4.py:206–214`。

当前 `torch.load(..., map_location=device)` 会把 checkpoint 中 CPU `torch_rng` ByteTensor 也映射到 CUDA，随后调用 `torch.set_rng_state(saved['torch_rng'])`。安装版本的该 API 明确仅接收 CPU RNG state。研究会话在独立 CPU 进程中用非 CPU tensor 验证其会拒绝，结合读取路径可确认这里存在设备契约不匹配。

**影响范围：未执行的完整 `--resume` 路径，非当前不中断运行。** update-20 的恢复探针没有调用 RNG 恢复，所以不会发现这个问题；不能把其成功扩大成“整套断点续训已验证”。

建议按独立提交修复：先 CPU 加载 checkpoint，让模型/optimizer state 正确迁移到训练设备；Torch CPU RNG 及 CUDA RNG state tensors 保持各恢复 API 需要的 CPU 格式。做隔离恢复验证，检查 next sample ID / augmentation / optimizer step / scheduler，而不覆盖原日志或复用正在训练的 run 目录。

另应为 epoch 完成后的 checkpoint 明确“下一个 epoch / 下一个 sampler position”的语义；`synthetic_best.pt` 当前保存完成 epoch 的索引且 position=0，不应直接据此重复该 epoch。它用于选权重没有问题，若允许从它续训则需统一下一步位置。

保持现有两个作业运行。不要为验证恢复而重启它们，也不要因修改恢复入口在报告中追溯改变当前 run 的执行代码提交。

## 6. 下一交付点

当前接受状态为“光流 dt1、seed0 已开训并完成一次 train→synthetic-validation→best checkpoint”，不是完成 45 epochs 或三个任务全部开始。

继续现有配对训练；后续读取每轮合成验证及 final/best checkpoint，完成后按既定规则排队其余种子。重建的实际启动需另给 PID、optimizer 更新和 checkpoint 证据，本次反馈尚不支持其已开训。不得据训练早期 loss 或已暴露 MVSEC 片段选择 generator 参数、训练时长或模型。

## 7. 本次独立材料

- `remote_audit_stdout.json`：01:51 的进程/GPU、CPU checkpoint 装载、元数据与 manifest。
- `data_audit_stdout.json`：01:55 的实际进度、完整 bank/64 NPZ 哈希、32 对数组、完整首轮采样及合成验证。
- `local_verification.json`：722 索引、first-20 原日志对照、灰度 helper 独立检查。
- 同目录 `audit_*.py`、`remote_*program.py`：核验方法。

报告另存入共享 WSL 仓库 `benchmark/sim2real/reviews/stage4_1895a15/`。本次只增加审查材料，不修改训练实现，不向 g1 写文件，不暂停作业；新增审查材料尚未提交或推送。
