# R1.5 在线 HDR 与缓存事件回放诊断

## 结论

在同一 seed 重放的 81 个时间点上，在线 Direct-HDR 与缓存重渲染帧逐像素完全相同。因此，这个窗口内的大幅事件差异不是图像帧不同造成的。

差异来自窗口开始时 DVS 比较器状态被重置。缓存回放只从 `0.5505 s` 开始，未继承此前事件造成的比较器参考电平。冷启动缓存回放产生 5899 个事件；按在线路径相同的 `_hdr_to_log_gray` 转换，并从发射帧和此前 324885 个事件重建比较器后，重放得到 6179/6179 个相同事件，事件时间完全相同。冷启动的事件数和定位结果不能代表同一在线输入。

正确回放没有修好旧 Ours checkpoint 的定位：7 个相邻读取时刻均有状态，平均平面位置误差为 58.69 mm、P95 为 108.33 mm；速度误差平均 16.55 mm/s。位置预测在相邻时刻明显跳动，而 GT 方块只缓慢移动。这说明模型在正确 Direct-HDR 事件上的空间定位仍然失败。现有证据尚不能区分训练事件分布问题与 checkpoint 的空间输出头/泛化问题。

另一条 `physical_log` 转换路径与在线函数只有微小数值差异（log 亮度绝对差均值 `8.12e-6`，P95 `2.76e-5`，最大 `7.80e-5`）。两种合理的比较器基线映射分别得到 6276 和 6431 个事件；两者都能匹配在线 6179 个事件中的 6175 个（99.94%），另有 101/256 个候选事件未匹配，净增 97/252 个。结果说明转换差异远小于冷启动历史造成的差异，却仍会因像素正好靠近阈值而改变少量事件；基线映射细节也值得后续统一。该对照共享在线历史事件计数，不能称为完整独立 `physical_log` 长时运行。

## 定位输出抽查

| 查询时间 | GT x | 预测 x | 平面位置误差 |
|---:|---:|---:|---:|
| 0.5855 s | 576.9 mm | 564.3 mm | 12.6 mm |
| 0.5865 s | 576.5 mm | 517.4 mm | 59.4 mm |
| 0.5875 s | 576.2 mm | 475.7 mm | 101.8 mm |
| 0.5885 s | 575.9 mm | 516.8 mm | 59.2 mm |
| 0.5895 s | 575.5 mm | 464.4 mm | 111.1 mm |
| 0.5905 s | 575.2 mm | 520.6 mm | 54.7 mm |

这些是同一 +55 mm episode 的相邻查询，不能当作独立样本或成功率统计。7/7 状态有效，问题表现为预测位置不稳，而非状态缺失。

## 复跑方法

在线 HDR capture 已在 g1 完成。原始帧、raw event、关键帧及源 episode 位于共享路径：

`/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r1_5_online_hdr_capture_20261029d/`

诊断只用 CPU，不运行 Genesis、不推进物理、不额外渲染；下面命令在 g1 的 `Eventbased_WAM` robot venv 中执行：

```bash
ROOT=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r1_5_online_hdr_capture_20261029d
SCRIPT=$ROOT/dvs_transform_state_mapped_20261029i/replay_r1_5_live_dvs_state.py
RUN_ID=rerun_20261029j
OUT=$ROOT/dvs_transform_history_seeded_${RUN_ID}/output
PYTHONPATH=/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a \
  /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python \
  "$SCRIPT" \
  --online-frames "$ROOT/online_hdr_capture/episode_000/online_hdr_frames_2khz.npz" \
  --online-events "$ROOT/source_replay/episodes/episode_000/direct_hdr_events_1khz_2khz.npz" \
  --online-snapshots "$ROOT/source_replay/episodes/episode_000/direct_hdr_key_snapshots.npz" \
  --cached-frames /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r1_5_cached_state_cadence_renderer_20261029c/artifacts/frames_msaa_on_dense2khz.npz \
  --cached-events /home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r1_5_cached_state_cadence_renderer_20261029c/artifacts/events_msaa_on_dense2khz.npz \
  --output "$OUT"
```

复跑时请将 `OUT` 改为新的、不存在的 run 子目录；脚本会拒绝覆盖已有输出。

## 版本、配置与哈希

- 分支：`exp/robot-closed-loop-v1`；最终诊断脚本提交：`dd42abb`。
- g1 源 episode runner commit：`2100645b0473c294760789bc4475b644f261d9cf`；HDR capture driver commit：`bac8463fea68dd0ae78b193190f2407e9f3ed85f`，driver SHA256 `b84c0ffc337d59e49f19e251f24c84f831de35239edc9c507d6f646929f2778b`；active launcher commit：`197e68ec69419fde0b421a2fa55acfcf3c5e5c7a`，launcher SHA256 `729cea67e984355d877870013cb70dda107542de773cc1da54fd66b8836ff031`；冻结 simulator core：`e79e656ac00f7847b98db9e12bf43b093f3afff1`。
- renderer checkout：`855b4b0bc7de96170b4bfee8d7002b370782f346`；场景配置 SHA256：`aa97f6f5a12af92561ef053407bb795bba2801582958b4ad1134bcd24497435a`。
- 在线 capture 配置 SHA256：`0609f01207ceb7679771a53296607bc3a038ba34af5f703079834dde5638c6cd`；最终诊断脚本 SHA256：`b6acac82d6878645eaa70917c3674e2cb0bd416e1a1c173f7106165f65f1958c`（commit `dd42abb`）。
- 在线 HDR 帧 SHA256：`986cbf19c6297b4d55db2d1e5ad7634f23c0c19634faa4c4e02bf2689a0e8c1c`；源 raw events SHA256：`b698817a8347ca48ad75015f0cb81c3bfc711958a60d3387d468c70456a7b2b6`；关键快照 SHA256：`d28c52a260eb82746b560e4c9435ae39b67842cc61f02377a7c6c59901e5f97b`。
- 缓存 HDR 帧 SHA256：`5ad675c7bf11b5404c8f05f1ef83bffd6f695ca084f7e6b138f75ce16df750f1`；缓存事件 SHA256：`39824192d1d9135714d4320b43807e068203ca7676c02b1f4bd44b4d07677049`。
- 最终 state-mapped 输出 summary SHA256：`39c56ddd9e2e271cd4bb4a9cc0ffce3a0cc8b190b06ba54bd07456f7aa2ef969`；逐样本 CSV SHA256：`5937404b05cc8347d1c66add29b64c318e4dff9d5236b37df4348ba95901e6de`；state-mapped physical events SHA256：`4b6a899304d6f73891a1a1cd7d78545d5255e392b8c99491226a4bf51057849f`。
- history-seeded online events SHA256：`93d17785b5c408aad543c542f9e1f37367c055f4d7ac9aa1a805e54c086761eb`；launch-baseline physical events SHA256：`6bb0f7d6defdb644102f051243e54d2df5c1fd828cebf9fa24379d1200fbf70e`。
- 远端 DVS 实现 `dvs_emulator.py` SHA256：`914dc83525ff96dc2c1a1373819de74491cfbb03007b65e8c93609f6cd1cf638`。
- 旧 Ours checkpoint SHA256：`d391fec293e2c6c50cdb267e15f742ed09e77e7b28e56a79fdf700f633e804e9`。
- 本地 `remote_data/dvs_transform_history_seeded_20261029h/output/` 与 `remote_data/dvs_transform_state_mapped_20261029i/output/` 保留逐样本 CSV、raw event 和完整 JSON；顶层 `SHA256SUMS` 覆盖本地归档文件。完整 HDR 文件留在上述共享 g1 路径，其远端哈希在 `remote_data/SHA256SUMS` 中。

早期临时对照曾把在线路径的 comparator 基线直接用于另一亮度坐标，导致窗口第一个子帧出现人为脉冲；该变体已排除，不用于表中结论。最终保留 launch 基线和相同起点残差两种状态映射，作为转换对照的敏感性范围。

## 下一步

先审计旧模型训练样本的事件窗口是否从整段 episode 连续生成，以及训练和评估是否使用完全相同的 HDR→log 函数、DVS 历史状态和时间采样。修好这条数据契约后，再用原预算做一次小重训和未参与拟合的新 episode 定位检查。当前结果不支持修改 frozen core，也不支持开始正式 Ours/V2E 排名。
