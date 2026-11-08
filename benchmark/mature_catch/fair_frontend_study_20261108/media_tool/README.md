# 盲测真实录像的代表回合与对照动图

`make_blind_demo.py` 分两步工作：先从全部完成的盲测结果生成可下载的实录清单，真实文件下载并完成远端哈希复核后，才制作并排 MP4、GIF、预览 PNG 和来源账本。不会连接 SSH，不推进物理，不重渲场景，不用生成图片代替实验录像。

## 固定选择规则

每个输入帧率独立看五种类别：仅 Ours 成功、仅 V2E 成功、双方成功、双方失败、含初始化失败。每类只选 `EVALUATION_PROTOCOL.json` 中 `block_index` 最小的一对；同一对命中多类时合并，四档最多 20 对。没有某类时明确记为 0，不用别的类别填满。始终同时展示两来源，因此 V2E 有利例会保留。代表录像用于说明行为，完整 320 回合统计才用于估计成功率。

所有 plan 只接受完整 blind collector：结果/完成凭据/CSV 一致，32 个配对 seed、四档两来源及各一次的两种参考齐全；参考不进入本代表对照视频。来源各自锁定的前端必须与盲测记录和实际 actor protocol 一致。主标题固定为“盲测代表回合｜固定7°原权重”。

## 实录、时间与失败的处理

- 正常执行用 `actor/episode_000/genesis.mp4`，同时下载完整 `trace.json`、`result.json` 与 actor 的 `protocol.json`。
- 初始化失败只用真实的 `migration_audit/sensor_episode_000/initial_rgb.png`，并显示“0 物理步”。如果该实录文件确实缺失，应取得另一份真实初始化截图后明确修订清单，不能生成或借用其他回合图像。
- 校验原录像 25 fps，帧数必须等于 reset 帧 + 每个控制步一帧。按 trace 的实际时间显示，不能自动拉伸播放速度。原控制间隔须为 40 ms。
- 两边完整回合从 reset 同时播放，短的一边结束后保留最终画面，并标明“已结束；之后没有新动作”。没有按结果裁切，也不把定格当作继续成功保持。双方初始化失败时仅将真实截图静置两秒，明确 0 步。
- MP4 保留每一原始录像帧。GIF 为约 12.5 fps 的缩略展示，覆盖同一完整时间范围，包含最后一帧。未加工原录像保留在 assets 里。
- 每个面板显示事件来源、锁定前端、输入帧率、严格判据结果和模拟时间。原权重 SHA、7°机位、来源、实际步骤和评分都再次与 actor 文件核对。

## 产物

- `DEMO_DOWNLOAD_PLAN.json`：可移植主清单，包含固定选择的全部分类决策、每对结果、来源远端绝对路径和本地相对路径。
- `PLAN_SHA256.json`：主清单哈希；改动清单后不能悄悄继续渲染。
- `DOWNLOAD_FETCH_PLAN.json`：兼容项目现有下载工具的 remote/local 列表。本文件含当时的本地绝对路径；搬目录后可 `rebase-fetch` 重新导出。
- 渲染目录：每对一个 MP4/GIF/预览 PNG，`DEMO_PROVENANCE.json` 记录真实输入与输出哈希、完整步骤和原录像帧数、时间范围及选择规则。

## 运行

计划阶段只需要 Python 标准库；渲染用项目现有 genesis Python 环境中的 Pillow、NumPy、imageio、imageio-ffmpeg。字体默认为 Windows 微软雅黑。下载是父会话已有流程的单独动作，不由本工具执行。

```powershell
$py = 'C:/Users/zhaoyuhan/miniconda3/envs/genesis/python.exe'
$tool = 'C:/Users/zhaoyuhan/Documents/ChatGPT/New project/robot_closed_loop_reuse_20261107/fair_frontend_blind_20261108/media_tool/make_blind_demo.py'

# 仅完整盲测收尾后生成实录清单，不制作虚构 demo
& $py -X utf8 $tool plan --stage '完整盲测collector目录' --selection 'FRONTEND_SELECTION.json路径' --output '新的媒体清单目录'

# 父会话使用现有下载工具取得 DOWNLOAD_FETCH_PLAN.json 中的真实文件。
# 完成后应有 DOWNLOAD_FETCH_PLAN.receipt.json，包含远端大小与 SHA256。
# 先只验原片、逐帧时间、来源和判据，不写视频
& $py -X utf8 $tool render --plan '媒体清单目录/DEMO_DOWNLOAD_PLAN.json' --output '新视频目录' --check-only

# 然后制作完整并排实录与动图
& $py -X utf8 $tool render --plan '媒体清单目录/DEMO_DOWNLOAD_PLAN.json' --output '新视频目录'

# 清单搬家后重新导出本地绝对下载目标，不变更科学选择
& $py -X utf8 $tool rebase-fetch --plan '媒体清单目录/DEMO_DOWNLOAD_PLAN.json' --assets-root '新的媒体清单目录' --output '新的媒体清单目录/DOWNLOAD_FETCH_PLAN.json'
```

父会话已实测 g1/g3 的 `/home` 为同一共享 NFS，本轮录像可以统一由 g1 已有 relay 下载。普通 plan 中不虚构额外 host 字段；若以后换成非共享存储，须按实际任务归属路由。

缺完整结果、原片、真实截图或远端哈希时返回 `NOT_READY`，不产占位演示。测试 `test_blind_demo.py` 只验证选择、去重、路径、时间与文件契约，不保存伪造场景演示。
