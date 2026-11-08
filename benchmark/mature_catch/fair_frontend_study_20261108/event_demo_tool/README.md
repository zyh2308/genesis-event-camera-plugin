# 真实事件输入与接取动作的开发单例

固定第一个开发seed `2026110841`、12.5Hz、Ours已锁定B前端与V2E已锁定A前端，不再按结果挑回合。上排显示两条完整Genesis实际场景录像，下排显示各自真实raw events。标题为“事件输入与实际接取｜开发示例”，不能以此单例声称独立泛化。

既有真实场景录像、trace、result和protocol复用 `delivery_local/development_media` 中已验证的原片。额外只需双方的 `raw_events.npz` 与 `packet_audit.json`，共4个文件；不下载HDR大缓存。`EXTRA_DOWNLOAD_FETCH_PLAN.json` 中expected_sha256来自完成阶段的真实SHA256SUMS，不使用历史OOM结果。

每个视频控制时刻只选择 `parent.available_s <= 当前真实模拟时间` 的最新包，按该包结束时刻的末尾40ms事件窗绘图。两来源正常活动期间使用同一绝对时间窗；包间和停止采集后保留原事件图，并显示其包时间、到达时间及旧窗口，不伪造持续事件更新。提前结束的场景录像也明确标为定格。全片保持原25fps及每一原视频帧，GIF覆盖同一完整时长。

两来源都用ON青、OFF红，每像素3事件饱和，不分别归一化亮度。事件仅来自已到达父包的原始数组，没有GT框、GT掩码、新物理画面或新HDR参考。上排是旁观相机场景实录，下排是事件传感器坐标，两者视角不同，字幕会明确说明。

```powershell
$py='C:/Users/zhaoyuhan/miniconda3/envs/genesis/python.exe'
$tool='C:/Users/zhaoyuhan/Documents/ChatGPT/New project/robot_closed_loop_reuse_20261107/fair_frontend_blind_20261108/event_demo_tool/make_event_input_demo.py'
# 真实额外下载plan已在 delivery_local/event_input_pair_media 下生成。
# 父会话用现有g1 fetch_many取得4个文件，保留其receipt。
& $py -X utf8 $tool render --plan '媒体计划目录/EVENT_DEMO_PLAN.json' --output '新演示目录' --check-only
& $py -X utf8 $tool render --plan '媒体计划目录/EVENT_DEMO_PLAN.json' --output '新演示目录'
```

成品包括MP4、GIF、固定时刻预览、逐帧事件可用性账本以及完整输入/输出哈希。脚本不运行SSH、物理、训练或重渲，不修改先前交付视频。
