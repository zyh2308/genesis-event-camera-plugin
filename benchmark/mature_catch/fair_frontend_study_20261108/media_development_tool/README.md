# 开发选型完整16回合 Demo

此工具独立于已封版的盲测媒体工具。输入为全部48回合开发交叉验证及其已锁定前端，从中固定使用 Ours `w10_fit0_gap100`、V2E `w0_fit1_gap100`，完整展示四个开发 seed 的25/12.5 Hz两档：共16回合，不按输赢挑选。原48回合仍全部保留。

标题为“前端对照｜开发样例（用于选型）”。一个来球占一个2×2画面：两列Ours/V2E，两行25/12.5Hz。四个来球按seed顺序完整衔接为MP4与GIF，另输出每个来球的一张预览图。面板显示来源、前端、帧率、严格判据结果和真实模拟时间。提前结束的一格保留末帧并标注定格；初始化失败只显示真实截图并标0步。

实际本批是Ours8/8、V2E6/8；这是用于选型的开发样例，不能当独立盲测或正式泛化结果。原权重、冻结仿真器、7°机位及全部原始结果不变。

已生成的真实下载计划位于 `../delivery_local/development_media/DOWNLOAD_FETCH_PLAN.json`，共64文件：每回合的实际genesis.mp4、trace、result与actor protocol。父会话通过既有g1 relay/fetch_many下载，共享NFS已确认。本工具不执行SSH。

```powershell
$py='C:/Users/zhaoyuhan/miniconda3/envs/genesis/python.exe'
$tool='C:/Users/zhaoyuhan/Documents/ChatGPT/New project/robot_closed_loop_reuse_20261107/fair_frontend_blind_20261108/media_development_tool/make_development_demo.py'

# plan阶段已执行；如需新建另一份计划，必须给新目录
& $py -X utf8 $tool plan --stage '完整frontend_validation目录' --selection 'FRONTEND_SELECTION.json路径' --output '新媒体计划目录'

# fetch_many生成DOWNLOAD_FETCH_PLAN.receipt.json并验证远端哈希后
& $py -X utf8 $tool render --plan '媒体计划目录/DEVELOPMENT_DEMO_PLAN.json' --output '新demo目录' --check-only
& $py -X utf8 $tool render --plan '媒体计划目录/DEVELOPMENT_DEMO_PLAN.json' --output '新demo目录'
```

所有真实片段先验证下载哈希、reset+每步帧数、25fps与40ms真实控制时间、7°机位、原权重和锁定前端，然后再写视频。输出的DEVELOPMENT_DEMO_PROVENANCE.json记录全部16来源、判据、步骤、帧数、原片和成品哈希。只加工真实实录，不生成场景图像，不按结果裁切。
