# 重建线主线只读审计

日期：2026年11月5日。范围：仓库与本地campaign资料；没有SSH、GPU操作或实际真实评分。服务器最新训练步数应由主会话远端收据补齐。

## 新旧结果不能混用

当前新版是Stage9，冻结core `62da4f35ce3c5c96f04268bd1687a27657af9238`，5种管线 × seed0/1/2，共15个重建正式run。`sim2real_status.md`及`TRAINING_LIVE_STATUS.md`顶部仍是11月3部署快照，不代表当前训练进度。

旧Stage5的EventGenesis/V2E seed0均已完成12000更新，并有HQF14全序列15498帧评分与图表。它们属于旧bank/core/训练兼容层，不能当成本轮Stage9结果，也不能把新旧差异解释成单独simulator core消融。

## 新版已注册的正式规格

- 模型：E2VIDRecurrent，5 signed temporal bins，3 encoders、base32、2 residual blocks、ConvLSTM、sum skip、upsample conv、无final activation、无norm。
- 每run 200 epoch / 12000 optimizer updates；每epoch60 updates。
- 40-frame train clip，112×112 crop，有效batch2 clips、microbatch1。
- Adam lr1e-4、AMSGrad、constant schedule；LPIPS-VGG权重1，TC权重2，L0=2。
- 24 train / 8 validation独立rollout；120 train clips，synthetic validation整段200帧，160×160中心裁剪。
- 每5epoch及最终验证；以synthetic validation LPIPS-VGG最低选择synthetic_best，禁止real选点。
- 每正式run先20次真实更新、独立进程恢复到21并核optimizer、样本、初始化，然后完成12000预算。
- 10ms flow直接从共享物理状态生成，不使用30ms标签除3。

依据：`benchmark/sim2real/campaigns/newcore_20261103/training_recipe.json`、`train_e2vid_stage9.py`。

## 仓库可证明的最近训练状态

11月4恢复README记录EventGenesis seed0在g1 V100S已经正式完成；其余seed0重建曾分流到g3 RTX4090，后续部分seed1/2也移交，另3个seed1回g1 GPU5。此历史分流记录不能证明现在仍运行，亦不构成当前访问g3许可。

主会话应从g1共享根读取：

`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/training/reconstruction/seed{0,1,2}/{eventgenesis,v2e,vid2e,film_esim,rife_esim}/`

逐run检查`completion.json`、`live_status.json`、`training_metrics.jsonl`末行、`synthetic_best.json/pt`、`last.pt`、wrapper退出码与恢复门禁。完成资格须同时核task/method/seed、12000预算、recipe/init hash、selected checkpoint及SHA，不能只看文件名或状态快照。

g3当前AGENTS仍禁用，不应SSH探测。g1可读共享NFS文件，但历史挂载元数据缓存可能延迟，无法确认的g3进程标“待确认”。

## 真实重建测试的注册角色

HQF全14序列是已记录研究决策的主测试；HS-ERGB是extension，尚无完整独立时序/评分合同，不阻塞HQF主线。此决定已在`tasks/RECON_FLOW_ACTION_20261019.md`记录，不能把旧HQF结果说成HS-ERGB已经跑通。

HQF此前已被评分和查看，Stage9复测属于已见benchmark，不是全新盲测。原合同：事件输入，native180×240、固定pad/crop；每序列开始重置一次循环状态；首帧仅作时间边界，不当网络图像输入；使用(前一图像时刻,当前图像时刻]的events；全14序列均等权。保留engineering_posters一个错误时间戳参考帧的事前排除规则，事件流不变。原始输出只clip一次[0,1]，不逐帧minmax或拟合real亮度。

最终主指标LPIPS-Alex，辅SSIM/PSNR/MSE；LPIPS-VGG只作合成选点。14序列每seed原始值保留，最终3seed均值、样本SD和按序列配对bootstrap。

## 明确存在的Stage9真实评测工程缺口

旧`run_hqf14_reconstruction_eval.py`及`score_hqf14_reconstruction.py`硬编码两组`EXPECTED_ARMS=(eventgenesis,v2e)`，默认冻结protocol绑定10月19 Stage5的两个旧checkpoint。它们不能直接用于Stage9五管线×三seed；不可直接改旧protocol或复用旧授权SHA。

Stage9恢复目录已有：

- `recovery_20261103/freeze_evaluation_checkpoints.py`：要求全部15run成功、预算/recipe/初始hash一致、selected cp实际SHA正确，然后写`reconstruction_frozen_checkpoints.json`，但scoring_authorized仍False。
- `real_evaluation_contract.json`：只有Stage9最终评价摘要，未给新15权重的HQF完整绑定；score_launch_enabled=False。
- `aggregate_paired_real_scores.py`：已有五method×三seed×每sequence的严格聚合，缺任何单元格会拒绝完整正式汇总。

有利条件：Stage9与旧评价的`reconstruction.model_args`字典逐字段完全相同，`synthetic_best.pt`保存模型在`model`键。旧`infer_hqf_e2vid_candidate.py`支持从注册model-config提供model_args并严格载入model，基本推理模块可复用，不必重写。

可并行补一个独立Stage9 HQF评价桥接：沿用旧数据/时间/metric合同，使用新权重列表与真实method/seed身份，冻结新的协议和源码哈希；旧协议和旧分数保留。先fake/synthetic fixture测试manifest校验、错误权重拒绝、5×3覆盖、时序和输出形状，再在确认权重/协议锁定后进行真实评分。

## 还缺什么

训练：由远端实时核对剩余的15个正式run；已完成run不得为了换效率版本重启。

真实测试：Stage9最终权重冻结、新HQF15run协议绑定、实际210个sequence-run推理及评分。HS-ERGB仍是另行扩展，缺原始数据完整性、目标时刻、独立协议与评价入口。

正式结果图表：当前可直接使用的图只属于旧Stage5，Stage9需重新输出5method×3seed的每序列LPIPS/SSIM/PSNR/MSE、各seed表、均值±SD主表、相对EventGenesis配对差/区间图、合成训练/验证曲线。若首轮seed0先完成，可输出清楚标注单seed的首轮表，不冒充三seed结果。

高效执行建议：训练队列继续；CPU同时实现新HQF绑定及汇总脚手架；已有checkpoint只用于合成/协议检查，协议锁定后再真实评测，不边看分数边改处理。
