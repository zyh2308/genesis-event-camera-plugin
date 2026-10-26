# Genesis 高纹理光流监督拟合：阶段状态

更新时间：2026-10-26 13:48 JST（04:48 UTC）  
分支：`exp/public-sim2real-v1`，运行数据与权重仍在 g1。冻结核心没有修改；没有读取真实测试分数。

## 当前结论

**Genesis 高纹理样本上的 STE-FlowNet GT 监督训练还没有启动。** 已完成的是共享 bank 的 CPU 预检：固定样本、标签/图像方向线索、事件活跃覆盖和独立验证 rollout 的数据支持统计。最近一次 g1 GPU 快照为 04:47 UTC，六张卡利用率 65–94%；没有启动训练，也没有抢占 GPU。

已有的 200-update GT 拟合来自旧 Stage-4 `s2_00/dt1_144` 单窗，不是本次 Genesis 高纹理 bank，不能当成本任务的拟合或泛化结果。

## 固定样本与方向检查

候选数据使用 g1 八条 rollout bank：
`/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-stage5-flow-coverage-v3-eight-rollouts/`，SHA256 `2f0db064a1fdf90417f31e345e99813bcdb036759cf2dbbdce344e9a7700352c`。

这套 bank 是项目自建的棋盘格/条纹地面和运动物体，不是从公共真实场景数据集导入的室内布局。本次结果只能核验现有 Genesis 物理样本的监督接口与可观测性，不能回应自然室内场景的覆盖问题。

初始固定在 `s5q_02_train/dt1_990`（0.99–1.02 秒）的中点窗。运动纹理只有 16 个有效像素，其中 EventGenesis 活跃的仅 1 个；因此没有用它开训。这个未通过支持检查的结果保留在 g1 的 `20261026-genesis-hightexture-prefit-audit-v1`。

随后冻结一个不依赖任何生成器的训练样本选择规则：在登记的 `s5q_02_train` 66 个窗中，选取“共享有效区内 GT 位移至少 0.5 px/30ms 且图像 Sobel 纹理梯度大于 1/255”像素数最高的窗，平局取最早窗。选择脚本只读取物理 GT、共享图像和 split，没有打开事件数据。它选中了 `s5q_02_train/dt1_000`（0.00–0.03 秒），GT 运动纹理交集为 32 像素；其中 7 个像素有 EventGenesis 事件。两者输入同一标签，不能按事件数再换窗。

方向/warp 证据目前是**有支持但还不充分**：

- 对最初中点窗，真值正向 warp 的光度残差为 0.00210，低于零流 0.00318 和反向流 0.00328；9 个 LK 角点的中位误差也更接近正向标签（0.0098 px，对反向 0.0206 px）。但这些角点很少。
- 最终预选窗 `dt1_000` 上，LK 没有找到可用的目标角点；在 32 个运动纹理像素上，正向真值 warp 并未优于零流（0.00244 对 0.00204），反向更差（0.00308）。这是小区域上的不确定结果，不能据此判定标签翻转或实现缺陷。
- 同一窗有效 GT 中有 910 个像素位移至少 0.5 px，但运动纹理交集仅 32 个；GT 位移中位数为 0，P90 为 0.0062 px，P99 为 0.965 px。可见可观测运动仍集中在少量像素。
- 独立解析平移 fixture 和旧单 rollout 双生成器审计验证了 warp 代码的数学方向约定；它们不能替代对这八条 bank 每个标签的独立验证。

## 独立 rollout 与事件支持

`s5q_02_val` 是登记的独立物理 rollout。对它的 66 个窗口都做了 EventGenesis 事件/GT/图像支持统计：有效 GT 区中事件活跃像素平均覆盖 0.976%；GT 运动纹理平均 210.3 像素，其中同时有事件的平均 102.4 像素，66/66 窗非空。这是**输入/标签可观测性统计**，不是模型的独立 rollout 结果。

八 rollout bank 目前只有 EventGenesis raw events。已找到的 V2E 文件属于旧的单 rollout smoke，绑定的是另一份物理 bank，因此没有拿来拼接或评估。仍需针对本 bank 生成并登记 V2E 的 `s5q_02_train` 与 `s5q_02_val`，之后才能做公平配对拟合和独立验证。

## 运行记录与复现

- 初始固定窗审计：g1 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261026-genesis-hightexture-prefit-audit-v1/`，UTC 04:42:58–04:43:15，exit 0；JSON SHA256 `0355a0d88788507617ca731ef7059e3725355cc3f1e0d0729b4783b6467434f6`。
- 物理信号选窗：g1 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261026-genesis-hightexture-fit-window-selection-v2-attempt02/`，UTC 04:46:12–04:46:17，exit 0；选择 JSON SHA256 `7aae5137ff01d0302a660bb8de2830a65fa20ee7082c904a27a619c528c1ac4f`。第一次部署的 `attempt01` 因远端脚本错误命名为 `select.py`、遮蔽 Python 标准库而退出 1；完整现场保留，第二次仅改文件名后通过。
- 预选训练窗及独立 rollout 审计：g1 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261026-genesis-hightexture-prefit-audit-v2/`，UTC 04:47:09–04:47:19，exit 0；JSON SHA256 `e2a1bb670321de0028a87d7248e30e01b19b5303e37f0d4cbf1a3ee624d67a87`。
- 运行环境：Python 3.11，NumPy 2.4.6、h5py 3.14.0、OpenCV 5.0.0。审计仅用 CPU；本轮未加载 STE 模型，未执行 optimizer 更新。

按冻结选窗配置执行：

```bash
python benchmark/sim2real/scripts/select_stage5_genesis_gt_fit_window.py \
  --bank-root /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-stage5-flow-coverage-v3-eight-rollouts \
  --config benchmark/sim2real/configs/stage5_genesis_gt_fit_window_selection_v2.json \
  --output /tmp/stage5_genesis_fit_sample_selection.json

python benchmark/sim2real/scripts/audit_stage5_genesis_high_texture_gt_fit.py \
  --bank-root /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-stage5-flow-coverage-v3-eight-rollouts \
  --config benchmark/sim2real/configs/stage5_genesis_high_texture_gt_fit_v2.json \
  --output /tmp/stage5_genesis_hightexture_audit.json
```

## 下一步

先等到一张经资源协调、确认空闲的 GPU；再用 pinned V2E 对**同一 bank manifest**生成 train/validation rollout 的事件，并验证事件时间窗、标签 hash 和共享掩码。之后才运行同初始化、同 200-update 短预算的配对 STE-FlowNet GT 拟合，并在完整 `s5q_02_val` rollout 做普通 `eval()` 推理。若真实配对覆盖仍很稀，先记录数据/场景限制，不把单窗拟合写成模型或生成器结论。
