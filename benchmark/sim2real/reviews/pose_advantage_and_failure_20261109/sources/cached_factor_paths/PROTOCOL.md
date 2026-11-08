GT 因子路径诊断；不能部署、不能作为候选模型或生成器排名。

固定 C1、两生成器、三 seed、三光照、四物理 rollout 的全部 1,152 个窗口。四格为预测/GT 中心 × 预测/GT anchor 深度；分别固定原 R，或在 GT 中心格使用原先保存的 `R_crop_gt_center`。裁剪 size 始终为该窗口原预测 size。不执行模型前向，不重新裁图。

原几何工具、PD 原始预测与资产、PG/PCR 完成收据和宏结果均锁定 SHA，开始/结束再次核验。七格复用既有缓存宏结果（126 cell）；只新增 GT 中心、预测深度、已保存 GT 中心 crop R 的 1,152 次 CPU 几何评分。两路径的原始控制共 36 cell 对原 18 个 native C1 摘要逐项校验。

ADD-S 使用冻结几何工具的 CAD 最近邻算法；成功门槛严格小于 0.1×直径；完整窗分母、逐 rollout 等权，再三 seed 平均和 sample SD。输出六个 method/light 宽表行、72 个 method/seed/light/rollout 宽表行。范围是 matched64，不能与 native clean384 混用。

这些条件性替换说明中心同时经平移射线和 R 裁剪路径影响完整位姿。由于预测因素相互依赖，不能把两路径相减视为唯一因果分解。

CLI（CPU，CUDA_VISIBLE_DEVICES=''）：
```
python -B factor_path.py --export-root RUNS/20261108-pose-factor-diagnostic-v1/export --geometry-root RUNS/20261108-pose-factor-geometry-v1/analysis --geometry-sources RUNS/20261108-pose-factor-geometry-v1/sources --crop-root RUNS/20261108-pose-rotation-crop-diagnostic-v1/results --out NEW_OUTPUT_DIRECTORY
```
实际本地 CPU 三项测试通过。它们验证非零旋转 anchor 的 origin 平移重算与 R 裁剪路径、同 ID 标签错误拒绝、非法旋转拒绝；不声称已执行服务器完整库存。
