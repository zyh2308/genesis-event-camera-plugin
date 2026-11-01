# 结论更正：高频 HDR 仅是阳性对照，不能计作稀疏重建修复

此前 candidate 的输入绕过了原稀疏几何重建。下列像素结果保留，但此前的“纠错完成／已验收”结论撤回。原稀疏路径尚需按同预算重训后的像素定位误差验收。

# R1.5 pixel-objective localization recovery

[Results and reproduction](../../benchmark/robot_closed_loop/results/r1_5_pixel_objective_repair_20261101a/README.md).

Primary 581 queries: old median/P95 8.18/106.72px, candidate 1.72/4.03px. Secondary 193 queries: 9.08/112.38px to 1.82/4.24px. Frozen checkpoints; secondary seeds disjoint from fit and primary bank. Dense actual-HDR observation path only; sparse warp and closed-loop success remain separate.
